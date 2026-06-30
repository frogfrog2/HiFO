import os
import torch
from tqdm import tqdm
import numpy as np
import trimesh
import logging
from pytorch3d.loss import chamfer_distance
from utils.tensor_util import to_device, to_numpy
from utils.misc import plot_with_std
from os import path as osp
from datasets import build_dataloader, build_dataset
from models import build_model
from utils import get_env_info, get_root_logger, get_time_str
from utils.options import dict2str, parse_options

class SSMPCA:
    def __init__(self, correspondences):
        """
        Compute the SSM based on eigendecomposition.
        Args:
            correspondences:    Corresponded shapes as a torch.Tensor
        """
        self.device = correspondences.device
        self.mean = torch.mean(correspondences, dim=0)

        data_centered = correspondences - self.mean
        cov_dual = torch.matmul(data_centered, data_centered.T) / (
            data_centered.shape[0] - 1
        )

        evals, evecs = torch.linalg.eigh(cov_dual)
        evecs = torch.matmul(data_centered.t(), evecs)
        # Normalize the col-vectors
        evecs /= torch.sqrt(torch.sum(evecs ** 2, dim=0))

        # Sort
        idx = torch.argsort(evals, descending=True)
        evecs = evecs[:, idx]
        evals = evals[idx]

        # Remove the last eigenpair (it should have zero eigenvalue)
        self.variances = evals[:-1]
        self.modes_norm = evecs[:, :-1]
        # Compute the modes scaled by corresp. std. dev.
        self.modes_scaled = self.modes_norm * torch.sqrt(self.variances)
        self.length = evecs.shape[0]

        self.compactness = torch.cumsum(self.variances, dim=0) / torch.sum(self.variances)
        output_path = "./results/fuss_pancreas_Pair_ssm/visualization"
        compactness = np.array(self.compactness.cpu())
        np.save(os.path.join(output_path, "compactness.npy"), compactness)

    def generate_random_samples(self, n_samples=1, n_modes=None):
        """
        Generate random samples from the SSM.
        Args:
            n_samples:  number of samples to generate
            n_modes:    number of modes to use
        Returns:
            samples:    Generated random samples as torch.Tensor
        """
        if n_modes is None:
            n_modes = self.modes_scaled.shape[1]
        weights = torch.randn(n_samples, n_modes).to(self.device).float()
        samples = (self.mean).float() + torch.matmul(weights, (self.modes_scaled).float().t()[:n_modes])
        return samples.squeeze()

    def get_reconstruction(self, shape, n_modes=None):
        """
        Project shape into the SSM to get a reconstruction
        Args:
            shape:      shape to reconstruct as torch.Tensor
            n_modes:    number of modes to use. If None, all relevant modes are used
        Returns:
            data_proj:  projected data as reconstruction as torch.Tensor
        """
        shape = shape.view(-1)
        data_proj = shape - self.mean
        if n_modes:
            # restrict to max number of modes
            if n_modes > self.length:
                n_modes = self.modes_scaled.shape[1]
            evecs = self.modes_norm[:, :n_modes]
        else:
            evecs = self.modes_norm
        evecs_t = evecs.t()
        data_proj_re = data_proj.view(-1, 1)
        weights = torch.matmul(evecs_t, data_proj_re)
        data_proj = self.mean + torch.matmul(weights.t(), evecs_t)
        data_proj = data_proj.view(-1, 3)
        return data_proj.float()

def calculate_generalization(ssm_model, dataloader_test, deformed_testing_shapes, logger, device, output_path, template):
    surface_distance = SurfaceDistance()

    generalizations_mean = []
    generalizations_std = []
    logger.info(f'Calculating Generalization')

    for mode in tqdm(range(1, ssm_model.variances.shape[0] + 1)):
        generalizations_per_mode = []

        for index, test_data in enumerate(dataloader_test):
            original_verts = (test_data['verts'].to(device)).float()
            original_areas = (test_data['face_area'].to(device)).float()
            original_verts = original_verts * original_areas
            recon_deformed_shape = (ssm_model.get_reconstruction(deformed_testing_shapes[index], n_modes=mode)
                                    .reshape(1, -1, 3).to(device)).float()

            original_mesh = trimesh.Trimesh(vertices=to_numpy(original_verts), faces=to_numpy(test_data['faces']))
            recon_mesh = trimesh.Trimesh(vertices=to_numpy(recon_deformed_shape),
                                         faces=to_numpy(template['faces']))
            surf_dist = surface_distance(original_mesh, recon_mesh)[0]
            generalizations_per_mode.append(surf_dist)


        generalization_per_mode_mean = np.mean(generalizations_per_mode)
        generalization_per_mode_std = np.std(generalizations_per_mode)
        generalizations_mean.append(generalization_per_mode_mean)
        generalizations_std.append(generalization_per_mode_std)
        logger.info(
            f'Generalizations for mode {mode} is {generalization_per_mode_mean:.4f} +/- {generalization_per_mode_std:.4f}')

    result_path = os.path.join(output_path, "generality.png")
    generalizations_mean = np.array(generalizations_mean)
    generalizations_std = np.array(generalizations_std)
    plot_with_std(np.array(list(range(1, ssm_model.variances.shape[0] + 1))),
                  generalizations_mean, generalizations_std,
                  "Generality in mm", result_path)
    np.save(os.path.join(output_path, "generalizations_mean.npy"), generalizations_mean)
    np.save(os.path.join(output_path, "generalizations_std.npy"), generalizations_std)



class SurfaceDistance():
    """This class calculates the symmetric vertex to surface distance of two
    trimesh meshes.
    """

    def __init__(self):
        pass

    def __call__(self, A, B):
        """
        Args:
          A: trimesh mesh
          B: trimesh mesh
        """
        _, A_B_dist, _ = trimesh.proximity.closest_point(A, B.vertices)
        _, B_A_dist, _ = trimesh.proximity.closest_point(B, A.vertices)
        distance = .5 * np.array(A_B_dist).mean() + .5 * \
            np.array(B_A_dist).mean()

        return np.array([distance])

def calculate_specificity(ssm_model, dataloader_train, logger, device, output_path):
    n_samples = 1000
    specificity_mean = []
    specificity_std = []
    logger.info(f'Calculating Specificity...')

    for mode in tqdm(range(1, ssm_model.variances.shape[0] + 1)):
        samples = ssm_model.generate_random_samples(n_samples=n_samples, n_modes=mode)
        samples = samples.reshape(n_samples, -1, 3).to(device)
        samples = samples - samples.mean(dim=1, keepdim=True)

        distances = np.zeros((n_samples, len(dataloader_train)))

        for index, data in enumerate(dataloader_train):
            data = to_device(data, device)
            target = (data['verts'].to(device))
            target_areas = data['face_area'].to(device)
            target = target * target_areas
            target = target.repeat(n_samples, 1, 1)

            loss, _ = chamfer_distance(target.float(), samples.float(), point_reduction=None, batch_reduction=None)
            distance = 0.5 * (loss[0].sqrt().mean(dim=1) + loss[1].sqrt().mean(dim=1))

            distance = to_numpy(distance)
            distances[:, index] = distance

        specificity_mean_value = distances.min(1).mean()
        specificity_std_value = distances.min(1).std()
        specificity_mean.append(specificity_mean_value)
        specificity_std.append(specificity_std_value)
        logger.info(f'Specificity for mode {mode} is {specificity_mean_value:.10f} +/- {specificity_std_value:.10f}')

    result_path = os.path.join(output_path, "specificity.png")
    specificity_mean = np.array(specificity_mean)
    specificity_std = np.array(specificity_std)
    plot_with_std(np.array(list(range(1, ssm_model.variances.shape[0] + 1))),
                  specificity_mean, specificity_std,
                  "Specificity in mm", result_path)
    np.save(os.path.join(output_path, "specificity_mean.npy"), specificity_mean)
    np.save(os.path.join(output_path, "specificity_std.npy"), specificity_std)

#################################################################################################################

@torch.no_grad()
def building_ssm(model, dataloader_reference, dataloader_train, dataloader_test):
    logger = get_root_logger()

    # get reference shape
    logger.info(f'Getting reference shape based on training set')
    #template_name = self.opt.get('template_name', None)
    model.get_reference_shape(dataloader_reference, '48')
    n_vertices = model.template["verts"].shape[0]
    logger.info(f'n_vertices of the chosen template: {model.template["name"]} is: {n_vertices}')

    trainlen = len(dataloader_train)
    testlen = len(dataloader_test)
    print(trainlen, testlen)
    path = "./results/fuss_pancreas_Pair/visualization/"
    #path = './results/kidney_flow/'

    # deforming template to all training shapes to run PCA
    deformed_training_shapes = torch.empty(0, n_vertices, 3).to(model.device)
    for i in range(trainlen):
        reconverts = torch.tensor(np.load(f'{path}displx_{i}.npy')).to(model.device) #displx_{i} #{i}_trans
        deformed_training_shapes = torch.cat((deformed_training_shapes, reconverts.unsqueeze(0)), dim=0)
    deformed_training_shapes = deformed_training_shapes.reshape(deformed_training_shapes.shape[0], -1)

    # build ssm using pca
    ssm_model = SSMPCA(deformed_training_shapes)

    # deformed test shapes
    deformed_testing_shapes = torch.empty(0, n_vertices, 3).to(model.device)
    for i in range(trainlen, trainlen + testlen):
        reconverts = torch.tensor(np.load(f'{path}displx_{i}.npy')).to(model.device)
        deformed_testing_shapes = torch.cat((deformed_testing_shapes, reconverts.unsqueeze(0)), dim=0)
    deformed_testing_shapes = deformed_testing_shapes.reshape(deformed_testing_shapes.shape[0], -1)

    logger = get_root_logger()
    calculate_generalization(ssm_model, dataloader_test, deformed_testing_shapes, logger, model.device, './results/fuss_pancreas_Pair_ssm/visualization', model.template)
    #calculate_specificity(ssm_model, dataloader_train, logger, model.device, './results/fuss_kidney_Pair_ssm/visualization')

    logger.info(f'Building SSM done!')

def test_pipeline(root_path):
    # parse options, set distributed setting, set random seed
    opt = parse_options(root_path, is_train=False)

    # initialize loggers
    log_file = osp.join(opt['path']['log'], f"test_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # create test dataset and dataloader
    test_loaders = []
    for _, dataset_opt in sorted(opt['datasets'].items()):
        test_set = build_dataset(dataset_opt)
        test_loader = build_dataloader(
            test_set, dataset_opt, phase='val', num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
        logger.info(f"Number of test images in {dataset_opt['name']}: {len(test_set)}")
        test_loaders.append(test_loader)

    # create model
    model = build_model(opt)

    # build_pca
    building_ssm(model, test_loaders[0], test_loaders[1], test_loaders[2])


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir))
    test_pipeline(root_path)
