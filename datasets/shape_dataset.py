import os, re, random
import numpy as np
import scipy.io as sio
from itertools import product
from glob import glob
import pandas as pd

import torch
from torch.utils.data import Dataset

from utils.shape_util import read_shape
from utils.geometry_util import get_operators
from utils.registry import DATASET_REGISTRY
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes

def sort_list(l):
    try:
        return list(sorted(l, key=lambda x: int(re.search(r'\d+(?=\.)', x).group())))
    except AttributeError:
        return sorted(l)


def get_spectral_ops(item, num_evecs, cache_dir=None):
    _, mass, L, evals, evecs, _, _ = get_operators(item['verts'], item.get('faces'),
                                                   k=num_evecs,
                                                   cache_dir=cache_dir)

    evecs_trans = evecs.T * mass[None]
    item['evecs'] = evecs[:, :num_evecs]
    item['evecs_trans'] = evecs_trans[:num_evecs]
    item['evals'] = evals[:num_evecs]
    item['mass'] = mass
    item['L'] = L.to_dense()


    return item

class SingleShapeDataset(Dataset):
    def __init__(self,
                 data_root, return_faces=True,
                 return_evecs=True, num_evecs=120):
        """
        Single Shape Dataset

        Args:
            data_root (str): Data root.
            return_evecs (bool, optional): Indicate whether return eigenfunctions and eigenvalues. Default True.
            return_faces (bool, optional): Indicate whether return faces. Default True.
            num_evecs (int, optional): Number of eigenfunctions and eigenvalues to return. Default 120.
        """
        # sanity check
        assert os.path.isdir(data_root), f'Invalid data root: {data_root}.'

        # initialize
        self.data_root = data_root
        self.return_faces = return_faces
        self.return_evecs = return_evecs
        self.num_evecs = num_evecs

        self.off_files = []

        self._init_data()

        # sanity check
        self._size = len(self.off_files)
        assert self._size != 0


    def _init_data(self):
        # check the data path contains .off files
        off_path = os.path.join(self.data_root, 'off')
        assert os.path.isdir(off_path), f'Invalid path {off_path} not containing .off files'
        self.off_files = sort_list(glob(f'{off_path}/*.off'))

        # check if mesh info file exists
        self.mesh_info_file = os.path.join(self.data_root, 'mesh_info.csv')
        assert os.path.isfile(self.mesh_info_file), f'Invalid file {self.mesh_info_file}'

    def __getitem__(self, index):
        item = dict()

        # get vertices
        off_file = self.off_files[index]
        basename = os.path.splitext(os.path.basename(off_file))[0]
        item['name'] = basename
        item['index'] = index
        verts, faces = read_shape(off_file)
        item['verts'] = torch.from_numpy(verts).float()
        if self.return_faces:
            item['faces'] = torch.from_numpy(faces).long()

        # get eigenfunctions/eigenvalues
        if self.return_evecs:
            item = get_spectral_ops(item, num_evecs=self.num_evecs,
                                    cache_dir=os.path.join(self.data_root, 'diffusion'))

        mesh_info = pd.read_csv(self.mesh_info_file, dtype={"file_name": str})
        current_mesh_info = np.array(mesh_info[mesh_info['file_name'] == basename])[0]
        item['face_area'] = current_mesh_info[4]
        item['mean'] = torch.from_numpy(np.array([current_mesh_info[1],
                                                 current_mesh_info[2],
                                                 current_mesh_info[3]])).float()
        return item

    def __len__(self):
        return self._size


class PairShapeDataset(Dataset):
    def __init__(self, dataset, n_combination, num_shapes=None): ##### delete n_combination for fine-tuning
        """
        Pair Shape Dataset

        Args:
            dataset (SingleShapeDataset): single shape dataset
        """
        assert isinstance(dataset, SingleShapeDataset), f'Invalid input data type of dataset: {type(dataset)}'
        self.dataset = dataset
        ##### fine-tune
        #n = len(dataset)
        #self.combinations = [(i, n-1) for i in range(n-1)]
        #self.num_shapes = num_shapes
        ##### pretrain with new template
        #if n_combination is not None:
        #    n = len(dataset)
        #    self.combinations = [(i, j) for i in range(n-1) for j in random.sample(range(n-1), n_combination)]
        #    self.combinations += [(i, n-1) for i in range(n-1)]
        #    self.num_shapes = num_shapes
        #else:
        #    self.combinations = list(product(range(len(dataset)), repeat=2))
        #self.num_shapes = num_shapes
        ##### pretrain
        if n_combination is not None:
            self.combinations = [(i, j) for i in range(len(dataset)) for j in random.sample(range(len(dataset)), n_combination)]
            #self.combinations = [(0, 20)]
            #self.combinations += [(48, j) for j in range(243)]
        else:
            self.combinations = list(product(range(len(dataset)), repeat=2))
        self.num_shapes = num_shapes

    def __getitem__(self, index):
        # get index
        first_index, second_index = self.combinations[index]

        item = dict()
        item['first'] = self.dataset[first_index]
        item['second'] = self.dataset[second_index]

        return item

    def __len__(self):
        ##### fine-tune
        #return self.num_shapes - 1
        ##### pretrain
        if self.num_shapes is not None:
            return self.num_shapes
        else:
            return len(self.combinations)
        
class IndiPairShapeDataset(Dataset):
    def __init__(self, dataset, template_index, pair_index, num_shapes=1):
        """
        Pair Shape Dataset

        Args:
            dataset (SingleShapeDataset): single shape dataset
        """
        assert isinstance(dataset, SingleShapeDataset), f'Invalid input data type of dataset: {type(dataset)}'
        self.dataset = dataset
        self.combinations = [(template_index, pair_index)]
        self.num_shapes = num_shapes

    def __getitem__(self, index):
        # get index
        first_index, second_index = self.combinations[index]

        item = dict()
        item['first'] = self.dataset[first_index]
        item['second'] = self.dataset[second_index]

        return item

    def __len__(self):
        if self.num_shapes is not None:
            return self.num_shapes
        else:
            return len(self.combinations)


@DATASET_REGISTRY.register()
class SinglePancreasDataset(SingleShapeDataset):
    def __init__(self, data_root,
                 phase, start_index, end_index, #addi_index1, addi_index2, addi_index3, addi_index4,
                 return_faces=True, return_evecs=True, num_evecs=120):
        super(SinglePancreasDataset, self).__init__(data_root, return_faces,
                                                 return_evecs, num_evecs)
        assert phase in ['train', 'test', 'full'], f'Invalid phase {phase}, only "train" or "test" or "full"'
        #assert len(self) == 273, f'Pancreas dataset should contain 273 shapes, but get {len(self)}.'
        cnt = 0

        if self.off_files:
            offs = self.off_files[start_index:end_index]
            #if addi_index1 is not None:
            #    cnt += 1
            #    #print('one')
            #    offs = [self.off_files[addi_index1]] + offs
            #if addi_index2 is not None:
            #    cnt += 1
            #    #print('two')
            #    offs = [self.off_files[addi_index2]] + offs
            #if addi_index3 is not None:
            #    cnt += 1
            #    #print('three')
            #    offs = offs + [self.off_files[addi_index3]]
            #if addi_index4 is not None:
            #    cnt += 1
            #    #print('four')
            #    offs = offs + [self.off_files[addi_index4]]
            self.off_files = offs
        self._size = end_index - start_index + cnt

@DATASET_REGISTRY.register()
class PairPancreasDataset(PairShapeDataset):
    def __init__(self, data_root,
                 phase,  start_index, end_index, n_combination=None,
                 return_faces=True, return_evecs=True, num_evecs=120):
        self.dataset = SinglePancreasDataset(data_root, phase, start_index, end_index,
                                        return_faces, return_evecs, num_evecs)
        ##### fine-tune
        #datalen = dataset._size
        #super(PairPancreasDataset, self).__init__(dataset, datalen)
        ##### pretrain
        super(PairPancreasDataset, self).__init__(self.dataset, n_combination)

@DATASET_REGISTRY.register()
class PairSpleenDataset(PairShapeDataset):
    def __init__(self, data_root, phase, start_index, end_index, n_combination=None, return_faces=True, return_evecs=True, num_evecs=120):
        dataset = SinglePancreasDataset(data_root, phase, start_index, end_index, return_faces, return_evecs, num_evecs)
        super(PairSpleenDataset, self).__init__(dataset, None)

@DATASET_REGISTRY.register()
class IndiPairPancreasDataset(IndiPairShapeDataset):
    def __init__(self, data_root, phase, start_index, end_index, template_index, pair_index, n_combination=None, return_faces=True, return_evecs=True, num_evecs=120):
        dataset = SinglePancreasDataset(data_root, phase, start_index, end_index, return_faces, return_evecs, num_evecs)
        super(IndiPairPancreasDataset, self).__init__(dataset, template_index, pair_index)
