# HiFO: High-Fidelity Organ Mesh Deformation via Global-to-Individual Learning

## Installation
```bash 
conda create -n fuss python=3.9 # create new viertual environment
conda activate fuss
conda install pytorch=1.13.0 torchvision pytorch-cuda=11.6 -c pytorch -c nvidia # install pytorch
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install pytorch3d -c pytorch3d # install pytorch3d
conda install pyg -c pyg # install torch_geometric
pip install -r requirements.txt # install other necessary libraries via pip
```

## Dataset
Put all datasets under ../data/. 
```Shell
├── data
    ├── pancreas
        ├── off 
```

## Data preparation
For data preprocessing, we provide *[preprocess.py](preprocess.py)* to compute all things we need.
Here is an example for pancreas.
```python
python preprocess.py --data_root ../data/pancreas/ --n_eig 200
```

## Train (Global Learning)
To train the model on a specified dataset.
```python
python train.py --opt options/train/pancreas.yaml 
```
You can visualize the training process in tensorboard.
```bash
tensorboard --logdir experiments/
```

## Test (Global Learning)
To test the model on a specified dataset.
```python
python test.py --opt options/test/pancreas.yaml 
```
The qualitative and quantitative results will be saved in [results](results) folder.

## Train (Individual Learning)
Run
```python
python trainindi.py --opt options/train/pancreas_indi.yaml 
```

## Test (Individual Learning)
Run
```python
python testindi.py --opt options/test/pancreas_indi.yaml 
```
Note: Before running the testindi command, make sure to modify ```models/base_model.py```. Inside the ```build_ssm``` function, set ```template_name``` to your desired template index (e.g., '30').

## Outputs
Inside the ```visualization``` directory, you will find template-related files and individual data files for each index $i$:
* Template Files:
    * ```area_temp.npy```: Normalization factor (```sqrt_area```) for the template mesh.
    * ```verts_temp.npy```: Vertex coordinates of the template mesh. Multiply ```verts_temp.npy``` by ```area_temp.npy``` to reverse the normalization.
    * ```faces_temp.npy```: Triangle connectivity information of the template mesh.

* Individual Data Files:
    * ```area_{i}.npy```: Normalization factor (```sqrt_area```) for the $i$-th original mesh.
    * ```verts_{i}.npy```: Vertex coordinates of the $i$-th original mesh. Multiply ```verts_{i}.npy``` by ```area_{i}.npy``` to reverse the normalization.
    * ```faces_{i}.npy```: Triangle connectivity information of the $i$-th original mesh.
    * ```displx_{i}.npy```: Unnormalized vertex coordinates of the deformed mesh (result of deforming the template mesh into the $i$-th mesh shape).

## Acknowledgement
This repository is adapted from [FUSS](https://github.com/NafieAmrani/FUSS).

## License
This repo is licensed under MIT licence.
