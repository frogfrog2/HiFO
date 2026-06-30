import open3d as o3d
import trimesh
import os
import glob
import numpy as np
import torch
from chamfer_distance import ChamferDistance

path = './results/fuss_pancreas_Pair_indionly/visualization/'
#path = './results/fuss_kidney/visualization/'

tempfaces = np.load(path + 'faces_temp.npy')

files = glob.glob(path + 'displx*.npy')
num = len(files)
num = 28 # 230 for kidney train data, 240 for pancreas
print(num)

means = 0.0
pers = 0.0
maxs = 0.0

p2mmeans = 0.0
p2mpers = 0.0
p2mmaxs = 0.0

s2s = []
p2m = []

for i in range(240, 268): #[0,1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,39,40]: # [2,36,37,38]
    print(f'Processing {i}...')
    oriverts = np.load(path + f'verts_{i}.npy')[0]
    orifaces = np.load(path + f'faces_{i}.npy')[0]
    oriarea = np.load(path + f'area_{i}.npy')[0]
    oriverts = oriverts * np.sum(oriarea)

    orimesh = o3d.geometry.TriangleMesh()
    orimesh.vertices = o3d.utility.Vector3dVector(oriverts)
    orimesh.triangles = o3d.utility.Vector3iVector(orifaces)
    orimesh.compute_vertex_normals()
    orimesh.paint_uniform_color([0.5, 0.5, 0.5])

    reconverts = np.load(f'{path}displx_{i}.npy')

    reconmesh = o3d.geometry.TriangleMesh()
    reconmesh.vertices = o3d.utility.Vector3dVector(reconverts)
    reconmesh.triangles = o3d.utility.Vector3iVector(tempfaces)
    reconmesh.compute_vertex_normals()
    reconmesh.paint_uniform_color([0.8, 0.8, 0.8])

    #orsampled_points = orimesh.sample_points_uniformly(number_of_points=20000)
    #recsampled_points = reconmesh.sample_points_uniformly(number_of_points=20000)
    #recontensor = torch.tensor(np.asarray(recsampled_points.points), dtype=torch.float).unsqueeze(0)  # (1, N, 3)
    #oritensor = torch.tensor(np.asarray(orsampled_points.points), dtype=torch.float).unsqueeze(0)
    #print(recontensor.shape)
    #print(oritensor.shape)

    #chamfer = ChamferDistance()
    #dist = chamfer(recontensor, oritensor, bidirectional=True, batch_reduction='mean', point_reduction='mean')
    #print(dist)

    ormesh = trimesh.Trimesh(vertices=oriverts, faces=orifaces)
    orprox = trimesh.proximity.ProximityQuery(ormesh)

    orsampled_points = orimesh.sample_points_uniformly(number_of_points=200000)
    recsampled_points = reconmesh.sample_points_uniformly(number_of_points=200000)
    recsampled_points.paint_uniform_color([0.8, 0.8, 0.8])
    orsampled_point = np.asarray(orsampled_points.points)
    recsampled_point = np.asarray(recsampled_points.points)
    recmesh = trimesh.Trimesh(vertices=reconverts, faces=tempfaces)
    recprox = trimesh.proximity.ProximityQuery(recmesh)
    s2sDist_1 = recprox.signed_distance(orsampled_point)
    s2sDist_2 = orprox.signed_distance(recsampled_point)
    #print(s2sDist_1.shape)
    #print(s2sDist_2.shape)
    mean1 = np.mean(np.abs(s2sDist_1))
    mean2 = np.mean(np.abs(s2sDist_2))
    per1 = np.percentile(np.abs(s2sDist_1), 95)
    per2 = np.percentile(np.abs(s2sDist_2), 95)
    max1 = np.max(np.abs(s2sDist_1))
    max2 = np.max(np.abs(s2sDist_2))
    print((mean1+mean2)/2, (per1+per2)/2, (max1+max2)/2)
    means += (mean1+mean2)/2
    pers += (per1+per2)/2
    maxs += (max1+max2)/2
    s2s.append([mean1, mean2, per1, per2, max1, max2])

    aa = np.asarray(orimesh.vertices)
    recmesh = trimesh.Trimesh(vertices=reconverts, faces=tempfaces)
    recprox = trimesh.proximity.ProximityQuery(recmesh)
    s2sDist_1 = recprox.signed_distance(aa)
    #bb = np.asarray(reconmesh.vertices)
    #ormesh = trimesh.Trimesh(vertices=oriverts, faces=orifaces)
    #orprox = trimesh.proximity.ProximityQuery(ormesh)
    #s2sDist_2 = orprox.signed_distance(bb)
    mean3 = np.mean(np.abs(s2sDist_1))
    per3 = np.percentile(np.abs(s2sDist_1), 95)
    max3 = np.max(np.abs(s2sDist_1))
    print(mean3, per3, max3)
    #print(np.mean(np.abs(s2sDist_2)), np.percentile(np.abs(s2sDist_2), 95), np.max(np.abs(s2sDist_2)))
    p2mmeans += mean3
    p2mpers += per3
    p2mmaxs += max3
    p2m.append([mean3, per3, max3])

print(f'Mean: {means/num}')
print(f'95th Percentile: {pers/num}')
print(f'Max: {maxs/num}')

print(f'Point-to-Mesh Mean: {p2mmeans/num}')
print(f'Point-to-Mesh 95th Percentile: {p2mpers/num}')
print(f'Point-to-Mesh Max: {p2mmaxs/num}')

print(s2s)
print(p2m)

np.save(path+'s2s.npy', np.array(s2s))
np.save(path+'p2m.npy', np.array(p2m))
