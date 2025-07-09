from torch.utils import data
import pickle
import numpy as np
import cv2
import torch
import SimpleITK as sitk

class TrainData(data.Dataset):
    def __init__(self, datapath, num_samples, scale_factor = 1):
        super().__init__()
        
        self.scale_factor = scale_factor
        file = open(datapath,'rb')
        data = pickle.load(file)
        self.img_gt = data['image']

        self.num_voxel = np.array(data["nVoxel"])
        self.size_voxel = np.array(data['dVoxel'])/1000*scale_factor

        self.num_samples = num_samples
        self.coordinates = self.get_coordinate().cuda()

    def __getitem__(self, index):

        coordinate = self.coordinates[index].reshape(-1,3)
        img = self.img_gt[index].reshape(-1)

        select_idx = np.random.choice(coordinate.shape[0], size=[self.num_samples], replace=False)
        select_coordinate = coordinate[select_idx]
        select_voxel = img[select_idx]

        return select_coordinate,select_voxel
    
    def __len__(self):
        return 256
    
    def get_coordinate(self):
        
        w,h,l = self.num_voxel
        wsize,hsize,lsize = self.size_voxel
        x = torch.tensor(np.linspace(-(w-1)/2*wsize, (w-1)/2*wsize, w))
        y = torch.tensor(np.linspace(-(h-1)/2*hsize, (h-1)/2*hsize, h))
        z = torch.tensor(np.linspace(-(l-1)/2*lsize, (l-1)/2*lsize, l))

        X, Y, Z = torch.meshgrid(x, y, z,indexing='ij')
        coordinate = torch.stack([X, Y, Z], dim=-1)

        return coordinate
    