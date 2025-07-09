from torch.utils import data
import pickle
import numpy as np
import cv2
import torch
import SimpleITK as sitk

class TrainData(data.Dataset):
    def __init__(self, datapath, num_samples, num_rays,scale_factor = 1):
        super().__init__()
        
        self.scale_factor = scale_factor
        self.num_samples = num_samples
        self.num_rays = num_rays

        file = open(datapath,'rb')  
        data = pickle.load(file)

        self.img_gt = data['image']
        self.DSD = data['DSD']/1000*scale_factor # distance from source to detector
        self.DSO = data['DSO']/1000*scale_factor # distance from source to origine
        self.num_detector = np.array(data['nDetector'])
        self.size_detector = np.array(data['dDetector'])/1000*scale_factor
        self.num_voxel = np.array(data["nVoxel"])
        self.size_voxel = np.array(data['dVoxel'])/1000*scale_factor
        self.angles = data['train']['angles']
        self.projections = torch.tensor(data['train']['projections']).cuda()
        self.n_samples = data["numTrain"]

        self.coordinates = self.GetPosition().cuda()

    def __getitem__(self, index):

        angle = self.angles[index]
        projection = self.projections[index].reshape(-1)
        coordinate = self.coordinates[index].reshape(-1,3)

        roi_index = torch.nonzero(projection > 0).squeeze()
        select_idx = np.random.choice(roi_index.cpu().numpy(), size=[self.num_rays], replace=False)

        select_coordinates = coordinate[select_idx]
        select_projections = projection[select_idx]        

        source = torch.tensor([np.cos(angle),np.sin(angle),0]).cuda()
        select_rays = self.get_rays(select_coordinates, source)

        return select_rays, select_projections, select_coordinates
    
    def __len__(self):
        return len(self.angles)

    def GetPosition(self):
        H, W = self.num_detector

        # init detect in yz
        y,z = torch.meshgrid(torch.linspace(0, W - 1, W),torch.linspace(0, H - 1, H), indexing="xy")
        x = -torch.ones_like(z)   
        z = z-W/2+0.5
        y = y-H/2+0.5
        position = torch.stack((x,y/self.DSD/1000*self.scale_factor,-z/self.DSD/1000*self.scale_factor), dim=-1)

        # rotate
        coordinates = []
        for angle in self.angles:
            rotate_matrix = torch.tensor([[np.cos(angle), np.sin(angle),  0],
                                        [-np.sin(angle), np.cos(angle), 0],
                                        [0,              0,             1]], dtype=torch.float32)
            coordinate = torch.matmul(position,rotate_matrix)
            coordinates.append(coordinate)

        coordinates = torch.stack(coordinates, dim=0)  

        return coordinates
    
    def get_rays(self, coordinate, source, perturb=False):
        source = source.expand(coordinate.shape)

        if perturb:
            random_tensor = ((self.size_detector[0] - 0) *torch.rand(coordinate.shape) - self.size_detector[0]//2).cuda()
            coordinate = coordinate+random_tensor

        dis_np = np.array([self.num_voxel[0] * self.size_voxel[0] / 2, self.num_voxel[1] * self.size_voxel[1] / 2])
        dis = torch.tensor(np.linalg.norm(dis_np)).cuda()
        near = self.DSO-dis-0.005
        far = dis+self.DSO+0.005
        temp = torch.linspace(0., 1., self.num_samples).cuda()
        self.dis = near * (1. - temp) + far * (temp)
        rays = coordinate[:,None,:]*self.dis[None,:,None]+source[:,None,:]
        
        return rays


