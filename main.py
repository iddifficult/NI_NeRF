import  torch
import  torch.optim 
import cv2
import os
import numpy as np
import commentjson as json
import SimpleITK as sitk
import model
import util
from dataset import TrainData
from torch.utils import data, tensorboard
from hashencoder import HashEncoder
from level_mask import get_mask
from tqdm import tqdm
from render import render, get_coordinate

# constrain CPU core
cpu_num = 4
os.environ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)

# import config
config_path = "config.json"
with open(config_path) as config_file:
        config = json.load(config_file)
# set GPU
gpu = config['train']['gpu']
torch.cuda.set_device(gpu)

# file root
input = config['file']['in_dir'] # input file
output = config['file']['out_dir'] #output file
ckp = config['file']['model_dir'] # checkpoint root
name = config['file']['name'] # case name

if not os.path.exists(ckp):
    os.makedirs(ckp)
if not os.path.exists(output):
    os.makedirs(output)

# NeRF hyperparameters
num_sample_ray = config['train']['num_sample_ray']
num_sample_point = config['train']['num_sample_point']
batch_size = config['train']['batch_size']
epochs = config['train']['epoch']
lr = config['train']['lr']
visualize = config['train']['visualize']

# init dataset
train_dataset = TrainData(input, num_sample_point, num_sample_ray)
TrainLoader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# constrcut NeRF
encoder = HashEncoder(input_dim=3, num_levels=10, level_dim=2, base_resolution=16, log2_hashmap_size=19).cuda()
NeRF = model.naf(out_size=1, hidden_dim=32, encoder=encoder)
mask = torch.ones((1,20)).cuda()

# initialize NeuralNetwork
if os.path.exists(config['file']['check_point']):
        NeRF.load_state_dict(torch.load(config['file']['check_point']), strict=False) # load weights
        NeRF.encoder = HashEncoder(input_dim=3, num_levels=10, level_dim=8, base_resolution=16, log2_hashmap_size=19).cuda() # reinitialize hashencoder
        print('initialize success')
else:
       print('haven\'t find pretrain weights')

NeRF.cuda()

# construct loss and optimizer
lossfn = torch.nn.L1Loss()
optimizer = torch.optim.Adam(params=NeRF.parameters(), betas=(0.9, 0.999),lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=config['train']['lr_decay_epoch'], gamma=config['train']['lr_decay_coefficient'])



# construct test datset
w,h,l = train_dataset.num_voxel
coordinate = get_coordinate(w,h,l,train_dataset.size_voxel)
coordinate = coordinate.float().cuda()
print('image shape:',coordinate.shape)
img_gt = train_dataset.img_gt

loop_tqdm = tqdm((range(epochs)), leave=True)
writer = tensorboard.SummaryWriter('log/{}'.format(name))

# start training
for i in loop_tqdm:
        
        NeRF.train()
        epoch_loss = 0

        for index, (ray,projection, ray_d) in enumerate(TrainLoader):
                
                projection = projection.reshape(-1).float().cuda()
                ray = ray.reshape(-1,3).float().cuda()

                u = NeRF(ray,mask).reshape(-1,num_sample_point)
                u = u.reshape(-1,num_sample_point)

                prediected_projection = render(u,ray_d.reshape(-1,3).float().cuda(), train_dataset.dis).reshape(-1)
                loss = lossfn(prediected_projection, projection)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss +=loss.item()       
                        

        scheduler.step()
        loop_tqdm.set_description('epoch_loss')
        loop_tqdm.set_postfix(lr=(scheduler.get_last_lr()[0]), loss=(epoch_loss))

        # mask useless hash level
        if (i+1) == 10: 
               mask = get_mask(NeRF, coordinate,mask)
               print(torch.sum(mask).item())
               
        # test       
        with torch.no_grad():
                
                # visualize middle slice
                if (i+1) % config['train']['visualize_epoch'] == 0 and visualize:

                    slice_w = coordinate[w//2,:,:,:].reshape(-1,3)
                    slice_h = coordinate[:,h//2,:,:].reshape(-1,3)
                    slice_l = coordinate[:,:,l//2,:].reshape(-1,3)

                    img_jpg_w = NeRF(slice_w,mask).reshape(h,l).float().cpu().detach().numpy()
                    img_jpg_h = NeRF(slice_h,mask).reshape(w,l).float().cpu().detach().numpy()
                    img_jpg_l = NeRF(slice_l,mask).reshape(w,h).float().cpu().detach().numpy()

                    img_jpg = img_jpg_w                    
                    max = np.max(img_jpg)
                    min = np.min(img_jpg)
                    img_jpg = (img_jpg-min)/(max-min)*255
                    img_jpg = np.where(img_jpg > 255, 255, img_jpg)
                    cv2.imwrite('{}/{}W.jpg'.format(output,name),img_jpg.astype(np.uint8))

                    img_jpg = img_jpg_h
                    max = np.max(img_jpg)
                    min = np.min(img_jpg)
                    img_jpg = (img_jpg-min)/(max-min)*255
                    img_jpg = np.where(img_jpg > 255, 255, img_jpg)
                    cv2.imwrite('{}/{}H.jpg'.format(output,name),img_jpg.astype(np.uint8))

                    img_jpg = img_jpg_l
                    max = np.max(img_jpg)
                    min = np.min(img_jpg)
                    img_jpg = (img_jpg-min)/(max-min)*255
                    img_jpg = np.where(img_jpg > 255, 255, img_jpg)
                    cv2.imwrite('{}/{}L.jpg'.format(output,name),img_jpg.astype(np.uint8))

                # save nii and test
                if (i+1) % (config['train']['save_epoch']) == 0:
                        
                        for slice_index in range(w):
                                slice = coordinate[slice_index].reshape(-1,3)
                                slice_pre = NeRF(slice,mask).reshape(1,h,l)
                                if slice_index==0:
                                        img_pre = slice_pre
                                else:
                                        img_pre = torch.cat((img_pre,slice_pre))

                        img_pre = img_pre.float().cpu().detach().numpy()
                        psnr = util.get_psnr_3d(img_gt,img_pre,device=gpu)
                        ssim = util.get_ssim_3d(img_gt,img_pre,device=gpu)
                        print(psnr,ssim)

                        writer.add_scalar('psnr',psnr, global_step=i)
                        writer.add_scalar('ssim',ssim, global_step=i)
                       
                        sitk.WriteImage(sitk.GetImageFromArray(img_pre), '{}/{}.nii.gz'.format(output,name))



NeRF=NeRF.cpu()
torch.save(NeRF.state_dict(), '{}/{}.pkl'.format(ckp,name))
torch.save({'model_state_dict': NeRF.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        }, '{}/{}checkpoint.pth'.format(ckp,name))






