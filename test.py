import torch
import torch.utils.data as data
from torch.utils.data import DataLoader

import argparse
import numpy as np
import os

import math
import copy
import pandas as pd
 
import h5py
import matplotlib
matplotlib.use('Agg')

from skimage.metrics import structural_similarity as ssim
from PIL import Image
from model import Net
#------------------------------------------------------------------------#
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#----------------------------------------------------------------------------------#
class Store_as_array(argparse._StoreAction):
    def __call__(self, parser, namespace, values, option_string=None):
        values = np.array(values)
        return super().__call__(parser, namespace, values, option_string)
        
# Test settings
parser = argparse.ArgumentParser(description="PyTorch LF Reconstruction -- test pretrained model")
parser.add_argument("--model_path", type=str, default="./Model/PrexNet2_Kalantari_2x2to7x7.pth", help="pretrained model path")
parser.add_argument("--angular_out", type=int, default=7, help="angular number of the dense light field [AngOut x AngOut]")
parser.add_argument("--angular_in", type=int, default=2, help="angular number of the sparse light field [AngIn x AngIn]")
parser.add_argument("--test_dataset", type=str, default="30Scenes", help="dataset for testing") #30Scenes, occlusions, reflective
parser.add_argument("--data_path", type=str, default="./TestData/test_30Scenes.h5",help="file path contained the dataset for testing")
parser.add_argument("--save_img", type=int, default=1,help="save image or not")
opt = parser.parse_args()
print(opt)

#-----------------------------------------------------------------------------------#
class DatasetFromHdf5(data.Dataset):
    def __init__(self, opt):
        super(DatasetFromHdf5, self).__init__()
        
        hf = h5py.File(opt.data_path)
        self.LFI_ycbcr = hf.get('LFI_ycbcr') # [N,ah,aw,h,w,3]        

        self.ang_out = opt.angular_out
        self.ang_in = opt.angular_in

    def __getitem__(self, index):

        H, W = self.LFI_ycbcr.shape[3:5]
        lfi_ycbcr = self.LFI_ycbcr[index]  #[ah,aw,h,w,3] 
        lfi_ycbcr = lfi_ycbcr[:opt.angular_out, :opt.angular_out, :].reshape(-1, H, W, 3) #[ah*aw,h,w,3]

        # Indexing
        ind_all = np.arange(self.ang_out*self.ang_out).reshape(self.ang_out, self.ang_out)
        delt = (self.ang_out-1) // (self.ang_in-1)
        ind_source = ind_all[0:self.ang_out:delt, 0:self.ang_out:delt]
        ind_source = ind_source.reshape(-1)

        inputs = lfi_ycbcr[ind_source, :, :, :]  # [num_source,H,W]
        inputs = torch.from_numpy(inputs.astype(np.float32) / 255.0)
        lfi_ycbcr = torch.from_numpy(lfi_ycbcr.astype(np.float32)/255.0) 
        
        return ind_source, inputs, lfi_ycbcr
        
    def __len__(self):
        return self.LFI_ycbcr.shape[0]
#-----------------------------------------------------------------------------------#
# Make paths
if not os.path.exists(opt.model_path):
    print('model folder is not found ')
if not os.path.exists('quan_results'):
    os.makedirs('quan_results')
if opt.save_img:
    save_dir = 'saveImg/resIm_{}'.format(opt.test_dataset)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
#-----------------------------------------------------------------------------------#
# Data loader
print('===> Loading test datasets')
test_set = DatasetFromHdf5(opt)
test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False)
print('loaded {} LFIs from {}'.format(len(test_loader), opt.data_path))
#-----------------------------------------------------------------------------------#
# Build model
print("building net")
opt.num_source = opt.angular_in * opt.angular_in
model = Net(opt).to(device)
#-----------------------------------------------------------------------------------#

def ycbcr2rgb(ycbcr):
    m = np.array([[ 65.481, 128.553, 24.966],
                  [-37.797, -74.203, 112],
                  [ 112, -93.786, -18.214]])
    shape = ycbcr.shape
    if len(shape) == 3:
        ycbcr = ycbcr.reshape((shape[0] * shape[1], 3))
    rgb = copy.deepcopy(ycbcr)
    rgb[:,0] -= 16. / 255.
    rgb[:,1:] -= 128. / 255.
    rgb = np.dot(rgb, np.linalg.inv(m.transpose()) * 255.)
    return rgb.clip(0, 1).reshape(shape).astype(np.float32)

def compt_psnr(img1, img2):
    bd=22
    mse = np.mean( (img1[bd:-bd, bd:-bd] - img2[bd:-bd, bd:-bd]) ** 2 )

    if mse == 0:
        return 100
    PIXEL_MAX = 1.0

    if mse > 1000:
        return -100
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def save_img(test_out, y, x, lfi_no):
    if opt.save_img:
        bd=22
        test_out_crop = test_out[bd:-bd, bd:-bd, :]
        img_name = '{}/SynLFI{}_view{}{}.png'.format(save_dir, lfi_no, y, x)
        img = (test_out_crop.clip(0, 1) * 255.0).astype(np.uint8)
        Image.fromarray(img).save(img_name)

def compute_quan(ind_source, pred, target, lfi_no, view_list, view_y, view_x, view_psnr, view_ssim):
    # Compute only Luminance
    for i in range(opt.angular_out * opt.angular_out):
        if i not in ind_source:
            cur_psnr = compt_psnr(target[i,:,:,0], pred[i,:,:,0])
            cur_ssim = ssim((target[i,:,:,0] * 255.0).astype(np.uint8), (pred[i,:,:,0] * 255.0).astype(np.uint8), gaussian_weights=True, sigma=1.5, use_sample_covariance=False)

            view_list.append(lfi_no)
            y = i//7
            x = i%7
            view_y.append(y)
            view_x.append(x)
            view_psnr.append(cur_psnr)
            view_ssim.append(cur_ssim)

            pred_rgb = ycbcr2rgb(pred[i, :, :, :])
            save_img(pred_rgb, y, x, lfi_no)

def test():

    csv_name = 'quan_results/res_{}_y.csv'.format(opt.test_dataset)
    
    scene_list = []
    y = []
    x = []
    psnrs = []
    ssims = []

    with torch.no_grad():
        for k, batch in enumerate(test_loader):
            # Network
            ind_source, inputs, lfi_ycbcr = batch[0], batch[1], batch[2]
            inputs = inputs[:,:,:,:,0].to(device)
            out = model(inputs, opt)

            # # for visualization
            _, d, h, w = out.shape
            out_y = out.view(d,h,w,1)
            lfi_ycbcr = lfi_ycbcr.view(d,h,w,3)

            out_y = out_y.cpu().numpy()
            lfi_ycbcr = lfi_ycbcr.cpu().numpy()

            cb = np.expand_dims(lfi_ycbcr[:,:,:,1], axis=3)
            cr = np.expand_dims(lfi_ycbcr[:,:,:,2], axis=3)
            out_ycb = np.concatenate((out_y, cb), axis=3)
            out_ycbcr = np.concatenate((out_ycb, cr), axis=3)

            # compute PSNR/SSIM
            compute_quan(ind_source, out_ycbcr, lfi_ycbcr, k, scene_list, y, x, psnrs, ssims)
            
        dataframe_lfi = pd.DataFrame({'LFI': scene_list, 'y': y, 'x': x, 'psnr':psnrs, 'ssim':ssims})
        dataframe_lfi.to_csv(csv_name, index=False, sep=',', mode='a')

# for epoch in test_epochs:
print('===> test')
checkpoint = torch.load(opt.model_path)
ckp_dict = checkpoint['model']

model_dict = model.state_dict()
ckp_dict = {k: v for k, v in ckp_dict.items() if k in model_dict}
model_dict.update(ckp_dict)
model.load_state_dict(model_dict)

print('loaded model {}'.format(opt.model_path))
model.eval()
test()
                  

