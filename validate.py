import os
import torch
import argparse
import numpy as np
import torch.nn as nn
import cv2 as cv
import nibabel as nib
import torch.nn.functional as F
import torchvision.models as models

from torch.autograd import Variable
from torch.utils import data
from tqdm import tqdm

from fcn_xu import fcn_mul
from data_loader import MR18loader_CV


def validate(args):
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu_id)
    print(args)
    #torch.manual_seed(1337)
    # get nii header
    srcvol=nib.load(data_path+'training/14/pre/reg_T1.nii.gz')
    # setup dataloader
    data_path='../../data/'
    v_loader=MR18loader_CV(root=data_path,val_num=args.val_num,is_val=True,is_transform=True,is_rotate=False,is_crop=True,is_histeq=True,forest=args.num_forest)
    n_classes=v_loader.n_classes
    valloader=data.DataLoader(v_loader,batch_size=1,num_workers=1,shuffle=False)
    # setup model
    model=fcn_mul(n_classes=n_classes)
    model.cuda()
    state = torch.load(args.model_path)['model_state']
    model.load_state_dict(state)
    model.eval()
    # start predict
    pred_out=np.zeros((240,240,48),np.uint8)
    for i_val,(regions,T1s,IRs,T2s,lbls) in tqdm(enumerate(valloader)):
        print(regions)
        T1s,IRs,T2s=Variable(T1s.cuda()),Variable(IRs.cuda()),Variable(T2s.cuda())
        with torch.no_grad():
            output_slice=model(T1s,IRs,T2s)[0,:,:,:]
        pred_slice=np.zeros((output_slice.shape[1],output_slice.shape[2]),np.uint8)
        pred_slice=output_slice.data.max(0)[1].cpu().numpy()
        pred_out[regions[0]:regions[1],regions[2]:regions[3],i_val]=    \
                pred_slice[0:regions[1]-regions[0],0:regions[3]-regions[2]]
        pred_out[:,:,i_val]=pred_out[:,:,i_val].transpose()
    nib.Nifti1Image(pred_out,srcvol.affine,srcvol.header).to_filename('evaluation/result.nii')
    print('predicted')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Params')
    parser.add_argument('--gpu_id', nargs='?', type=int, default=-1,
                        help='GPU id, -1 for cpu')
    parser.add_argument('--model_path', nargs='?', type=str, default='FCN_MR13_best_model.pkl',
                        help='Path to the saved model')
    parser.add_argument('--val_num', nargs='?', type=int, default=1,
                        help='which sample to be validated')
    parser.add_argument('--num_forest', nargs='?', type=int, default=3,
                        help='how much slices to be stacked')
    args = parser.parse_args()
    validate(args)


