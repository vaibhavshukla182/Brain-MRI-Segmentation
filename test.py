import numpy as np
import nibabel as nib
import cv2 as cv
import torch
from torch.utils import data
from torchvision.transforms import transforms
from data_loader.preprocess import readVol,to_uint8,IR_to_uint8,histeq,preprocessed,get_stacked,rotate,calc_crop_region,calc_max_region_list,crop,get_edge

import os
import argparse
from torch.autograd import Variable
from models.fcn_xu import fcn_mul

class MR18loader_test(data.Dataset):
    def __init__(self,T1_path,IR_path,T2_path,is_transform,is_crop,is_hist,forest):
        self.T1_path=T1_path
        self.IR_path=IR_path
        self.T2_path=T2_path
        self.is_transform=is_transform
        self.is_crop=is_crop
        self.is_hist=is_hist
        self.forest=forest
        self.n_classes=11 
        self.T1mean=0.0
        self.IRmean=0.0
        self.T2mean=0.0
        #read data
        T1_nii=to_uint8(readVol(self.T1_path))
        IR_nii=IR_to_uint8(readVol(self.IR_path))
        T2_nii=to_uint8(readVol(self.T2_path))
        #histeq
        if self.is_hist:
            T1_nii=histeq(T1_nii)
        #stack 
        T1_stack_list=get_stacked(T1_nii,self.forest)
        IR_stack_list=get_stacked(IR_nii,self.forest)
        T2_stack_list=get_stacked(T2_nii,self.forest)
        #crop
        if self.is_crop:
            region_list=calc_max_region_list(calc_crop_region(T1_stack_list,50,5),self.forest)
            self.region_list=region_list
            T1_stack_list=crop(T1_stack_list,region_list)
            IR_stack_list=crop(IR_stack_list,region_list)
            T2_stack_list=crop(T2_stack_list,region_list)
        #get mean
        T1mean,IRmean,T2mean=0.0,0.0,0.0
        for samples in T1_stack_list:
            for stacks in samples:
                T1mean=T1mean+np.mean(stacks)
        self.T1mean=T1mean/(len(T1_stack_list)*len(T1_stack_list[0]))
        for samples in IR_stack_list:
            for stacks in samples:
                IRmean=IRmean+np.mean(stacks)
        self.IRmean=IRmean/(len(IR_stack_list)*len(IR_stack_list[0]))
        for samples in T2_stack_list:
            for stacks in samples:
                T2mean=T2mean+np.mean(stacks)
        self.T2mean=T2mean/(len(T2_stack_list)*len(T2_stack_list[0]))

        #transform
        if self.is_transform:
            for stack_index in range(len(T1_stack_list)):
                T1_stack_list[stack_index],  \
                IR_stack_list[stack_index],  \
                T2_stack_list[stack_index]=  \
                self.transform(               \
                T1_stack_list[stack_index],  \
                IR_stack_list[stack_index],  \
                T2_stack_list[stack_index]) 

        # data ready
        self.T1_stack_list=T1_stack_list
        self.IR_stack_list=IR_stack_list
        self.T2_stack_list=T2_stack_list

    def __len__(self):
        return 48
    def __getitem__(self,index):
        return self.region_list[index],self.T1_stack_list[index],self.IR_stack_list[index],self.T2_stack_list[index]
    
    def transform(self,imgT1,imgIR,imgT2):
        imgT1=torch.from_numpy((imgT1.transpose(2,0,1).astype(np.float)-self.T1mean)/255.0).float()
        imgIR=torch.from_numpy((imgIR.transpose(2,0,1).astype(np.float)-self.IRmean)/255.0).float()
        imgT2=torch.from_numpy((imgT2.transpose(2,0,1).astype(np.float)-self.T2mean)/255.0).float()
        return imgT1,imgIR,imgT2


def test(args):
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu_id)
    # io vols
    srcvol=nib.load(args.T1_path)
    outvol=np.zeros((240,240,48),np.uint8)
    # data loader
    loader=MR18loader_test(T1_path=args.T1_path,IR_path=args.IR_path,T2_path=args.T2_path,is_transform=True,is_crop=True,is_hist=True,forest=3)
    testloader=data.DataLoader(loader,batch_size=1,num_workers=1,shuffle=False)
    # model setup
    n_classes = loader.n_classes
    model_1=fcn_mul(n_classes=n_classes)
    model_2=fcn_mul(n_classes=n_classes)
    model_3=fcn_mul(n_classes=n_classes)
    model_4=fcn_mul(n_classes=n_classes)
    model_5=fcn_mul(n_classes=n_classes)
    model_6=fcn_mul(n_classes=n_classes)
    model_7=fcn_mul(n_classes=n_classes)
    model_1.cuda()
    model_2.cuda()
    model_3.cuda()
    model_4.cuda()
    model_5.cuda()
    model_6.cuda()
    model_7.cuda()
    state_1 = torch.load(args.model_path_1)['model_state']
    state_2 = torch.load(args.model_path_2)['model_state']
    state_3 = torch.load(args.model_path_3)['model_state']
    state_4 = torch.load(args.model_path_4)['model_state']
    state_5 = torch.load(args.model_path_5)['model_state']
    state_6 = torch.load(args.model_path_6)['model_state']
    state_7 = torch.load(args.model_path_7)['model_state']
    model_1.load_state_dict(state_1)
    model_2.load_state_dict(state_2)
    model_3.load_state_dict(state_3)
    model_4.load_state_dict(state_4)
    model_5.load_state_dict(state_5)
    model_6.load_state_dict(state_6)
    model_7.load_state_dict(state_7)
    model_1.eval()
    model_2.eval()
    model_3.eval()
    model_4.eval()
    model_5.eval()
    model_6.eval()
    model_7.eval()
    # test
    for i_t,(regions_t,T1s_t,IRs_t,T2s_t) in enumerate(testloader):
        T1s_t,IRs_t,T2s_t=Variable(T1s_t.cuda()),Variable(IRs_t.cuda()),Variable(T2s_t.cuda())
        with torch.no_grad():
            out_1=model_1(T1s_t,IRs_t,T2s_t)[0,:,:,:]
            out_2=model_2(T1s_t,IRs_t,T2s_t)[0,:,:,:]
            out_3=model_3(T1s_t,IRs_t,T2s_t)[0,:,:,:]
            out_4=model_4(T1s_t,IRs_t,T2s_t)[0,:,:,:]
            out_5=model_5(T1s_t,IRs_t,T2s_t)[0,:,:,:]
            out_6=model_6(T1s_t,IRs_t,T2s_t)[0,:,:,:]
            out_7=model_7(T1s_t,IRs_t,T2s_t)[0,:,:,:]
        pred_1 = out_1.data.max(0)[1].cpu().numpy()
        pred_2 = out_2.data.max(0)[1].cpu().numpy()
        pred_3 = out_3.data.max(0)[1].cpu().numpy()
        pred_4 = out_4.data.max(0)[1].cpu().numpy()
        pred_5 = out_5.data.max(0)[1].cpu().numpy()
        pred_6 = out_6.data.max(0)[1].cpu().numpy()
        pred_7 = out_7.data.max(0)[1].cpu().numpy()
        h,w=pred_1.shape[0],pred_1.shape[1]
        pred=np.zeros((h,w),np.uint8)
        # vote in 7 results
        for y in range(h):
            for x in range(w):
                pred_list=np.array([pred_1[y,x],pred_2[y,x],pred_3[y,x],pred_4[y,x],pred_5[y,x],pred_6[y,x],pred_7[y,x]])
                pred[y,x]=np.argmax(np.bincount(pred_list))
        # padding to 240x240
        pred_pad=np.zeros((240,240),np.uint8)
        pred_pad[regions_t[0]:regions_t[1],regions_t[2]:regions_t[3]]=pred[0:regions_t[1]-regions_t[0],0:regions_t[3]-regions_t[2]]
        outvol[:,:,i_t]=pred_pad.transpose()
    # write nii.gz
    nib.Nifti1Image(outvol, srcvol.affine, srcvol.header).to_filename(args.outpath)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Params')
    parser.add_argument('--gpu_id', nargs='?', type=int, default=-1,help='GPU id, -1 for cpu')
    parser.add_argument('--T1_path',nargs='?',type=str,default='')
    parser.add_argument('--IR_path',nargs='?',type=str,default='')
    parser.add_argument('--T2_path',nargs='?',type=str,default='')
    parser.add_argument('--outpath',nargs='?',type=str,default='')
    parser.add_argument('--model_path_1', nargs='?', type=str, default='./CV-models/FCN_MR13_val1.pkl')
    parser.add_argument('--model_path_2', nargs='?', type=str, default='./CV-models/FCN_MR13_val2.pkl')
    parser.add_argument('--model_path_3', nargs='?', type=str, default='./CV-models/FCN_MR13_val3.pkl')
    parser.add_argument('--model_path_4', nargs='?', type=str, default='./CV-models/FCN_MR13_val4.pkl')
    parser.add_argument('--model_path_5', nargs='?', type=str, default='./CV-models/FCN_MR13_val5.pkl')
    parser.add_argument('--model_path_6', nargs='?', type=str, default='./CV-models/FCN_MR13_val5.pkl')
    parser.add_argument('--model_path_7', nargs='?', type=str, default='./CV-models/FCN_MR13_val5.pkl')
    args = parser.parse_args()
    test(args)
