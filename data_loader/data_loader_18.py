import os
import torch
import numpy as np
import math
import random
import cv2 as cv
import nibabel as nib
import torch
from torch.utils import data
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pandas as pd

from data_loader.preprocess import readVol,to_uint8,IR_to_uint8,histeq,preprocessed,get_stacked,rotate,calc_crop_region,calc_max_region_list,crop,get_edge

class MR18loader_CV(data.Dataset):
    def __init__(self,root='../../data/',val_num=5,is_val=False,
                 is_transform=False,is_flip=False,is_rotate=False,is_crop=False,is_histeq=False,forest=5):
        self.root=root
        self.val_num=val_num
        self.is_val=is_val
        self.is_transform=is_transform
        self.is_flip=is_flip
        self.is_rotate=is_rotate
        self.is_crop=is_crop
        self.is_histeq=is_histeq
        self.forest=forest
        self.n_classes=11
        # Back: Background
        # GM:   Cortical GM(red), Basal ganglia(green)
        # WM:   WM(yellow), WM lesions(blue)
        # CSF:  CSF(pink), Ventricles(light blue)
        # Back: Cerebellum(white), Brainstem(dark red)
        self.color=np.asarray([[0,0,0],[0,0,255],[0,255,0],[0,255,255],[255,0,0],\
                [255,0,255],[255,255,0],[255,255,255],[0,0,128],[0,128,0],[128,0,0]]).astype(np.uint8)
        # Back , CSF , GM , WM
        self.label_test=[0,2,2,3,3,1,1,0,0]
        # nii paths
        self.T1path=[self.root+'training/'+name+'/pre/reg_T1.nii.gz' for name in ['1','4','5','7','14','070','148']]
        self.IRpath=[self.root+'training/'+name+'/pre/IR.nii.gz' for name in ['1','4','5','7','14','070','148']]
        self.T2path=[self.root+'training/'+name+'/pre/FLAIR.nii.gz' for name in ['1','4','5','7','14','070','148']]
        self.lblpath=[self.root+'training/'+name+'/segm.nii.gz' for name in ['1','4','5','7','14','070','148']]

        # val path
        self.val_T1path=self.T1path[self.val_num-1]
        self.val_IRpath=self.IRpath[self.val_num-1]
        self.val_T2path=self.T2path[self.val_num-1]
        self.val_lblpath=self.lblpath[self.val_num-1]
        # train path
        self.train_T1path=[temp for temp in self.T1path if temp not in [self.val_T1path]]
        self.train_IRpath=[temp for temp in self.IRpath if temp not in [self.val_IRpath]]
        self.train_T2path=[temp for temp in self.T2path if temp not in [self.val_T2path]]
        self.train_lblpath=[temp for temp in self.lblpath if temp not in [self.val_lblpath]]
        
        if self.is_val==False:
            print('training data')
            T1_nii=[to_uint8(readVol(path)) for path in self.train_T1path]
            IR_nii=[IR_to_uint8(readVol(path)) for path in self.train_IRpath]
            T2_nii=[to_uint8(readVol(path)) for path in self.train_T2path]
            lbl_nii=[readVol(path) for path in self.train_lblpath]
            
            if self.is_flip:
                vol_num=len(T1_nii)
                for nums in range(vol_num):
                    T1_nii.append(np.array([cv.flip(slice_,1) for slice_ in T1_nii[nums]]))
                    IR_nii.append(np.array([cv.flip(slice_,1) for slice_ in IR_nii[nums]]))
                    T2_nii.append(np.array([cv.flip(slice_,1) for slice_ in T2_nii[nums]]))
                    lbl_nii.append(np.array([cv.flip(slice_,1) for slice_ in lbl_nii[nums]]))

            if self.is_histeq:
                print('hist equalizing......')
                T1_nii=[histeq(vol) for vol in T1_nii]
                IR_nii=[vol for vol in IR_nii]
                T2_nii=[vol for vol in T2_nii]

            print('get stacking......')
            T1_stack_lists=[get_stacked(vol,self.forest) for vol in T1_nii]
            IR_stack_lists=[get_stacked(vol,self.forest) for vol in IR_nii]
            T2_stack_lists=[get_stacked(vol,self.forest) for vol in T2_nii]
            lbl_stack_lists=[get_stacked(vol,self.forest) for vol in lbl_nii]

            if self.is_rotate:
                print('rotating......')
                angle_list=[5,-5,10,-10,15,-15]
                sample_num=len(T1_stack_lists)
                for angle in angle_list:
                    for sample_index in range(sample_num):
                        T1_stack_lists.append(rotate(T1_stack_lists[sample_index],angle,interp=cv.INTER_CUBIC).copy())
                        IR_stack_lists.append(rotate(IR_stack_lists[sample_index],angle,interp=cv.INTER_CUBIC).copy())
                        T2_stack_lists.append(rotate(T2_stack_lists[sample_index],angle,interp=cv.INTER_CUBIC).copy())
                        lbl_stack_lists.append(rotate(lbl_stack_lists[sample_index],angle,interp=cv.INTER_NEAREST).copy())

            if self.is_crop:
                print('cropping......')
                region_lists=[calc_max_region_list(calc_crop_region(T1_stack_list,50,5),self.forest) for T1_stack_list in T1_stack_lists]
                self.region_lists=region_lists
                T1_stack_lists=[crop(stack_list,region_lists[list_index]) for list_index,stack_list in enumerate(T1_stack_lists)]
                IR_stack_lists=[crop(stack_list,region_lists[list_index]) for list_index,stack_list in enumerate(IR_stack_lists)]
                T2_stack_lists=[crop(stack_list,region_lists[list_index]) for list_index,stack_list in enumerate(T2_stack_lists)]
                lbl_stack_lists=[crop(stack_list,region_lists[list_index]) for list_index,stack_list in enumerate(lbl_stack_lists)]
            '''
            print('len=',len(T1_stack_lists))
            T1_path_list=[]
            IR_path_list=[]
            T2_path_list=[]
            lbl_path_list=[]
            range_list=[]
            name=['1','4','5','7','14','070','148']
            f_n=['n','f']
            ang=['0','5','-5','10','-10','15','-15']
            save_path='../../../../data/'
            for sam_i,sample in enumerate(T1_stack_lists):
                for img_j,img in enumerate(sample):
                    T1_path_list.append('imgs/'+'T1/'+'{}_{}_{}_{}.png'.format(name[sam_i%7],f_n[(int(sam_i/7))%2],ang[int(sam_i/14)],img_j))
                    path=save_path+'imgs/'+'T1/'+'{}_{}_{}_{}.png'.format(name[sam_i%7],f_n[(int(sam_i/7))%2],ang[int(sam_i/14)],img_j)
                    cv.imwrite(path,img)
            for sam_i,sample in enumerate(IR_stack_lists):
                for img_j,img in enumerate(sample):
                    IR_path_list.append('imgs/'+'IR/'+'{}_{}_{}_{}.png'.format(name[sam_i%7],f_n[(int(sam_i/7))%2],ang[int(sam_i/14)],img_j))
                    path=save_path+'imgs/'+'IR/'+'{}_{}_{}_{}.png'.format(name[sam_i%7],f_n[(int(sam_i/7))%2],ang[int(sam_i/14)],img_j)
                    cv.imwrite(path,img)
            for sam_i,sample in enumerate(T2_stack_lists):
                for img_j,img in enumerate(sample):
                    T2_path_list.append('imgs/'+'T2/'+'{}_{}_{}_{}.png'.format(name[sam_i%7],f_n[(int(sam_i/7))%2],ang[int(sam_i/14)],img_j))
                    path=save_path+'imgs/'+'T2/'+'{}_{}_{}_{}.png'.format(name[sam_i%7],f_n[(int(sam_i/7))%2],ang[int(sam_i/14)],img_j)
                    cv.imwrite(path,img)
            for sam_i,sample in enumerate(lbl_stack_lists):
                for img_j,img in enumerate(sample):
                    lbl_path_list.append('lbls/'+'{}_{}_{}_{}.png'.format(name[sam_i%7],f_n[(int(sam_i/7))%2],ang[int(sam_i/14)],img_j))
                    path=save_path+'lbls/'+'{}_{}_{}_{}.png'.format(name[sam_i%7],f_n[(int(sam_i/7))%2],ang[int(sam_i/14)],img_j)
                    print(img.shape)
                    cv.imwrite(path,img)
            for sam_i,sample in enumerate(region_lists):
                for img_j,img in enumerate(sample):
                    range_list.append(img)
            range_array=np.array(range_list)
            y_min_list=range_array[:,0]
            y_max_list=range_array[:,1]
            x_min_list=range_array[:,2]
            x_max_list=range_array[:,3]
            df=pd.DataFrame({   'T1':T1_path_list,'IR':IR_path_list,'T2':T2_path_list,'lbl':lbl_path_list,
                                'y_min':y_min_list,'y_max':y_max_list,'x_min':x_min_list,'x_max':x_max_list})
            print(df)
            df.to_csv("index.csv")
            '''
            # get means
            T1mean,IRmean,T2mean=0.0,0.0,0.0
            for samples in T1_stack_lists:
                for stacks in samples:
                    T1mean=T1mean+np.mean(stacks)
            T1mean=T1mean/(len(T1_stack_lists)*len(T1_stack_lists[0]))
            print('T1 mean = ',T1mean)
            self.T1mean=T1mean
            for samples in IR_stack_lists:
                for stacks in samples:
                    IRmean=IRmean+np.mean(stacks)
            IRmean=IRmean/(len(IR_stack_lists)*len(IR_stack_lists[0]))
            print('IR mean = ',IRmean)
            self.IRmean=IRmean
            for samples in T2_stack_lists:
                for stacks in samples:
                    T2mean=T2mean+np.mean(stacks)
            T2mean=T2mean/(len(T2_stack_lists)*len(T2_stack_lists[0]))
            print('T2 mean = ',T2mean)
            self.T2mean=T2mean

            # get edegs
            print('getting edges')
            edge_stack_lists=[]
            for samples in lbl_stack_lists:
                edge_stack_lists.append(get_edge(samples))

            # transform
            if self.is_transform:
                print('transforming')
                for sample_index in range(len(T1_stack_lists)):
                    for stack_index in range(len(T1_stack_lists[0])):
                        T1_stack_lists[sample_index][stack_index],  \
                        IR_stack_lists[sample_index][stack_index],  \
                        T2_stack_lists[sample_index][stack_index],  \
                        lbl_stack_lists[sample_index][stack_index], \
                        edge_stack_lists[sample_index][stack_index]=\
                        self.transform(                             \
                        T1_stack_lists[sample_index][stack_index],  \
                        IR_stack_lists[sample_index][stack_index],  \
                        T2_stack_lists[sample_index][stack_index],  \
                        lbl_stack_lists[sample_index][stack_index], \
                        edge_stack_lists[sample_index][stack_index])
        
        else:
            print('validating data')
            T1_nii=to_uint8(readVol(self.val_T1path))
            IR_nii=IR_to_uint8(readVol(self.val_IRpath))
            T2_nii=to_uint8(readVol(self.val_T2path))
            lbl_nii=readVol(self.val_lblpath)

            if self.is_histeq:
                print('hist equalizing......')
                T1_nii=histeq(T1_nii)
                IR_nii=IR_nii
                T1_nii=T1_nii

            print('get stacking......')
            T1_stack_lists=get_stacked(T1_nii,self.forest)
            IR_stack_lists=get_stacked(IR_nii,self.forest)
            T2_stack_lists=get_stacked(T2_nii,self.forest)
            lbl_stack_lists=get_stacked(lbl_nii,self.forest)

            if self.is_crop:
                print('cropping......')
                region_lists=calc_max_region_list(calc_crop_region(T1_stack_lists,50,5),self.forest)
                self.region_lists=region_lists
                T1_stack_lists=crop(T1_stack_lists,region_lists)
                IR_stack_lists=crop(IR_stack_lists,region_lists)
                T2_stack_lists=crop(T2_stack_lists,region_lists)
                lbl_stack_lists=crop(lbl_stack_lists,region_lists)

            # get means
            T1mean,IRmean,T2mean=0.0,0.0,0.0
            for stacks in T1_stack_lists:
                T1mean=T1mean+np.mean(stacks)
            T1mean=T1mean/(len(T1_stack_lists))
            print('T1 mean = ',T1mean)
            self.T1mean=T1mean
            for stacks in IR_stack_lists:
                IRmean=IRmean+np.mean(stacks)
            IRmean=IRmean/(len(IR_stack_lists))
            print('IR mean = ',IRmean)
            self.IRmean=IRmean
            for stacks in T2_stack_lists:
                T2mean=T2mean+np.mean(stacks)
            T2mean=T2mean/(len(T2_stack_lists))
            print('T2 mean = ',T2mean)
            self.T2mean=T2mean

            # get edges
            print('getting edges')
            edge_stack_lists=get_edge(lbl_stack_lists)

            # transform
            if self.is_transform:
                print('transforming')
                for stack_index in range(len(T1_stack_lists)):
                    T1_stack_lists[stack_index],  \
                    IR_stack_lists[stack_index],  \
                    T2_stack_lists[stack_index],  \
                    lbl_stack_lists[stack_index], \
                    edge_stack_lists[stack_index]=\
                    self.transform(               \
                    T1_stack_lists[stack_index],  \
                    IR_stack_lists[stack_index],  \
                    T2_stack_lists[stack_index],  \
                    lbl_stack_lists[stack_index], \
                    edge_stack_lists[stack_index])

        # data ready
        self.T1_stack_lists=T1_stack_lists
        self.IR_stack_lists=IR_stack_lists
        self.T2_stack_lists=T2_stack_lists
        self.lbl_stack_lists=lbl_stack_lists
        self.edge_stack_lists=edge_stack_lists


    def __len__(self):
        return (self.is_val)and(48)or(48*6*7*2)
    def __getitem__(self,index):
        # get train or validation data
        if self.is_val==False:
            set_index=range(len(self.T1_stack_lists))
            img_index=range(len(self.T1_stack_lists[0]))
            return  \
                self.region_lists[set_index[int(index/48)]][img_index[int(index%48)]],  \
                self.T1_stack_lists[set_index[int(index/48)]][img_index[int(index%48)]],\
                self.IR_stack_lists[set_index[int(index/48)]][img_index[int(index%48)]],\
                self.T2_stack_lists[set_index[int(index/48)]][img_index[int(index%48)]],\
                self.lbl_stack_lists[set_index[int(index/48)]][img_index[int(index%48)]]
                #self.edge_stack_lists[set_index[int(index/48)]][img_index[int(index%48)]]

        else:
            img_index=range(len(self.T1_stack_lists))
            return  \
                self.region_lists[img_index[int(index)]],   \
                self.T1_stack_lists[img_index[int(index)]], \
                self.IR_stack_lists[img_index[int(index)]], \
                self.T2_stack_lists[img_index[int(index)]], \
                self.lbl_stack_lists[img_index[int(index)]]
                #self.edge_stack_lists[img_index[int(index)]]

    
    
    
    def transform(self,imgT1,imgIR,imgT2,lbl,edge):
        imgT1=torch.from_numpy((imgT1.transpose(2,0,1).astype(np.float)-self.T1mean)/255.0).float()
        imgIR=torch.from_numpy((imgIR.transpose(2,0,1).astype(np.float)-self.IRmean)/255.0).float()
        imgT2=torch.from_numpy((imgT2.transpose(2,0,1).astype(np.float)-self.T2mean)/255.0).float()
        lbl=torch.from_numpy(lbl.transpose(2,0,1)).long()
        edge=torch.from_numpy(edge.transpose(2,0,1)/255).float()
        return imgT1,imgIR,imgT2,lbl,edge
    def decode_segmap(self,label_mask):
        r,g,b=label_mask.copy(),label_mask.copy(),label_mask.copy()
        for ll in range(0,self.n_classes):
            r[label_mask==ll]=self.color[ll,2]
            g[label_mask==ll]=self.color[ll,1]
            b[label_mask==ll]=self.color[ll,0]
        rgb=np.zeros((label_mask.shape[0],label_mask.shape[1],3))
        rgb[:,:,0],rgb[:,:,1],rgb[:,:,2]=r,g,b
        return rgb
    def lbl_totest(self,pred):
        pred_test=np.zeros((pred.shape[0],pred.shape[1]),np.uint8)
        for ll in range(9):
            pred_test[pred==ll]=self.label_test[ll]
        return pred_test

if __name__=='__main__':
    path='../../../../data/'
    MRloader=MR18loader_CV(root=path,val_num=7,is_val=False,is_transform=True,is_flip=True,is_rotate=True,is_crop=True,is_histeq=True,forest=3)
    loader=data.DataLoader(MRloader, batch_size=1, num_workers=1, shuffle=True)
    for i,(regions,T1s,IRs,T2s,lbls) in enumerate(MRloader):
        print(i)
        #print(T1s.shape)
        #print(regions)
        #print(lbls.min())
        #print(lbls.max())
        #cv.imwrite(str(i)+'.png',T1s[:,:,1])
        #print(region)
        #print(imgT1.shape)
        #print(imgIR.shape)
        #print(imgT2.shape)
        #print(lbl.shape)

        #print('[{},{},{},{}]'.format(imgT1[0,2,40,40],imgIR[0,2,40,40],imgT2[0,2,40,40],lbl[0,2,40,40]))
        
        #cv.imwrite('T1-'+str(i)+'.png',imgT1[2])
        #cv.imwrite('IR-'+str(i)+'.png',imgIR[2])
        #cv.imwrite('T2-'+str(i)+'.png',imgT2[2])
        #cv.imwrite('lbl-'+str(i)+'.png',MRloader.decode_segmap(lbl[2]))
        
