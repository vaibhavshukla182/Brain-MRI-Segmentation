import os
import numpy as np
import math
import random
import cv2 as cv
import nibabel as nib
import torch

# in: volume path
# out: volume data in array
def readVol(volpath):
    return nib.load(volpath).get_data()

# in: volume array
# out: comprise to uint8, put 0 where number<0
def to_uint8(vol):
    vol=vol.astype(np.float)
    vol[vol<0]=0
    return ((vol-vol.min())*255.0/vol.max()).astype(np.uint8)

# in: volume array
# out: comprise to uint8, put 0 where number<800
def IR_to_uint8(vol):
    vol=vol.astype(np.float)
    vol[vol<0]=0
    return ((vol-800)*255.0/vol.max()).astype(np.uint8)

# in: volume array
# out: hist equalized volume arrray
def histeq(vol):
    for slice_index in range(vol.shape[2]):
        vol[:,:,slice_index]=cv.equalizeHist(vol[:,:,slice_index])
    return vol

# in: volume array
# out: preprocessed array
def preprocessed(vol):
    for slice_index in range(vol.shape[2]):
        cur_slice=vol[:,:,slice_index]
        sob_x=cv.Sobel(cur_slice,cv.CV_16S,1,0)
        sob_y=cv.Sobel(cur_slice,cv.CV_16S,0,1)
        absX=cv.convertScaleAbs(sob_x)
        absY=cv.convertScaleAbs(sob_y)
        sob=cv.addWeighted(absX,0.5,absY,0.5,0)
        dst=cur_slice+0.5*sob
        vol[:,:,slice_index]=dst
    return vol

# in: index of slice, stack number, slice number
# out: which slice should be stacked
def get_stackindex(slice_index, stack_num, slice_num):
    assert stack_num%2==1, 'stack numbers must be odd!'
    query_list=[0]*stack_num
    for stack_index in range(stack_num):
        query_list[stack_index]=(slice_index+(stack_index-int(stack_num/2)))%slice_num
    return query_list

# in: volume array, stack number
# out: stacked img in list
def get_stacked(vol,stack_num):
    stack_list=[]
    stacked_slice=np.zeros((vol.shape[0],vol.shape[1],stack_num),np.uint8)
    for slice_index in range(vol.shape[2]):
        query_list=get_stackindex(slice_index,stack_num,vol.shape[2])
        for index_query_list,query_list_content in enumerate(query_list):
            stacked_slice[:,:,index_query_list]=vol[:,:,query_list_content].transpose()
        stack_list.append(stacked_slice.copy())
    return stack_list

# in: stacked img, rotate angle
# out: rotated imgs
def rotate(stack_list,angle,interp):
    for stack_list_index,stacked in enumerate(stack_list):
        raws,cols=stacked.shape[0:2]
        M=cv.getRotationMatrix2D(((cols-1)/2.0,(raws-1)/2.0),angle,1)
        stack_list[stack_list_index]=cv.warpAffine(stacked,M,(cols,raws),flags=interp)
    return stack_list

# in: T1 volume, foreground threshold, margin pixel numbers
# out: which region should be cropped
def calc_crop_region(stack_list_T1,thre,pix):
    crop_region=[]
    for stack_list_index,stacked in enumerate(stack_list_T1):
        _,threimg=cv.threshold(stacked[:,:,int(stacked.shape[2]/2)].copy(),thre,255,cv.THRESH_TOZERO)
        pix_index=np.where(threimg>0)
        if not pix_index[0].size==0:
            y_min,y_max=min(pix_index[0]),max(pix_index[0])
            x_min,x_max=min(pix_index[1]),max(pix_index[1])
        else:
            y_min,y_max=pix,pix
            x_min,x_max=pix,pix
        y_min=(y_min<=pix)and(0)or(y_min)
        y_max=(y_max>=stacked.shape[0]-1-pix)and(stacked.shape[0]-1)or(y_max)
        x_min=(x_min<=pix)and(0)or(x_min)
        x_max=(x_max>=stacked.shape[1]-1-pix)and(stacked.shape[1]-1)or(x_max)
        crop_region.append([y_min,y_max,x_min,x_max])
    return crop_region

# in: crop region for each slice, how many slices in a stack
# out: max region in a stacked img
def calc_max_region_list(region_list,stack_num):
    max_region_list=[]
    for region_list_index in range(len(region_list)):
        y_min_list,y_max_list,x_min_list,x_max_list=[],[],[],[]
        for stack_index in range(stack_num):
            query_list=get_stackindex(region_list_index,stack_num,len(region_list))
            region=region_list[query_list[stack_index]]
            y_min_list.append(region[0])
            y_max_list.append(region[1])
            x_min_list.append(region[2])
            x_max_list.append(region[3])
        max_region_list.append([min(y_min_list),max(y_max_list),min(x_min_list),max(x_max_list)])
    return max_region_list

# in: size, devider
# out: padded size which can be devide by devider
def calc_ceil_pad(x,devider):
    return math.ceil(x/float(devider))*devider

# in: stack img list, maxed region list
# out: cropped img list
def crop(stack_list,region_list):
    cropped_list=[]
    for stack_list_index,stacked in enumerate(stack_list):
        y_min,y_max,x_min,x_max=region_list[stack_list_index]
        cropped=np.zeros((calc_ceil_pad(y_max-y_min,16),calc_ceil_pad(x_max-x_min,16),stacked.shape[2]),np.uint8)
        cropped[0:y_max-y_min,0:x_max-x_min,:]=stacked[y_min:y_max,x_min:x_max,:]
        cropped_list.append(cropped.copy())
    return cropped_list

# in: stack lbl list, dilate iteration
# out: stack edge list
def get_edge(stack_list,kernel_size=(3,3),sigmaX=0):
    edge_list=[]
    for stacked in stack_list:
        edges=np.zeros((stacked.shape[0],stacked.shape[1],stacked.shape[2]),np.uint8)
        for slice_index in range(stacked.shape[2]):
            edges[:,:,slice_index]=cv.Canny(stacked[:,:,slice_index],1,1)
            edges[:,:,slice_index]=cv.GaussianBlur(edges[:,:,slice_index],kernel_size,sigmaX)
        edge_list.append(edges)
    return edge_list





if __name__=='__main__':
    T1_path='../../data/training/1/pre/reg_T1.nii.gz'
    vol=to_uint8(readVol(T1_path))
    print(vol.shape)
    print('vol[100,100,20]= ', vol[100,100,20])
    histeqed=histeq(vol)
    print('vol[100,100,20]= ', vol[100,100,20])
    print('query list: ', get_stackindex(1,5,histeqed.shape[2]))
    stack_list=get_stacked(histeqed,5)
    print(len(stack_list))
    print(stack_list[0].shape)
    angle=random.uniform(-15,15)
    print('angle= ', angle)
    rotated=rotate(stack_list,angle)
    print(len(rotated))
    region=calc_crop_region(rotated,50,5)
    max_region=calc_max_region_list(region,5)
    print(region)
    print(max_region)
    cropped=crop(rotated,max_region)
    for i in range(48):
        print(cropped[i].shape)
