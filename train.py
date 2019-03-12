import os
import time
import gc
import cv2 as cv
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ion()

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable
from torch.utils import data

from models.fcn_xu import fcn_mul
from data_loader.data_loader_18 import MR18loader_CV
from metrics import runningScore
from loss import cross_entropy2d,loss_ce_t,  weighted_loss,  dice_loss,dice_coeff,  bce2d_hed

from models.fcn_xu import fcn_xu,fcn_xu_19,fcn_nopool,fcn_xu_dilated
from models.unet import unet
from models.PAN import PAN_seg
from models.resnet import FCN_res

from models.segnet import segnet
from models.densenet import DenseNet,DenseNetSeg
from models.tiramisu import tiramisu

def adjust_learning_rate(optimizer, epoch):
    lr = args.lr * (0.1 ** (epoch // 5))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(args):
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu_id)
    #torch.manual_seed(1337)
    print(args)
    # setup dataloader
    t_loader=MR18loader_CV(root=args.data_path,val_num=args.val_num,is_val=False,is_transform=True,is_flip=True,is_rotate=True,is_crop=True,is_histeq=True,forest=args.num_forest)
    v_loader=MR18loader_CV(root=args.data_path,val_num=args.val_num,is_val=True,is_transform=True,is_flip=False,is_rotate=False,is_crop=True,is_histeq=True,forest=args.num_forest)
    n_classes = t_loader.n_classes
    trainloader = data.DataLoader(t_loader, batch_size=args.batch_size, num_workers=1, shuffle=True)
    valloader = data.DataLoader(v_loader, batch_size=1, num_workers=1,shuffle=True)
    # setup Metrics
    running_metrics_single = runningScore(n_classes)
    running_metrics_single_test = runningScore(4)
    # setup Model
    model=fcn_mul(n_classes=n_classes)
    vgg16 = models.vgg16(pretrained=True)
    model.init_vgg16_params(vgg16)
    model.cuda()
    # setup optimizer and loss
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.99, weight_decay=5e-4)
    loss_ce = cross_entropy2d
    #loss_ce_weight = weighted_loss
    #loss_dc = dice_loss
    #loss_hed= bce2d_hed
    # resume
    best_iou=-100.0
    if args.resume is not None:
        if os.path.isfile(args.resume):
            print("Loading model and optimizer from checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            best_iou=checkpoint['best_iou']
            model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            print("Loaded checkpoint '{}' (epoch {}), best_iou={}"
                  .format(args.resume, checkpoint['epoch'],best_iou))
        else:
            best_iou=-100.0
            print("No checkpoint found at '{}'".format(args.resume))
    # visualization
    t = []
    loss_seg_list=[]
    loss_hed_list=[]
    Dice_mean=[]
    Dice_CSF=[]
    Dice_GM=[]
    Dice_WM=[]
    t_pre=time.time()
    print('training prepared, cost {} seconds\n\n'.format(t_pre-t_begin))
    for epoch in range(args.n_epoch):
        t.append(epoch+1)
        model.train()
        adjust_learning_rate(optimizer,epoch)
        #loss_sum=0.0
        loss_epoch=0.0
        t_epoch=time.time()
        for i_train, (regions,T1s,IRs,T2s,lbls) in enumerate(trainloader):
            T1s=Variable(T1s.cuda())
            IRs,T2s=Variable(IRs.cuda()),Variable(T2s.cuda())
            lbls=Variable(lbls.cuda()[:,int(args.num_forest/2),:,:].unsqueeze(1))
            #edges=Variable(edges.cuda()[:,int(args.num_forest/2),:,:].unsqueeze(1))
            optimizer.zero_grad()
            outputs=model(T1s,IRs,T2s)
            seg_out=F.log_softmax(outputs,dim=1)
            max_prob,_=torch.max(seg_out,dim=1)
            max_prob=-max_prob.detach().unsqueeze(1)
            loss_seg_value=loss_ce(input=outputs,target=lbls) 
                    #+0.5*loss_dc(input=outputs,target=lbls)
                    #+0.5*loss_ce_weight(input=outputs,target=lbls,weight=max_prob)\
                    #+0.5*loss_ce_weight(input=outputs,target=lbls,weight=edges)\
            #loss_hed_value=loss_hed(input=outputs[1],target=edges)
                #+0.5*loss_hed(input=outputs[2],target=edges) \
                #+0.5*loss_hed(input=outputs[3],target=edges) \
                #+0.5*loss_hed(input=outputs[4],target=edges) \
                #+0.5*loss_hed(input=outputs[5],target=edges)
            loss=loss_seg_value
            #loss=loss_seg_value+loss_hed_value
            # loss average
            #loss_sum+=loss
            #if (i_train+1)%args.loss_avg==0:
            #    loss_sum/=args.loss_avg
            #    loss_sum.backward()
            #    optimizer.step()
            #    loss_sum=0.0
            loss.backward()
            optimizer.step()
            loss_epoch+=loss.item()
            # visualization
            if i_train==40:
                ax1=plt.subplot(241)
                ax1.imshow(T1s[0,1,:,:].data.cpu().numpy(),cmap ='gray')
                ax1.set_title('train_img')
                ax1.axis('off')
                ax2=plt.subplot(242)
                ax2.imshow(t_loader.decode_segmap(lbls[0,0,:,:].data.cpu().numpy()).astype(np.uint8))
                ax2.set_title('train_label')
                ax2.axis('off')
                ax3=plt.subplot(243)
                model.eval()
                train_show=model(T1s,IRs,T2s)
                ax3.imshow(t_loader.decode_segmap(train_show[0].data.max(0)[1].cpu().numpy()).astype(np.uint8))
                ax3.set_title('train_predict')
                ax3.axis('off')
                ax4=plt.subplot(244)
                ax4.imshow(max_prob[0,0].cpu().numpy())
                ax4.set_title('uncertainty')
                ax4.axis('off')
                model.train()
        loss_epoch/=i_train
        loss_seg_list.append(loss_epoch)
        loss_hed_list.append(0)
        t_train=time.time()
        print('epoch: ',epoch+1)
        print('--------------------------------Training--------------------------------')
        print('average loss in this epoch: ',loss_epoch)
        print('final loss in this epoch: ',loss.data.item())
        print('cost {} seconds up to now'.format(t_train-t_begin))
        print('cost {} seconds in this train epoch'.format(t_train-t_epoch))

        model.eval()
        for i_val, (regions_val,T1s_val,IRs_val,T2s_val,lbls_val) in enumerate(valloader):
            T1s_val=Variable(T1s_val.cuda())
            IRs_val,T2s_val=Variable(IRs_val.cuda()),Variable(T2s_val.cuda())
            with torch.no_grad():
                outputs_single=model(T1s_val,IRs_val,T2s_val)[0,:,:,:]
            # get predict
            pred_single=outputs_single.data.max(0)[1].cpu().numpy()
            # pad to 240
            pred_pad=np.zeros((240,240),np.uint8)
            pred_pad[regions_val[0]:regions_val[1],regions_val[2]:regions_val[3]]=  \
                    pred_single[0:regions_val[1]-regions_val[0],0:regions_val[3]-regions_val[2]]
            # convert to 3 classes
            pred_single_test=np.zeros((240,240),np.uint8)
            pred_single_test=v_loader.lbl_totest(pred_pad)
            # get gt
            gt = lbls_val[0][int(args.num_forest/2)].numpy()
            # pad to 240
            gt_pad=np.zeros((240,240),np.uint8)
            gt_pad[regions_val[0]:regions_val[1],regions_val[2]:regions_val[3]]=  \
                    gt[0:regions_val[1]-regions_val[0],0:regions_val[3]-regions_val[2]]
            # convert to 3 classes
            gt_test=np.zeros((240,240),np.uint8)
            gt_test=v_loader.lbl_totest(gt_pad)
            # metrics update
            running_metrics_single.update(gt_pad, pred_pad)
            running_metrics_single_test.update(gt_test, pred_single_test)
            # visualization
            if i_val==40:
                ax5=plt.subplot(245)
                ax5.imshow((T1s_val[0,int(args.num_forest/2),:,:].data.cpu().numpy()*255+t_loader.T1mean).astype(np.uint8),cmap ='gray')
                ax5.set_title('src_img')
                ax5.axis('off')
                ax6=plt.subplot(246)
                ax6.imshow(t_loader.decode_segmap(gt).astype(np.uint8))
                ax6.set_title('gt')
                ax6.axis('off')
                ax7=plt.subplot(247)
                ax7.imshow(t_loader.decode_segmap(pred_single).astype(np.uint8))
                ax7.set_title('pred_single')
                ax7.axis('off')
                ax8=plt.subplot(248)
                ax8.imshow(pred_single_test[regions_val[0]:regions_val[1],regions_val[2]:regions_val[3]].astype(np.uint8))
                ax8.set_title('pred_single_test')
                ax8.axis('off')
                plt.tight_layout()
                plt.subplots_adjust(wspace=.1,hspace=.3)
                plt.savefig('./fig_out/val_{}_out_{}.png'.format(str(args.val_num),epoch+1))
        # compute dice coefficients during validation
        score_single, class_iou_single = running_metrics_single.get_scores()
        score_single_test, class_iou_single_test = running_metrics_single_test.get_scores()
        Dice_mean.append(score_single['Mean Dice : \t'])
        Dice_CSF.append(score_single_test['Dice : \t'][1])
        Dice_GM.append(score_single_test['Dice : \t'][2])
        Dice_WM.append(score_single_test['Dice : \t'][3])
        print('--------------------------------All tissues--------------------------------')
        print('Back: Background,')
        print('GM: Cortical GM(red), Basal ganglia(green),')
        print('WM: WM(yellow), WM lesions(blue),')
        print('CSF: CSF(pink), Ventricles(light blue),')
        print('Back: Cerebellum(white), Brainstem(dark red)')
        print('single predict: ')
        for k, v in score_single.items():
            print(k, v)
        print('--------------------------------Only tests--------------------------------')
        print('tissue : Back , CSF , GM , WM')
        print('single predict: ')
        for k, v in score_single_test.items():
            print(k, v)
        t_test=time.time()
        print('cost {} seconds up to now'.format(t_test-t_begin))
        print('cost {} seconds in this validation epoch'.format(t_test-t_train))
        # save model at best validation metrics
        if score_single['Mean Dice : \t'] >= best_iou:
            best_iou = score_single['Mean Dice : \t']
            state = {'epoch': epoch+1,
                     'model_state': model.state_dict(),
                     'optimizer_state' : optimizer.state_dict(),
                     'best_iou':best_iou}
            torch.save(state, "val_{}_best.pkl".format(str(args.val_num)))
            print('model saved!!!')
        # save model every 10 epochs
        if (epoch+1)%10==0:
            state = {'epoch': epoch+1,
                     'model_state': model.state_dict(),
                     'optimizer_state' : optimizer.state_dict(),
                     'score':score_single}
            torch.save(state, "val_{}_e_{}.pkl".format(str(args.val_num),epoch+1))
        # plot curve
        ax1=plt.subplot(211)
        ax1.plot(t,loss_seg_list,'g')
        ax1.plot(t,loss_hed_list,'r')
        ax1.set_title('train loss')
        ax2=plt.subplot(212)
        ax2.plot(t,Dice_mean,'k')
        ax2.plot(t,Dice_CSF,'r')
        ax2.plot(t,Dice_GM,'g')
        ax2.plot(t,Dice_WM,'b')
        ax2.set_title('validate Dice, R/G/B for CSF/GM/WM')
        plt.tight_layout()
        plt.subplots_adjust(wspace=0,hspace=.3)
        plt.savefig('./fig_out/val_{}_curve.png'.format(str(args.val_num)))
        # metric reset
        running_metrics_single.reset()
        running_metrics_single_test.reset()
        print('\n\n')

if __name__ == '__main__':
    t_begin=time.time()
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--gpu_id', nargs='?', type=int, default=-1,
                        help='GPU id, -1 for cpu')
    parser.add_argument('--data_path', nargs='?', type=str, default='/home/canpi/canpi/MRBrainS18/data/',
                        help='dataset path')
    parser.add_argument('--val_num', nargs='?', type=int, default=1,
                        help='which set is left for validation')
    
    parser.add_argument('--n_epoch', nargs='?', type=int, default=20,
                        help='# of the epochs')
    parser.add_argument('--batch_size', nargs='?', type=int, default=1,
                        help='Batch Size')
    parser.add_argument('--num_forest', nargs='?', type=int, default=3,
                        help='number of stacked slice')
    #parser.add_argument('--loss_avg', nargs='?', type=int, default=1,
    #                    help='loss average')
    parser.add_argument('--lr', nargs='?', type=float, default=1e-3,
                        help='Learning Rate')
    parser.add_argument('--resume', nargs='?', type=str, default=None,
                        help='Path to previous saved model to restart from')
    args = parser.parse_args()
    train(args)
