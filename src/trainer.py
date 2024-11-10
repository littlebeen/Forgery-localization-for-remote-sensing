import os
from decimal import Decimal
from utils import imgproc
import utility

import torch
import torch.nn.utils as utils
import matplotlib.pyplot as plt
import numpy as np
import math

from utils.miou import get_iou,get_Acc
interp = torch.nn.Upsample(size=(256, 256), mode='bilinear', align_corners=True)


myid = ['24_1','24_2','24_3', '24_4', '1_1','1_2','1_3','1_4','5_2','5_3','65_1','65_2']

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = int(args.scale)

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        print('Training set:' + str(len(loader.loader_train)))
        print('Test set:' +str(len(loader.loader_test)))
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)
        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))

        self.error_last = 1e8

    def testtrain(self, is_train=True):
        best_psnr_index= 0
        best_ssim_index=0
        all_accuracy=[0]
        all_F1Score=[0]
        for e in range(self.args.epochs):
            if(is_train):
                self.train()
                accuracy,F1Score, MIoU=self.test(all_accuracy[best_psnr_index])
            else:
                accuracy,F1Score, MIoU=self.test(all_F1Score[best_psnr_index])
                break
            if(e==0):
                all_accuracy[0]=accuracy
                all_F1Score[0]=F1Score
            else:
                all_accuracy.append(accuracy)
                all_F1Score.append(F1Score)
            if accuracy>all_accuracy[best_psnr_index]:
                best_psnr_index=e
            if F1Score>all_F1Score[best_ssim_index]:
                best_ssim_index=e
        if(is_train):
            self.ckp.write_log('Best PSNR epoch{}:PSNR:{:.4f} SSIM:{:.4f}'.format(best_psnr_index+1,all_accuracy[best_psnr_index],all_F1Score[best_psnr_index]))
            self.ckp.write_log('Best SSIM epoch{}:PSNR:{:.4f} SSIM:{:.4f}'.format(best_ssim_index+1,all_accuracy[best_ssim_index],all_F1Score[best_ssim_index]))

    def train(self):
        epoch = self.optimizer.get_last_epoch() + 1
        lr = self.optimizer.get_lr()

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        #self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()
        # TEMP
        for batch, (lr, hr,real, filename,) in enumerate(self.loader_train):
            lr, hr = self.prepare(lr, hr)
            real = real.cuda()
            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            out = self.model(lr) 
            loss = self.loss.loss_calc(out,hr,real, self.args.model)
            loss.backward()

            self.optimizer.step()

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                  self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    loss.item(),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        self.optimizer.schedule()
    
    def test(self,best):
        torch.set_grad_enabled(False)

        epoch = self.optimizer.get_last_epoch()
        self.ckp.write_log('\nEvaluation:')
        self.model.eval()
        timer_test = utility.timer()
        data_list=[]
        data_real=[]
        data_pre=[]
        data_avg_result = []
        mask_avg = []

        if self.args.save_results: self.ckp.begin_background()
        for idx_data, (lr, hr, real,filename)  in enumerate(self.loader_test):
            lr, hr = self.prepare(lr,hr)
            hr_cacu = np.asarray(hr[0].cpu().numpy(), dtype=int)

            result = self.model(lr)
            pred, out = self.loss.test_handle(result,self.args.model)
            if(out!= None):
                data_real.append(int(real[0]))
                data_pre.append(out.item()>0.5)      

            if(pred is not None):
                # if(real[0]!=1.0):
                #data_avg_result.append(out.item())
                data_list.append([hr_cacu.flatten(), pred.flatten()])
                mask_avg.append( 1-hr_cacu.mean())
                pred = 1-pred
                pred = np.round(pred*255).clip(min=0, max=255).astype(np.uint8)
                save_list = [pred]
                if self.args.save_results: #and filename[0] in myid:
                    self.ckp.save_results(self.args.data_train, filename, save_list, self.scale)
        if(len(data_list)!=0):
            MIoU = get_iou(data_list, 2)
        if(len(data_real)!=0):
            Acc = get_Acc(data_real,data_pre)
        # print('Number:' +str(len(data_list)))
        # print('Value result: {:.2f}'.format(np.mean(data_avg_result)))
        
        # print('Mask: {:.4f}'.format(np.mean(mask_avg)))

        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')

        if self.args.save_results:
            self.ckp.end_background()

        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=0.1>best)

        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

        torch.set_grad_enabled(True)
        return 0.1,0.1, 0.1
    
    def testimgae(self,best):
        torch.set_grad_enabled(False)

        epoch = self.optimizer.get_last_epoch()
        self.ckp.write_log('\nEvaluation:')
        self.model.eval()
        timer_test = utility.timer()
        all_psnr = []

        if self.args.save_results: self.ckp.begin_background()
        for idx_data, (lr, hr, real,filename)  in enumerate(self.loader_test):
            lr, hr = self.prepare(lr,hr)

            result,_ = self.model(lr)
            pred = result.cpu().numpy()
            hr = hr.cpu().numpy()
            if(real.item()<0.5):
                psnr = self.loss.calc_psnr(pred,hr)
                all_psnr.append(psnr)

            pred = np.round(pred[0]*255).clip(min=0, max=255).astype(np.uint8).transpose(1, 2, 0)
            save_list = [pred]
            if self.args.save_results:
                self.ckp.save_results(self.args.data_train, filename, save_list, self.scale)
        
        print('PSNR: {:.4f}'.format(np.mean(all_psnr)))
        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')

        if self.args.save_results:
            self.ckp.end_background()

        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=0.1>best)

        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

        torch.set_grad_enabled(True)
        return 0.1,0.1, 0.1
    
    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]