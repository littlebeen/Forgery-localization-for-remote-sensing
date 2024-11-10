import os
from importlib import import_module

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .adversarial import GANLoss
from collections import namedtuple
import torchvision.models as models
from torch.nn.modules.loss import _Loss
import math
from skimage.metrics import peak_signal_noise_ratio as psnr1 

class SoftDiceLoss(_Loss):
    '''
    Soft_Dice = 2*|dot(A, B)| / (|dot(A, A)| + |dot(B, B)| + eps)
    eps is a small constant to avoid zero division,
    '''
    def __init__(self, *args, **kwargs):
        super(SoftDiceLoss, self).__init__()

    def forward(self,y_pred, y_true, eps=1e-8):
        y_pred = torch.squeeze(y_pred)
        y_true = torch.squeeze(y_true)
        assert y_pred.size() == y_true.size(), "the size of predict and target must be equal."
        intersection = torch.sum(torch.mul(y_pred, y_true))
        union = torch.sum(torch.mul(y_pred, y_pred)) + torch.sum(torch.mul(y_true, y_true)) + eps

        dice = 2 * intersection / union
        dice_loss = 1.0 - dice
        return dice_loss

class CrossEntropy2d(nn.Module):

    def __init__(self, ignore_label=255):
        super(CrossEntropy2d, self).__init__()
        self.ignore_label = ignore_label
#weight = torch.tensor([2.0, 1.0]).cuda()
    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        if not target.data.dim():
            return Variable(torch.zeros(1))
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        #loss = F.cross_entropy(predict, target, weight=weight, reduction='elementwise_mean')
        loss = F.cross_entropy(predict, target, weight=weight, reduction='mean')	
        return loss

class Loss_fake():
    def __init__(self,):
        self.criterion = CrossEntropy2d(ignore_label=255).cuda()  # Ignore label ??
        self.gan = GANLoss().cuda()
        self.fake = nn.CrossEntropyLoss().cuda()
        self.criterion2 = torch.nn.BCELoss().cuda()
        self.SoftDice = SoftDiceLoss().cuda()
        self.interp = torch.nn.Upsample(size=(256, 256), mode='bilinear', align_corners=True)
        self.Predictions = namedtuple('predictions', ['vote', 'before_softmax',
                                                 'after_softmax', 'raw'])
        self.softmax = torch.nn.Softmax(dim=1)
        self.image_loss_function = nn.L1Loss()
    def loss_calc(self,out,label, out_label, model):
        out_label=out_label.type(torch.long)
        if(model=='crnet' or model=='my2'):
            pred, real_or = out
            b,c,w,h = pred.size()
            label = Variable(label.long()).cuda()
            corss = self.criterion(pred, label) 
            gan=self.fake(real_or,out_label)
            loss = corss+gan
            # loss = gan
            # loss = corss
        if(model=='movenet'):
            pred = out
            b,c,w,h = pred.size()
            label = Variable(label.long()).cuda()
            corss = self.criterion(pred, label) 
            loss = corss
        if(model=='scunet' or model=='mvss' or model=='dfcn'):
            pred = out
            #b,c,w,h = pred.size()
            #label = Variable(label.long()).cuda()
            corss = 0.2*self.criterion2(pred.view(pred.size(0), -1), label.view(label.size(0), -1)) + 0.8*self.SoftDice(pred.view(pred.size(0), -1), label.view(label.size(0), -1))
            loss = corss
        if(model =='face' or model =='deepfake'):
            loss = self.fake(out,out_label)
        if(model=='patch'):
            assert(len(out.shape) == 4)
            assert(out.shape[1] == 2)
            n, c, h, w = out.shape
            labels = out_label.view(-1, 1, 1).expand(n, h, w)
            loss=self.fake(out, labels)
        if(model =='capsule'):
            out_label =  Variable(out_label, requires_grad=False)
            classes, class_ = out
            loss = self.fake(classes[:,0,:], out_label)
            for i in range(classes.size(1) - 1):
                loss = loss + self.fake(classes[:,i+1,:], out_label)
        if(model == 'restore'):
            img,_ = out
            loss =  self.image_loss_function(img, label)
        return loss
    
    def test_handle(self,out, model):
        if(model=='crnet' or model=='my2'):
            pred, real_or = out
            pred = pred.cpu().data[0].numpy()
            pred = np.asarray(np.argmax(pred, axis=0), dtype=int)
            _, predicted_eval = torch.max(real_or.data, 1)
            return pred, predicted_eval
        if(model=='movenet'):
            pred = out
            pred = pred.cpu().numpy()
            return pred[0], None
        if(model=='scunet' or model=='mvss' or model=='dfcn'):
            pred = out
            pred = pred.cpu().data[0].numpy()
            pred[np.where(pred<0.5)] = int(0)
            pred[np.where(pred>=0.5)] = int(1)
            pred = np.asarray(pred, dtype=int)
            return pred[0], None
        if(model =='face' or model =='deepfake'):
            _, predicted_eval = torch.max(out.data, 1)
            return None, predicted_eval
        if(model =='capsule'):
            classes, class_ = out
            _, predicted_eval = torch.max(class_.data, 1)
            return None, predicted_eval

    def calc_psnr(self,sr, hr):
        sr_cacu = np.round(sr[0]*255).clip(min=0, max=255).astype(np.uint8)
        hr_cacu = np.round(hr[0]*255).clip(min=0, max=255).astype(np.uint8)
        sr_cacu = sr_cacu.transpose(1, 2, 0)
        hr_cacu = hr_cacu.transpose(1, 2, 0)
        psnr = psnr1(sr_cacu, hr_cacu, data_range=255)
        return psnr
    
class Loss(nn.modules.loss._Loss):
    def __init__(self, args, ckp):
        super(Loss, self).__init__()
        print('Preparing loss function:')

        self.n_GPUs = args.n_GPUs
        self.loss = []
        self.loss_module = nn.ModuleList()
        for loss in args.loss.split('+'):
            weight, loss_type = loss.split('*')
            if loss_type == 'MSE':
                loss_function = nn.MSELoss()
            elif loss_type == 'L1':
                loss_function = nn.L1Loss()
            elif loss_type.find('VGG') >= 0:
                module = import_module('loss.vgg')
                loss_function = getattr(module, 'VGG')(
                    loss_type[3:],
                    rgb_range=args.rgb_range
                )
            elif loss_type.find('GAN') >= 0:
                module = import_module('loss.adversarial')
                loss_function = getattr(module, 'Adversarial')(
                    args,
                    loss_type
                )

            self.loss.append({
                'type': loss_type,
                'weight': float(weight),
                'function': loss_function}
            )
            if loss_type.find('GAN') >= 0:
                self.loss.append({'type': 'DIS', 'weight': 1, 'function': None})

        if len(self.loss) > 1:
            self.loss.append({'type': 'Total', 'weight': 0, 'function': None})

        for l in self.loss:
            if l['function'] is not None:
                print('{:.3f} * {}'.format(l['weight'], l['type']))
                self.loss_module.append(l['function'])

        self.log = torch.Tensor()

        device = torch.device('cpu' if args.cpu else 'cuda')
        self.loss_module.to(device)
        if args.precision == 'half': self.loss_module.half()
        if not args.cpu and args.n_GPUs > 1:
            self.loss_module = nn.DataParallel(self.loss_module,range(args.n_GPUs))

        if args.load != '': self.load(ckp.dir, cpu=args.cpu)

    def forward(self, sr, hr):
        losses = []
        for i, l in enumerate(self.loss):
            if l['function'] is not None:
                loss = l['function'](sr, hr)
                effective_loss = l['weight'] * loss
                losses.append(effective_loss)
                self.log[-1, i] += effective_loss.item()
            elif l['type'] == 'DIS':
                self.log[-1, i] += self.loss[i - 1]['function'].loss

        loss_sum = sum(losses)
        if len(self.loss) > 1:
            self.log[-1, -1] += loss_sum.item()

        return loss_sum

    def step(self):
        for l in self.get_loss_module():
            if hasattr(l, 'scheduler'):
                l.scheduler.step()

    def start_log(self):
        self.log = torch.cat((self.log, torch.zeros(1, len(self.loss))))

    def end_log(self, n_batches):
        self.log[-1].div_(n_batches)

    def display_loss(self, batch):
        n_samples = batch + 1
        log = []
        for l, c in zip(self.loss, self.log[-1]):
            log.append('[{}: {:.4f}]'.format(l['type'], c / n_samples))

        return ''.join(log)

    def plot_loss(self, apath, epoch):
        axis = np.linspace(1, epoch, epoch)
        for i, l in enumerate(self.loss):
            label = '{} Loss'.format(l['type'])
            fig = plt.figure()
            plt.title(label)
            plt.plot(axis, self.log[:, i].numpy(), label=label)
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.savefig(os.path.join(apath, 'loss_{}.pdf'.format(l['type'])))
            plt.close(fig)

    def get_loss_module(self):
        if self.n_GPUs == 1:
            return self.loss_module
        else:
            return self.loss_module.module

    def save(self, apath):
        torch.save(self.state_dict(), os.path.join(apath, 'loss.pt'))
        torch.save(self.log, os.path.join(apath, 'loss_log.pt'))

    def load(self, apath, cpu=False):
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}

        self.load_state_dict(torch.load(
            os.path.join(apath, 'loss.pt'),
            **kwargs
        ))
        self.log = torch.load(os.path.join(apath, 'loss_log.pt'))
        for l in self.get_loss_module():
            if hasattr(l, 'scheduler'):
                for _ in range(len(self.log)): l.scheduler.step()

