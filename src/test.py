import torch
import os
import utility
import data
import model
import loss
from option import args
from trainer import Trainer
import glob
from thop import profile
import time
import torch.optim as optim

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args) 

os.environ["CUDA_VISIBLE_DEVICES"] = "0" #在卡2调试
def main():
    global model
    if checkpoint.ok:                  
        loader = data.Data(args)
        _model = model.Model(args, checkpoint)
        _loss = loss.Loss_fake() if not args.test_only else None
        t = Trainer(args, loader, _model, _loss, checkpoint)
        t.testtrain(False)
        checkpoint.done()

if __name__ == '__main__':
    main()
    