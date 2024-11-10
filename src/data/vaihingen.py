import os
import glob
from .srdata import SRData
from .common import np2Tensor
from utils import imgproc
import numpy as np
from PIL import Image
from utils.tools import draw_spectrum

class Vaihingen(SRData):
    def __init__(self, args,name='Vaihingen', train=True, benchmark=False):
        super(Vaihingen, self).__init__(
            args, name, train=train, benchmark=benchmark
        )

    def _scan(self):
        if(self.train):
            names_lr = sorted(
                glob.glob(os.path.join(self.dir_lrl, '*' + '.png'))+ 
                glob.glob(os.path.join(self.dir_lrr, '*' + '.png'))
            )
            names_hr = []
            for f in names_lr:
                filename, _ = os.path.splitext(os.path.basename(f))
                names_hr.append(os.path.join(
                    self.dir_hr, '{}{}'.format(
                        filename, '.png'
                    )
                ))
            names_lr_gt = sorted(
                glob.glob(os.path.join(self.dir_lr_gt, '*' + '.png'))
            )
            names_hr_gt=[]
            for f in names_lr_gt:
                names_hr_gt.append(os.path.join(self.apath, 'gt_mask','gt_mask.png'))
        else:
            names_lr = sorted(
                glob.glob(os.path.join(self.dir_test_lrl, '*' + '.png'))+
                glob.glob(os.path.join(self.dir_test_lrr, '*' + '.png'))
            )
            names_hr = []
            for f in names_lr:
                filename, _ = os.path.splitext(os.path.basename(f))
                names_hr.append(os.path.join( 
                    self.dir_test_hr, '{}{}'.format(
                        filename, '.png'
                    )
                ))
            names_lr_gt = sorted(
                glob.glob(os.path.join(self.dir_test_lr_gt, '*' + '.png'))
            )
            names_hr_gt = []
            for f in names_lr_gt:
                names_hr_gt.append(os.path.join(self.apath, 'gt_mask','gt_mask.png'))
        names_hr=names_hr_gt+names_hr
        names_lr=names_lr_gt+names_lr
        return names_hr, names_lr

    def __getitem__(self, idx):
        lr, label, filename = self._load_file(idx)  #whc 
        lr,label = np2Tensor(*[lr,label], rgb_range=self.args.rgb_range) #归一化外加转成cwh
        real = not label.min()==0
        if(real):
            real=np.float32(1.0)
        else:
            real=np.float32(0.0)
        return lr, label[0,:,:],real,filename  #cwh
    

    def _set_filesystem(self, dir_data):
        super(Vaihingen, self)._set_filesystem(dir_data)
        self.dir_hr = os.path.join(self.apath, 'train/inpainted_mask')
        self.dir_lrl = os.path.join(self.apath, 'train/lama')
        self.dir_lrr = os.path.join(self.apath, 'train/repaint')
        self.dir_lr_gt = os.path.join(self.apath, 'train/gt')
        self.dir_test_hr = os.path.join(self.apath, 'test/inpainted_mask')
        self.dir_test_lrr = os.path.join(self.apath, 'test/repaint')
        self.dir_test_lrl = os.path.join(self.apath, 'test/lama')
        self.dir_test_lr_gt = os.path.join(self.apath, 'test/gt')
        self.dir_gt_mask = os.path.join(self.apath, 'gt_mask')


