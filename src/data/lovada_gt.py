import os
import glob
from .srdata import SRData
from .common import np2Tensor
import numpy as np
import random

def get_patch(*args, patch_size=512):
    ih, iw = args[0].shape[1:3]
    tp = patch_size
    ip = patch_size

    ix = random.randrange(0, iw - ip + 1)   #480-96+1  385
    iy = random.randrange(0, ih - ip + 1)   #480-96+1

    tx, ty = ix, iy

    ret = [
        args[0][:,iy:iy + ip, ix:ix + ip],
        *[a[:,ty:ty + tp, tx:tx + tp] for a in args[1:]]
    ]

    return ret

class Lovadagt(SRData):
    def __init__(self, args,name='Lovada', train=True, benchmark=False):
        super(Lovadagt, self).__init__(
            args, name, train=train, benchmark=benchmark
        )

    def _scan(self):
        if(self.train):
            names_lr = sorted(
                glob.glob(os.path.join(self.dir_lr, '*' + '.png'))
            )
            names_lr2 = sorted(
                glob.glob(os.path.join(self.dir_lr.replace('lama','repaint'), '*' + '.png'))
            )
            names_lr=names_lr+names_lr2
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
            names_hr_gt=names_lr_gt
        else:
            names_lr = sorted(
                glob.glob(os.path.join(self.dir_test_lr, '*' + '.png'))
            )
            names_lr2 = sorted(
                glob.glob(os.path.join(self.dir_test_lr.replace('lama','repaint'), '*' + '.png'))
            )
            names_lr=names_lr+names_lr2
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
            names_hr_gt = names_lr_gt
          
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

        if(self.train):
            lr,label  = get_patch(*[lr,label])
        return lr, label,real,filename  #cwh
    

    def _set_filesystem(self, dir_data):
        super(Lovadagt, self)._set_filesystem(dir_data)
        self.dir_hr = os.path.join(self.apath, 'train/gt_generated_place')
        self.dir_lr = os.path.join(self.apath, 'train/lama/inpainted')
        self.dir_lr_gt = os.path.join(self.apath, 'train/gt')
        self.dir_test_hr = os.path.join(self.apath, 'test/gt_generated_place')
        self.dir_test_lr = os.path.join(self.apath, 'test/lama/inpainted')
        self.dir_test_lr_gt = os.path.join(self.apath, 'test/gt')


