import os
import sys
sys.path.append('../lib')

from .backbones import ResNet50
from .modules import AdjustMultiLayer, MultiRegClsModules, AlignHead, RegClsModuleX
from .siamx import SiamX_


class SiamX(SiamX_):
    def __init__(self, align=False):
        super(SiamX, self).__init__()
        self.features = ResNet50(used_layers=[3])   
        self.neck = AdjustMultiLayer(in_channels=[512,1024], out_channels=[256,256])
        self.pred_head = RegClsModuleX(in_channels=[256,256,256], towernum=4)
        #self.pred_head = MultiRegClsModules(in_channels=[256], towernum=4)
        self.align_head = AlignHead(in_channels=256) if align else None
