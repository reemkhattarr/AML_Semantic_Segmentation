import torch.nn.functional as F
from .pidnet import PIDNet, get_seg_model
import torch
import torchvision
import logging

class PIDNetMulti(PIDNet):
    def __init__(self, *args, **kwargs):
        super(PIDNetMulti, self).__init__(*args, **kwargs)
        # Ensure you have the auxiliary heads defined as in your pidnet.py

    def forward(self, x):
        # Adapting your forward to explicitly output multiple predictions for multi-level adversarial learning
        width_output = x.shape[-1] // 8
        height_output = x.shape[-2] // 8
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.relu(self.layer2(self.relu(x)))
        x_ = self.layer3_(x)
        x_d = self.layer3_d(x)
        x = self.relu(self.layer3(x))
        x_ = self.pag3(x_, self.compression3(x))
        x_d = x_d + F.interpolate(self.diff3(x), size=[height_output, width_output], mode='bilinear', align_corners=False)

        temp_p = None
        temp_d = None
        if self.augment:
            temp_p = x_

        x = self.relu(self.layer4(x))
        x_ = self.layer4_(self.relu(x_))
        x_d = self.layer4_d(self.relu(x_d))
        x_ = self.pag4(x_, self.compression4(x))
        x_d = x_d + F.interpolate(self.diff4(x), size=[height_output, width_output], mode='bilinear', align_corners=False)

        if self.augment:
            temp_d = x_d

        x_ = self.layer5_(self.relu(x_))
        x_d = self.layer5_d(self.relu(x_d))
        x = F.interpolate(self.spp(self.layer5(x)), size=[height_output, width_output], mode='bilinear', align_corners=False)

        # Final prediction head
        pred_final = self.final_layer(self.dfm(x_, x, x_d))

        # Auxiliary heads for intermediate levels
        if self.augment:
            pred_aux_p = self.seghead_p(temp_p)
            pred_aux_d = self.seghead_d(temp_d)
            return [pred_aux_p, pred_final, pred_aux_d]

        else:
            return [pred_final]

def get_seg_model_multi(cfg, imgnet_pretrained):
    # Instantiate PIDNetMulti with augment=True for multi-level output heads
    if 's' in cfg.MODEL.NAME:
        model = PIDNetMulti(m=2, n=3, num_classes=cfg.DATASET.NUM_CLASSES, planes=32, ppm_planes=96, head_planes=128, augment=True)
    elif 'm' in cfg.MODEL.NAME:
        model = PIDNetMulti(m=2, n=3, num_classes=cfg.DATASET.NUM_CLASSES, planes=64, ppm_planes=96, head_planes=128, augment=True)
    else:
        model = PIDNetMulti(m=3, n=4, num_classes=cfg.DATASET.NUM_CLASSES, planes=64, ppm_planes=112, head_planes=256, augment=True)

    if imgnet_pretrained:
        pretrained_state = torch.load(cfg.MODEL.PRETRAINED, map_location='cpu')['state_dict'] 
        model_dict = model.state_dict()
        pretrained_state = {k: v for k, v in pretrained_state.items() if (k in model_dict and v.shape == model_dict[k].shape)}
        model_dict.update(pretrained_state)
        msg = 'Loaded {} parameters!'.format(len(pretrained_state))
        logging.info('Attention!!!')
        logging.info(msg)
        logging.info('Over!!!')
        model.load_state_dict(model_dict, strict = False)
    else:
        pretrained_dict = torch.load(cfg.MODEL.PRETRAINED, map_location='cpu')
        if 'state_dict' in pretrained_dict:
            pretrained_dict = pretrained_dict['state_dict']
        model_dict = model.state_dict()
        pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() if (k[6:] in model_dict and v.shape == model_dict[k[6:]].shape)}
        msg = 'Loaded {} parameters!'.format(len(pretrained_dict))
        logging.info('Attention!!!')
        logging.info(msg)
        logging.info('Over!!!')
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict = False)
    
    return model

