# pem/add_albu_cfg.py
from detectron2.config import CfgNode as CN

def add_albu_cfg(cfg):
    cfg.INPUT.ALBU = CN()
    cfg.INPUT.ALBU.COLOR_JITTER = CN()
    cfg.INPUT.ALBU.COLOR_JITTER.ENABLED = False
    cfg.INPUT.ALBU.COLOR_JITTER.P = 0.0
    cfg.INPUT.ALBU.COLOR_JITTER.BRIGHTNESS = 0.2
    cfg.INPUT.ALBU.COLOR_JITTER.CONTRAST = 0.2
    cfg.INPUT.ALBU.COLOR_JITTER.SATURATION = 0.2
    cfg.INPUT.ALBU.COLOR_JITTER.HUE = 0.1
