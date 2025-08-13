# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------
import torch
import torch.nn as nn
from torch.nn import functional as F
from configs import config

from .lovasz_loss import lovasz_softmax

class CrossEntropyLovasz(nn.Module):
    """
    CE (multi-scale) + Lovasz-Softmax sul logit semantico principale.
    Pesa 0.5/0.5 come richiesto.
    - Filtra automaticamente eventuali head non-semantiche (es. boundary a 1 canale).
    """
    def __init__(self, num_classes, ignore_label=-1, balance_weights=None,
                 use_ohem=False, ohem_thres=0.9, ohem_keep=131072,
                 weight=None, ce_weight=0.5, lovasz_weight=0.5,
                 per_image=False, classes='present'):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.balance = balance_weights if balance_weights is not None else [1.0]
        self.ce_weight = float(ce_weight)
        self.lv_weight = float(lovasz_weight)
        self.per_image = per_image
        self.classes = classes

        # riusa le tue CE
        if use_ohem:
            self.ce_fn = OhemCrossEntropy(ignore_label=ignore_label,
                                          thres=ohem_thres, min_kept=ohem_keep,
                                          weight=weight)
        else:
            self.ce_fn = CrossEntropy(ignore_label=ignore_label, weight=weight)

    def _as_list(self, score):
        if isinstance(score, (list, tuple)):
            return list(score)
        return [score]

    # def forward(self, score, target):
    #     # score può essere lista/tupla di logits vari (semantici, boundary, ecc.)
    #     outs = self._as_list(score)

    #     # prendi solo i logits semantici (C == num_classes)
    #     sem_logits = [x for x in outs if x.dim() >= 2 and x.size(1) == self.num_classes]
    #     if len(sem_logits) == 0:  # fallback: usa l'ultimo
    #         sem_logits = [outs[-1]]

    #     # --- CE multi-scale (come le tue CE/OHEM), pesi in self.balance
    #     ce_loss = 0.0
    #     for i, x in enumerate(sem_logits):
    #         w = self.balance[i] if i < len(self.balance) else 1.0
    #         ce_loss = ce_loss + w * self.ce_fn(x, target)

    #     # --- Lovasz sul logit "principale": l'ultimo semantico (di solito il più fine)
    #     main_logit = sem_logits[-1]
    #     probas = torch.softmax(main_logit, dim=1)
    #     lv_loss = lovasz_softmax(probas, target, classes=self.classes,
    #                              per_image=self.per_image, ignore=self.ignore_label)

    #     return self.ce_weight * ce_loss + self.lv_weight * lv_loss

    def forward(self, score, target):
        # normalizza in lista di logits semantici
        outs = list(score) if isinstance(score, (list, tuple)) else [score]
        sem_logits = [x for x in outs if x.dim() >= 4 and x.size(1) == self.num_classes]
        if len(sem_logits) == 0:
            sem_logits = [outs[-1]]

        H, W = target.shape[-2], target.shape[-1]

        # --- CE (multi-scale) upsampled
        ce_loss = 0.0
        for i, x in enumerate(sem_logits):
            if x.shape[-2:] != (H, W):
                x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
            w = self.balance[i] if i < len(self.balance) else 1.0

            # OHEM si aspetta una lista di logits
            if self.ce_fn.__class__.__name__.lower().startswith('ohem'):
                ce_loss = ce_loss + w * self.ce_fn([x], target)
            else:
                ce_loss = ce_loss + w * self.ce_fn(x, target)

        # --- Lovasz sul logit principale (upsampled)
        main_logit = sem_logits[-1]
        if main_logit.shape[-2:] != (H, W):
            main_logit = F.interpolate(main_logit, size=(H, W), mode='bilinear', align_corners=True)
        probas = torch.softmax(main_logit, dim=1)
        lv_loss = lovasz_softmax(probas, target, classes=self.classes,
                                per_image=self.per_image, ignore=self.ignore_label)

        return self.ce_weight * ce_loss + self.lv_weight * lv_loss

class CrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, weight=None):
        super(CrossEntropy, self).__init__()
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_label
        )

    def _forward(self, score, target):

        loss = self.criterion(score, target)

        return loss

    def forward(self, score, target):

        if config.MODEL.NUM_OUTPUTS == 1:
            score = [score]

        balance_weights = config.LOSS.BALANCE_WEIGHTS
        sb_weights = config.LOSS.SB_WEIGHTS
        if len(balance_weights) == len(score):
            return sum([w * self._forward(x, target) for (w, x) in zip(balance_weights, score)])
        elif len(score) == 1:
            return sb_weights * self._forward(score[0], target)
        
        else:
            raise ValueError("lengths of prediction and target are not identical!")

        


class OhemCrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, thres=0.7,
                 min_kept=100000, weight=None):
        super(OhemCrossEntropy, self).__init__()
        self.thresh = thres
        self.min_kept = max(1, min_kept)
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_label,
            reduction='none'
        )

    def _ce_forward(self, score, target):


        loss = self.criterion(score, target)

        return loss

    def _ohem_forward(self, score, target, **kwargs):

        pred = F.softmax(score, dim=1)
        pixel_losses = self.criterion(score, target).contiguous().view(-1)
        mask = target.contiguous().view(-1) != self.ignore_label

        tmp_target = target.clone()
        tmp_target[tmp_target == self.ignore_label] = 0
        pred = pred.gather(1, tmp_target.unsqueeze(1))
        pred, ind = pred.contiguous().view(-1,)[mask].contiguous().sort()
        min_value = pred[min(self.min_kept, pred.numel() - 1)]
        threshold = max(min_value, self.thresh)

        pixel_losses = pixel_losses[mask][ind]
        pixel_losses = pixel_losses[pred < threshold]
        return pixel_losses.mean()

    def forward(self, score, target):
        
        if not (isinstance(score, list) or isinstance(score, tuple)):
            score = [score]

        balance_weights = config.LOSS.BALANCE_WEIGHTS
        sb_weights = config.LOSS.SB_WEIGHTS
        if len(balance_weights) == len(score):
            functions = [self._ce_forward] * \
                (len(balance_weights) - 1) + [self._ohem_forward]
            return sum([
                w * func(x, target)
                for (w, x, func) in zip(balance_weights, score, functions)
            ])
        
        elif len(score) == 1:
            return sb_weights * self._ohem_forward(score[0], target)
        
        else:
            raise ValueError("lengths of prediction and target are not identical!")


def weighted_bce(bd_pre, target):
    n, c, h, w = bd_pre.size()
    log_p = bd_pre.permute(0,2,3,1).contiguous().view(1, -1)
    target_t = target.view(1, -1)

    pos_index = (target_t == 1)
    neg_index = (target_t == 0)

    weight = torch.zeros_like(log_p)
    pos_num = pos_index.sum()
    neg_num = neg_index.sum()
    sum_num = pos_num + neg_num
    weight[pos_index] = neg_num * 1.0 / sum_num
    weight[neg_index] = pos_num * 1.0 / sum_num

    loss = F.binary_cross_entropy_with_logits(log_p, target_t, weight, reduction='mean')

    return loss


class BondaryLoss(nn.Module):
    def __init__(self, coeff_bce = 20.0):
        super(BondaryLoss, self).__init__()
        self.coeff_bce = coeff_bce
        
    def forward(self, bd_pre, bd_gt):

        bce_loss = self.coeff_bce * weighted_bce(bd_pre, bd_gt)
        loss = bce_loss
        
        return loss
    
if __name__ == '__main__':
    a = torch.zeros(2,64,64)
    a[:,5,:] = 1
    pre = torch.randn(2,1,16,16)
    
    Loss_fc = BondaryLoss()
    loss = Loss_fc(pre, a.to(torch.uint8))

        
        
        


