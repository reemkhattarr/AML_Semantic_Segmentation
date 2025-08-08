import numpy as np
import torch
from typing import Optional, Tuple, List, Dict

class MeanIoU:
    """
    Computes Mean Intersection over Union (mIoU) for semantic segmentation.
    Supports ignore_index for unlabeled pixels.
    Reference: [19] "A Survey on Deep Learning-based Architectures for Semantic Segmentation on 2D Images"
    """

    def __init__(self, num_classes: int, ignore_index: Optional[int] = None):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        Update confusion matrix with a batch of predictions and ground truths.
        Args:
            preds: (N, H, W) or (H, W) tensor of predicted class indices
            targets: (N, H, W) or (H, W) tensor of ground truth class indices
        """
        preds = preds.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()
        if preds.ndim == 2:
            preds = preds[None, ...]
            targets = targets[None, ...]
        for p, t in zip(preds, targets):
            mask = np.ones_like(t, dtype=bool)
            if self.ignore_index is not None:
                mask = t != self.ignore_index
            self.confusion_matrix += np.bincount(
                self.num_classes * t[mask].astype(int) + p[mask].astype(int),
                minlength=self.num_classes ** 2
            ).reshape(self.num_classes, self.num_classes)

    def compute(self, return_per_class: bool = False) -> Tuple[float, Optional[np.ndarray]]:
        """
        Returns:
            mIoU: mean intersection over union
            per_class_iou: IoU for each class (if return_per_class)
        """
        h = self.confusion_matrix
        tp = np.diag(h)
        fp = h.sum(axis=0) - tp
        fn = h.sum(axis=1) - tp
        denom = tp + fp + fn
        iou = np.zeros(self.num_classes, dtype=np.float32)
        for c in range(self.num_classes):
            if denom[c] == 0:
                iou[c] = np.nan  # Class not present in GT or prediction
            else:
                iou[c] = tp[c] / denom[c]
        if self.ignore_index is not None and self.ignore_index < self.num_classes:
            valid = [i for i in range(self.num_classes) if i != self.ignore_index]
        else:
            valid = list(range(self.num_classes))
        mIoU = np.nanmean(iou[valid])
        if return_per_class:
            return mIoU, iou
        else:
            return mIoU

    def summary(self) -> Dict[str, float]:
        mIoU, per_class = self.compute(return_per_class=True)
        return {
            "mIoU": float(mIoU),
            **{f"IoU_class_{i}": float(iou) for i, iou in enumerate(per_class)}
        }


def pixel_accuracy(preds: torch.Tensor, targets: torch.Tensor, ignore_index: Optional[int] = None) -> float:
    """
    Computes pixel accuracy (overall).
    Args:
        preds: (N, H, W) or (H, W) tensor of predicted class indices
        targets: (N, H, W) or (H, W) tensor of ground truth class indices
        ignore_index: label to ignore in accuracy computation
    Returns:
        accuracy: float
    """
    preds = preds.detach().cpu()
    targets = targets.detach().cpu()
    if preds.ndim == 2:
        preds = preds[None, ...]
        targets = targets[None, ...]
    correct = 0
    labeled = 0
    for p, t in zip(preds, targets):
        if ignore_index is not None:
            mask = t != ignore_index
            correct += (p[mask] == t[mask]).sum().item()
            labeled += mask.sum().item()
        else:
            correct += (p == t).sum().item()
            labeled += t.numel()
    return correct / (labeled + 1e-10)


def compute_flops_and_params(model: torch.nn.Module, input_shape: Tuple[int, int, int, int]) -> Tuple[float, float]:
    """
    Computes FLOPs and parameter count for a model.
    Args:
        model: torch.nn.Module
        input_shape: (N, C, H, W)
    Returns:
        flops: number of multiply-adds (MACs)
        params: number of parameters
    """
    try:
        from fvcore.nn import FlopCountAnalysis, parameter_count_table
        dummy = torch.randn(*input_shape)
        flops = FlopCountAnalysis(model, dummy).total()
        params = sum(p.numel() for p in model.parameters())
        return flops, params
    except ImportError:
        try:
            from thop import profile
            dummy = torch.randn(*input_shape)
            flops, params = profile(model, inputs=(dummy,), verbose=False)
            return flops, params
        except ImportError:
            print("Install fvcore or thop for FLOPs/params computation.")
            return -1, -1

