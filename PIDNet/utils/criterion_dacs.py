import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropyLoss2dPixelWiseWeighted(nn.Module):
    """
    Per-pixel weighted Cross Entropy Loss for DACS-style domain adaptation.
    """
    def __init__(self, weight=None, ignore_index=255, reduction='none'):
        super(CrossEntropyLoss2dPixelWiseWeighted, self).__init__()
        self.CE = nn.CrossEntropyLoss(
            weight=weight, 
            ignore_index=ignore_index, 
            reduction=reduction
        )
        self.ignore_index = ignore_index
        
    def forward(self, output, target, pixel_weights):
        """
        Forward pass for pixel-wise weighted cross entropy loss.
    
        Args:
            output (torch.Tensor): Predicted logits with shape [B, C, H, W].
            target (torch.Tensor): Ground-truth labels with shape [B, H, W].
            pixel_weights (torch.Tensor): Per-pixel weights, same shape as loss output [B, H, W].
    
        Returns:
            torch.Tensor: Weighted loss scalar.
        """
        n_classes = output.shape[1]  # number of classes predicted
    
        # Fix a common DataParallel quirk:
        # if output batch size is 1 but output channels dimension equals target batch size,
        # then output is likely [1, N, H, W] but needs to be [N, C, H, W]
        if output.shape[0] == 1 and output.shape[1] == target.shape[0]:
            output = output.permute(1, 0, 2, 3).contiguous()
    
        # Check batch size consistency
        if output.shape[0] != target.shape[0]:
            raise ValueError(f"Batch size mismatch: output batch size {output.shape[0]} != target batch size {target.shape[0]}")
    
        # Check shapes: output should be 4D, target should be 3D
        if output.dim() != 4 or target.dim() != 3:
            raise ValueError(f"Expected output shape [B, C, H, W] and target shape [B, H, W], got {output.shape} and {target.shape}")
    
        # Validate target label values: must be in [0, n_classes-1] or equal to ignore_index
        # This check prevents CUDA assert errors in loss calculation
        if (target.min() < 0) or (target.max() >= n_classes and target.max() != self.ignore_index):
            raise ValueError(f"Target tensor contains invalid class indices outside [0, {n_classes - 1}] or ignore_index {self.ignore_index}")
    
        # Calculate per-pixel cross entropy loss (reduction='none' keeps shape)
        pixel_loss = self.CE(output, target)  # shape: [B, H, W]
    
        # Move pixel weights to the same device as the loss
        pixel_weights = pixel_weights.to(pixel_loss.device)
    
        # Pixel weights must match the pixel loss shape
        if pixel_weights.shape != pixel_loss.shape:
            raise ValueError(f"Pixel weights shape {pixel_weights.shape} does not match pixel loss shape {pixel_loss.shape}")
    
        # Mask for valid pixels (excluding ignore_index)
        valid_mask = (target != self.ignore_index).float()
    
        # Compute weighted loss masked on valid pixels
        weighted_loss = pixel_loss * pixel_weights * valid_mask
    
        # Normalize by sum of weights of valid pixels to keep scale consistent
        total_weight = (pixel_weights * valid_mask).sum() + 1e-8
    
        return weighted_loss.sum() / total_weight
    
    
