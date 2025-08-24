import torch
import torch.nn.functional as F
import numpy as np
import random

def generate_pseudo_labels(model, images, threshold=0.968, ignore_label=255):
    model.eval()
    with torch.no_grad():
        outputs = model(images)
        if isinstance(outputs, (list, tuple)):
            # select the main output (usually the second last)
            outputs = outputs[-2] if len(outputs) > 1 else outputs[0]
        if outputs.shape[2:] != images.shape[2:]:
            outputs = F.interpolate(outputs, size=images.shape[2:], mode='bilinear', align_corners=False)
        probs = F.softmax(outputs, dim=1)
        conf, pseudo_labels = torch.max(probs, dim=1)
        mask = conf.ge(threshold).float()
        pseudo_labels[mask == 0] = ignore_label
    model.train()
    return pseudo_labels
    
import torch
import numpy as np

def classmix(
    source_img, source_lbl, target_img, target_plbl,
    ignore_label=255, classmix_frac=0.5, min_valid_pixels=100
):
    """
    Robust ClassMix: Mixes source and target images/labels based on randomly selected classes from the source label,
    ensuring the mixed label always contains valid (non-ignore) pixels to prevent OHEM errors.

    Args:
        source_img: [B, C, H, W] Source images
        source_lbl: [B, H, W]    Source labels
        target_img: [B, C, H, W] Target images
        target_plbl: [B, H, W]   Target pseudo-labels
        ignore_label: int, label to ignore (default: 255)
        classmix_frac: float, fraction of source classes to select
        min_valid_pixels: int, minimum valid pixels required in the mixed label

    Returns:
        mixed_img: [B, C, H, W]
        mixed_lbl: [B, H, W]
        source_mask: [B, H, W] (bool) mask of source pixels in the mix
    """
    B, _, H, W = source_img.shape
    mixed_img = source_img.clone()
    mixed_lbl = source_lbl.clone()
    source_mask = torch.zeros_like(source_lbl, dtype=torch.bool)

    for i in range(B):
        # Count valid pixels in source and target
        source_valid = torch.sum(source_lbl[i] != ignore_label).item()
        target_valid = torch.sum(target_plbl[i] != ignore_label).item()

        # SAFEGUARD 1: If both source and target lack valid pixels, create a minimal valid region
        if source_valid < min_valid_pixels and target_valid < min_valid_pixels:
            # Emergency fallback: create a small valid region
            mixed_lbl[i, 0:10, 0:10] = 1  # Set a small region to class 1
            source_mask[i] = torch.ones_like(source_lbl[i], dtype=torch.bool)
            continue

        # Get unique valid classes in source (excluding ignore and background)
        classes = torch.unique(source_lbl[i])
        classes = classes[(classes != ignore_label) & (classes != 0)]

        # SAFEGUARD 2: If no valid source classes, use target if possible
        if len(classes) == 0:
            if target_valid > 0:
                mixed_img[i] = target_img[i]
                mixed_lbl[i] = target_plbl[i]
                source_mask[i] = torch.zeros_like(source_lbl[i], dtype=torch.bool)
            else:
                mixed_lbl[i, 0:10, 0:10] = 1  # Emergency valid region
                source_mask[i] = torch.ones_like(source_lbl[i], dtype=torch.bool)
            continue

        # Try multiple attempts to create a valid mask
        mask = None
        best_valid_count = 0
        best_mask = None

        for attempt in range(10):
            n_select = max(1, int(len(classes) * classmix_frac))
            selected = np.random.choice(classes.cpu().numpy(), n_select, replace=False)
            temp_mask = torch.zeros_like(source_lbl[i], dtype=torch.bool)
            for c in selected:
                temp_mask |= (source_lbl[i] == c)

            if temp_mask.sum() > 0:
                # Test the mixed label
                test_mixed = torch.where(temp_mask, source_lbl[i], target_plbl[i])
                ignore_mask = (source_lbl[i] == ignore_label) | (target_plbl[i] == ignore_label)
                test_mixed[ignore_mask] = ignore_label
                valid_count = torch.sum(test_mixed != ignore_label).item()
                if valid_count >= min_valid_pixels:
                    mask = temp_mask
                    break
                elif valid_count > best_valid_count:
                    best_valid_count = valid_count
                    best_mask = temp_mask.clone()

        # SAFEGUARD 3: Use best available mask or fallback to valid regions
        if mask is None:
            if best_mask is not None and best_valid_count > 0:
                mask = best_mask
            else:
                # Use all valid source pixels if possible, else all valid target pixels
                source_valid_mask = (source_lbl[i] != ignore_label) & (source_lbl[i] != 0)
                target_valid_mask = (target_plbl[i] != ignore_label) & (target_plbl[i] != 0)
                mask = source_valid_mask if source_valid_mask.sum() >= target_valid_mask.sum() else ~target_valid_mask

        # Mix images and labels
        mask_3ch = mask.unsqueeze(0).repeat(source_img.size(1), 1, 1)
        mixed_img[i] = torch.where(mask_3ch, source_img[i], target_img[i])
        mixed_lbl[i] = torch.where(mask, source_lbl[i], target_plbl[i])
        ignore_mask = (source_lbl[i] == ignore_label) | (target_plbl[i] == ignore_label)
        mixed_lbl[i][ignore_mask] = ignore_label

        # SAFEGUARD 4: Final validation and emergency correction
        final_valid_count = torch.sum(mixed_lbl[i] != ignore_label).item()
        if final_valid_count < min_valid_pixels:
            # Find any available valid class from source or target
            available_classes = []
            for cls in torch.cat([torch.unique(source_lbl[i]), torch.unique(target_plbl[i])]):
                if cls != ignore_label and cls != 0:
                    available_classes.append(cls.item())
            emergency_class = available_classes[0] if available_classes else 1
            emergency_size = max(10, int(np.sqrt(min_valid_pixels)))
            mixed_lbl[i, 0:emergency_size, 0:emergency_size] = emergency_class

        source_mask[i] = mask

    return mixed_img, mixed_lbl, source_mask



'''
def classmix(source_img, source_lbl, target_img, target_plbl, ignore_label=255, classmix_frac=0.5):
    """
    ClassMix: Mixes source and target images/labels based on randomly selected classes from the source label.
    Args:
        source_img: [B, C, H, W] Source images
        source_lbl: [B, H, W]    Source labels
        target_img: [B, C, H, W] Target images
        target_plbl: [B, H, W]   Target pseudo-labels
        ignore_label: int, label to ignore
    Returns:
        mixed_img: [B, C, H, W]
        mixed_lbl: [B, H, W]
        source_mask: [B, H, W] (bool) mask of source pixels in the mix
    """
    B, _, H, W = source_img.shape
    mixed_img = source_img.clone()
    mixed_lbl = source_lbl.clone()
    source_mask = torch.zeros_like(source_lbl, dtype=torch.bool)

    for i in range(B):
        # Get unique classes in this source label (excluding ignore)
        classes = torch.unique(source_lbl[i])
        classes = classes[classes != ignore_label]
        # if we want to ignore background
        # classes = classes[(classes != 0)]
        if len(classes) == 0:
            continue
        for attempt in range(5):
            n_select = max(1, int(len(classes) * classmix_frac))
            selected = np.random.choice(classes.cpu(), n_select, replace=False)
            mask = torch.zeros_like(source_lbl[i], dtype=torch.bool)
            for c in selected:
                mask |= (source_lbl[i] == c)
            if mask.sum() > 0:
                break
        else:
            # If after max_tries the mask is still empty, just use all source
            mask = torch.ones_like(source_lbl[i], dtype=torch.bool)
        # Mix: where mask is True, keep source; where False, use target
        mask_3ch = mask.unsqueeze(0).repeat(source_img.size(1), 1, 1)
        mixed_img[i] = torch.where(mask_3ch, source_img[i], target_img[i])
        mixed_lbl[i] = torch.where(mask, source_lbl[i], target_plbl[i])
        # set ignore label where either input is ignore
        mixed_lbl[i][(source_lbl[i] == ignore_label) | (target_plbl[i] == ignore_label)] = ignore_label
        source_mask[i] = mask

    return mixed_img, mixed_lbl, source_mask
'''

def generate_edge_map(label, edge_size=3, ignore_label=255):
    # label: [B, H, W]
    # Returns: [B, H, W] binary edge map
    edge = torch.zeros_like(label, dtype=torch.float)
    for i in range(label.size(0)):
        lbl = label[i].cpu().numpy()
        # Use simple morphological gradient (dilation - erosion)
        from scipy.ndimage import binary_dilation, binary_erosion
        mask = (lbl != ignore_label)
        dil = binary_dilation(mask, iterations=edge_size)
        ero = binary_erosion(mask, iterations=edge_size)
        edge_map = (dil ^ ero).astype(np.float32)
        edge[i] = torch.from_numpy(edge_map)
    return edge

