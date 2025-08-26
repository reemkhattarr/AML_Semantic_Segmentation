import torch
import torch.nn.functional as F
import numpy as np
import random

# def generate_pseudo_labels(model, images, threshold=0.968, ignore_label=255):
#     model.eval()
#     with torch.no_grad():
#         outputs = model(images)
#         if isinstance(outputs, (list, tuple)):
#             # select the main output (usually the second last)
#             outputs = outputs[-2] if len(outputs) > 1 else outputs[0]
#         if outputs.shape[2:] != images.shape[2:]:
#             outputs = F.interpolate(outputs, size=images.shape[2:], mode='bilinear', align_corners=False)
#         probs = F.softmax(outputs, dim=1)
#         conf, pseudo_labels = torch.max(probs, dim=1)
#         mask = conf.ge(threshold).float()
#         pseudo_labels[mask == 0] = ignore_label
#     model.train()
#     return pseudo_labels
    
def generate_pseudo_labels(model, images, threshold=0.968, ignore_label=255):
    model.eval()
    with torch.no_grad():
        outputs = model(images)
        if isinstance(outputs, (list, tuple)):
            outputs = outputs[-2] if len(outputs) > 1 else outputs[0]
        if outputs.shape[2:] != images.shape[2:]:
            outputs = F.interpolate(outputs, size=images.shape[2:], mode='bilinear', align_corners=False)
        probs = F.softmax(outputs, dim=1)
        conf, pseudo_labels = torch.max(probs, dim=1)
        mask = conf.ge(threshold).float()
    model.train()
    return pseudo_labels, mask



# def classmix(source_img, source_lbl, target_img, target_plbl, ignore_label=255, classmix_frac=0.5):
#     """
#     ClassMix: Mixes source and target images/labels based on randomly selected classes from the source label.
#     Args:
#         source_img: [B, C, H, W] Source images
#         source_lbl: [B, H, W]    Source labels
#         target_img: [B, C, H, W] Target images
#         target_plbl: [B, H, W]   Target pseudo-labels
#         ignore_label: int, label to ignore
#     Returns:
#         mixed_img: [B, C, H, W]
#         mixed_lbl: [B, H, W]
#         source_mask: [B, H, W] (bool) mask of source pixels in the mix
#     """
#     B, _, H, W = source_img.shape
#     mixed_img = source_img.clone()
#     mixed_lbl = source_lbl.clone()
#     source_mask = torch.zeros_like(source_lbl, dtype=torch.bool)

#     for i in range(B):
#         # Get unique classes in this source label (excluding ignore)
#         classes = torch.unique(source_lbl[i])
#         classes = classes[classes != ignore_label]
#         # if we want to ignore background
#         # classes = classes[(classes != 0)]
#         if len(classes) == 0:
#             continue
#         for attempt in range(5):
#             n_select = max(1, int(len(classes) * classmix_frac))
#             selected = np.random.choice(classes.cpu(), n_select, replace=False)
#             mask = torch.zeros_like(source_lbl[i], dtype=torch.bool)
#             for c in selected:
#                 mask |= (source_lbl[i] == c)
#             if mask.sum() > 0:
#                 break
#         else:
#             # If after max_tries the mask is still empty, just use all source
#             mask = torch.ones_like(source_lbl[i], dtype=torch.bool)
#         # Mix: where mask is True, keep source; where False, use target
#         mask_3ch = mask.unsqueeze(0).repeat(source_img.size(1), 1, 1)
#         mixed_img[i] = torch.where(mask_3ch, source_img[i], target_img[i])
#         mixed_lbl[i] = torch.where(mask, source_lbl[i], target_plbl[i])
#         # set ignore label where either input is ignore
#         mixed_lbl[i][(source_lbl[i] == ignore_label) | (target_plbl[i] == ignore_label)] = ignore_label
#         source_mask[i] = mask

#     return mixed_img, mixed_lbl, source_mask


def classmix(source_img, source_lbl, target_img, target_plbl, 
             source_weight=None, target_weight=None, 
             classmix_frac=0.5, ignore_label=255):
    """
    ClassMix (code 2 style): For each image in the batch, randomly select a subset of classes
    from the source label, and create a binary mask. Use this mask to mix source and target images/labels.
    Optionally, mix per-pixel weights as well.
    """
    B, C, H, W = source_img.shape
    mixed_img = torch.zeros_like(source_img)
    mixed_lbl = torch.zeros_like(source_lbl)
    source_mask = torch.zeros_like(source_lbl, dtype=torch.bool)
    mixed_weight = torch.ones_like(source_lbl, dtype=torch.float) if (source_weight is not None or target_weight is not None) else None

    for i in range(B):
        # 1. Get valid classes (ignore ignore_label and optionally background)
        classes = torch.unique(source_lbl[i])
        classes = classes[(classes != ignore_label) & (classes != 0)]
        if len(classes) == 0:
            # fallback: keep target
            mixed_img[i] = target_img[i]
            mixed_lbl[i] = target_plbl[i]
            if mixed_weight is not None and target_weight is not None:
                mixed_weight[i] = target_weight[i]
            continue

        # 2. Randomly select half the classes (rounded up)
        n_select = max(1, int(np.ceil(len(classes) / 2)))
        selected = np.random.choice(classes.cpu().numpy(), n_select, replace=False)
        selected = torch.from_numpy(selected).to(source_lbl.device)

        # 3. Generate class mask: 1 where pixel belongs to selected classes, else 0
        mask = torch.isin(source_lbl[i], selected)
        mask_3ch = mask.unsqueeze(0).expand(C, H, W)

        # 4. Mix images and labels
        mixed_img[i] = torch.where(mask_3ch, source_img[i], target_img[i])
        mixed_lbl[i] = torch.where(mask, source_lbl[i], target_plbl[i])
        mixed_lbl[i][(source_lbl[i] == ignore_label) | (target_plbl[i] == ignore_label)] = ignore_label
        source_mask[i] = mask

        # 5. Mix weights if provided
        if mixed_weight is not None:
            if source_weight is not None and target_weight is not None:
                mixed_weight[i] = torch.where(mask, source_weight[i], target_weight[i])
            elif target_weight is not None:
                mixed_weight[i] = target_weight[i]
            # else: default to 1

    if mixed_weight is not None:
        return mixed_img, mixed_lbl, source_mask, mixed_weight
    else:
        return mixed_img, mixed_lbl, source_mask




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

