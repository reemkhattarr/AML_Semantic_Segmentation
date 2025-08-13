# utils/lovasz_loss.py
import torch
import torch.nn.functional as F

def lovasz_grad(gt_sorted):
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if gt_sorted.numel() > 1:
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard

def flatten_probas(probas, labels, ignore=None):
    if probas.dim() == 4:
        probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, probas.size(1))
    elif probas.dim() == 3:
        probas = probas.view(-1, probas.size(2))
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = (labels != ignore)
    return probas[valid], labels[valid]

def lovasz_softmax_flat(probas, labels, classes='present'):
    if probas.numel() == 0:
        return probas * 0.
    C = probas.size(1)
    losses, class_to_sum = [], (range(C) if classes in ['all','present'] else classes)
    for c in class_to_sum:
        fg = (labels == c).float()
        if classes == 'present' and fg.sum() == 0:
            continue
        prob = probas[:, 0] if C == 1 and labels.size(0) > 0 else (probas.squeeze() if C == 1 else probas[:, c])
        errors = (fg - prob).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        fg_sorted = fg[perm]
        grad = lovasz_grad(fg_sorted)
        losses.append(torch.dot(errors_sorted, grad))
    if len(losses) == 0:
        return torch.tensor(0., device=probas.device)
    return sum(losses) / len(losses)

def lovasz_softmax(probas, labels, classes='present', per_image=False, ignore=None):
    if per_image:
        loss = 0.
        for prob, lab in zip(probas, labels):
            loss += lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes)
        return loss / probas.size(0)
    return lovasz_softmax_flat(*flatten_probas(probas, labels, ignore), classes=classes)
