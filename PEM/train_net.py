# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
MaskFormer Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""
try:
    # ignore ShapelyDeprecationWarning from fvcore
    from shapely.errors import ShapelyDeprecationWarning
    import warnings
    warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)
    warnings.filterwarnings('ignore')
except:
    warnings.filterwarnings('ignore')
    pass

import copy
import itertools
import logging
import os

from collections import OrderedDict
from typing import Any, Dict, List, Set

import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_train_loader, build_detection_test_loader, DatasetCatalog
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
)
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger

# MaskFormer
from pem import (
    COCOInstanceNewBaselineDatasetMapper,
    COCOPanopticNewBaselineDatasetMapper,
    MaskFormerInstanceDatasetMapper,
    MaskFormerPanopticDatasetMapper,
    MaskFormerSemanticDatasetMapper,
    SemanticSegmentorWithTTA,
    add_maskformer2_config,
)

from register_loveda_areas import register_loveda_areas
from register_loveda_areas_raw import register_loveda_areas_raw
import math

from detectron2.utils.visualizer import Visualizer
from detectron2.utils.file_io import PathManager
from detectron2.data.detection_utils import read_image
import numpy as np
import cv2

#Albumentation
from pem.add_albu_cfg import add_albu_cfg

from typing import List

# -------- helpers (niente etichette, solo colori coerenti) --------
def _get_palette(metadata, num_classes: int) -> List[List[int]]:
    pal = metadata.get("stuff_colors", None)
    if pal is not None and len(pal) >= num_classes:
        return [list(map(int, c)) for c in pal[:num_classes]]
    rng = np.random.default_rng(0)
    pal = rng.integers(0, 255, size=(num_classes, 3), dtype=np.uint8)
    pal[0] = np.array([0, 0, 0], dtype=np.uint8)  # classe 0 nera (puoi cambiare)
    return pal.tolist()

def colorize_mask(mask: np.ndarray, palette: List[List[int]], ignore_id: int = 255) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    # ignora i pixel "ignore"
    valid = mask != ignore_id
    for cls_id, rgb in enumerate(palette):
        out[(mask == cls_id) & valid] = rgb
    return out

def overlay(img_rgb: np.ndarray, color_mask: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    return (img_rgb * (1 - alpha) + color_mask * alpha).astype(np.uint8)

# -------- main function --------
@torch.no_grad()
def dump_val_visuals(cfg, model, dataset_name=None, out_subdir="vis_val", max_images=50, ignore_id=255):
    """
    Salva immagini affiancate [INPUT | GT | PRED], senza etichette.
    Usa la stessa palette per GT e Pred. Output: <OUTPUT_DIR>/<out_subdir>/*.jpg
    """
    from detectron2.data import DatasetCatalog
    dataset_name = dataset_name or cfg.DATASETS.TEST[0]
    metadata = MetadataCatalog.get(dataset_name)

    # <<< FIX BN in eval >>>
    model.eval()
    for m in model.modules():
        if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.SyncBatchNorm)):
            m.eval()
    # <<<

    out_dir = os.path.join(cfg.OUTPUT_DIR, out_subdir)
    os.makedirs(out_dir, exist_ok=True)

    # lookup GT path per ogni immagine
    dataset_dicts = DatasetCatalog.get(dataset_name)
    gt_lookup = {d["file_name"]: d.get("sem_seg_file_name", None) for d in dataset_dicts}

    loader = build_detection_test_loader(cfg, dataset_name)
    count = 0

    for batch in loader:
        inp = batch[0]
        file_name = inp["file_name"]

        # --- INPUT ---
        img = read_image(file_name, format=metadata.get("image_format", "BGR"))
        img_rgb = img[:, :, ::-1] if img.shape[-1] == 3 else img

        # --- GT ---
        gt_mask = None
        gt_path = gt_lookup.get(file_name, None)
        if gt_path is not None:
            with PathManager.open(gt_path, "rb") as f:
                gt_gray = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_UNCHANGED)
            if gt_gray is not None:
                if gt_gray.ndim == 3:
                    gt_gray = gt_gray[:, :, 0]
                gt_mask = gt_gray.astype(np.int32)

        # --- PRED ---
        outputs = model(batch)
        sem_logits = outputs[0]["sem_seg"]            # [C,H,W]
        num_classes = sem_logits.shape[0]
        pred = sem_logits.argmax(dim=0).to(torch.int32)
        pred_np = pred.cpu().numpy()

        # # --- auto-fix offset classi (±1) ---
        # if gt_mask is not None:
        #     u = np.unique(gt_mask)
        #     if u.min() == 1 and u.max() <= num_classes:
        #         gt_mask = gt_mask - 1
        #     gt_mask = np.where(gt_mask == 255, 255, np.clip(gt_mask, 0, num_classes - 1))

        # --- palette condivisa ---
        palette = _get_palette(metadata, num_classes)

        # --- visualizzazioni ---
        if gt_mask is not None:
            gt_color = colorize_mask(gt_mask, palette, ignore_id=ignore_id)
            gt_vis = overlay(img_rgb, gt_color, alpha=0.6)
        else:
            gt_vis = np.zeros_like(img_rgb)

        pred_color = colorize_mask(pred_np, palette, ignore_id=ignore_id)
        pred_vis = overlay(img_rgb, pred_color, alpha=0.6)

        # concat [INPUT | GT | PRED]
        h = max(img_rgb.shape[0], gt_vis.shape[0], pred_vis.shape[0])
        def pad_h(x):
            if x.shape[0] == h: return x
            return np.pad(x, ((0, h - x.shape[0]), (0, 0), (0, 0)), mode="constant")
        triptych = np.concatenate([pad_h(img_rgb), pad_h(gt_vis), pad_h(pred_vis)], axis=1)

        base = os.path.splitext(os.path.basename(file_name))[0]
        out_path = os.path.join(out_dir, f"{base}_pred_gt.jpg")
        cv2.imwrite(out_path, triptych[:, :, ::-1])  # RGB->BGR per imwrite

        count += 1
        if count >= max_images:
            break

    print(f"[dump_val_visuals] salvate {count} immagini in {out_dir}")


class Trainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to MaskFormer.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each
        builtin dataset. For your own dataset, you can simply create an
        evaluator manually in your script and do not have to worry about the
        hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        # semantic segmentation
        if evaluator_type in ["sem_seg", "ade20k_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                )
            )
        # instance segmentation
        if evaluator_type == "coco":
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        # panoptic segmentation
        if evaluator_type in [
            "coco_panoptic_seg",
            "ade20k_panoptic_seg",
            "cityscapes_panoptic_seg",
            "mapillary_vistas_panoptic_seg",
        ]:
            if cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON:
                evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        # COCO
        if evaluator_type == "coco_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type == "coco_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON:
            evaluator_list.append(SemSegEvaluator(dataset_name, distributed=True, output_dir=output_folder))
        # Mapillary Vistas
        if evaluator_type == "mapillary_vistas_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON:
            evaluator_list.append(SemSegEvaluator(dataset_name, distributed=True, output_dir=output_folder))
        # Cityscapes
        if evaluator_type == "cityscapes_instance":
            assert (
                torch.cuda.device_count() > comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                torch.cuda.device_count() > comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        if evaluator_type == "cityscapes_panoptic_seg":
            if cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON:
                assert (
                    torch.cuda.device_count() > comm.get_rank()
                ), "CityscapesEvaluator currently do not work with multiple machines."
                evaluator_list.append(CityscapesSemSegEvaluator(dataset_name))
            if cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
                assert (
                    torch.cuda.device_count() > comm.get_rank()
                ), "CityscapesEvaluator currently do not work with multiple machines."
                evaluator_list.append(CityscapesInstanceEvaluator(dataset_name))
        # LVIS
        if evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, output_dir=output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        # Semantic segmentation dataset mapper
        if cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_semantic":
            mapper = MaskFormerSemanticDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # Panoptic segmentation dataset mapper
        elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_panoptic":
            mapper = MaskFormerPanopticDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # Instance segmentation dataset mapper
        elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_instance":
            mapper = MaskFormerInstanceDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # coco instance segmentation lsj new baseline
        elif cfg.INPUT.DATASET_MAPPER_NAME == "coco_instance_lsj":
            mapper = COCOInstanceNewBaselineDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # coco panoptic segmentation lsj new baseline
        elif cfg.INPUT.DATASET_MAPPER_NAME == "coco_panoptic_lsj":
            mapper = COCOPanopticNewBaselineDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        else:
            mapper = None
            return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
                ):
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA.
        logger.info("Running inference with test-time augmentation ...")
        model = SemanticSegmentorWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res

def compute_iters_per_epoch(cfg):
    train_sets = cfg.DATASETS.TRAIN
    # Conta le immagini effettive nei dataset di training
    n_images = sum(len(DatasetCatalog.get(name)) for name in train_sets)
    iters_per_epoch = math.ceil(n_images / cfg.SOLVER.IMS_PER_BATCH)
    return n_images, iters_per_epoch

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)

    #Add Albumentation config
    add_albu_cfg(cfg)

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "mask_former" module
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask2former")
    return cfg


def main(args):
    # IMG_ROOT = "/content/drive/MyDrive/AML_Semantic_Segmentation/data/LoveDA"
    # REMAPPED = "/content/drive/MyDrive/AML_Semantic_Segmentation/data/LoveDA_remapped"

    # # fallo PRIMA di setup(cfg)/Trainer, nello stesso processo del training
    # register_loveda_areas(
    #     img_root=IMG_ROOT,
    #     mask_root=None,                 # le maschere sorgenti sono nello stesso root di LoveDA
    #     remap_cache_root=REMAPPED,      # abilita remap + cache
    #     ignore=255,                     # (default se remap è 255)
    #     make_combined=True,             # anche loveda_train/val
    #     warm_cache=True                 # crea subito Train e Val rimappati
    # )

    register_loveda_areas_raw("/content/drive/MyDrive/AML_Semantic_Segmentation/data/LoveDA", make_combined=False)

    #Verifica creazione dataset
    print(DatasetCatalog.list())

    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
            dump_val_visuals(cfg, model, dataset_name=cfg.DATASETS.TEST[0],
                            out_subdir="vis_val_2", max_images=50, ignore_id=255)
        return res


    #Calcolo dinamico epoche
    n_images, iters_per_epoch = compute_iters_per_epoch(cfg)
    epochs = 20

    cfg.defrost()
    
    cfg.SOLVER.MAX_ITER = epochs * iters_per_epoch

    # Comodo: valuta e salva un checkpoint ogni "epoca"
    cfg.TEST.EVAL_PERIOD = iters_per_epoch
    cfg.SOLVER.CHECKPOINT_PERIOD = iters_per_epoch

    cfg.freeze()

    print(f"[Auto-epochs] N={n_images}, ims_per_batch={cfg.SOLVER.IMS_PER_BATCH}, "
          f"iters/epoch={iters_per_epoch}, MAX_ITER={cfg.SOLVER.MAX_ITER}")

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url='auto',
        args=(args,),
    )
