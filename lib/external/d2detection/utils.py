import os
import shutil
import itertools
from typing import Any, Dict, List, Set

import torch

from detectron2.solver import build_optimizer
from detectron2.engine import default_setup, DefaultTrainer
from detectron2.evaluation import COCOEvaluator, CityscapesInstanceEvaluator
from detectron2.data import (
    DatasetCatalog, MetadataCatalog, DatasetMapper,
    build_detection_train_loader, build_detection_test_loader, 
)
import detectron2.data.transforms as T
from detectron2.data.datasets import register_coco_instances # register_pascal_voc
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.data.datasets.pascal_voc import CLASS_NAMES, load_voc_instances

from .model import appendCfg
from .datasetMapper import DetrDatasetMapper
from .evaluator import PascalVOCDetectionEvaluator


def setup(args):
    cfg = appendCfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    
    # remove folder if exist
    if os.path.isdir(cfg.OUTPUT_DIR):
        shutil.rmtree(cfg.OUTPUT_DIR)
        
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def registerVOC(name, dirname, split, year, class_names=CLASS_NAMES, ColourMaps=None):
    # add ColourMaps registration
    DatasetCatalog.register(name, lambda: load_voc_instances(dirname, split, class_names))
    MetadataCatalog.get(name).set(
        thing_classes=list(class_names), thing_colors=ColourMaps, stuff_colors=ColourMaps,
        dirname=dirname, year=year, split=split,
    )


def registerMyD2Dataset():
    DatasetRoot = "./dataset"
    # VOC-like datasets, needs to add meta.year in evaluator
    # spermtrack
    registerVOC(
        name="spermtrack_2023_train", 
        dirname="%s/spermtrack" % DatasetRoot, 
        split="train", 
        year=2023,
        class_names=["sperm"],
        ColourMaps=[[0, 0, 255]], # register sperm bbox colours
    )
    
    registerVOC(
        name="spermtrack_2023_test", 
        dirname="%s/spermtrack" % DatasetRoot, 
        split="test", 
        year=2023,
        class_names=["sperm"],
        ColourMaps=[[0, 0, 255]],
    )

    # BCCD
    registerVOC(
        name="bccd_2021_train", 
        dirname="%s/BCCD" % DatasetRoot, 
        split="train", 
        year=2021,
        class_names=["RBC", "WBC", "Platelets"],
    )
    
    registerVOC(
        name="bccd_2021_test", 
        dirname="%s/BCCD" % DatasetRoot, 
        split="test", 
        year=2021,
        class_names=["RBC", "WBC", "Platelets"],
    )

    # COCO-like datasets
    # UTDAC2020
    register_coco_instances(
        name="utdac_2020_train", 
        metadata={},
        image_root="%s/UTDAC2020/train" % DatasetRoot, 
        json_file="%s/UTDAC2020/annotations/instances_train.json" % DatasetRoot,
    )
    
    register_coco_instances(
        name="utdac_2020_val", 
        metadata={},
        image_root="%s/UTDAC2020/val" % DatasetRoot, 
        json_file="%s/UTDAC2020/annotations/instances_val.json" % DatasetRoot,
    )
    
    # BDD100K
    # has errors, needs coco format
    register_coco_instances(
        name="bdd100k_train", 
        metadata={},
        image_root="%s/BDD100K/train" % DatasetRoot, 
        json_file="%s/BDD100K/annotations/bdd100k_coco_train.json" % DatasetRoot,
    )
    
    register_coco_instances(
        name="bdd100k_val", 
        metadata={},
        image_root="%s/UTDAC2020/val" % DatasetRoot, 
        json_file="%s/BDD100K/annotations/bdd100k_coco_val.json" % DatasetRoot,
    )

    return


class TrainEngine(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        # if "Detr" == cfg.MODEL.META_ARCHITECTURE:
        #     mapper = DetrDatasetMapper(cfg, True)
        # else:
        #     mapper = None
        if 'vit' not in cfg.MODEL.BACKBONE.NAME:
            return build_detection_train_loader(cfg)
        else:
            ResizeShape = cfg.MODEL.VIT.AUG_SHAPE
            Mapper = DatasetMapper(
                cfg, is_train=True, augmentations=[
                    T.Resize((ResizeShape, ResizeShape)),
                    T.RandomFlip(),
                ]
            )
            return build_detection_train_loader(cfg, mapper=Mapper)
        
    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        if 'vit' not in cfg.MODEL.BACKBONE.NAME:
            return build_detection_test_loader(cfg, dataset_name)
        else:
            ResizeShape = cfg.MODEL.VIT.AUG_SHAPE
            Mapper = DatasetMapper(
                cfg, is_train=False, augmentations=[T.Resize((ResizeShape, ResizeShape))]
            )
            return build_detection_test_loader(cfg, dataset_name, mapper=Mapper)
    
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
            
        COCOLikeDatasets = ["coco", "utdac", "bdd100k"]
        VOCLikeDatasets = ["voc", "spermtrack", "bccd"]
        if any([n in dataset_name for n in COCOLikeDatasets]):
            return COCOEvaluator(dataset_name, cfg, True, output_folder)
        elif any([n in dataset_name for n in VOCLikeDatasets]):
            return PascalVOCDetectionEvaluator(dataset_name)
        elif "city" in dataset_name:
            return CityscapesInstanceEvaluator(dataset_name)
    
    @classmethod
    def build_optimizer(cls, cfg, model):
        if cfg.MODEL.META_ARCHITECTURE != "Detr":
            return build_optimizer(cfg, model)
        else:
            # for DETR detection
            params: List[Dict[str, Any]] = []
            memo: Set[torch.nn.parameter.Parameter] = set()
            for key, value in model.named_parameters(recurse=True):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)
                lr = cfg.SOLVER.BASE_LR
                weight_decay = cfg.SOLVER.WEIGHT_DECAY
                if "backbone" in key:
                    lr = lr * cfg.SOLVER.BACKBONE_MULTIPLIER
                params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

            def maybe_add_full_model_gradient_clipping(optim):  # optim: the optimizer class
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