#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import logging

# try:
#     # Although we set the device by overriding MODEL.DEVICE, we still need to
#     # import pytorch_xla here to make sure that __init__ runs
#     import torch_xla
#     print('Successfully loaded XLA.')
# except ImportError:
#     print('XLA unavailable!')


from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.data import datasets as detectron2_datasets
from detectron2.layers.batch_norm import FrozenBatchNorm2d

# hacky way to register
from ubteacher.modeling import *
from ubteacher.engine import *
from ubteacher import add_ubteacher_config

# from detectron2.utils.logger import setup_logger
# setup_logger()

def build_argument_parser():
    parser = default_argument_parser()
    parser.add_argument('--disable-amp', action='store_true',
                        help='Override YAML setting for AMP, to enable CPU training.')
    parser.add_argument('--use-cpu', action='store_true',
                        help='Train on CPU rather than GPU.')
    parser.add_argument('--freeze-bn', action='store_true',
                        help='Replace batch norm with sync batch norm.')
    parser.add_argument('--ddp-teacher', action='store_true',
                        help='Wrap teacher in DDP')

    return parser

def custom_setup(args):
    """
    Nonstandard configuration for our environment.
    """
    detectron2_datasets.register_coco_instances(
        name="nyu_dl_train", metadata={},
        json_file="data/annotations/annotations_training.json",
        image_root="data/labeled_data/training/images")
    detectron2_datasets.register_coco_instances(
        name="nyu_dl_val", metadata={},
        json_file="data/annotations/annotations_validation.json",
        image_root="data/labeled_data/validation/images")
    detectron2_datasets.register_coco_instances(
        name="nyu_dl_unlabeled", metadata={},
        json_file="data/annotations/annotations_unlabeled.json",
        image_root="data/unlabeled_data/")



def setup(args):
    """
    Create configs and perform basic setups.
    """
    custom_setup(args)
    cfg = get_cfg()
    add_ubteacher_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)

    if args.disable_amp:
        print(f'Disabling AMP. Was: {cfg.SOLVER.AMP.ENABLED}.')
        cfg.defrost()
        cfg.SOLVER.AMP.ENABLED = False
        cfg.freeze()

    return cfg

def do_freeze(module):
    bn_module = torch.nn.modules.batchnorm
    bn_module = (bn_module.BatchNorm2d, bn_module.SyncBatchNorm)
    res = module
    if isinstance(module, bn_module):
        for param in module.parameters():
            param.requires_grad = False
    else:
        for child in module.children():
            do_freeze(child)



def maybe_freeze_bn(model, args):
    logger = logging.getLogger(__name__)
    print(f"Going to freeze batch norm? {args.freeze_bn}")
    logger.info(f"Going to freeze batch norm? {args.freeze_bn}")
    if args.freeze_bn:
        do_freeze(model)


    #     # https://detectron2.readthedocs.io/en/latest/modules/layers.html#detectron2.layers.FrozenBatchNorm2d.convert_frozen_batchnorm
    #     FrozenBatchNorm2d.convert_frozen_batchnorm(model)
    #     logger.info("Froze bathchnorm: ", model)
    #     print("Froze bathchnorm: ", model)

def main(args):
    cfg = setup(args)
    if cfg.SEMISUPNET.Trainer == "ubteacher":
        Trainer = UBTeacherTrainer
    elif cfg.SEMISUPNET.Trainer == "ubteacher_rcnn":
        Trainer = UBRCNNTeacherTrainer
    else:
        raise ValueError("Trainer Name is not found.")

    if args.eval_only:
        if cfg.SEMISUPNET.Trainer == "ubteacher":
            model = Trainer.build_model(cfg)
            model_teacher = Trainer.build_model(cfg)
            ensem_ts_model = EnsembleTSModel(model_teacher, model)

            DetectionCheckpointer(
                ensem_ts_model, save_dir=cfg.OUTPUT_DIR
            ).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
            res = Trainer.test(cfg, ensem_ts_model.modelTeacher)

        else:
            model = Trainer.build_model(cfg)
            DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
                cfg.MODEL.WEIGHTS, resume=args.resume
            )
            res = Trainer.test(cfg, model)
        return res

    trainer = Trainer(cfg, args)
    trainer.resume_or_load(resume=args.resume)
    # maybe_freeze_bn(trainer.ensem_ts_model, args)

    return trainer.train()


if __name__ == "__main__":
    args = build_argument_parser().parse_args()

    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
