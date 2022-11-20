#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.data import datasets as detectron2_datasets

# hacky way to register
from ubteacher.modeling import *
from ubteacher.engine import *
from ubteacher import add_ubteacher_config


def build_argument_parser():
    parser = default_argument_parser()
    parser.add_argument('--disable-amp', action='store_true',
                        help='Override YAML setting for AMP, to enable CPU training.')
    return parser

def custom_setup(args):
    """
    Nonstandard configuration for our environment.
    """
    detectron2_datasets.register_coco_instances(
        name="nyu_dl_train", metadata={},
        json_file="data/labeled_data/annotations_training.json",
        image_root="data/labeled_data/training/images")
    detectron2_datasets.register_coco_instances(
        name="nyu_dl_val", metadata={},
        json_file="data/labeled_data/annotations_validation.json",
        image_root="data/labeled_data/validation/images")



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
