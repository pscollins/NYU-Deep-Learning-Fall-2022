
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup
from detectron2.checkpoint import DetectionCheckpointer

from wrap_model import ModelWrapper

# HACK: make the imports in unbiased_teacher_v2/ work properly
import sys
sys.path.insert(0, 'unbiased_teacher_v2')

from ubteacher import add_ubteacher_config
from ubteacher.engine import trainer
from ubteacher.modeling.meta_arch.ts_ensemble import EnsembleTSModel

CONFIG_FILE_PATH = 'data/weights/config.yaml'
CONFIG_OVERRIDES_PATH = 'data/config_overrides.yaml'
CHECKPOINT_FILE_PATH = 'data/weights/checkpoint.pth'
# CHECKPOINT_FILE_PATH = 'data/weights/checkpoint.pth'

def _setup():
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_ubteacher_config(cfg)
    cfg.merge_from_file(CONFIG_FILE_PATH)
    cfg.merge_from_file(CONFIG_OVERRIDES_PATH)
    cfg.freeze()

    # default_setup(cfg, None)

    print('Built cfg: ', cfg)
    # if args.disable_amp:
    #     print(f'Disabling AMP. Was: {cfg.SOLVER.AMP.ENABLED}.')
    #     cfg.defrost()
    #     cfg.SOLVER.AMP.ENABLED = False
    #     cfg.freeze()

    return cfg

def get_model():
    cfg = _setup()

    if cfg.SEMISUPNET.Trainer == "ubteacher":
        Trainer = trainer.UBTeacherTrainer
    elif cfg.SEMISUPNET.Trainer == "ubteacher_rcnn":
        Trainer = trainer.UBRCNNTeacherTrainer

    if cfg.SEMISUPNET.Trainer == "ubteacher":
        model = Trainer.build_model(cfg)
        model_teacher = Trainer.build_model(cfg)
        ensem_ts_model = EnsembleTSModel(model_teacher, model)

        DetectionCheckpointer(
            ensem_ts_model, save_dir=cfg.OUTPUT_DIR, save_to_disk=False
        ).resume_or_load(cfg.MODEL.WEIGHTS, resume=True)

        res = ensem_ts_model.modelTeacher

    else:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR, save_to_disk=False).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=True,
        )
        res = model
    wrapped = ModelWrapper(res, cfg)
    wrapped.eval()
    return wrapped
