
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

def modernize_cfg(cfg):
    # define some variables that otherwise cause issues
    _C = cfg

    _C.SOLVER.BASE_LR_END = 0.0
    _C.SOLVER.RESCALE_INTERVAL = False
    _C.SOLVER.NUM_DECAYS = 3

    _C.MODEL.ROI_BOX_HEAD.USE_SIGMOID_CE = False
    _C.MODEL.ROI_BOX_HEAD.USE_FED_LOSS = False
    _C.MODEL.ROI_BOX_HEAD.FED_LOSS_NUM_CLASSES = 50
    _C.MODEL.ROI_BOX_HEAD.FED_LOSS_FREQ_WEIGHT_POWER = .5


def _setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_ubteacher_config(cfg)
    modernize_cfg(cfg)
    cfg.merge_from_file(CONFIG_FILE_PATH)
    cfg.merge_from_file(CONFIG_OVERRIDES_PATH)
    if args is not None:
        cfg.merge_from_list(args.opts)
    cfg.freeze()
    print('Built cfg: ', cfg)

    return cfg

def get_model(args=None):
    cfg = _setup(args)

    if cfg.SEMISUPNET.Trainer == "ubteacher":
        Trainer = trainer.UBTeacherTrainer
    elif cfg.SEMISUPNET.Trainer == "ubteacher_rcnn":
        Trainer = trainer.UBRCNNTeacherTrainer

    if cfg.SEMISUPNET.Trainer == "ubteacher":
        model = Trainer.build_model(cfg)
        model_teacher = Trainer.build_model(cfg)
        ensem_ts_model = EnsembleTSModel(model_teacher, model)

        checkpoint = DetectionCheckpointer(
            ensem_ts_model, save_dir=cfg.OUTPUT_DIR, save_to_disk=False
        ).resume_or_load(cfg.MODEL.WEIGHTS, resume=True)

        # WARNING: Before unsupervised training begins, the teacher model is
        # left uninitialized, and only the student is trained. After supervised
        # training begins, the student is copied to the teacher, and the teacher
        # becomes "ground truth." So we choose which model to use based on the
        # iteration reported in the checkpoint.
        iteration = checkpoint.get('iteration', -1) + 1
        if iteration == 0:
            print('WARNING: Failed to load checkpoint!')
        if iteration <= (cfg.SEMISUPNET.BURN_UP_STEP + 100):
            print(f'Iteration {iteration} near burn in time ({cfg.SEMISUPNET.BURN_UP_STEP}): use student')
            res = ensem_ts_model.modelStudent
        else:
            print(f'Iteration {iteration} not near burn in time ({cfg.SEMISUPNET.BURN_UP_STEP}): use teacher.')
            res = ensem_ts_model.modelTeacher

    else:
        model = Trainer.build_model(cfg)
        model_teacher = Trainer.build_model(cfg)
        ensem_ts_model = EnsembleTSModel(model_teacher, model)
        checkpoint = DetectionCheckpointer(
            ensem_ts_model, save_dir=cfg.OUTPUT_DIR, save_to_disk=False
        ).resume_or_load(cfg.MODEL.WEIGHTS, resume=True)

        res = ensem_ts_model.modelTeacher

    wrapped = ModelWrapper(res, cfg)
    wrapped.eval()
    return wrapped
