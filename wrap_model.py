import torch

import detectron2.data.detection_utils as utils
import detectron2.data.transforms as T
import torchvision.transforms as torchvision_T


def instance_to_dict(instance):
    def get_all(key):
        return get(instances.get_fields()[key])

    fields = instance.get_fields()
    # extract Boxes.tensor
    # https://detectron2.readthedocs.io/en/latest/_modules/detectron2/structures/boxes.html
    bboxes = fields['pred_boxes'].tensor
    labels = fields['pred_classes']
    scores = fields['scores']

    # if bboxes.shape[0] != 0:
    #     # print('found #', bboxes.shape)
    #     print('labels: ', labels)

    return {
        'boxes': bboxes,
        'labels': labels,
        'scores': scores,
    }

# def build_resize(cfg):
#     min_size = cfg.INPUT.MIN_SIZE_TEST
#     max_size = cfg.INPUT.MAX_SIZE_TEST
    # sample_style = "choice"
    # return T.ResizeShortestEdge(min_size, max_size, sample_style)
#     return

class ModelWrapper(torch.nn.Module):
    def __init__(self, inner_model, cfg):
        super().__init__()
        self.inner_model = inner_model
        # self.augmentation = utils.build_augmentation(cfg, is_train=False)
        # self.resize = build_resize(cfg)
        if cfg.INPUT.MIN_SIZE_TEST:
            self.resize = torchvision_T.Resize(size=cfg.INPUT.MIN_SIZE_TEST,
                                           max_size=cfg.INPUT.MAX_SIZE_TEST)
        else:
            self.resize = lambda x: x
        self.cfg = cfg

    def _transform(self, img):
        assert img.dim() == 3
        # rescale back to [0, 255]
        img = img * 255.0
        # resize
        img = self.resize(img)
        # cast to uint8
        img = img.type(torch.uint8)

        # default format in detectron2 is BGR (configured with INPUT.FORMAT), so
        # we trained with the channels flipped relative to what eval.py expects
        img = torch.flip(img, [0])
        return img


    def build_image_dict(self, image):
        _, old_h, old_w = image.shape
        return {
            # resize + rescale to [0, 255]
            'image': self._transform(image),
            # populating H and W lets detectron2 do the un-resize
            'height': old_h,
            'width': old_w,
        }

    def forward(self, images):
        image_dict = [self.build_image_dict(x) for x in images]
        if self.cfg.SEMISUPNET.Trainer == 'ubteacher':
            preds =  self.inner_model(image_dict, nms_method=self.cfg.MODEL.FCOS.NMS_CRITERIA_TEST)
        elif self.cfg.SEMISUPNET.Trainer == 'ubteacher_rcnn':
            preds =  self.inner_model(image_dict)
        else:
            raise ValueError('Unknown trainer: ', self.cfg.SEMISUPNET.Trainer)

        result = [instance_to_dict(pred['instances']) for pred in preds]
        return result
