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
#     # sample_style = "choice"
#     # return T.ResizeShortestEdge(min_size, max_size, sample_style)
#     return

class ModelWrapper(torch.nn.Module):
    def __init__(self, inner_model, cfg):
        super().__init__()
        self.inner_model = inner_model
        # self.augmentation = utils.build_augmentation(cfg, is_train=False)
        # self.resize = build_resize(cfg)
        self.resize = torchvision_T.Resize(size=cfg.INPUT.MIN_SIZE_TEST,
                                                          max_size=cfg.INPUT.MAX_SIZE_TEST)

    def _transform(self, img):
        assert img.dim() == 3
        # rescale back to [0, 255]
        img = img * 255.0
        # resize
        img = self.resize(img)

        # transform = self.resize.get_transform(img)
        # # img = transform.apply_image(torchvision_T.ToPILImage()(img))
        # img = transform.apply_image(img.cpu().numpy())
        return img


    def forward(self, images):
        # print('images: ', images)
        # unwrap along the batch axis and fix up
        image_dict = [{'image': self._transform(x)} for x in images]
        preds =  self.inner_model(image_dict)
        # print('got preds: ', preds)
        # TODO: SCALE BBOXES BACK DOWN
        result = [instance_to_dict(pred['instances']) for pred in preds]
        # print('built result: ', result)
        return result
