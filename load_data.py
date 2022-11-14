from typing import List

import collections
import os

import PIL
import torch
import torchvision
import dataclasses
import yaml

from PIL import Image


DATA_ROOT = 'data/'
TRAINING_DATA_ROOT = os.path.join(DATA_ROOT, 'labeled_data', 'training')
VALIDATION_DATA_ROOT = os.path.join(DATA_ROOT, 'labeled_data', 'validation')

UNLABLED_DATA_ROOT = os.path.join(DATA_ROOT, 'unlabeled_data')

def _get_relative_paths_below(root):
    return [
        os.path.join(root, f)
        for f in os.listdir(root)
    ]

def get_labeled_image_paths(root):
    return _get_relative_paths_below(os.path.join(root, 'images'))


def get_labeled_label_paths(root):
    return _get_relative_paths_below(os.path.join(root, 'labels'))


def load_class_index():
    location = os.path.join(DATA_ROOT, 'class_index.yaml')
    with open(location, 'r') as f:
        result = yaml.safe_load(f)
    assert len(result) == 100
    return result

def load_inverted_class_index():
    class_index = load_class_index()
    return {
        v: k for k, v in class_index.items()
    }

def collect_paths_by_index(paths):
    def clean(path):
        basename = os.path.basename(path)
        name, ext = os.path.splitext(basename)
        assert ext, 'no extension!'
        return name
    result = {
        int(clean(path)): path
        for path in paths
    }
    assert len(result) == len(paths), 'data loss'
    return result


def collect_examples_by_index(image_paths, label_paths):
    images_by_index = collect_paths_by_index(image_paths)
    labels_by_index = collect_paths_by_index(label_paths)
    assert len(images_by_index) == len(labels_by_index), 'image/label mismatch'
    result = {
        index: (images_by_index[index], labels_by_index[index])
        for index in images_by_index
    }
    assert len(result) == len(images_by_index), 'data loss'
    return result


@dataclasses.dataclass
class Label:
    bbox: List[int]
    label: str
    # TODO(pscollins): Any use for the image size?

def parse_labels(labels_yaml):
    parsed = yaml.safe_load(labels_yaml)
    bboxes = parsed['bboxes']
    labels = parsed['labels']
    assert len(labels) == len(bboxes)
    return [
        Label(bbox=bbox, label=label)
        for bbox, label
        in zip(bboxes, labels)
    ]

def crop_tensor_to_bbox(img, bbox):
    # TODO(pscollins): consider batching
    if isinstance(bbox, torch.Tensor):
        x0, y0, x1, y1 = bbox.tolist()
    else:
        x0, y0, x1, y1 = bbox
    dx = x1 - x0
    dy = y1 - y0

    cropped = torchvision.transforms.functional.crop(
        img,
        top=y0,
        left=x0,
        height=dy,
        width=dx)
    return cropped


# https://pytorch.org/vision/main/_modules/torchvision/datasets/folder.html#ImageFolder
def pil_loader(path: str):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")

class LabeledDataset(torch.utils.data.Dataset):
    # transforms must have the signature
    #   (image_tensor, bboxes_tensor, class_tensor) ->
    #     (image_tensor, bboxes_tensor, class_tensor)
    def __init__(self, root_dir, load_image=torchvision.io.read_image,
                 transform=lambda *x: x):
        unsorted_examples_by_index = collect_examples_by_index(
            image_paths=get_labeled_image_paths(root_dir),
            label_paths=get_labeled_label_paths(root_dir)
        )
        # Sort for determinism, drop the key
        self.examples_by_index = list(
            (image_path, label_path)
            for key, (image_path, label_path)
            in sorted(unsorted_examples_by_index.items())
        )

        self.class_index = load_class_index()
        self.transform = transform
        self.load_image = load_image

        # TODO(pscollins): Add an option to prefetch into memory for faster
        # loading.

    def __len__(self):
        return len(self.examples_by_index)


    # returns (image, bboxes, classes)
    # with shape:
    #  ([C, H, W], [N, H, W], [N])
    #
    # where N is the number of bounding boxes in the specified image.
    def __getitem__(self, idx):
        image_path, label_path = self.examples_by_index[idx]
        # image_tensor = torchvision.io.read_image(image_path)
        image_tensor = self.load_image(image_path)

        with open(label_path, 'r') as f:
            labels = parse_labels(f.read())

        bbox_tensor = torch.tensor([l.bbox for l in labels])
        class_tensor = torch.tensor([self.class_index[l.label] for l in labels])

        result = self.transform(image_tensor, bbox_tensor, class_tensor)

        return result



# Returns crops from the labeled dataset corresponding to a single bounding
# box. For images containing multiple bounding boxes, a random one is chosen.
class ClassifierDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, load_image=torchvision.io.read_image,
                 inner_transform=lambda *x: x,
                 outer_transform=lambda x: x, dataset_factory=LabeledDataset):
        # inner_transform must have the signature for LabeledDataset's transform.
        #
        # outer_transform must have the signature
        #   (image_tensor) -> (image_tensor)
        self.labeled_dataset = dataset_factory(root_dir,
                                               load_image=load_image,
                                               transform=inner_transform)
        self.outer_transform = outer_transform

    def __len__(self):
        # just 1 crop per image, so no need to count bboxes
        return len(self.labeled_dataset)

    # returns (image, class)
    # with shape
    #   ([C, H, W], [1])
    def __getitem__(self, idx):
        img, bboxes, classes = self.labeled_dataset[idx]

        bbox_idx = torch.randint(low=0, high=len(bboxes), size=(1,)).item()
        cropped = crop_tensor_to_bbox(img, bboxes[bbox_idx])
        cropped = self.outer_transform(cropped)
        return (cropped, classes[bbox_idx])


# Returns data of the form
#    (augmented, original)
# where
#   original = transform(load_image(path))
#   augmented = augment(original)
class UnlabeledDataset(torch.utils.data.Dataset):
    # transform and augment
    def __init__(self, root_dir=UNLABLED_DATA_ROOT, transform=lambda x: x, augment=lambda x: x,
                 read_image=torchvision.io.read_image):
        # sort for determinism
        if root_dir is None:
            root_dir = UNLABLED_DATA_ROOT
        # TODO(pscollins): shuffle?
        self.image_paths = list(sorted(_get_relative_paths_below(root_dir)))
        self.transform = transform
        self.augment = augment

        # SWaV expects PIL images, so for SWAaV we override this function with PIL.image.open
        self.read_image = read_image

        # TODO(pscollins): Prefetching?

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]

        # work around corrupt images in input
        while True:
            try:
                # img = torchvision.io.read_image(path)
                img = self.read_image(path)
                break
            except:
                print(f'WARNING: corrupt image at {path}')
                # use a different, random image instead to work around corruption
                path = self.image_paths[torch.randint(low=0, high=len(self.image_paths),
                                                      size=(1,))[0]]

        img = self.transform(img)
        augmented = self.augment(img)
        return (augmented, img)


class DetrCocoWrapper(torch.utils.data.Dataset):
    # transform has the signature
    #   (image, target_dict) ->
    #     (image, target_dict)
    def __init__(self, labeled_dataset, transform=lambda *x: x):
        self.labeled_dataset = labeled_dataset
        self.transform = transform


    def __len__(self):
        return len(self.labeled_dataset)

    def __getitem__(self, idx):
        image, bboxes, classes = self.labeled_dataset[idx]
        assert bboxes.shape[0] == classes.shape[0]

        if isinstance(image, torch.Tensor):
            c, h, w = image.shape
        else:
            assert isinstance(image, PIL.Image.Image)
            w, h = image.size

        # just fill in what's needed for
        # https://github.com/facebookresearch/detr/blob/8a144f83a287f4d3fece4acdf073f387c5af387d/models/detr.py#L83
        target = {
            'image_id': torch.tensor(idx),
            # [N, 4]
            'boxes': bboxes,
            # [N]
            'labels': classes,
            # for compatibility with 'evaluate' in engine.py
            'orig_size': torch.tensor([h, w])
            }

        # ([C, H, W], {...})
        return self.transform(image, target)
