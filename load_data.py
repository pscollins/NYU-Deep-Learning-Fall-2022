from typing import List

import collections
import os

import torch
import torchvision
import dataclasses
import yaml



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

# def crop_tensor_to_bbox(img, bbox):
#     col0, row0, 1, y1 = bbox[0]


class LabeledDataset(torch.utils.data.Dataset):
    # transforms must have the signature
    #   (image_tensor, bboxes_tensor, class_tensor) ->
    #     (image_tensor, bboxes_tensor, class_tensor)
    def __init__(self, root_dir, transform=lambda *x: x):
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
        image_tensor = torchvision.io.read_image(image_path)

        with open(label_path, 'r') as f:
            labels = parse_labels(f.read())

        bbox_tensor = torch.tensor([l.bbox for l in labels])
        class_tensor = torch.tensor([self.class_index[l.label] for l in labels])

        result = self.transform(image_tensor, bbox_tensor, class_tensor)

        return result



# Returns crops from the labeled dataset corresponding to a single bounding
# box. For images containing multiple bounding boxes, a random one is chosen.
class ClassifierDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, inner_transform=lambda *x: x,
                 outer_transform=lambda *x: x):
        # inner_transform must have the signature for LabeledDataset's transform.
        #
        # outer_transform must have the signature
        #   (image_tensor) -> (image_tensor)
        self.labeled_dataset = LabeledDataset(root_dir,
                                              transform=inner_transform)
        self.outer_transform = outer_transform



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
                img = torchvision.io.read_image(path)
                break
            except:
                print(f'WARNING: corrupt image at {path}')
                # use a different, random image instead to work around corruption
                path = self.image_paths[torch.randint(low=0, high=len(self.image_paths),
                                                      size=(1,))[0]]

        img = self.read_image(path)
        img = self.transform(img)
        augmented = self.augment(img)
        return (augmented, img)
