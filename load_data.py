from typing import List

import os

import torch
import dataclasses
import yaml



DATA_ROOT = 'data/'
TRAINING_DATA_ROOT = os.path.join(DATA_ROOT, 'labeled_data', 'training')
VALIDATION_DATA_ROOT = os.path.join(DATA_ROOT, 'labeled_data', 'validation')

UNLABLED_DATA_ROOT = os.path.join(DATA_ROOT, 'unlabeled_data')

def get_labeled_image_paths(root):
    image_path = os.path.join(root, 'images')
    return os.listdir(image_path)


def get_labeled_label_paths(root):
    label_path = os.path.join(root, 'labels')
    return os.listdir(label_path)


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

class LabeledDataSet(torch.utils.data.IterableDataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
