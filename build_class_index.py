import itertools
import os

import load_data
import yaml

OUT_PATH = os.path.join(load_data.DATA_ROOT, 'class_index.yaml')

# Creates a mapping from label name to int and stores it in
# `data/class_index.yaml`. This is important to get right since nondeterminism
# in python set operations could otherwise end up changing the class labels on
# each invocation of the process and break compatibility with old checkpoints.
def main():
    if os.path.isfile(OUT_PATH):
        raise ValueError(f'Class index already exists at path {OUT_PATH}. Delete it first.')

    all_labels_paths = load_data.get_labeled_label_paths(
        load_data.TRAINING_DATA_ROOT)

    def load_labels(path):
        with open(path, 'r') as f:
            return {
                l.label
                for l in load_data.parse_labels(f.read())
            }

    all_labels = set(itertools.chain.from_iterable(map(load_labels, all_labels_paths)))

    # Sort for determinism
    label_to_index = {
        label: idx
        for idx, label
        in enumerate(sorted(all_labels))
    }

    with open(OUT_PATH, 'w') as f:
        f.write(yaml.dump(label_to_index))


if __name__ == '__main__':
    main()
