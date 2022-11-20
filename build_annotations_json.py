import json
import os
import sys

import load_data

# usage:
#  python3 build_annotations_json.py data/labeled_data/training data/labeled_data/annotations_training.json
def main(argv):
    assert len(argv) == 3, f'Unexpected argument count: {len(argv)}'
    in_path = argv[1]
    out_path = argv[2]

    if os.path.isfile(out_path):
        raise ValueError(f'Annotations JSON already exists at {out_path}. Delte it first.')

    builder = load_data.CocoAnnotationBuilder(load_data.LabeledDataset(in_path))
    result = builder.build_coco()

    with open(out_path, 'w') as f:
        json.dump(result, f)

if __name__ == '__main__':
    main(sys.argv)
