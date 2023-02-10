import argparse
import json
import os
import sys

import load_data

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_path', default='')
    parser.add_argument('--out_path', default='')
    parser.add_argument('--is_unlabeled', action='store_true',
                        help='If set, do not populate class/bbox annotations.')

    return parser


# usage:
#  $ python3 build_annotations_json.py --in_path=data/labeled_data/training --out_path=data/labeled_data/annotations_training.json
#  or
#  $ python3 build_annotations_json.py --in_path=data/unlabeled_data  --out_path=data/annotations_unlabeled.json --is_unlabeled
def main(args):
    if os.path.isfile(args.out_path):
        raise ValueError(f'Annotations JSON already exists at {out_path}. Delte it first.')

    if args.is_unlabeled:
        ds = load_data.UnlabeledDatasetLabelShim(args.in_path)
    else:
        ds = load_data.LabeledDataset(args.in_path)

    builder = load_data.CocoAnnotationBuilder(ds)
    result = builder.build_coco()

    with open(args.out_path, 'w') as f:
        json.dump(result, f)

if __name__ == '__main__':
    main(build_parser().parse_args())
