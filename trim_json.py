import os
import sys

import json

KEEP_COUNT = 10

def main(argv):
    _, infile, outfile = argv

    if os.path.isfile(outfile):
        raise ValueError(f'Annotations JSON already exists at {outfile}. Delte it first.')

    with open(infile, 'r') as f:
        all_annotations = json.loads(f.read())


    all_annotations['images'] = all_annotations['images'][:KEEP_COUNT]

    keep_anns = [
        ann for ann in all_annotations['annotations']
        if ann['image_id'] <= KEEP_COUNT
    ]
    all_annotations['annotations'] = keep_anns

    with open(outfile, 'w') as f:
        f.write(json.dumps(all_annotations))

if __name__ == '__main__':
    main(sys.argv)
