#! /bin/bash

rm zip_files/team_19.zip
zip -r zip_files/team_19.zip . -x "data/labeled_data/*" \
    "data/unlabeled_data/*" \
    "*__pycache__/*" \
    "zip_files/*" \
    "env/*" \
    ".git/*" \
    "data/weights/.git/*"
