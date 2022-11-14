
python -m torch.distributed.launch \
       --nproc_per_node=1 \
       --use_env detr/main.py \
       --dataset_file custom_cocolike


# python -m torch.distributed.launch --nproc_per_node=1 --use_env detr/main.py --coco_path /mnt/sdc/code/ml_datasets/coco
