python -m torch.distributed.launch \
       --nproc_per_node=1 \
       --use_env detr/main.py \
       --dataset_file custom_cocolike \
       --swav_model_arch resnet50 \
       --swav_checkpoint_path /tmp/semisup_out/checkpoint.pth.tar \
       "$@"

#python -m torch.distributed.launch \
#       --nproc_per_node=1 \
#       --use_env detr/main.py \
#       --dataset_file custom_cocolike "$@"


# python -m torch.distributed.launch --nproc_per_node=1 --use_env detr/main.py --coco_path /mnt/sdc/code/ml_datasets/coco
