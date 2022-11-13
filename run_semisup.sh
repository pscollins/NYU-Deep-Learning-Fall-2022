python -m torch.distributed.launch --nproc_per_node=1 swav/eval_semisup.py \
       --use_imagenet=False \
       --train_data_path=data/labeled_data/training \
       --val_data_path=data/labeled_data/training \
       --data_path /mnt/sdc/code/ml_datasets/imagenet/data/train.X1/ \
       --pretrained_from_hub=True  \
       --lr 0.01 \
       --lr_last_layer 0.2 \
       --dump_path=/tmp/semisup_out
