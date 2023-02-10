set -x

python -m torch.distributed.launch --nproc_per_node=1 swav/eval_semisup.py \
       --train_data_path data/labeled_data/training \
       --val_data_path=data/labeled_data/training \
       --data_path /mnt/sdc/code/ml_datasets/imagenet/data/train.X1/ \
       --lr 0.01 \
       --lr_last_layer 0.2 \
       --batch_size 32 \
       --dump_path=/tmp/semisup_out_2
       # --pretrained_from_hub=True  \
