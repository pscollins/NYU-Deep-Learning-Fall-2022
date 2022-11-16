export XRT_TPU_CONFIG="localservice;0;localhost:51011"
export XLA_USE_BF16=1
# export PT_XLA_DEBUG=1
python3 swav/main_swav.py \
--data_path /mnt/disks/data/unlabeled_data/ \
--unlabeled_dataset_path /mnt/disks/data/unlabeled_data/ \
--epochs 100 \
--base_lr 0.6 \
--final_lr 0.0006 \
--warmup_epochs 0 \
--batch_size 512 \
--size_crops 224 96 \
--nmb_crops 2 6 \
--min_scale_crops 0.14 0.05 \
--max_scale_crops 1. 0.14 \
--use_fp16 false \
--freeze_prototypes_niters 5005 \
--queue_length 3840 \
--workers 2 \
--epoch_queue_starts 15 --use_unlabeled_dataset=true --dump_path=/mnt/disks/data/exp_1 --no_use_dist "$@"
