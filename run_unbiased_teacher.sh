python3 unbiased_teacher_v2/train_net.py  --num-gpus 1  --disable-amp --config-file unbiased_teacher_v2/configs/FCOS/coco-standard/fcos_R_50_ut2_custom_run0.yaml SOLVER.IMG_PER_BATCH_LABEL 8 SOLVER.IMG_PER_BATCH_UNLABEL 8 MODEL.DEVICE 'gpu' "$@"
