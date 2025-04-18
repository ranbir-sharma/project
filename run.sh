# Removing all previous sample images
rm -rf samples/*
# rm -rf models/*

# python pcnn_train.py \
# --batch_size 16 \
# --sample_batch_size 16 \
# --sampling_interval 50 \
# --save_interval 50 \
# --dataset cpen455 \
# --nr_resnet 1 \
# --nr_filters 40 \
# --nr_logistic_mix 5 \
# --lr_decay 0.999995 \
# --max_epochs 500 \
# --en_wandb False \

python pcnn_train.py \
--batch_size 16 \
--sample_batch_size 16 \
--sampling_interval 50 \
--save_interval 50 \
--dataset cpen455 \
--nr_resnet 5 \
--nr_filters 40 \
--nr_logistic_mix 7 \
--lr_decay 0.999995 \
--max_epochs 800 \
--en_wandb True \
--load_params models/pcnn_cpen455_from_scratch_249.pth
>> output.txt