#!/bin/bash
CIFAR='--data_path ./data/ --log_every 3125 --dataset cifar100 --cuda --log_dir logs/'
IMGNET='--data_path ./tiny-imagenet-200/ --regularization_layers 6 --log_every 3125 --dataset tinyimagenet --cuda --log_dir logs/'
FIVEDATA='--data_path ./data/ --log_every 3125 --dataset cifar100 --dataset_list cifar10-mnist-svhn-fmnist-notmnist --cuda --log_dir logs/'
SEED=0
GPU=0


## CIFAR-100 (20 Tasks)
CUDA_VISIBLE_DEVICES=$GPU python main.py $CIFAR --model amphibian --expt_name amphibian_cifar --batch_size 10 \
                    --glances 5 --opt_lr 1.0 --alpha_init 0.5 --n_epochs 1 --loader class_incremental_loader \
                    --arch "pc_cnn_gpm" --inner_batches 5 --learn_lr --class_order random --increment 5 \
                    --mask_type "relu_STE" --seed $SEED --grad_clip_norm 1.0 --calc_test_accuracy --validation 0.1 \
                    --weight_reg 0.0 --data_per_class 500 --num_fs_updates 10 \
                    --unique_id 'Amph_r0_CF' 

# TinyImageNet (40 Tasks)
CUDA_VISIBLE_DEVICES=$GPU python main.py $IMGNET --model amphibian --expt_name amphibian_tinyimagenet --batch_size 10 \
                    --glances 2 --opt_lr 0.5 --alpha_init 0.25 --n_epochs 1 --loader class_incremental_loader \
                    --arch "pc_cnn_gpm" --inner_batches 5 --learn_lr --class_order random --increment 5 \
                    --mask_type "relu_STE" --seed $SEED --grad_clip_norm 1.0 --calc_test_accuracy --validation 0.1 \
                    --weight_reg 0.0 --data_per_class 500 --num_fs_updates 10 \
                    --unique_id 'Amph_r0_TI' 

# 5-Datasets (25 Tasks)
CUDA_VISIBLE_DEVICES=$GPU python main_seq.py $FIVEDATA --model amphibian --expt_name amphibian_5D --batch_size 10 \
                    --glances 1 --opt_lr 1.0 --alpha_init 0.5 --n_epochs 1 --loader seq_class_incremental_loader \
                    --arch "pc_cnn_gpm_wide" --regularization_layers 5 --inner_batches 5 --learn_lr --class_order random --increment 2  \
                    --mask_type "relu_STE" --seed $SEED --grad_clip_norm 1.0 --calc_test_accuracy --validation 0.1 \
                    --weight_reg 0.0 --data_per_class 500 --num_fs_updates 10 \
                    --unique_id 'Amph_r0_5D' 


