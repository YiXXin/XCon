PYTHON='python'

hostname
nvidia-smi

export CUDA_VISIBLE_DEVICES=0

# Get unique log file,
SAVE_DIR=./XCon_outputs/outputs/

EXP_NUM=$(ls ${SAVE_DIR} | wc -l)
EXP_NUM=$((${EXP_NUM}+1))
echo $EXP_NUM

${PYTHON} -m methods.representation_learning.representation_learning \
            --dataset_name 'cub' \
            --batch_size 256 \
            --grad_from_block 11 \
            --epochs 200 \
            --base_model vit_dino \
            --num_workers 4 \
            --use_ssb_splits 'True' \
            --sup_con_weight 0.35 \
            --weight_decay 5e-5 \
            --contrast_unlabel_only 'False' \
            --transform 'imagenet' \
            --lr 0.1 \
            --eval_funcs 'v2' \
            --val_epoch_size 10 \
            --use_best_model 'True' \
            --use_global_con 'True' \
            --expert_weight 0.1 \
            --max_kmeans_iter 200 \
            --k_means_init 100 \
            --best_new 'False' \
            --pretrain_model 'dino' \
> ${SAVE_DIR}logfile_${EXP_NUM}.out