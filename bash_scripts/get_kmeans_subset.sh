PYTHON='python'

hostname
nvidia-smi

export CUDA_VISIBLE_DEVICES=0

# Get unique log file,
SAVE_DIR=./XCon_outputs/outputs/

EXP_NUM=$(ls ${SAVE_DIR} | wc -l)
EXP_NUM=$((${EXP_NUM}+1))
echo $EXP_NUM

${PYTHON} -m methods.partitioning.kmeans_subset \
            --dataset_name 'cub' \
            --batch_size 256 \
            --grad_from_block 11 \
            --epochs 100 \
            --base_model vit_dino \
            --num_workers 4 \
            --use_ssb_splits 'True' \
            --weight_decay 5e-5 \
            --transform 'imagenet' \
            --lr 0.1 \
            --eval_funcs 'v2' \
            --pretrain_model 'dino' \
            --experts_num 8 \