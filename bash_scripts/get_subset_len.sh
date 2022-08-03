PYTHON='python'

hostname
nvidia-smi

export CUDA_VISIBLE_DEVICES=0

# Get unique log file,
SAVE_DIR=./XCon_outputs/outputs/

EXP_NUM=$(ls ${SAVE_DIR} | wc -l)
EXP_NUM=$((${EXP_NUM}+1))
echo $EXP_NUM

${PYTHON} -m methods.partitioning.subset_len \
            --dataset_name 'cub' \
            --batch_size 256 \
            --num_workers 4 \
            --use_ssb_splits 'True' \
            --transform 'imagenet' \
            --eval_funcs 'v2' \
            --experts_num 8 \