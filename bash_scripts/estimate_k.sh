PYTHON='python'

hostname
export CUDA_VISIBLE_DEVICES=0

# Get unique log file
SAVE_DIR=./XCon_outputs/outputs/

EXP_NUM=$(ls ${SAVE_DIR} | wc -l)
EXP_NUM=$((${EXP_NUM}+1))
echo $EXP_NUM

${PYTHON} -m methods.estimate_k.estimate_k --max_classes 1000 --dataset_name cub --search_mode other \
 --warmup_model_exp_id '(02.08.2022_|_25.076)' --use_best_model 'True' \
 > ${SAVE_DIR}logfile_${EXP_NUM}.out