PYTHON='python'

hostname
nvidia-smi

export CUDA_VISIBLE_DEVICES=0

# Get unique log file
SAVE_DIR=./XCon_outputs/outputs/

EXP_NUM=$(ls ${SAVE_DIR} | wc -l)
EXP_NUM=$((${EXP_NUM}+1))
echo $EXP_NUM

${PYTHON} -m methods.clustering.ssk_means --dataset 'cub' --semi_sup 'True' --use_ssb_splits 'True' \
 --use_best_model 'True' --max_kmeans_iter 200 --k_means_init 100 --warmup_model_exp_id '(02.08.2022_|_25.076)'\
 > ${SAVE_DIR}logfile_${EXP_NUM}.out