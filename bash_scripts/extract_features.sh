PYTHON='python'

hostname
nvidia-smi

export CUDA_VISIBLE_DEVICES=0

${PYTHON} -m methods.clustering.extract_features --dataset cub --use_best_model 'True' \
 --warmup_model_dir './XCon_outputs/metric_learn_gcd/log/(02.08.2022_|_25.076)/checkpoints/model.pt'