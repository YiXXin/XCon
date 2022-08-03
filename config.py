# -----------------
# DATASET ROOTS
# -----------------
cifar_10_root = '/data/dataset/cifar10'
cifar_100_root = '/data/dataset/cifar100'
cub_root = '/data/dataset/cub200'
aircraft_root = '/data/user-data/fgvc/aircraft/fgvc-aircraft-2013b'
imagenet_root = '/data/user-data/imagenet'
cars_root = '/data/user-data/fgvc/cars'

pets_root = '/data/user-data/fgvc/pets'
flower_root = '/data/user-data/fgvc/flower102'
food_root = '/data/user-data/fgvc/food-101'

# OSR Split dir
osr_split_dir = './data/ssb_splits'

# -----------------
# PRETRAIN PATHS
# -----------------
dino_pretrain_path = './pretrained_models/dino/dino_vitbase16_pretrain.pth'
moco_pretrain_path = './pretrained_models/moco/vit-b-300ep.pth.tar'
mae_pretrain_path = './pretrained_models/mae/mae_pretrain_vit_base.pth'

# Dataset partitioning paths
km_label_path = './partition_out/km_labels'
subset_len_path = './partition_out/subset_len'

# -----------------
# OTHER PATHS
# -----------------
feature_extract_dir = './XCon_outputs/extracted_features'     # Extract features to this directory
exp_root = './XCon_outputs'          # All logs and checkpoints will be saved here