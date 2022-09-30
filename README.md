# XCon: Learning with Experts for Fine-grained Category Discovery
This repo contains the implementation of our paper: "XCon: Learning with Experts for Fine-grained Category Discovery". ([arXiv](https://arxiv.org/abs/2208.01898))
## Abstract
We address the problem of generalized category discovery (GCD) in this paper, 
i.e. clustering the unlabeled images leveraging the information from a set of
seen classes, where the unlabeled images could contain both seen classes and
unseen classes. The seen classes can be seen as an implicit criterion of
classes, which makes this setting different from unsupervised clustering where
the cluster criteria may be ambiguous. We mainly concern the problem of
discovering categories within a fine-grained dataset since it is one of the
most direct applications of category discovery, i.e. helping experts discover
novel concepts within an unlabeled dataset using the implicit criterion set
forth by the seen classes. State-of-the-art methods for generalized category
discovery leverage contrastive learning to learn the representations, but the
large inter-class similarity and intra-class variance pose a challenge for the
methods because the negative examples may contain irrelevant cues for
recognizing a category so the algorithms may converge to a local-minima. We
present a novel method called Expert-Contrastive Learning (XCon) to help the
model to mine useful information from the images by first partitioning the
dataset into sub-datasets using k-means clustering and then performing
contrastive learning on each of the sub-datasets to learn fine-grained
discriminative features. Experiments on fine-grained datasets show a clear
improved performance over the previous best methods, indicating the
effectiveness of our method.

![image](https://github.com/YiXXin/XCon/blob/master/assets/overview.png)

## Requirements
- Python 3.8
- Pytorch 1.10.0
- torchvision 0.11.1
```
pip install -r requirements.txt
```

## Datasets
In our experiments, we use generic image classification datasets including [CIFAR-10/100](https://www.cs.toronto.edu/~kriz/cifar.html) and [ImageNet](https://image-net.org/download.php).

We also use fine-grained image classification datasets including [CUB-200](https://www.kaggle.com/datasets/coolerextreme/cub-200-2011/versions/1), [Stanford-Cars](http://ai.stanford.edu/~jkrause/cars/car_dataset.html), [FGVC-Aircraft](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/) and [Oxford-Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/).

## Pretrained Checkpoints
Our model is initialized with the parameters pretrained by DINO on ImageNet.
The DINO checkpoint of ViT-B-16 is available at [here](https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain_full_checkpoint.pth).

## Training and Evaluation Instructions
### Step 1. Set config
Set the path of datasets and the directory for saving outputs in ```config.py```.
### Step 2. Dataset partitioning
- Get the k-means labels for partitioning the dataset.
```
bash bash_scripts/get_kmeans_subset.sh
```
- Get the length of k expert sub-datasets.
```
bash bash_scripts/get_subset_len.sh
```
### Step 3. Representation learning
Fine-tune the model with the evaluation of semi-supervised k-means.
```
bash bash_scripts/representation_learning.sh
```
### Step 4. Semi-supervised k-means
To run the semi-supervised k-means alone by first running
```
bash bash_scripts/extract_features.sh
```
and then running
```
bash bash_scripts/ssk_means.sh
```
### Step 5. Estimate the number of classes
To estimate the number of classes in the unlabeled dataset by first running
```
bash bash_scripts/extract_features.sh
```
and then running
```
bash bash_scripts/estimate_k.sh
```

## Results
Results of our method are reported as below. You can download our model checkpoint by the link.
| **Datasets**       | **All** | **Old** | **New** | **Models** |
|:------------|:--------:|:---------:|:---------:|:------:|
| CIFAR10 | 96.0 | 97.3 | 95.4 | [ckpt](https://pan.baidu.com/s/1XKHioJp002Lm7P1xmM5Htg?pwd=xhwq) |
| CIFAR100 | 74.2 | 81.2 | 60.3 | [ckpt](https://pan.baidu.com/s/1DbUpDpFj-dlO58w6GqhyKw?pwd=rvkd) |
| ImageNet-100 | 77.6 | 93.5 | 69.7 | [ckpt](https://pan.baidu.com/s/1G1mY85up1ji2LLxMNBJjrw?pwd=rc7o) |
| CUB-200 | 52.1 | 54.3 | 51.0 | [ckpt](https://pan.baidu.com/s/1gtuPMF-itQvt9r5kW7Y32Q?pwd=pg9m) |
| Stanford-Cars | 40.5 | 58.8 | 31.7 | [ckpt](https://pan.baidu.com/s/1PDVhatM6qVUZZwjBVwSgTg?pwd=6337) |
| FGVC-Aircraft | 47.7 | 44.4 | 49.4 | [ckpt](https://pan.baidu.com/s/1SwkobAaT8l-TTlYn7IXWyQ?pwd=06u1) |
| Oxford-Pet | 86.7 | 91.5 | 84.1 | [ckpt](https://pan.baidu.com/s/1kCUfebbKmws9EgYrvgF5Aw?pwd=ck3k) |

## Citation
If you find this repo useful for your research, please consider citing our paper:
```
@inproceedings{fei2022xcon,
        title = {XCon: Learning with Experts for Fine-grained Category Discovery}, 
        author = {Yixin Fei and Zhongkai Zhao and Siwei Yang and Bingchen Zhao},
        booktitle={British Machine Vision Conference (BMVC)},
        year = {2022}
}
```