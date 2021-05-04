env CUDA_VISIBLE_DEVICES=2 python train.py --gpu --batch_norm --task cifar10 --net vgg11 --optimizer nero --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 0.01
