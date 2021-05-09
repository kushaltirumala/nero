env CUDA_VISIBLE_DEVICES=1 python train.py --batch_norm --gpu --task mnist --net resnet18 --optimizer nero --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 0.01 --warm 0 
