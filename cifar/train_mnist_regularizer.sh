env CUDA_VISIBLE_DEVICES=2 python train.py --batch_norm --gpu --task mnist --net resnet50 --optimizer nero --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 0.01 --gainlr 10.0 --gamma 0.2 --warm 0
