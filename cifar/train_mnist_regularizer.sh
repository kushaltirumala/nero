# env CUDA_VISIBLE_DEVICES=0 python train.py --batch_norm --gpu --task mnist --net resnet50 --optimizer nero --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 0.01 --gainlr 0.0001 --gamma 0 --warm 0
env CUDA_VISIBLE_DEVICES=1 python train.py --batch_norm --gpu --task mnist --net resnet50 --optimizer nero --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 0.01 --gainlr 0.001 --gamma 0 --warm 0
# env CUDA_VISIBLE_DEVICES=3 python train.py --batch_norm --gpu --task mnist --net resnet50 --optimizer nero --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 0.01 --gainlr 0.01 --gamma 0 --warm 0
# env CUDA_VISIBLE_DEVICES=3 python train.py --batch_norm --gpu --task mnist --net resnet50 --optimizer nero --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 0.01 --gainlr 1.0 --gamma 0 --warm 0
