# env CUDA_VISIBLE_DEVICES=3 python train.py --gpu --batch_norm --task cifar10 --net resnet --depth 10 --width 16 --optimizer nero --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 0.01
# env CUDA_VISIBLE_DEVICES=3 python train.py --gpu --batch_norm --task cifar10 --net resnet --depth 10 --width 16 --optimizer nero --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 0.01
#


# for depth in 50 42 34 26 18 10
# do
#   for width in 16 32 64 128 256
#   do
#     echo "ON DEPTH $depth and WIDTH $width \n"
#     env CUDA_VISIBLE_DEVICES=3 python train.py --gpu --batch_norm --task cifar10 --net resnet --depth $depth --width $width --optimizer sgd --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 0.01 --warm 0
#   done
# done

for depth in 34 26
do
  for width in 16 32 64 128 256
  do
    echo "ON DEPTH $depth and WIDTH $width \n"
    env CUDA_VISIBLE_DEVICES=2 python train.py --gpu --batch_norm --task cifar10 --net resnet --depth $depth --width $width --optimizer sgd --momentum 0.0 --beta 0.999 --wd 0.0 --seed 0  --lr 0.001 --warm 0
  done
done
