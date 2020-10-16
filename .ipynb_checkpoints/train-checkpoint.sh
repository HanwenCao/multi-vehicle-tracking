set -e
dir=$(dirname $(readlink -fn --  $0))
# after prepare dataset

# python ./identifier/train.py --train_sets Aic --test_set Aic --train-batch-size 128 \
#  --test-batch-size 256 -a resnet101 --save-dir models/resnet101-Aic --use-avai-gpus \
#  --root ./datasets

# 零基础训练
python ./identifier/train.py --train_sets Aic --test_set Aic --train-batch-size 64 \
 --test-batch-size 256 -a resnet101 --save-dir /output/resnet101-Aic --use-avai-gpus \
 --root /data/caohw9/track3_intermediate
 
# 在repo提供的模型参数基础上训练
# python ./identifier/train.py --train_sets Aic --test_set Aic --train-batch-size 64 \
#  --test-batch-size 256 -a resnet101 --save-dir /output/resnet101-Aic --use-avai-gpus \
#  --root /data/caohw9/track3_intermediate --load-weights '/model/caohw9/track3_model/model.pth.tar-9'
 
 
# croped training dataset
#  python ./identifier/train.py --train_sets Aic_crop --test_set Aic_crop --train-batch-size 128 \
#  --test-batch-size 256 -a resnet101 --save-dir /data/caohw9/track3_intermediate/resnet101-Aic --use-avai-gpus \
#  --root /data/caohw9/track3_intermediate
