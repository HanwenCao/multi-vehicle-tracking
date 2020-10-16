set -e
dir=$(dirname $(readlink -fn --  $0))
cd ../
cd $dir


#使用自己训练的reid模型
python ./identifier/test_ruiyan.py --test-batch-size 256 --test_set aic_test \
--use-avai-gpus --load-weights /model/caohw9/track3_model/my_model.pth.tar-21 \
--config_file='configs/aicity20.yml' \
MODEL.PRETRAIN_CHOICE "('self')"\


