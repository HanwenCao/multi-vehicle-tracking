# set environment variable so that the Detectron2 weights won't be downloaded
#export FVCORE_CACHE=/data/caohw9/track3_torch



#please download the data set and locate it under ./datasets
set -e
dir=$(dirname $(readlink -fn --  $0))
cd ../
#加载视频，检测跟踪，结果写入exp/tracklets.txt，
#格式 [相机号 carID 帧号 box位置信息]
#复杂的多进程没看懂
#python -m ELECTRICITY-MTMC.utils.test  
# python -m ELECTRICITY-MTMC.detector.detectron2-examine  #检查目标检测的结果


cd $dir
#根据跟踪结果，从test视频里对应的剪裁出小图，命名，作为query和gallery
#输入exp/tracklets.txt，输出query和gallery文件夹

# python ./identifier/preprocess/extract_img.py  
# python ./identifier/preprocess/extract_img_bigimg.py  



#--test_set aic_test 参考identifier/datasets/aic_test.py，这里规定test数据集地址

# python ./identifier/test-Copy1.py --test-batch-size 256 --test_set aic_test \
#  --use-avai-gpus --load-weights /model/caohw9/track3_model/model.pth.tar-9  #用图片做REID

#使用自己训练的reid模型
python ./identifier/test-Copy1.py --test-batch-size 256 --test_set aic_test \
 --use-avai-gpus --load-weights /model/caohw9/track3_model/my_model.pth.tar-21  #用图片做REID