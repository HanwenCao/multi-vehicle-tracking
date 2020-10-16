from __future__ import print_function
from __future__ import division

import os
import sys
import time
import datetime
import os.path as osp
import numpy as np
import warnings

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from opts import argument_parser, testset_kwargs
from datasets.dm_infer import ImageDataManager
import models
from losses import CrossEntropyLoss, TripletLoss, DeepSupervision
from utils.io import check_isfile
from utils.avgmeter import AverageMeter
from utils.log import Logger, RankLogger
from utils.torch_func import count_num_param,load_pretrained_weights
from utils.seed import set_random_seed
from postprocess.postprocess import calc_reid,update_output

from tqdm import tqdm
from lib.modeling import build_model  #/code/lib/modeling/__init__.py
from lib.config import cfg
import argparse
from lib.utils.logger import setup_logger
from lib.utils.post_process import *

# global variables
parser = argument_parser()
parser.add_argument(
        "--config_file", default="configs/aicity20.yml", help="path to config file", type=str
    )
parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
args = parser.parse_args()

if args.config_file != "":
    cfg.merge_from_file(args.config_file)
cfg.merge_from_list(args.opts)
cfg.freeze()

output_dir = cfg.OUTPUT_DIR
if output_dir and not os.path.exists(output_dir):
    mkdir(output_dir)

num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
logger = setup_logger("reid_baseline", output_dir, 0)
logger.info("Using {} GPUS".format(num_gpus))
# logger.info(args)

if args.config_file != "":
    logger.info("Loaded configuration file {}".format(args.config_file)) #--config_file='/code/configs/aicity20.yml'
    with open(args.config_file, 'r') as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
# logger.info("Running with config:\n{}".format(cfg))


def main():
    global args
    if not args.use_avai_gpus:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    use_gpu = torch.cuda.is_available()
    if args.use_cpu:
        use_gpu = False

    if use_gpu:
        print('Currently using GPU {}'.format(args.gpu_devices))
        cudnn.benchmark = True
    else:
        warnings.warn('Currently using CPU, however, GPU is highly recommended')

    '''创建test loader'''
    print('Initializing image data manager')
    dm = ImageDataManager(use_gpu, **testset_kwargs(args))  # 创建 data manager
    testloader_dict = dm.return_dataloaders()   # 创建data loader字典（包含query和gallery）

    '''创建模型'''
    num_classes = 400
    model = build_model(cfg, num_classes)  #创建reid模型（ResNet+Gem+BN+分类器）/lib/modeling/baseline.py
    print('Model size: {:.3f} M'.format(count_num_param(model)))
    model.load_param(cfg.TEST.WEIGHT)  #给模型添加reid训练得到的参数 TEST.WEIGHT "('/data/caohw9/aicity20/0409-ensemble/r50-320-circle/best.pth')"

    model = nn.DataParallel(model).cuda()
    
    print('Matching {} ...'.format(args.test_set))  #split好的query和gallery
    queryloader = testloader_dict['query']
    galleryloader = testloader_dict['test']
    distmat, q_pids, g_pids, q_camids, g_camids, distmat_np, feats = run(model, queryloader, galleryloader, use_gpu, return_distmat=True) #RE-ID（加了后处理）
    ''''存储'''
    # args.save_npy= ./exp/distmatrix.npz
#     np.savez(args.save_npy,distmat=distmat,q_pids=q_pids,g_pids=g_pids,q_camids=q_camids,g_camids=g_camids) #存储原始结果到distmatrix.npz后续可以本地跑出track3.txt
    np.savez(args.save_npy,distmat=distmat_np,q_pids=q_pids,g_pids=g_pids,q_camids=q_camids,g_camids=g_camids) #存储后处理结果
#     np.savez('./exp/distmat_np.npz', distmat_np)  #保存距离矩阵distmat_np
#     np.savez('./exp/feats.npz', feats)  #保存特征feats.cpu().numpy()
    
    
    '''
    print('Initializing model: {}'.format(args.arch))  # args.arch默认是'resnet101'
    model = models.init_model(name=args.arch, loss={'xent', 'htri'},  
                              pretrained=not args.no_pretrained, use_gpu=use_gpu)  #models/__init__.py -> models/resnet.py/def resnet101
    print('Model size: {:.3f} M'.format(count_num_param(model)))

    if args.load_weights and check_isfile(args.load_weights):
        load_pretrained_weights(model, args.load_weights)  #下载好的model.pth.tar-9文件

    model = nn.DataParallel(model).cuda() if use_gpu else model  #多GPU

    print('Matching {} ...'.format(args.test_set))  #split好的query和gallery
    queryloader = testloader_dict['query']
    galleryloader = testloader_dict['test']
    distmat, q_pids, g_pids, q_camids, g_camids = run(model, queryloader, galleryloader, use_gpu, return_distmat=True) #RE-ID
    np.savez(args.save_npy,distmat=distmat,q_pids=q_pids,g_pids=g_pids,q_camids=q_camids,g_camids=g_camids) #存储结果到distmatrix.npz后续可以本地跑出track3.txt
    # 到这为止，利用定义的reid模型，读取extract的img，得到距离矩阵distmat和 行所代表的queryID+camID 和列所代表的galleryID+camID
    
    #下一步，后处理，解析距离矩阵的物理含义，得到MTMC的结果
    result  = np.load(args.save_npy) #存储完马上加载distmatrix.npz
    reid_dict,rm_dict = calc_reid(result) #得到reid可能匹配结果reid_dict和reid无匹配结果rm_dict
    print(rm_dict,reid_dict)  
    with open(args.tracklet_path,"r") as f: #读取tracklets.txt
        or_tracks = f.readlines()
    g = open(args.track3_path,"w") #打开track3.txt
    update_output(or_tracks,reid_dict,rm_dict,g) #更新tracklets.txt->track3.txt
    '''







def run(model, queryloader, galleryloader, use_gpu, ranks=[1, 5, 10, 20], return_distmat=False):
    batch_time = AverageMeter()

    model.eval()

    with torch.no_grad():
        qf, q_pids, q_camids = [], [], []
        for batch_idx, (imgs, pids, camids, _) in enumerate(tqdm(queryloader)):
            if use_gpu:
                imgs = imgs.cuda()

            end = time.time()
            features = model(imgs)
            batch_time.update(time.time() - end)

            features = features.data.cpu()
            qf.append(features) 
            q_pids.extend(pids) #extend()用于在列表末尾一次性追加另一个序列
            q_camids.extend(camids)
        qf = torch.cat(qf, 0) #query images features
        q_pids = np.asarray(q_pids) #query image ids
        q_camids = np.asarray(q_camids) # query camera ids

        print('Extracted features for query set, obtained {}-by-{} matrix'.format(qf.size(0), qf.size(1)))

        gf, g_pids, g_camids = [], [], []
        for batch_idx, (imgs, pids, camids, _) in enumerate(tqdm(galleryloader)):
            if use_gpu:
                imgs = imgs.cuda()

            end = time.time()
            features = model(imgs)
            batch_time.update(time.time() - end)

            features = features.data.cpu()
            gf.append(features)
            g_pids.extend(pids)
            g_camids.extend(camids)
        gf = torch.cat(gf, 0) #gallery image feature
        g_pids = np.asarray(g_pids) #gallery image ids
        g_camids = np.asarray(g_camids) #gallery camera ids

        print('Extracted features for gallery set, obtained {}-by-{} matrix'.format(gf.size(0), gf.size(1)))

    print('=> BatchTime(s)/BatchSize(img): {:.3f}/{}'.format(batch_time.avg, args.test_batch_size))

    m, n = qf.size(0), gf.size(0)
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, qf, gf.t()) #从features计算距离矩阵
    distmat = distmat.numpy()

    
    '''补充'''
    feat_norm = 0 #1 #这个要做
    do_DBA = 0 #好像ruiyan不做这个？
    
    feats = torch.cat([qf,gf])
    
    if feat_norm:
        feats = torch.nn.functional.normalize(feats, dim=1, p=2)
        print("The test feature is normalized")
    if do_DBA:
        feats = database_aug(feats, top_k=6)
        
    '''后处理'''
    query_expansion = 0 #1
    TRACK_AUG = 0
    USE_VOC = 0
    do_rerank = 1 #1
    TRACK_RERANK = 0
    
    if query_expansion:
        print('before query_expansion',qf.shape)
        qf = average_query_expansion(qf, feats, top_k=6)  #后处理query expansion
        print('Query expansion done.')
        print('after query_expansion',qf.shape)
    # if TRACK_AUG:
    #     gf = track_aug(gf, self.dataset.test_tracks, self.img_paths[self.num_query:]) #后处理track_aug
    if USE_VOC:
        print('using VOC-ReID')
        # cam_dist = np.load(self.cfg.TEST.CAM_DIST_PATH)  #读取之前计算好的camREID的距离矩阵
        # ori_dist = np.load(self.cfg.TEST.ORI_DIST_PATH)  #方向reid
    else:
        cam_dist = None
        ori_dist = None
    if do_rerank:  # 后处理re-ranking（用到了camReid和方向reid） [50, 15, 0.5]
        distmat_np = re_ranking(qf, gf, k1=50, k2=15, lambda_value=0.5, USE_VOC=USE_VOC, cam_dist=cam_dist, ori_dist=ori_dist)
        print('re-ranking done')
    else:  #不做re-ranking的话
        print('no re-ranking')
        distmat_np, indices = comput_distmat(qf, gf)  #直接用qf gf计算距离矩阵
        distmat_np = distmat_np.cpu().numpy()  #后缀_np的含义 
    indices_np = np.argsort(distmat_np, axis=1)  #距离矩阵保持行顺序固定，列排序，得到排序index就是indices_np（返回）
    # if TRACK_RERANK:
    #     rerank_indice_by_track(indices_np, self.img_paths[self.num_query:], self.dataset.test_tracks)
    print('post-process done')
    '''后处理结束'''
#     print(type(distmat),type(distmat_np))
    print((distmat==distmat_np).all())

    return distmat, q_pids, g_pids, q_camids, g_camids, distmat_np, feats.cpu().numpy()


if __name__ == '__main__':
    print('args.save_npy=',args.save_npy)
    args.tracklet_path = './exp/tracklets_mask.txt'  #如果不更改，默认是./exp/tracklets.txt （忘了在哪里设置了就在这里简单改一下）
    print('args.tracklet_path=',args.tracklet_path)
    print('args.track3_path=',args.track3_path)
    main()