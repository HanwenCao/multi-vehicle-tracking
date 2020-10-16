# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""
import logging
import time
import torch
import torch.nn as nn
from lib.utils.reid_eval import evaluator
''' 重要
    测试时调用 indices_np = inference(cfg, model, val_loader, num_query, dataset)
'''
def inference(
        cfg,
        model,
        val_loader,
        num_query,
        dataset
):
    device = cfg.MODEL.DEVICE
    model.to(device) #model转GPU
    logger = logging.getLogger("reid_baseline.inference")
    logger.info("Enter inferencing")
    metric = evaluator(num_query, dataset, cfg, max_rank=50)  #reid评估指标算子  /lib/utils/reid_eval.py
    model.eval()  #固定参数
    start = time.time()
    with torch.no_grad():
        for batch in val_loader:  #分batch取出测试数据
            data, pid, camid, img_path = batch #一个batch包含这些信息
            data = data.cuda() #一个batch的img转GPU
            feats = model(data) #前向计算，得到feats特征
            if cfg.TEST.FLIP_TEST:  #测试技巧，flip图像后再次计算特征，与原图特征求平均
                data_flip = data.flip(dims=[3])  # NCHW  水平flip？
                feats_flip = model(data_flip)
                feats = (feats + feats_flip) / 2  #求平均
            #完成一个batch的特征计算
            output = [feats, pid, camid, img_path] #只用于计算评估指标
            metric.update(output)  #更新reid评估指标算子
            #结束一个batch
    #结束所有batch
    end = time.time()
    logger.info("inference takes {:.3f}s".format((end - start)))
    torch.cuda.empty_cache()
    cmc, mAP, indices_np = metric.compute()  #计算reid评估指标，计算距离矩阵和特征并保存，其中indices_np是距离矩阵的列排序index，间接表示哪个gallery距离每一行的query最近
    logger.info('Validation Results')
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return indices_np

def select_topk(indices, query, gallery, topk=10):
    results = []
    for i in range(indices.shape[0]):
        ids = indices[i][:topk]
        results.append([query[i][0]] + [gallery[id][0] for id in ids])
    return results


def extract_features(cfg, model, loader):
    device = cfg.MODEL.DEVICE
    model.to(device)
    model.eval()
    feats = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            data, pid, camid, img_path = batch
            data = data.cuda()
            feat = model(data)
            feats.append(feat)
    feats = torch.cat(feats, dim=0)
    feats = torch.nn.functional.normalize(feats, dim=1, p=2)
    return feats