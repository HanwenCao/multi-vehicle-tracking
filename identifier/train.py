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

from opts import argument_parser, trainset_kwargs, optimizer_kwargs, lr_scheduler_kwargs
from datasets.dm import ImageDataManager
import models
from losses import CrossEntropyLoss, TripletLoss, DeepSupervision
from utils.io import check_isfile
from utils.avgmeter import AverageMeter
from utils.log import Logger
from utils.torch_func import count_num_param, accuracy, \
    load_pretrained_weights, save_checkpoint, resume_from_checkpoint, \
    init_optimizer,init_lr_scheduler
from utils.seed import set_random_seed
from utils.eval import evaluate

# global variables
parser = argument_parser()
args = parser.parse_args()
'''
adam_beta1=0.9, adam_beta2=0.999, arch='resnet101', color_aug=False, color_jitter=False, eval_freq=1, gamma=0.1, gpu_devices='0', height=128, label_smooth=False, lambda_htri=1, lambda_xent=1, load_weights='', lr=0.001, lr_scheduler='multi_step', margin=0.3, max_epoch=25, momentum=0.9, no_pretrained=False, num_instances=4, optim='amsgrad', print_freq=100, query_remove=True, random_erase=False, resume='', rmsprop_alpha=0.99, root='/data/caohw9/track3_intermediate', save_dir='/data/caohw9/track3_intermediate/resnet101-Aic', save_npy='./exp/distmatrix.npz', seed=1, sgd_dampening=0, sgd_nesterov=False, start_epoch=0, start_eval=0, stepsize=[10, 15, 20], test_batch_size=256, test_set='Aic', track3_path='./exp/track3.txt', tracklet_path='./exp/tracklets.txt', train_batch_size=128, train_sampler='RandomSampler', train_sets=['Aic'], use_avai_gpus=True, use_cpu=False, weight_decay=0.0005, width=256, workers=4
'''
def main():

    # 为了看看repo提供的model.pth.tar-9在validation集的mAp和rank-1
    # 我自己训练的tar-9只有mAP: 15.1%； Rank-1: 23.3% ，不知道为什么
    # 更改args.load_weights = '/model/caohw9/track3_model/model.pth.tar-9'

    global args
    print(args)

    set_random_seed(args.seed)
    if not args.use_avai_gpus:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    use_gpu = torch.cuda.is_available()
    if args.use_cpu:
        use_gpu = False
    sys.stdout = Logger(osp.join(args.save_dir, "log.txt"))
    if use_gpu:
        print('Currently using GPU {}'.format(args.gpu_devices))
        cudnn.benchmark = True
    else:
        warnings.warn('Currently using CPU, however, GPU is highly recommended')

    # 初始化loader
    print('Initializing image data manager')
    dm = ImageDataManager(use_gpu, **trainset_kwargs(args))
    trainloader, testloader_dict = dm.return_dataloaders()  #trainloader用于训练，testloader_dict包含['query']和['gallery']2个loader
    print('suffessfully initialize loaders!')

    # 初始化模型
    print('Initializing model: {}'.format(args.arch))  #args.arch default='resnet101'
    model = models.init_model(name=args.arch, num_classes=dm.num_train_pids, loss={'xent', 'htri'},
                              pretrained=not args.no_pretrained, use_gpu=use_gpu)
    print('Model size: {:.3f} M'.format(count_num_param(model)))

    # 加载预训练参数
    if args.load_weights and check_isfile(args.load_weights):
        load_pretrained_weights(model, args.load_weights)
        #加载训练过的模型后，先看看validation
        print('=> Validation')
        print('Evaluating {} ...'.format(args.test_set))  #args.test_set应该是指的validation set？
        queryloader = testloader_dict['query']
        galleryloader = testloader_dict['test']
        model = nn.DataParallel(model).cuda() if use_gpu else model
        rank1 = test(model, queryloader, galleryloader, use_gpu)  #validation！

    # 多GPU训练
    else:
        model = nn.DataParallel(model).cuda() if use_gpu else model

    # 定义loss,optimizer, lr_scheduler
    criterion_xent = CrossEntropyLoss(num_classes=dm.num_train_pids, use_gpu=use_gpu, label_smooth=args.label_smooth)
    criterion_htri = TripletLoss(margin=args.margin)
    optimizer = init_optimizer(model, **optimizer_kwargs(args))
    scheduler = init_lr_scheduler(optimizer, **lr_scheduler_kwargs(args))

    # 是否是resume训练
    if args.resume and check_isfile(args.resume):
        args.start_epoch = resume_from_checkpoint(args.resume, model, optimizer=optimizer) #获取中断时刻的epoch数

    # 开始训练！
    time_start = time.time()
    print('=> Start training')

    for epoch in range(args.start_epoch, args.max_epoch):
        train(epoch, model, criterion_xent, criterion_htri, optimizer, trainloader, use_gpu)  #训练

        scheduler.step()  #更新lr

        # 当epoch数超过args.start_eval，每隔一定频率args.eval_freq，或者达到最后一个epoch，进行validation+存储checkpoint
        if (epoch + 1) > args.start_eval and args.eval_freq > 0 and (epoch + 1) % args.eval_freq == 0 or (
                epoch + 1) == args.max_epoch:            
            print('=> Validation')
            print('Evaluating {} ...'.format(args.test_set))  #args.test_set应该是指的validation set？
            queryloader = testloader_dict['query']
            galleryloader = testloader_dict['test']
            rank1 = test(model, queryloader, galleryloader, use_gpu)  #validation！

            save_checkpoint({
                'state_dict': model.state_dict(),  #模型的状态字典
                'rank1': rank1,
                'epoch': epoch + 1,
                'arch': args.arch,  #default='resnet101'
                'optimizer': optimizer.state_dict(),  #优化器对象的状态字典，包含优化器的状态和超参数(如lr, momentum,weight_decay等)
            }, args.save_dir)  #validation同时保存checkpoint

    # 训练结束！
    elapsed = round(time.time() - time_start)  #持续时间
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print('Elapsed {}'.format(elapsed))

'''
一个典型的训练函数：
for t in range(100):
    x is a batch
    prediction = net(x)     # 喂给net训练数据x, 输出预测值
    loss = loss_func(prediction, y)     # 计算两者的误差
    optimizer.zero_grad()   # 清空上一步的残余更新参数值
    loss.backward()         # 误差反向传播, 计算参数更新值
    optimizer.step()        # 将参数更新值施加到net的parameters上
'''
def train(epoch, model, criterion_xent, criterion_htri, optimizer, trainloader, use_gpu):
    xent_losses = AverageMeter()  #AverageMeter()：Computes and stores the average and current value
    htri_losses = AverageMeter()  # .update(curr_value)
    accs = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()  #设置为训练模式
    for p in model.parameters():
        p.requires_grad = True    # open all layers 打开自动求导

    end = time.time()
    for batch_idx, (imgs, pids, _, _) in enumerate(trainloader):  #加载一个batch的训练图像
        data_time.update(time.time() - end)

        if use_gpu:
            imgs, pids = imgs.cuda(), pids.cuda()

        # 前向计算 
        outputs, features = model(imgs)  #imgs -> y,v (https://github.com/KevinQian97/ELECTRICITY-MTMC/blob/ce5f173aabdc9ae6733ca36d1fdcfc53fa3d3d6e/identifier/models/resnet.py#L221)
        # 计算2种loss
        if isinstance(outputs, (tuple, list)):
            xent_loss = DeepSupervision(criterion_xent, outputs, pids)
        else:
            xent_loss = criterion_xent(outputs, pids)
        if isinstance(features, (tuple, list)):
            htri_loss = DeepSupervision(criterion_htri, features, pids)
        else:
            htri_loss = criterion_htri(features, pids)
        #综合loss=2种loss求和
        loss = args.lambda_xent * xent_loss + args.lambda_htri * htri_loss
        optimizer.zero_grad()  #对于每个batch，梯度（loss关于weight的导数）置零，目的是清空上一步的残余更新参数值
        loss.backward()  #反向传播
        optimizer.step()  #更新参数

        batch_time.update(time.time() - end)

        xent_losses.update(xent_loss.item(), pids.size(0))  #记录
        htri_losses.update(htri_loss.item(), pids.size(0))
        accs.update(accuracy(outputs, pids)[0])

        #隔一段时间，打印训练状态
        if (batch_idx + 1) % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.4f} ({data_time.avg:.4f})\t'
                  'Xent {xent.val:.4f} ({xent.avg:.4f})\t'
                  'Htri {htri.val:.4f} ({htri.avg:.4f})\t'
                  'Acc {acc.val:.2f} ({acc.avg:.2f})\t'.format(
                epoch + 1, batch_idx + 1, len(trainloader),
                batch_time=batch_time,
                data_time=data_time,
                xent=xent_losses,
                htri=htri_losses,
                acc=accs
            ))

        end = time.time()


def test(model, queryloader, galleryloader, use_gpu, ranks=[1, 5, 10, 20], return_distmat=False):
    batch_time = AverageMeter()

    model.eval()

    with torch.no_grad():
        qf, q_pids, q_camids = [], [], []
        for batch_idx, (imgs, pids, camids, _) in enumerate(queryloader):
            if use_gpu:
                imgs = imgs.cuda()

            end = time.time()
            features = model(imgs)
            batch_time.update(time.time() - end)

            features = features.data.cpu()
            qf.append(features)
            q_pids.extend(pids)
            q_camids.extend(camids)
        qf = torch.cat(qf, 0)
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)

        print('Extracted features for query set, obtained {}-by-{} matrix'.format(qf.size(0), qf.size(1)))

        gf, g_pids, g_camids = [], [], []
        for batch_idx, (imgs, pids, camids, _) in enumerate(galleryloader):
            if use_gpu:
                imgs = imgs.cuda()

            end = time.time()
            features = model(imgs)
            batch_time.update(time.time() - end)

            features = features.data.cpu()
            gf.append(features)
            g_pids.extend(pids)
            g_camids.extend(camids)
        gf = torch.cat(gf, 0)
        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)

        print('Extracted features for gallery set, obtained {}-by-{} matrix'.format(gf.size(0), gf.size(1)))

    print('=> BatchTime(s)/BatchSize(img): {:.3f}/{}'.format(batch_time.avg, args.test_batch_size))

    m, n = qf.size(0), gf.size(0)
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, qf, gf.t())
    distmat = distmat.numpy()

    print('Computing CMC and mAP')
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)

    print('Results ----------')
    print('mAP: {:.1%}'.format(mAP))
    print('CMC curve')
    for r in ranks:
        print('Rank-{:<3}: {:.1%}'.format(r, cmc[r - 1]))
    print('------------------')

    if return_distmat:
        return distmat
    return cmc[0]


if __name__ == '__main__':
    main()
