import os
import cv2
from multiprocessing import Pool
import argparse
import shutil

def argument_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--tracklet_path', type=str, default="./exp/tracklets.txt",
                        help='path to the tracklets')
    # parser.add_argument('--data_path', type=str, default="./datasets/aic_20_trac3",
                        # help='path to the aicity 2020 track 3 folders')
    parser.add_argument('--data_path', type=str, default="/data/caohw9/track3_dataset",
                        help='path to the aicity 2020 track 3 folders')
    # parser.add_argument('--output_path', type=str, default="./exp/imgs/aic_test",
    #                     help='path to the output dictionaries')
    parser.add_argument('--output_path', type=str, default="/output/exp/imgs/aic_test",
                        help='path to the output dictionaries')
    parser.add_argument("--njobs",type = int,default=3,help="number of pools to extract imgs")
    return parser

#输入跟踪结果，输出整理后的结果
def sort_tracklets(gts):
    sorted_gts = {} 
    #sorted_gts={frame1: 多个[left,top,right,bot,car_id,query]，frame2: 多个[left,top,right,bot,car_id,query] ...}
    car_list = []
    for line in gts:#遍历gts行
        # line = [camID,car_id,frame,left,top,width,hight,0,0]，gts包含[camID ID号 帧号 位置信息]
        line = line.strip().split(" ")[:-2]  #[:-2]丢弃最后2个元素0 0
        frame = int(line[2])
        left = int(line[3])
        top = int(line[4])
        right = left +int(line[5])
        bot = top+int(line[6])
        car_id = int(line[1])
        query = False
        if car_id not in car_list:
            car_list.append(car_id) #不重复的记录目前为止出现过的carID
            query = True #如果这个ID的第一次出现，设为query，否则设置为gallery（关键）
        if frame not in list(sorted_gts.keys()):#不重复的记录所有frame号
            sorted_gts[frame] = []
        sorted_gts[frame].append([left,top,right,bot,car_id,query])
    print(len(car_list))
    return sorted_gts

    

def extract_im_api(args): #args=[数据集地址，输出query和gallery地址，"test",S06，camID，跟踪结果]
    base_path = args[0]
    data_path = args[1]
    split = args[2]
    scene = args[3]
    cam = args[4]
    gts = args[5]
    extrac_im(base_path,data_path,split,scene,cam,gts)

def extrac_im(base_path,data_path,split,scene,cam,gts):
    print("start cam:"+cam)
    scene_dir = os.path.join(base_path,split,scene)  # S06
    cam_dir = os.path.join(scene_dir,cam)
    cap = cv2.VideoCapture(os.path.join(cam_dir,"vdo.avi"))  #找到测试集1个具体的视频
    sorted_gts = sort_tracklets(gts)  #1.按照帧整理gts 2.标注哪些box为query
    #sorted_gts={frame1: 多个[left,top,right,bot,car_id,query]，frame2: 多个[left,top,right,bot,car_id,query] ...}
    fr_id=0 #遍历视频的帧num变量
    state,im = cap.read() #读取第一帧，state表示有没有读取到图片，im是图片
    frames = list(sorted_gts.keys()) #所有frameID的list
    while(state):
        if fr_id not in frames or im is None: #如果视频当前帧没有任何跟踪结果：下一帧
            state,im = cap.read() #读取下一帧
            fr_id+=1 #读取下一帧
        else:#有跟踪结果：
            tracks = sorted_gts[fr_id] #找出跟踪结果中 对应当前帧的 所有检测跟踪结果：多个[left,top,right,bot,car_id,query]
            for track in tracks: #1个[left,top,right,bot,car_id,query]
                left,top,right,bot,car_id,query=track
                clip = im[top:bot,left:right] #剪裁处小图
                im_name = str(car_id).zfill(5)+"_"+cam+"_"+str(fr_id).zfill(4)+".jpg" #命名 0..0carID_camID_0..0frameID.jpg
                if query:
                    if not os.path.exists(os.path.join(data_path,"image_query")):
                        os.makedirs(os.path.join(data_path,"image_query"))
                    cv2.imwrite(os.path.join(data_path,"image_query",im_name),clip) #存储小图到到输出地址的image_query文件夹
                else:
                    if not os.path.exists(os.path.join(data_path,"image_test")):
                        os.makedirs(os.path.join(data_path,"image_test"))
                    cv2.imwrite(os.path.join(data_path,"image_test",im_name),clip)  #存储小图到到输出地址的image_test文件夹（gallery）
            state,im = cap.read() #读取下一帧
            fr_id+=1 #读取下一帧


def main(args):
    gts_path = args.tracklet_path
    base_path = args.data_path #注意这里命名变化，后面的代码中base_path指的是数据集地址
    data_path = args.output_path #data_path指的是输出query和gallery小图的地址
    if os.path.exists(os.path.join(data_path)):
        shutil.rmtree(os.path.join(data_path)) 
    os.makedirs(os.path.join(data_path))
    splits = ["test"]

    with open(gts_path,"r") as f:
        or_tracks = f.readlines() #读取tracklets.txt
    args_list = []
    for split in splits:
        split_dir = os.path.join(base_path,split)  # test folder
        scenes = os.listdir(split_dir)  #'S06'  (only this folder under split_dir)
        for scene in scenes:
            scene_dir = os.path.join(split_dir,scene)  # test/S06
            cams = os.listdir(scene_dir)  # 6 cams-folders under test/S06
            for cam in cams:#遍历test集的所有相机（视频）
                gts = []
                camid = int(cam.split("c")[1]) # c041,...,c046 -> 41,...,46
                for track in or_tracks: #取出tracklets的每一行
                    if int(track.split(" ")[0]) == camid: #找到对应这个camID的所有跟踪结果，记录在gts里
                        gts.append(track)  #gts包含[camID ID号 帧号 位置信息]
                args_list.append([base_path,data_path,split,scene,cam,gts])#合并所有相机（视频）的跟踪结果和视频地址到args_list
                #感觉args_list只是：tracklets信息原封不动（gts）+输入视频地址（base_path）+输出地址（data_path），方便后续处理
    n_jobs = args.njobs
    pool = Pool(n_jobs)
    pool.map(extract_im_api, args_list) #把args_list送入提取图片的程序
    pool.close()



if __name__ == "__main__":
    parser = argument_parser()
    args = parser.parse_args()
    main(args)