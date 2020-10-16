import os
import numpy as np
# result_path = "../output.npz"
# input_path = "../tracklets.txt"
# output_path = "../track3.txt"
# result = np.load(result_path)
# dis_thre = 12
# dis_remove = 100
# max_length = 20

def calc_reid(result,dis_remove=100,dis_thre=12):
    distmat=result["distmat"]  # dist matrix,行对应着q_pids的每个query ID，列对应g_pids里的每个gallery ID
    q_pids = result["q_pids"]  # 所有query ID的列表[363 793 1193 ...]
    g_pids = result["g_pids"]  # 所有gallery ID的列表[1067 36 53 1247 ...]
    q_camids=result["q_camids"]+1  # q_pids对应的相机编号[42 41 42 42 ...]
    g_camids=result["g_camids"]+1  # 为什么+1？
    new_id = np.max(g_pids)  # 1363
    # print(np.max(g_pids))
    # print(np.max(q_pids))
    rm_dict = {}
    reid_dict = {}
    indices = np.argsort(distmat, axis=1) #按照距离从近至远，行不动，列排序距离矩阵（还没排）
    num_q, num_g = distmat.shape #963，123698
    # print(np.min(distmat))
    for q_idx in range(num_q):
        q_pid = q_pids[q_idx]  # 当前的quieryID 363
        q_camid = q_camids[q_idx]  # 当前query对应的相机 42
        order = indices[q_idx]  # 针对当前行（当前queryID）的distmat列排序顺序
        # g_pids[order]=[363 363 363 363 164 232 115 282 363 ...164...]将所有galleryID按照距离当前query的距离排序，前几名应该是当前query同一track的后续几帧
        # g_pids[order] == q_pid 找出galleryID中跟当前query属于同一track（被跟踪算法赋予同一ID的帧），这些已经是同一ID/trarck，不做reid
        # g_camids[order] == q_camid 找出galleryID中跟当前query属于同一相机的，这些可能是单相机的MOT结果，不做reid
        # distmat[q_idx][order]>dis_thre 当前行按照距离排序，去掉距离当前query太远的gallery，认为无论如何不是同一辆车，不做reid
        remove = (g_pids[order] == q_pid) | (g_camids[order] == q_camid) | (distmat[q_idx][order]>dis_thre) #distmat[q_idx][order]当前itr中的query距离所有galleries的距离从小到大排序
        keep = np.invert(remove) # dis_thre=12这个条件有点严格，keep容易全false

        remove_hard = (g_pids[order] == q_pid) | (g_camids[order] == q_camid) | (distmat[q_idx][order]>dis_remove) #阈值不一样dis_remove=100更宽容
        keep_hard = np.invert(remove_hard)  # false是要去除的，ture是要保留的潜在reid candidate
        if True not in keep_hard:  # 如果一个要保留的都没有（宽松条件下）
            if q_camid not in list(rm_dict.keys()):
                rm_dict[q_camid] = {}  # 这只是创建rm_dict[q_camid]字典的方法，值得学习
            rm_dict[q_camid][q_pid] = True  # 一个要保留的都没有（当前queryID没有reid对象）大概率找不到跨相机的对应track，考虑去除
        # 如果当前queryID还是有希望找到跨相机匹配的 或者没有希望 也都执行下面
        sel_g_dis = distmat[q_idx][order][keep]  # 严格条件下依旧保留的的galleryID对应的距离
        sel_g_pids = g_pids[order][keep]  # 严格条件下依旧保留的的galleryID
        sel_g_camids = g_camids[order][keep]  # ...对应的相机
        sel_g_pids_list = []
        sel_g_camids_list = []
        selg_dis_list = []
        for i in range(sel_g_pids.shape[0]):  #对严格条件下生存下来的gallery（潜在的reid目标）(可能有多个或没有)
            sel_pid =  sel_g_pids[i]  # galleryID
            sel_cam = sel_g_camids[i]  # camID
            sel_dis = sel_g_dis[i]  #dist 
            if sel_cam not in sel_g_camids_list and sel_cam!=q_camid:  # 再次确认reid目标和当前query不是同一个相机
                sel_g_pids_list.append(sel_pid)  # 记录reid目标的galleryID
                sel_g_camids_list.append(sel_cam)  # 无重复记录reid目标的cam
                selg_dis_list.append(sel_dis)  # 记录reid目标的dist
                

        if len(selg_dis_list)>0:  # 如果有reid目标：记录在reid_dict里（注意只保留了所有有希望找到reid的query的最小距离）
            new_id+=1  # 可能是为了不和之前的ID冲突
            if q_camid in list(reid_dict.keys()):
                if q_pid in list(reid_dict[q_camid]):  # 如果之前记录过这个query
                    if reid_dict[q_camid][q_pid]["dis"]>min(selg_dis_list):  # 且即将记录的reid目标更好
                        reid_dict[q_camid][q_pid]["dis"] = min(selg_dis_list)  # 更新之前的记录
                        reid_dict[q_camid][q_pid]["id"] = new_id
                else:
                    reid_dict[q_camid][q_pid] = {"dis":min(selg_dis_list),"id":new_id}  # 记录所有query的最近dist和newid
            else:
                reid_dict[q_camid] = {}
                reid_dict[q_camid][q_pid] = {"dis":min(selg_dis_list),"id":new_id}  # 记录所有query的最近dist和newid（queryID是reid_dict字典的key）


        for i in range(len(sel_g_pids_list)):  # 遍历当前query的reid目标gallery
            if sel_g_camids_list[i] in list(reid_dict.keys()):
                if sel_g_pids_list[i] in list(reid_dict[sel_g_camids_list[i]]):
                    if reid_dict[sel_g_camids_list[i]][sel_g_pids_list[i]]["dis"]>selg_dis_list[i]:
                        reid_dict[sel_g_camids_list[i]][sel_g_pids_list[i]]["dis"] = selg_dis_list[i]
                        reid_dict[sel_g_camids_list[i]][sel_g_pids_list[i]]["id"] = new_id
                else:
                    reid_dict[sel_g_camids_list[i]][sel_g_pids_list[i]] = {"dis":selg_dis_list[i],"id":new_id}  
                    #反向记录：暂时把当前query的每个目标gallery的ID当做reid_dict字典的key（当做queryID），记录每个目标的距离（不只是记录最小值了）
                    # 也可能是因为如果要融合两段跟踪的话，不仅query对应的track要改id为new_id,gallery也是 
            else:
                reid_dict[sel_g_camids_list[i]] = {}
                reid_dict[sel_g_camids_list[i]][sel_g_pids_list[i]] = {"dis":selg_dis_list[i],"id":new_id}
    # 结束后
    # reid_dict[camid][id]记录有希望reid的query的距离  rm_dict[q_camid][q_pid]记录哪些query可能找不到跨相机匹配，应予删除
    return reid_dict,rm_dict

def calc_length(output):
    calc_dict = {}
    for line in output:
        line = line.strip().split(" ")
        cam_id = int(line[0])
        track_id = int(line[1])
        if cam_id not in list(calc_dict.keys()):
            calc_dict[cam_id] = {}
        if track_id not in list(calc_dict[cam_id].keys()):
            calc_dict[cam_id][track_id] = 1
        else:
            calc_dict[cam_id][track_id]+=1
    return calc_dict


def update_output(output,reid_dict,rm_dict,f,max_length=20):
    calc_dict = calc_length(output)  # output=tracklets.txt的所有行（单相机跟踪的输出）
    for line in output: # tracklets.txt的一行（要逐行更新为跨相机跟踪）
        line = line.strip().split(" ")
        cam_id = int(line[0])
        track_id = int(line[1])
        if cam_id in list(rm_dict.keys()): #是否remove
            if track_id in list(rm_dict[cam_id].keys()):  # 判断当前行是否需要写入track3.txt（如果没有找到跨相机匹配就算了）
                continue
        if calc_dict[cam_id][track_id] < max_length: #好像是单相机的每个目标的跟踪长度
            continue
        if cam_id in list(reid_dict.keys()):  #注意trackID和camID是关联的，单独拿出两个trackID相同是否没有意义？
            if track_id in list(reid_dict[cam_id].keys()):  # 意思是哪个cam下的那个id，应该改为new_id
                line[1] = str(reid_dict[cam_id][track_id]["id"]) #修改单相机跟踪结果中的目标id->reid_dict[cam_id][track_id]["id"]，其实是new_id
        f.write(" ".join(line)+"\n")  # 写入目标文件track3.txt






# if __name__ == "__main__":
#     reid_dict,rm_dict = calc_reid(result)  # result是np.savez保存的dist matrix和行列对应的图片信息
#     print(rm_dict,reid_dict)
#     with open(input_path,"r") as f:
#         or_tracks = f.readlines()  # 打开tracklets.txt的所有行（单相机跟踪）
#     g = open(output_path,"w")  # 打开目标文件track3.txt
#     update_output(or_tracks,reid_dict,rm_dict,g)  # 单相机跟踪结果用reid结果更新成多相机跟踪



