from collections import defaultdict

import numpy as np
import torch

from ..utils import pack_tensor
from .base import Frame, ObjectType, Tracker
from .deep_sort import nn_matching
from .deep_sort.detection import Detection
from .deep_sort.tracker import Tracker as dsTracker

TRACK_STATES = ['Tentative', 'Confirmed']


class DeepSort(Tracker):

    def __init__(self, video_name, fps,min_iou = 0.1):
        super().__init__(video_name, fps)
        self.trackers = {}
        for obj_type in ObjectType:
            metric = nn_matching.NearestNeighborDistanceMetric(
                "cosine", 0.5, 5)
            self.trackers[obj_type] = dsTracker(
                metric, 1-min_iou, int(2 * fps))  #针对每种目标类型，创建DeepSORT跟踪器
        self.finished_tracks = []

    def group_instances(self, instances):
        grouped_instances = defaultdict(list)
        for obj_i in range(len(instances)):
            obj_type = ObjectType(instances.pred_classes[obj_i].item())
            bbox = instances.pred_boxes.tensor[obj_i]  #MaskRCNN的bbox
            feature = instances.roi_features[obj_i].numpy().copy() #MaskRCNN的bbox内的feature
            detection = Detection(
                bbox, instances.scores[obj_i], feature, obj_i) #把MaskRCNN的bbox和feature等送入DeepSORT，创建DeepSORT检测器
            grouped_instances[obj_type].append(detection)  # 对每种目标类型，分别创建检测器
        return grouped_instances  #返回检测器列表

    def get_tracked_instances(self, instances):
        track_ids = torch.zeros((len(instances)), dtype=torch.int)
        states = torch.zeros((len(instances)), dtype=torch.int)
        track_boxes = torch.zeros((len(instances), 4))
        image_speeds = torch.zeros((len(instances), 2))
        for obj_type, tracker in self.trackers.items():
            for track in tracker.tracks:
                if track.time_since_update > 0:
                    continue
                obj_i = track.current_detection.obj_index
                track_ids[obj_i] = track.track_id
                states[obj_i] = track.state
                track_boxes[obj_i] = torch.as_tensor(
                    track.to_tlbr(), dtype=torch.float)
                speed = torch.as_tensor([
                    track.mean[4], track.mean[5] + track.mean[7] / 2])
                image_speeds[obj_i] = speed * self.fps
            for track in tracker.deleted_tracks:
                self.finished_tracks.append((obj_type, track.track_id))
        instances.track_ids = track_ids #跟踪ID
        instances.track_states = states #跟踪状态
        instances.track_boxes = track_boxes #跟踪box
        instances.image_speeds = image_speeds #速度
        if len(instances) > 0:
            instances.finished_tracks = pack_tensor(
                torch.as_tensor(self.finished_tracks), len(instances)) #删除/完成的跟踪
            self.finished_tracks.clear()
        else:
            instances.finished_tracks = torch.zeros((0, 0, 0))
        return instances

    #跟踪一帧
    def track(self, frame):
        grouped_instances = self.group_instances(frame.instances) #用MaskRCNN的结果创建针对每种目标的DeepSORT检测器
        for obj_type, tracker in self.trackers.items():#循环每种目标类型
            print(obj_type)
            tracker = self.trackers[obj_type] #提取得到这种目标的跟踪器
            tracker.predict()  #用KF预测当前帧的目标位置
            tracker.update(grouped_instances[obj_type])  #匹配当前帧的检测和预测，更新各种信息
        instances = self.get_tracked_instances(frame.instances) #记录正在跟踪的目标的信息（ID，状态，box，速度）和完成的跟踪
        return Frame(frame.image_id, frame.image, instances)
