from os.path import join
import cv2
from pytorch_i3d import InceptionI3d
import torch
import numpy as np
import torchvision.transforms as tf
from PIL import Image
import os
from tqdm import tqdm

def to_segments(data, num=32):
    """
    借鉴于：https://github.com/ekosman/AnomalyDetectionCVPR2018-Pytorch/blob/master/feature_extractor.py
	These code is taken from:
	https://github.com/rajanjitenpatel/C3D_feature_extraction/blob/b5894fa06d43aa62b3b64e85b07feb0853e7011a/extract_C3D_feature.py#L805
	:param data: list of features of a certain video
	:return: list of 32 segments
    """
    data = np.array(data)
    Segments_Features = []
    thirty2_shots = np.round(np.linspace(0, len(data) - 1, num=num + 1)).astype(int)
    for ss, ee in zip(thirty2_shots[:-1], thirty2_shots[1:]):
        if ss == ee:
            temp_vect = data[min(ss, data.shape[0] - 1), :]
        else:
            temp_vect = data[ss:ee, :].mean(axis=0)

        temp_vect = temp_vect / np.linalg.norm(temp_vect)
        if np.linalg.norm == 0:
            logging.error("Feature norm is 0")
            exit()
        if len(temp_vect) != 0:
            Segments_Features.append(temp_vect.tolist())

    return Segments_Features

class Feature_Extractor(object):
    def __init__(self, mode='RGB', load_from='./models/rgb_imagenet.pt'):
        if mode == 'flow':
            self.model = InceptionI3d(400, in_channels=2)
        else:
            self.model = InceptionI3d(400, in_channels=3)
        # 原始的rgb_imagenet模型使用的logits个400,157为提取rgb_charades.pt模型使用
        # self.model.replace_logits(157)
        self.model.load_state_dict(torch.load(load_from))
        self.model.cuda()
        self.model.eval()

        self.trans = tf.Compose([
                tf.Resize(size=(224, 224)),
                tf.ToTensor(),
                tf.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])

    def extract(self, inputs):
        with torch.no_grad():
            high_level = self.model.extract_features(inputs).squeeze()
        return high_level


    def processing_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        high_levels = []

        ret = True
        while ret:
            frames = []
            for i in range(16):
                ret, frame_ori = cap.read()
                if ret == False:
                    break
                frame = Image.fromarray(frame_ori.astype('uint8'))
                w, h = frame.size
                # 对图像进行裁剪
                #  frame = frame.crop((200, 0, int(w*0.55), h*0.88))
                # 进行显示
                #  cv2.imshow('test', np.array(frame))
                #  cv2.waitKey(1)
                frame = self.trans(frame)
                #  print(frame.shape)  # torch.Size([3, 224, 224])
                #  print(frame.unsqueeze(1).shape)  # torch.Size([3, 1, 224, 224])
                frames.append(frame.unsqueeze(1))
            if len(frames) != 16:
                break
            inputs = torch.cat(frames, axis=1)
            #  print(inputs.shape) # torch.Size([3, 16, 224, 224])
            inputs = inputs.unsqueeze(0).cuda()
            #  print(inputs.shape) # torch.Size([1, 3, 16, 224, 224]) 代表 (batch, channel, t, h, w)
            high_level = self.extract(inputs)
            high_levels.append(high_level.cpu().numpy())

        return np.array(high_levels)



if __name__=='__main__':
    #  video_dir = '../data/anomaly/'
    video_dir = '../data/'
    save_dir = '../features/'

    if not video_dir.endswith('/'):
        video_dir += '/'
    if not save_dir.endswith('/'):
        save_dir += '/'
    
    
    # TODO video_path need reimplement

    # reimplement for video_path extract 
    video_path = []
    for root, dirs, files in os.walk(video_dir):
        for file in files:
            if file.endswith('.mp4'):
                video_path.append(os.path.join(root, file))

    if not video_path:
        print("video_path 为空，目录中没有对应的视频")
    else:
        #  进行特征提取           
        server = Feature_Extractor()

        for video_name in tqdm(video_path):
            #  print(video_name)  # ../data/anomaly/2_1_1.mp4
            high_levels = server.processing_video(video_name)
            #  print(high_levels.shape) # (187, 1024)
            #  print(type(high_levels)) # numpy.ndarray
            # 将提取的视频特征变为32个snippet
            high_levels = to_segments(high_levels)
            #  print(np.array(high_levels).shape) # (32, 1024)
            save_path = os.path.join(save_dir, video_name[len(video_dir):].split('.')[0]+'.npy')
            #  save_path = os.path.join(save_dir, video_name[len(video_dir):])
            #  save_path = save_dir + video_name[len(video_dir):]
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.save(save_path, high_levels)
    

