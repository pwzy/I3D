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
                tf.Scale(size=(224, 224)),
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
                frame = frame.crop((200, 0, int(w*0.55), h*0.88))
                cv2.imshow('', np.array(frame))
                cv2.waitKey(0)
                frame = self.trans(frame)
                frames.append(frame.unsqueeze(1))
            if len(frames) != 16:
                break
            inputs = torch.cat(frames, axis=1)
            inputs = inputs.unsqueeze(0).cuda()
            high_level = self.extract(inputs)
            high_levels.append(high_level.cpu().numpy())

        return np.array(high_levels)



if __name__=='__main__':
    video_dir = '../data/video/anomaly/'
    save_dir = './features/RGB/anomaly/'
    server = Feature_Extractor()

    for video_name in tqdm(os.listdir(video_dir)):
        video_path = os.path.join(video_dir, video_name)
        high_levels = server.processing_video(video_path)

        np.save(os.path.join(save_dir, video_path.split('/')[-1].split('.')[0]+'.npy'),
                high_levels)

    video_dir = '../data/video/normal/'
    save_dir = './features/RGB/normal/'
    server = Feature_Extractor()

    for video_name in tqdm(os.listdir(video_dir)):
        video_path = os.path.join(video_dir, video_name)
        high_levels = server.processing_video(video_path)

        np.save(os.path.join(save_dir, video_path.split('/')[-1].split('.')[0] + '.npy'),
                high_levels)



