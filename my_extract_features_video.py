import cv2
from pytorch_i3d import InceptionI3d
import torch
import numpy as np
import torchvision.transforms as tf
from PIL import Image
import os
from tqdm import tqdm

class Feature_Extractor(object):
    def __init__(self, mode='RGB', load_from='./models/rgb_charades.pt'):
        if mode == 'flow':
            self.model = InceptionI3d(400, in_channels=2)
        else:
            self.model = InceptionI3d(400, in_channels=3)
        self.model.replace_logits(157)
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



