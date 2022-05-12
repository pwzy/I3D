import cv2
from pytorch_i3d import InceptionI3d
import torch
import numpy as np
import torchvision.transforms as tf
from PIL import Image
import os
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt

class Feature_Extractor(object):
    def __init__(self, mode='RGB', load_from='./models/rgb_charades.pt'):
        self.load_from = load_from
        if mode == 'FLOW':
            self.model = InceptionI3d(400, in_channels=2)
        else:
            self.model = InceptionI3d(400, in_channels=3)
        self.model.replace_logits(400)
        self.model.load_state_dict(torch.load(load_from))
        self.model.cuda()
        self.model.eval()

    def extract(self, inputs):
        with torch.no_grad():
            high_level = self.model.extract_features(inputs).squeeze()

        return high_level

    def frame(self, frame_dir):
        trans = tf.Compose([
            tf.Scale(size=(224, 224)),
            tf.ToTensor(),
            tf.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        files = glob(os.path.join(frame_dir, '*.jpg'))
        files.sort(key=lambda x:x)
        files_num = len(files)
        high_levels = []
        for i in tqdm(range(int(files_num//16))):
            frames = []
            for j in range(16):
                frame_ori = Image.open(files[i*16 + j]).convert('RGB')
                frame = trans(frame_ori)
                frames.append(frame.unsqueeze(1))
            inputs = torch.cat(frames, axis=1)
            inputs = inputs.unsqueeze(0).cuda()
            high_level = self.extract(inputs)
            high_levels.append(high_level.cpu().numpy())

        np.save(os.path.join('./features/RGB', frame_dir.split('/')[-1] + '.npy'),
                np.array(high_levels))
        print(np.array(high_levels).shape)

    def flow(self, flow_dir):
        files = glob(os.path.join(flow_dir, '*.npy'))
        files.sort(key=lambda x: x)
        files_num = len(files)
        high_levels = []
        for i in tqdm(range(int(files_num//16))):
            flows = []
            for j in range(16):
                # print(files[i*16 + j])
                flow = np.load(files[i*16 + j])/20.0
                # plt.imshow(flow[..., 0])
                # plt.pause(0.01)
                flow = np.transpose(flow, [2,0,1])
                flow = torch.tensor(flow, dtype=torch.float32)
                flows.append(flow.unsqueeze(1))
            inputs = torch.cat(flows, axis=1)
            # print(inputs.size())
            inputs = inputs.unsqueeze(0).cuda()
            high_level = self.extract(inputs)
            high_levels.append(high_level.cpu().numpy())

        np.save(os.path.join('./features/FLOW', flow_dir.split('/')[-1] + '.npy'),
                np.array(high_levels))


if __name__=='__main__':
    # RGB
    server = Feature_Extractor(mode='RGB')
    server.frame(frame_dir='../dataset/frame/video1')

    # FLOW
    # server = Feature_Extractor(mode='FLOW', load_from='./models/flow_imagenet.pt')
    # server.flow(flow_dir='../dataset/flow/video1')


