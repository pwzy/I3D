import os
import numpy as np
import cv2


def rgb2gray(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def cpt_of(prev, now, TVL1):
    '''
        TVL1 = cv2.createOptFlow_DualTVL1()
    '''
    prev_ = rgb2gray(prev.copy())
    now_  = rgb2gray(now.copy())
    return np.array(TVL1.calc(prev_, now_, None), dtype=np.float32)

video_name = 'video1'

video_path = os.path.join('../dataset/video', video_name+'.mp4')
frame_dir = os.path.join('../dataset/frame', video_name)
flow_dir  = os.path.join('../dataset/flow', video_name)

if not os.path.exists(frame_dir):
    os.mkdir(frame_dir)
if not os.path.exists(flow_dir):
    os.mkdir(flow_dir)


cap = cv2.VideoCapture(video_path)

ret, frame_prev = cap.read()
frame_prev = cv2.resize(frame_prev, (224, 224))

TVL1 = cv2.createOptFlow_DualTVL1()
cnt = 0
while ret:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (224, 224))
    flow  = cpt_of(frame_prev, frame, TVL1)

    cv2.imshow('frame', frame)
    cv2.imshow('flow', flow[..., 0])
    cv2.waitKey(1)

    cv2.imwrite(os.path.join(frame_dir, str(cnt).zfill(4)+'.jpg'), frame)
    np.save(os.path.join(flow_dir, str(cnt).zfill(4) + '.npy'), flow)

    frame_prev = frame
    cnt += 1

    print(cnt)






