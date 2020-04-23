import torch
from utils import *
from models import TransformerNet
import os
from torchvision import transforms
import time
import cv2
from cv2 import VideoWriter, VideoWriter_fourcc


TITLE = 'gogh'
STYLE_TRANSFORM_PATH = "gogh.pth"
device = ("cuda" if torch.cuda.is_available() else "cpu")

net = TransformerNet()
net.load_state_dict(torch.load(STYLE_TRANSFORM_PATH, map_location=torch.device(device)))
net = net.to(device)

videofile = "input.avi"
#videofile = 0
cap = cv2.VideoCapture(videofile)

if videofile is not 0:
    w = int(cap.get(3))
    h = int(cap.get(4))
    fps = cap.get(5)
    fourcc = VideoWriter_fourcc(*'mp4v')
    writer = VideoWriter('output.mp4', fourcc, fps, (w, h))

count = 0
while cap.isOpened():
    ret, frame = cap.read()
    count += 1
    if ret:
        with torch.no_grad():
          content = itot(frame)
          generated = net(content)
          generated_img = ttoi(generated)
          generated_img = generated_img * 255
          frame = generated_img.astype(np.uint8)
          if videofile == 0:
              cv2.imshow(TITLE, frame)
              k = cv2.waitKey(1) & 0xff
              if k == 27:
                  break
          else:
              writer.write(frame)
        if count % int(fps) == 0:
          print("processing {} second(s)".format(count//fps))
    else:
        break
        
        
if videofile is not 0:
    writer.release()
cap.release()