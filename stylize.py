import torch
from utils import *
from models import TransformerNet
import os
from torchvision import transforms
import time
import cv2

STYLE_TRANSFORM_PATH = "models/transformer_weight.pth"

device = ("cuda" if torch.cuda.is_available() else "cpu")

def stylize():

    net = TransformerNet()
    net.load_state_dict(torch.load(STYLE_TRANSFORM_PATH))
    net = net.to(device)

    with torch.no_grad():
        while(1):
            torch.cuda.empty_cache()
            print("Stylize Image~ Press Ctrl+C and Enter to close the program")
            content_image_path = input("Enter the image path: ")
            content_image = cv2.imread(content_image_path)
            content_tensor = itot(content_image)
            generated_tensor = net(content_tensor)
            generated_image = ttoi(generated_tensor)
            generated_image = cv2.cvtColor(generated_image, cv2.COLOR_BGR2RGB)
            plt.imshow(generated_image)
            plt.show()
            
stylize()