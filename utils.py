import numpy as np
import torch
from torchvision import transforms, datasets
import cv2
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

def gram(tensor):
	b, c, h, w = tensor.shape
	x = tensor.view(b, c, h*w)
	x_t = x.transpose(1, 2)
	return torch.bmm(x, x_t) / (c*h*w)

def itot(img):
    itot_t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tensor = itot_t(img).unsqueeze(0)
    return tensor.to(device)
    
def ttoi(tensor):
    denorm = transforms.Normalize(mean=[-2.12, -2.04, -1.80], std=[4.37, 4.46, 4.44])
    tensor = tensor.clone().detach().squeeze()
    tensor = denorm(tensor)
    img = tensor.cpu().numpy()
    img = img.transpose(1, 2, 0)
    return np.clip(img, 0, 1)