import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
import timeit
import os
import cv2
from datetime import datetime
from tensorboardX import SummaryWriter
from models import *
from utils import *


DATASET="dataset/"
SAVE_PATH=os.path.join("run/","run_{}/".format(len(os.listdir("run/"))+1))
os.mkdir(SAVE_PATH)
BATCH_SIZE=8
STYLE_IMG="images/gogh.jpg"
CONTENT_WEIGHT=1
STYLE_WEIGHT=100000
LR=1e-3

device = "cuda" if torch.cuda.is_available() else "cpu"

def train():
	train_dataset = datasets.ImageFolder(DATASET, transform=transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop(256),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	]))
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE)
	train_size = len(train_loader.dataset)

	VGG = VGGNet().to(device)
	style = cv2.imread(STYLE_IMG)
	style = itot(style)
	style = style.repeat(BATCH_SIZE, 1, 1, 1)
	style_features = VGG(style)
	style_gram = {}
	for layer, value in style_features.items():
		style_gram[layer] = gram(value)

	Transformer = TransformerNet().to(device)
	optimizer = optim.Adam(Transformer.parameters(), lr=LR)

	log_dir = os.path.join(SAVE_PATH, 'log', datetime.now().strftime('%b%d_%H-%M-%S'))
	writer = SummaryWriter(log_dir=log_dir)

	batch = 1

	print("Start Training...")
	start_time = timeit.default_timer()

	for content, _ in train_loader:
		batch_size = content.shape[0]
		torch.cuda.empty_cache()
		optimizer.zero_grad()
		
		content = content[:, [2,1,0]].to(device)
		generated = Transformer(content)
		content_features = VGG(content)
		generated_features = VGG(generated)

		MSELoss = nn.MSELoss().to(device)
		content_loss = CONTENT_WEIGHT * MSELoss(generated_features['8'], content_features['8'])

		style_loss = 0
		for layer, value in generated_features.items():
			style_loss += MSELoss(gram(value), style_gram[layer][:batch_size])
		style_loss *= STYLE_WEIGHT

		total_loss = content_loss + style_loss
		total_loss.backward()
		optimizer.step()
		
		writer.add_scalar('data/style_loss', style_loss, batch)
		writer.add_scalar('data/content_loss', content_loss, batch)
		writer.add_scalar('data/total_loss', total_loss, batch)

		print("Batch: {}/{} Style_loss: {} Content_loss: {}".format(batch, train_size//BATCH_SIZE, style_loss, content_loss))
		batch += 1

	stop_time = timeit.default_timer()
	print("End of Training. Time: {}".format(stop_time-start_time))
	torch.save(Transformer.state_dict(), os.path.join(SAVE_PATH,"transformer_weight.pth"))

train()