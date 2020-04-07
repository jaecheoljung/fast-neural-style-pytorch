import torch
import torch.nn as nn
from torchsummary import summary
import torchvision.models as models

class VGGNet(nn.Module):
    def __init__(self):
        super(VGGNet, self).__init__()
        self.select = ['3', '8', '17', '22', '26', '35'] 
        self.vgg = models.vgg19(pretrained=True).features
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, x):
        features = {}
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.select:
                features[name] = x
        return features


class TransformerNet(nn.Module):
	def __init__(self):
		super(TransformerNet, self).__init__()
		self.ConvBlock = nn.Sequential(
			Conv(3, 32, kernel=9, stride=1),
			nn.ReLU(),
			Conv(32, 64, kernel=3, stride=2),
			nn.ReLU(),
			Conv(64, 128, kernel=3, stride=2),
			nn.ReLU()
		)
		self.ResBlock = nn.Sequential(
			Res(ch=128, kernel=3),
			Res(ch=128, kernel=3),
			Res(ch=128, kernel=3),
			Res(ch=128, kernel=3),
			Res(ch=128, kernel=3)
		)
		self.DeconvBlock = nn.Sequential(
			Deconv(128, 64, kernel=3, stride=2, padding=1),
			nn.ReLU(),
			Deconv(64, 32, kernel=3, stride=2, padding=1),
			nn.ReLU(),
			Conv(32, 3, kernel=9, stride=1, norm="None")
		)

	def forward(self, x):
		x = self.ConvBlock(x)
		x = self.ResBlock(x)
		out = self.DeconvBlock(x)
		return out

class Conv(nn.Module):
	def __init__(self, in_channel, out_channel, kernel, stride, norm="batch"):
		super(Conv, self).__init__()
		padding = kernel // 2
		self.pad1 = nn.ReflectionPad2d(padding)
		self.conv2 = nn.Conv2d(in_channel, out_channel, kernel, stride)
		self.norm3 = nn.InstanceNorm2d(out_channel, affine=True)
		self.norm = norm

	def forward(self, x):
		x = self.pad1(x)
		x = self.conv2(x)
		if self.norm is "batch":
			out = self.norm3(x)
		else:
			out = x
		return out

class Res(nn.Module):
	def __init__(self, ch, kernel):
		super(Res, self).__init__()
		self.conv1 = Conv(ch, ch, kernel, stride=1)
		self.conv2 = Conv(ch, ch, kernel, stride=1)
		self.relu = nn.ReLU()

	def forward(self, x):
		tmp = x
		x = self.relu(self.conv1(x))
		x = self.conv2(x)
		out = x + tmp
		return out

class Deconv(nn.Module):
	def __init__(self, in_channel, out_channel, kernel, stride, padding):
		super(Deconv, self).__init__()
		self.convT1 = nn.ConvTranspose2d(in_channel, out_channel, kernel, stride, kernel//2, padding)
		self.norm2 = nn.InstanceNorm2d(out_channel, affine=True)

	def forward(self, x):
		x = self.convT1(x)
		out = self.norm2(x)
		return out

if __name__=="__main__":
	model = TransformerNet()
	print(model)
	summary(model, (3, 256, 256))