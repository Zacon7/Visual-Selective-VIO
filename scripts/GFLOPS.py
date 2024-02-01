import sys
sys.path.append('/home/zacon/code_projects/Visual-Selective-VIO')

from train import args
from model import DeepVIO
import torch
from thop import profile


model = DeepVIO(args)
ckpt_model = torch.load('results/train/fastflow_hard_flow6/checkpoints/best_6.56.pth', map_location='cpu')
model.load_state_dict(ckpt_model, strict=False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

imgs = torch.randn((1, 11, 3, 256, 512)).to(device)
imus = torch.randn((1, 101, 6)).to(device)

flops, params = profile(model, inputs=(imgs, imus, ))
print("GFLOPs :{:.2f}, Params : {:.2f}".format(flops / 1e9, params / 1e6))  # flops单位G，para单位M
