import sys
sys.path.append('/home/zacon/code_projects/Visual-Selective-VIO')

from model import DeepVIO
import torch
from thop import profile
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_dir', type=str, default='./data', help='path to the dataset')
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--save_dir', type=str, default='./results', help='path to save the result')
parser.add_argument('--seq_len', type=int, default=11, help='sequence length for LSTM')

parser.add_argument('--train_seq', type=list, default=['00', '01', '02', '04', '06', '08', '09'], help='sequences for training')
parser.add_argument('--val_seq', type=list, default=['05', '07', '10'], help='sequences for validation')
parser.add_argument('--seed', type=int, default=0, help='random seed')

parser.add_argument('--img_h', type=int, default=256, help='image height')
parser.add_argument('--img_w', type=int, default=512, help='image width')
parser.add_argument('--v_f_len', type=int, default=512, help='visual feature length')
parser.add_argument('--i_f_len', type=int, default=256, help='imu feature length')
parser.add_argument('--imu_dropout', type=float, default=0.2, help='dropout for the IMU encoder')
parser.add_argument('--imu_encoder', type=str, default='separable', help='encoder type [original, separable]')
parser.add_argument('--fuse_method', type=str, default='EFA', help='fusion method [cat, soft, hard, EFA]')

parser.add_argument('--rnn_hidden_size', type=int, default=1024, help='size of the LSTM latent')
parser.add_argument('--rnn_dropout_out', type=float, default=0.2, help='dropout for the LSTM output layer')
parser.add_argument('--rnn_dropout_between', type=float, default=0.2, help='dropout within LSTM')

parser.add_argument('--ckpt_model', type=str, default='pretrain_models/flownet_cat_3e-05.model', help='path to the checkpoint model')
parser.add_argument('--flow_encoder', type=str, default='flownet', help='choose to use the flownet or fastflownet')
parser.add_argument('--flownetBN', default=True, help='choose to use the flownetS or flownetS_BN')

args = parser.parse_args()

model = DeepVIO(args)
model.load_state_dict(torch.load(args.ckpt_model), strict=False)
print('load model from %s' % args.ckpt_model)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

imgs = torch.randn((1, 11, 3, 256, 512)).to(device)
imus = torch.randn((1, 101, 6)).to(device)

flops, params = profile(model, inputs=(imgs, imus))
print("GFLOPs :{:.2f}, Params : {:.2f}".format(flops / 1e9, params / 1e6))  # flops单位G，para单位M
