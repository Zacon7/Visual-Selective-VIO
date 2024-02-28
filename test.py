import argparse
import torch
import logging
from path import Path
from model import DeepVIO
from utils.kitti_eval import KITTI_tester
import numpy as np

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_dir', type=str, default='./data', help='path to the dataset')
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--save_dir', type=str, default='./results', help='path to save the result')
parser.add_argument('--seq_len', type=int, default=11, help='sequence length for LSTM')

parser.add_argument('--train_seq', type=list, default=['00', '01', '02', '04', '06', '08', '09'], help='sequences for training')
parser.add_argument('--val_seq', type=list, default=['00', '01', '02', '04', '05', '06', '07', '08', '09', '10'], help='sequences for validation')
# parser.add_argument('--val_seq', type=list, default=['05', '07', '10'], help='sequences for validation')
parser.add_argument('--seed', type=int, default=0, help='random seed')

parser.add_argument('--img_h', type=int, default=256, help='image height')
parser.add_argument('--img_w', type=int, default=512, help='image width')
parser.add_argument('--v_f_len', type=int, default=512, help='visual feature length')
parser.add_argument('--i_f_len', type=int, default=256, help='imu feature length')
parser.add_argument('--imu_dropout', type=float, default=0.2, help='dropout for the IMU encoder')
parser.add_argument('--imu_encoder', type=str, default='original', help='encoder type [original, separable]')
parser.add_argument('--fuse_method', type=str, default='soft', help='fusion method [cat, soft, hard]')

parser.add_argument('--rnn_hidden_size', type=int, default=1024, help='size of the LSTM latent')
parser.add_argument('--rnn_dropout_out', type=float, default=0.2, help='dropout for the LSTM output layer')
parser.add_argument('--rnn_dropout_between', type=float, default=0.2, help='dropout within LSTM')

parser.add_argument('--experiment_name', type=str, default='flownet_soft', help='experiment name')
parser.add_argument('--ckpt_model', type=str, default='results/train/flownet/flownet_soft_new/checkpoints/best_5.35.pth', help='path to the checkpoint model')
parser.add_argument('--flow_encoder', type=str, default='flownet', help='choose to use the flownet or fastflownet')
parser.add_argument('--flownetBN', default=True, help='choose to use the flownetS or flownetS_BN')
parser.add_argument('--workers', type=int, default=1, help='number of workers')

args = parser.parse_args()

# Set the random seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)


def main():

    # Create Dir
    experiment_dir = Path('./results/test')
    ckpt_name = Path(args.ckpt_model).stem
    experiment_dir.mkdir_p()
    result_dir = experiment_dir.joinpath('{}/{}'.format(args.experiment_name, ckpt_name))
    result_dir.makedirs_p()

    # Create logs
    logger = logging.getLogger(args.experiment_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler = logging.FileHandler(str(result_dir) + '/%s.log' % ckpt_name)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # GPU selections
    str_ids = args.gpu_ids.split(',')
    gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            gpu_ids.append(id)
    if len(gpu_ids) > 0:
        torch.cuda.set_device(gpu_ids[0])

    # Initialize the tester
    tester = KITTI_tester(args)

    # Model initialization
    model = DeepVIO(args)

    model.load_state_dict(torch.load(args.ckpt_model), strict=False)
    print('load model from %s' % args.ckpt_model)
    logger.info('load model from %s' % args.ckpt_model)

    # Feed model to GPU
    model.cuda(gpu_ids[0])
    model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    model.eval()

    errors = tester.eval(model, 'gumbel-softmax', num_gpu=len(gpu_ids))
    tester.generate_plots(result_dir, 30)
    tester.save_text(result_dir)

    for i, seq in enumerate(args.val_seq):
        message = f"Seq: {seq}, t_rel: {errors[i]['t_rel']:.4f}, r_rel: {errors[i]['r_rel']:.4f}, "
        message += f"t_rmse: {errors[i]['t_rmse']:.4f}, r_rmse: {errors[i]['r_rmse']:.4f}, "
        message += f"usage: {errors[i]['usage']:.4f}"
        print(message)
        logger.info(message)

    t_rel = np.mean([errors[i]['t_rel'] for i in range(len(errors))])
    r_rel = np.mean([errors[i]['r_rel'] for i in range(len(errors))])
    t_rmse = np.mean([errors[i]['t_rmse'] for i in range(len(errors))])
    r_rmse = np.mean([errors[i]['r_rmse'] for i in range(len(errors))])
    usage = np.mean([errors[i]['usage'] for i in range(len(errors))])
    message = f'Model evaluation finished\n' \
            f't_rel: {t_rel:.4f}, r_rel: {r_rel:.4f},  ' \
            f't_rmse: {t_rmse:.4f}, r_rmse: {r_rmse:.4f},  ' \
            f'usage: {usage:.4f}'
    print(message)
    logger.info(message)
if __name__ == "__main__":
    main()
