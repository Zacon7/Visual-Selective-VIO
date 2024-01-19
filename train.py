import argparse
import torch
import logging
from path import Path
from utils import custom_transform
from dataset.KITTI_dataset import KITTI
from model import DeepVIO
from collections import defaultdict
from utils.kitti_eval import KITTI_tester
import numpy as np
import math
import pickle

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_dir', type=str, default='./data', help='path to the dataset')
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--save_dir', type=str, default='./results', help='path to save the result')

parser.add_argument('--train_seq', type=list, default=['00', '01', '02', '04', '06', '08', '09'], help='sequences for training')
parser.add_argument('--val_seq', type=list, default=['05', '07', '10'], help='sequences for validation')
parser.add_argument('--seed', type=int, default=0, help='random seed')

parser.add_argument('--img_h', type=int, default=256, help='image height')
parser.add_argument('--img_w', type=int, default=512, help='image width')
parser.add_argument('--v_f_len', type=int, default=512, help='visual feature length')
parser.add_argument('--i_f_len', type=int, default=256, help='imu feature length')
parser.add_argument('--fuse_method', type=str, default='hard', help='fusion method [cat, soft, hard]')
parser.add_argument('--imu_dropout', type=float, default=0, help='dropout for the IMU encoder')

parser.add_argument('--rnn_hidden_size', type=int, default=1024, help='size of the LSTM latent')
parser.add_argument('--rnn_dropout_out', type=float, default=0.2, help='dropout for the LSTM output layer')
parser.add_argument('--rnn_dropout_between', type=float, default=0.2, help='dropout within LSTM')

parser.add_argument('--weight_decay', type=float, default=5e-6, help='weight decay for the optimizer')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--seq_len', type=int, default=11, help='sequence length for LSTM')
parser.add_argument('--workers', type=int, default=16, help='number of workers')
parser.add_argument('--epochs_warmup', type=int, default=40, help='number of epochs for warmup')
parser.add_argument('--epochs_joint', type=int, default=40, help='number of epochs for joint training')
parser.add_argument('--epochs_fine', type=int, default=20, help='number of epochs for finetuning')
parser.add_argument('--lr_warmup', type=float, default=5e-4, help='learning rate for warming up stage')
parser.add_argument('--lr_joint', type=float, default=5e-5, help='learning rate for joint training stage')
parser.add_argument('--lr_fine', type=float, default=1e-6, help='learning rate for finetuning stage')
parser.add_argument('--eta', type=float, default=0.05, help='exponential decay factor for temperature')
parser.add_argument('--temp_init', type=float, default=5, help='initial temperature for gumbel-softmax')
parser.add_argument('--alpha', type=float, default=100, help='weight to balance the translational loss and rotational loss.')
parser.add_argument('--Lambda', type=float, default=3e-5, help='penalty factor for the visual encoder usage')

parser.add_argument('--experiment_name', type=str, default='flownet_hard_new', help='experiment name')
parser.add_argument('--optimizer', type=str, default='Adam', help='type of optimizer [Adam, SGD]')

parser.add_argument('--load_cache', default=True, help='whether to load the pickle dataset cache')
parser.add_argument('--pkl_path', type=str, default='./dataset/kitti.pkl', help='whether to load the pickle dataset cache')

parser.add_argument('--ckpt_model', type=str, default=None, help='path to the checkpoint model')
parser.add_argument('--flow_encoder',type=str, default='flownet', help='choose to use the flownet or fastflownet')
parser.add_argument('--pretrain_flownet',type=str, default='pretrain_models/flownets_bn_EPE2.459.pth.tar', help='wehther to use the pre-trained flownet')
parser.add_argument('--hflip', default=False, action='store_true', help='whether to use horizonal flipping as augmentation')
parser.add_argument('--color', default=False, action='store_true', help='whether to use color augmentations')

parser.add_argument('--print_frequency', type=int, default=10, help='print frequency for loss values')
parser.add_argument('--weighted', default=False, action='store_true', help='whether to use weighted sum')

args = parser.parse_args()

# Set the random seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)

def update_status(epoch, args, model):
    # Warmup stage
    if epoch < args.epochs_warmup:
        lr = args.lr_warmup
        selection = 'random'
        temp = args.temp_init
        for param in model.module.Policy_net.parameters(): # Disable the policy network
            param.requires_grad = False

    # Joint training stage
    elif epoch >= args.epochs_warmup and epoch < (args.epochs_warmup + args.epochs_joint):
        lr = args.lr_joint
        selection = 'gumbel-softmax'
        temp = args.temp_init * math.exp(-args.eta * (epoch - args.epochs_warmup))
        for param in model.module.Policy_net.parameters(): # Enable the policy network
            param.requires_grad = True

    # Finetuning stage
    elif epoch >= args.epochs_warmup + args.epochs_joint:
        lr = args.lr_fine
        selection = 'gumbel-softmax'
        temp = args.temp_init * math.exp(-args.eta * (epoch - args.epochs_warmup))
        
    return lr, selection, temp

def train_epoch(model, optimizer, train_loader, image_cache, selection, temp, logger, ep, p=0.5, weighted=False):
    
    mse_losses = []
    penalties = []
    data_len = len(train_loader)

    for i, (imgs, imus, gts, rot, weights) in enumerate(train_loader):
        if image_cache is not None:
            all_imgs = []
            for seq_imgs in imgs:
                batch_imgs = [image_cache[img_path] for img_path in seq_imgs]   # len(batch): 3, H, W
                all_imgs.append(torch.stack(batch_imgs, dim=0)) # len(11): batch, 3, H, W
            imgs = torch.stack(all_imgs, dim=1) # imgs: (batch, seq_len=11, 3, H, W)


        imgs = imgs.cuda().float()          # imgs: (batch, seq_len=11, 3, H, W)
        imus = imus.cuda().float()          # imus: (batch, 101, 6)
        gts = gts.cuda().float()            # gts:  (batch, 10, 6)
        weights = weights.cuda().float()    # weights: (batch, 1)

        optimizer.zero_grad()
                
        poses, decisions, probs, _ = model(imgs, imus, is_first=True, hc=None, temp=temp, selection=selection, p=p)
        
        if not weighted:
            angle_loss = torch.nn.functional.mse_loss(poses[:,:,:3], gts[:, :, :3])
            translation_loss = torch.nn.functional.mse_loss(poses[:,:,3:], gts[:, :, 3:])
        else:
            weights = weights/weights.sum()
            angle_loss = (weights.unsqueeze(-1).unsqueeze(-1) * (poses[:,:,:3] - gts[:, :, :3]) ** 2).mean()
            translation_loss = (weights.unsqueeze(-1).unsqueeze(-1) * (poses[:,:,3:] - gts[:, :, 3:]) ** 2).mean()
        
        pose_loss = args.alpha * angle_loss + translation_loss        
        penalty_loss = (decisions[:,:,0].float()).sum(-1).mean()
        loss = pose_loss + args.Lambda * penalty_loss 
        
        loss.backward()
        optimizer.step()
        
        if i % args.print_frequency == 0: 
            message = f'Epoch: {ep}, batch: {i}/{data_len}, pose_loss: {pose_loss.item():.6f}, penalty_loss: {penalty_loss.item():.6f}, total loss: {loss.item():.6f}'
            print(message)
            logger.info(message)

        mse_losses.append(pose_loss.item())
        penalties.append(penalty_loss.item())

    return np.mean(mse_losses), np.mean(penalties)


def main():

    # Create Dir
    experiment_dir = Path('./results/train')
    experiment_dir.mkdir_p()
    file_dir = experiment_dir.joinpath('{}/'.format(args.experiment_name))
    file_dir.mkdir_p()
    checkpoints_dir = file_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir_p()
    log_dir = file_dir.joinpath('logs/')
    log_dir.mkdir_p()
    
    # Create logs
    logger = logging.getLogger(args.experiment_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(str(log_dir) + '/%s.txt'%args.experiment_name)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info('----------------------------------------TRAINING----------------------------------')
    logger.info('PARAMETER ...')
    logger.info(args)
    
    # Load the dataset
    transform_train = [
        custom_transform.ToTensor(),
        custom_transform.Resize((args.img_h, args.img_w)),
    ]
    if args.hflip:
        transform_train += [custom_transform.RandomHorizontalFlip()]
    if args.color:
        transform_train += [custom_transform.RandomColorAug()]
    transform_train = custom_transform.Compose(transform_train)

    image_cache = None
    if args.load_cache:
        # Load the dataset from the .pkl file
        with open(args.pkl_path, 'rb') as f:
            image_cache = pickle.load(f)
            
    train_dataset = KITTI(
        args.data_dir,
        sequence_length=args.seq_len,
        train_seqs=args.train_seq,
        transform=transform_train,
        load_cache=args.load_cache
    )
    logger.info('train_dataset: ' + str(train_dataset))
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True
    )
    
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

    # Initialize the model
    model = DeepVIO(args)

    # Continual training or not
    if args.ckpt_model is not None:
        model.load_state_dict(torch.load(args.ckpt_model))
        print('load checkpoint from %s'%args.ckpt_model)
        logger.info('load checkpoint from %s'%args.ckpt_model)
    else:
        print('Training from scratch')
        logger.info('Training from scratch')
    
    # Use the pre-trained flownet (only if training from scratch) or not
    if args.ckpt_model is None and args.pretrain_flownet is not None:
        model_dict = model.Feature_net.state_dict()
        pretrained_flownet = torch.load(args.pretrain_flownet, map_location='cpu')
        if args.flow_encoder == 'flownet':
            update_dict = {k: v for k, v in pretrained_flownet['state_dict'].items() if k in model_dict}
        elif args.flow_encoder == 'fastflownet':
            update_dict = {k: v for k, v in pretrained_flownet.items() if k in model_dict}
        model_dict.update(update_dict)
        model.Feature_net.load_state_dict(model_dict)

    # Feed model to GPU
    model.cuda(gpu_ids[0])
    model = torch.nn.DataParallel(model, device_ids = gpu_ids)
    
    # Initialize or restore the training epoch
    init_epoch = int(args.ckpt_model[-7:-4]) + 1 if args.ckpt_model is not None else 0    
    
    # Initialize the optimizer
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), 
                                     eps=1e-08, weight_decay=args.weight_decay)
    
    best = 10000

    # Start training
    for epoch in range(init_epoch, args.epochs_warmup + args.epochs_joint + args.epochs_fine):
        
        lr, selection, temp = update_status(epoch, args, model)
        optimizer.param_groups[0]['lr'] = lr
        message = f'Epoch: {epoch}, lr: {lr}, selection: {selection}, temperaure: {temp:.5f}'
        print(message)
        logger.info(message)

        model.train()
        avg_pose_loss, avg_penalty_loss = train_epoch(model, optimizer, train_loader, image_cache, selection, temp, logger, epoch, p=0.5)
        
        # Save the model after training
        torch.save(model.module.state_dict(), f'{checkpoints_dir}/{epoch:003}.pth')
        message = f'Epoch {epoch} training finished, pose loss: {avg_pose_loss:.6f}, penalty_loss: {avg_penalty_loss:.6f}, model saved'
        print(message)
        logger.info(message)
        
        if epoch > args.epochs_warmup+args.epochs_joint:
            # Evaluate the model
            print('Evaluating the model')
            logger.info('Evaluating the model')
            with torch.no_grad(): 
                model.eval()
                errors = tester.eval(model, selection='gumbel-softmax', num_gpu=len(gpu_ids))
        
            t_rel = np.mean([errors[i]['t_rel'] for i in range(len(errors))])
            r_rel = np.mean([errors[i]['r_rel'] for i in range(len(errors))])
            t_rmse = np.mean([errors[i]['t_rmse'] for i in range(len(errors))])
            r_rmse = np.mean([errors[i]['r_rmse'] for i in range(len(errors))])
            usage = np.mean([errors[i]['usage'] for i in range(len(errors))])

            if t_rel < best:
                best = t_rel 
                torch.save(model.module.state_dict(), f'{checkpoints_dir}/best_{best:.2f}.pth')
        
            message = f'Epoch {epoch} evaluation finished , t_rel: {t_rel:.4f}, r_rel: {r_rel:.4f}, t_rmse: {t_rmse:.4f}, r_rmse: {r_rmse:.4f}, usage: {usage:.4f}, best t_rel: {best:.4f}'
            logger.info(message)
            print(message)
    
    message = f'Training finished, best t_rel: {best:.4f}'
    logger.info(message)
    print(message)

if __name__ == "__main__":
    main()




