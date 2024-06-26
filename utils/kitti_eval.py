import glob
import numpy as np
import scipy.io as sio
import torch
from PIL import Image
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from utils.utils import *
from tqdm import tqdm


class data_partition():
    def __init__(self, opt, folder):
        super(data_partition, self).__init__()
        self.opt = opt
        self.data_dir = opt.data_dir
        self.seq_len = opt.seq_len
        self.folder = folder
        self.load_data()

    def load_data(self):
        image_dir = self.data_dir + '/sequences/'
        imu_dir = self.data_dir + '/imus/'
        pose_dir = self.data_dir + '/poses/'

        # img_paths: len(2761)    imus: (27601, 6)    abs_pose_gt: (2761, 4, 4)    rel_pose_gt: (2760, 6)
        self.img_paths = glob.glob('{}{}/image_2/*.png'.format(image_dir, self.folder))
        self.imus = sio.loadmat('{}{}.mat'.format(imu_dir, self.folder))['imu_data_interp']
        self.abs_pose_gt, self.rel_pose_gt = read_pose_from_text('{}{}.txt'.format(pose_dir, self.folder))
        self.img_paths.sort()

        self.img_paths_list, self.poses_list, self.imus_list = [], [], []
        start = 0
        n_frames = len(self.img_paths)
        while start + self.seq_len < n_frames:
            self.img_paths_list.append(self.img_paths[start:start + self.seq_len])      # append: len(11)
            self.poses_list.append(self.rel_pose_gt[start:start + self.seq_len - 1])    # append: (10, 6)
            self.imus_list.append(self.imus[start * 10:(start + self.seq_len - 1) * 10 + 1])    # append: (101, 6)
            start += self.seq_len - 1
        self.img_paths_list.append(self.img_paths[start:])  # len(276), with image seires len(11)
        self.poses_list.append(self.rel_pose_gt[start:])    # len(276), with rel_pose(10, 6)
        self.imus_list.append(self.imus[start * 10:])       # len(276), with imus (101, 6)

    def __len__(self):
        return len(self.img_paths_list)

    def __getitem__(self, i):
        image_path_sequence = self.img_paths_list[i]    # image_path_sequence: len(11)
        image_sequence = []
        for img_path in image_path_sequence:
            img_as_img = Image.open(img_path)
            img_as_img = TF.resize(img_as_img, size=(self.opt.img_h, self.opt.img_w))
            img_as_tensor = TF.to_tensor(img_as_img) - 0.5
            img_as_tensor = img_as_tensor.unsqueeze(0)
            image_sequence.append(img_as_tensor)
        image_sequence = torch.cat(image_sequence, 0)           # image_sequence: (11, 3, 256, 512)
        imu_sequence = torch.FloatTensor(self.imus_list[i])     # imu_sequence: (101, 6)
        # gt_sequence = self.poses_list[i][:, :6]                 # gt_sequence: (10, 6)
        return image_sequence, imu_sequence


class KITTI_tester():
    def __init__(self, args):
        super(KITTI_tester, self).__init__()

        # generate data loader for each path
        self.dataloader = []
        for seq in args.val_seq:
            self.dataloader.append(data_partition(args, seq))

        self.args = args

    def test_one_path(self, net, data_path, selection, num_gpu=1, p=0.5):
        hc = None
        pose_list, decision_list, probs_list = [], [], []
        for i, (image_seq, imu_seq) in tqdm(enumerate(data_path), total=len(data_path), smoothing=0.9):
            x_in = image_seq.unsqueeze(0).repeat(num_gpu, 1, 1, 1, 1).cuda()    # x_in: (1, 11, 3, 256, 512)
            i_in = imu_seq.unsqueeze(0).repeat(num_gpu, 1, 1).cuda()            # i_in: (1, 101, 6)
            with torch.no_grad():
                pose, decision, probs, hc = net(x_in, i_in, is_first=(i == 0), hc=hc, selection=selection, p=p)
            pose_list.append(pose[0, :, :].detach().cpu().numpy())
            decision_list.append(decision[0, :, :].detach().cpu().numpy()[:, 0])
            probs_list.append(probs[0, :, :].detach().cpu().numpy())
        pose_est = np.vstack(pose_list)         # pose_est: (2760, 6)
        dec_est = np.hstack(decision_list)      # dec_est:  (2759,)
        prob_est = np.vstack(probs_list)        # prob_est: (2759, 2)
        return pose_est, dec_est, prob_est

    def eval(self, net, selection, num_gpu=1, p=0.5):
        self.est = []
        self.errors = []
        for i, seq in enumerate(self.args.val_seq):
            print(f'testing sequence {seq}')
            rel_pose_est, dec_est, prob_est = self.test_one_path(net, self.dataloader[i], selection, num_gpu=num_gpu, p=p)
            # rel_pose_est:(2760, 6)   dec_est:(2759,)   prob_est:(2759, 2)

            abs_pose_est, abs_pose_gt, t_rel, r_rel, t_rmse, r_rmse, usage, speed = kitti_eval(
                rel_pose_est, self.dataloader[i].rel_pose_gt, self.dataloader[i].abs_pose_gt, dec_est)

            self.est.append({'abs_pose_est': abs_pose_est, 'abs_pose_gt': abs_pose_gt,
                            'decs': dec_est, 'probs': prob_est, 'speed': speed})
            self.errors.append({'t_rel': t_rel, 'r_rel': r_rel, 't_rmse': t_rmse, 'r_rmse': r_rmse, 'usage': usage})

        return self.errors

    def generate_plots(self, save_dir, window_size):
        for i, seq in enumerate(self.args.val_seq):
            plotPath_2D(seq,
                        self.est[i]['abs_pose_gt'],
                        self.est[i]['abs_pose_est'],
                        save_dir,
                        self.est[i]['decs'],
                        self.est[i]['speed'],
                        window_size)

    def save_text(self, save_dir):
        for i, seq in enumerate(self.args.val_seq):
            path = save_dir / '{}_pred.txt'.format(seq)
            saveSequence(self.est[i]['abs_pose_est'], path)
            print('Seq {} saved'.format(seq))


def kitti_eval(rel_pose_est, rel_pose_gt, abs_pose_mat_gt, dec_est):
    '''
    input:
        rel_pose_est: (2760, 6), estimation Relative pose in R6 throughout the trajectory
        rel_pose_gt:  (2760, 6), ground-truth Relative pose in R6 throughout the trajectory
        abs_pose_mat_gt:  (2761, 4, 4), ground-truth Absolute pose in SE(3) throughout the trajectory
        dec_est: (2759,), The decision at every time step except the first frame throughout the trajectory
    return:
        abs_pose_mat_est: len(2761) with estimated Absolute pose in 4x4 matrix SE(3), starting from the first frame
        abs_pose_mat_gt:  len(2761) with ground-truth Absolute pose in 4x4 matrix SE(3), starting from the first frame
        t_rmse, r_rmse: A real number, the RMSE of translation and rotation (euler angles) error of Relative Pose throughout the trajectory, aka RPE
        t_rel, r_rel: A percentage score, average pose error (per hundred meters) for all valid segments throughout the trajectory, aka KITTI_RPE
        usage: Average percentage of visual features enabled across the entire trajectory
        speed: len(2761), the speed of each frame of picture
    '''
    # First decision is always true
    dec_est = np.insert(dec_est, 0, 1)  # dec_est: (27560,)

    # Calculate Relative Pose Error, RPE
    # Calculate the translational and rotational RMSE of Relative pose throughout the trajectory
    t_rmse, r_rmse = rmse_err_cal(rel_pose_est, rel_pose_gt)

    # Transfer R6 relative pose to 4x4 absolute pose matrix SE(3)
    abs_pose_mat_est = path_accu(rel_pose_est)
    # abs_pose_mat_gt = path_accu(rel_pose_gt)

    # Using KITTI metric
    err_list, t_rel, r_rel, speed = kitti_metric(abs_pose_mat_est, abs_pose_mat_gt)

    # Convert errors to percentage and Convert rotation error from radians to angles
    t_rel = t_rel * 100
    r_rel = r_rel / np.pi * 180 * 100
    r_rmse = r_rmse / np.pi * 180
    usage = np.mean(dec_est) * 100  # Average percentage of visual features enabled across the entire trajectory

    return abs_pose_mat_est, abs_pose_mat_gt, t_rel, r_rel, t_rmse, r_rmse, usage, speed


def kitti_metric(abs_pose_est, abs_pose_gt):
    '''
    Traverse all the poses at both ends of the fixed distance, and calculate their errors.
    input:
        abs_pose_est: len(2761) with estimated Absolute pose in SE(3), starting from the first frame
        abs_pose_gt:  len(2761) with ground-truth Absolute pose in SE(3), starting from the first frame
    return:
        err: len(n < 2761) with average errors per meter for all subsequences of different lengths
        t_rel, r_rel: a real number, average error per meter for all subsequences of different lengths throughout the trajectory
        np.asarray(speed):
    '''
    subsequences = [100, 200, 300, 400, 500, 600, 700, 800]

    err = []
    # Calculate the accumulate distance and current speed for each frame
    dist, speed = trajectoryDistances(abs_pose_gt)   # dist, speed: len(2761)
    step_size = 10  # 10Hz

    for i in range(0, len(abs_pose_gt), step_size):    # Iterate through all frames with step size

        for len_ in subsequences:   # Iterate through all subsequences

            # Find the last frame index legnth away from current frame
            j = lastFrameFromSegmentLength(dist, i, len_)

            # Continue if sequence not long enough
            if j == -1 or j >= len(abs_pose_est) or i >= len(abs_pose_est):
                continue

            # Calculate the Relative pose error between ground-truth and estimation from end to end
            pose_delta_gt = np.dot(np.linalg.inv(abs_pose_gt[i]), abs_pose_gt[j])
            pose_delta_est = np.dot(np.linalg.inv(abs_pose_est[i]), abs_pose_est[j])
            pose_error = np.dot(np.linalg.inv(pose_delta_est), pose_delta_gt)

            # Calculate Relative Pose Error, RPE
            # Calculate the rotation error and translational distances of Relative Pose
            r_err = rotationError(pose_error)
            t_err = translationError(pose_error)

            # Calculate the average error per meter between two pose matrices
            err.append([i, r_err / len_, t_err / len_, len_])

    # Calculate average error per meter for all subsequences of different lengths throughout the trajectory
    t_rel, r_rel = computeOverallErr(err)
    return err, t_rel, r_rel, np.asarray(speed)


def plotPath_2D(seq, abs_pose_gt, abs_pose_est, plot_path_dir, decision, speed, window_size):

    # Apply smoothing to the decision
    decision = np.insert(decision, 0, 1)
    decision = moving_average(decision, window_size)

    fontsize_ = 10
    plot_keys = ["Ground Truth", "Ours"]
    styles = ['r-', 'b-']

    # get the value
    x_gt = np.asarray([pose[0, 3] for pose in abs_pose_gt])
    y_gt = np.asarray([pose[1, 3] for pose in abs_pose_gt])
    z_gt = np.asarray([pose[2, 3] for pose in abs_pose_gt])

    x_pred = np.asarray([pose[0, 3] for pose in abs_pose_est])
    y_pred = np.asarray([pose[1, 3] for pose in abs_pose_est])
    z_pred = np.asarray([pose[2, 3] for pose in abs_pose_est])

    # Plot 2d trajectory estimation map
    fig = plt.figure(figsize=(6, 6), dpi=100)
    ax = plt.gca()
    plt.plot(x_gt, z_gt, styles[0], label=plot_keys[0])
    plt.plot(x_pred, z_pred, styles[1], label=plot_keys[1])
    plt.plot(0, 0, 'ko', label='Start Point')
    plt.legend(loc="upper right", prop={'size': fontsize_})
    plt.xlabel('X (m)', fontsize=fontsize_)
    plt.ylabel('Z (m)', fontsize=fontsize_)

    # set the range of x and y
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xmean = np.mean(xlim)
    ymean = np.mean(ylim)
    plot_radius = max([abs(lim - mean_)
                       for lims, mean_ in ((xlim, xmean),
                                           (ylim, ymean))
                       for lim in lims])
    ax.set_xlim([xmean - plot_radius, xmean + plot_radius])
    ax.set_ylim([ymean - plot_radius, ymean + plot_radius])

    plt.title('2D path')
    png_title = "{}_path_2d".format(seq)
    plt.savefig(plot_path_dir + "/" + png_title + ".png", bbox_inches='tight', pad_inches=0.1)
    plt.close()

    # Plot decision hearmap
    fig = plt.figure(figsize=(8, 6), dpi=100)
    ax = plt.gca()
    cout = np.insert(decision, 0, 0) * 100
    cax = plt.scatter(x_pred, z_pred, marker='o', c=cout, cmap='inferno')
    plt.xlabel('X (m)', fontsize=fontsize_)
    plt.ylabel('Z (m)', fontsize=fontsize_)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xmean = np.mean(xlim)
    ymean = np.mean(ylim)
    ax.set_xlim([xmean - plot_radius, xmean + plot_radius])
    ax.set_ylim([ymean - plot_radius, ymean + plot_radius])
    max_usage = max(cout)
    min_usage = min(cout)
    ticks = np.floor(np.linspace(min_usage, max_usage, num=5))
    cbar = fig.colorbar(cax, ticks=ticks)
    cbar.ax.set_yticklabels([str(i) + '%' for i in ticks])

    plt.title('Visual Modality Usage Heat Map')
    png_title = "{}_decision_smoothed".format(seq)
    plt.savefig(plot_path_dir + "/" + png_title + ".png", bbox_inches='tight', pad_inches=0.1)
    plt.close()

    # Plot the speed map
    fig = plt.figure(figsize=(8, 6), dpi=100)
    ax = plt.gca()
    cout = speed
    cax = plt.scatter(x_pred, z_pred, marker='o', c=cout, cmap='inferno')
    plt.xlabel('X (m)', fontsize=fontsize_)
    plt.ylabel('Z (m)', fontsize=fontsize_)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xmean = np.mean(xlim)
    ymean = np.mean(ylim)
    ax.set_xlim([xmean - plot_radius, xmean + plot_radius])
    ax.set_ylim([ymean - plot_radius, ymean + plot_radius])
    max_speed = max(cout)
    min_speed = min(cout)
    ticks = np.floor(np.linspace(min_speed, max_speed, num=5))
    cbar = fig.colorbar(cax, ticks=ticks)
    cbar.ax.set_yticklabels([str(i) + 'm/s' for i in ticks])

    plt.title('Speed Heat Map')
    png_title = "{}_speed".format(seq)
    plt.savefig(plot_path_dir + "/" + png_title + ".png", bbox_inches='tight', pad_inches=0.1)
    plt.close()
