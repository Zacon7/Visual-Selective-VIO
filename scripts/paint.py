import sys
from turtle import color
sys.path.append('/home/zacon/code_projects/Visual-Selective-VIO')
from utils.utils import read_pose_from_text
import numpy as np
from path import Path
from matplotlib import pyplot as plt


def plotPath_2D(seq, abs_pose_gt, abs_poses_est, labels, colors, plot_path_dir):

    fontsize_ = 9
    fig = plt.figure(figsize=(6, 6), dpi=100)
    ax = plt.gca()
    plt.plot(0, 0, 'ko', label='Start Point')


    # Plot 2d trajectory ground-truth map
    x_gt = np.asarray([pose[0, 3] for pose in abs_pose_gt])
    y_gt = np.asarray([pose[1, 3] for pose in abs_pose_gt])
    z_gt = np.asarray([pose[2, 3] for pose in abs_pose_gt])
    plt.plot(x_gt, z_gt, '--', color='black', label='Ground Truth')

    # Plot 2d trajectory estimation map
    for i in range(len(abs_poses_est)):
        x_pred = np.asarray([pose[0, 3] for pose in abs_poses_est[i]])
        y_pred = np.asarray([pose[1, 3] for pose in abs_poses_est[i]])
        z_pred = np.asarray([pose[2, 3] for pose in abs_poses_est[i]])

        plt.plot(x_pred, z_pred, '-', color=colors[i], label=labels[i])

    
    plt.legend(loc="upper left", prop={'size': fontsize_})
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

    png_title = "{}_path_2d".format(seq)
    plt.savefig(plot_path_dir + "/" + png_title + ".png", bbox_inches='tight', pad_inches=0.1)
    plt.close()


def main():
    path_seq = 10
    path_dir = Path('./results/paths')

    # set the estimation label & line-color
    labels = [
        'Baseline',
        'Baseline+FE',
        'Baseline+FE+IE',
        'Baseline+FE+IE+AG'
    ]
    colors = ['blue', 'green', 'brown', 'red']

    # Path 05
    if path_seq == 5:
        gt_pose_path = 'data/poses/05.txt'
        est_pose_paths = [
            'results/test/jointloss/flownet_jointloss_ft1/best_4.14/05_pred.txt',
            'results/test/All_Seq/fastflow_hard_flow6/best_5.69/05_pred.txt',
            'results/test/jointloss/flownet_jointloss_ft1/best_5.55/05_pred.txt',
            'results/test/flownet/flownet_hard/05_pred.txt'
        ]

    # Path 07
    elif path_seq == 7:
        gt_pose_path = 'data/poses/07.txt'
        est_pose_paths = [
            'results/test/fastflownet/fastflow_hard_3e-5/best_11.16/07_pred.txt',
            'results/test/jointloss/flownet_inertial/best_6.03/07_pred.txt',
            'results/test/fastflownet/fastflow_hard_flow6/best_6.56/07_pred.txt',
            'results/test/flownet/flownet_hard/07_pred.txt'
        ]

    # Path 10
    elif path_seq == 10:
        labels = [
            'Rel-Loss Only',
            'Joint Loss (α=1)',
            'Joint Loss (α=20)',
            'Joint Loss (α=100)',
            'Joint Loss (α=200)'
        ]
        colors = ['blue', 'green', 'brown', 'red', 'purple']

        gt_pose_path = 'data/poses/10.txt'
        est_pose_paths = [
            'results/test/All_Seq/flownet_soft/best_5.90/10_pred.txt',
            'results/test/All_Seq/fastflow_hard_flow6/best_5.59/10_pred.txt',
            'results/test/All_Seq/fastflow_hard_flow6/best_5.52/10_pred.txt',
            'results/test/pretrain_model/flownet_cat_5e-05/10_pred.txt',
            'results/test/fastflownet/fastflow_hard_flow6_dwconv/10_pred.txt'
        ]

    gt_poses, _ = read_pose_from_text(gt_pose_path)
    est_poses = [read_pose_from_text(path)[0] for path in est_pose_paths]
    plotPath_2D(path_seq, gt_poses, est_poses, labels, colors, path_dir)


if __name__ == "__main__":
    main()
