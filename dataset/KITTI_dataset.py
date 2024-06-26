from scipy.ndimage import convolve1d
from scipy.signal.windows import triang
from scipy.ndimage import gaussian_filter1d
from collections import Counter
from utils.utils import rotationError, read_pose_from_text, poses_SE3_to_quaternion
from path import Path
import scipy.io as sio
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import sys
sys.path.append('..')

IMU_FREQ = 10


class KITTI(Dataset):
    def __init__(self,
                 data_root,
                 sequence_length=11,
                 train_seqs=['00', '01', '02', '04', '06', '08', '09'],
                 transform=None,
                 load_cache=False
                 ):

        self.data_root = Path(data_root)
        self.sequence_length = sequence_length
        self.transform = transform
        self.train_seqs = train_seqs
        self.load_cache = load_cache
        self.make_dataset()

    def make_dataset(self):
        sequence_set = []
        for folder in self.train_seqs:

            # poses_abs: (img_nums, 4, 4)       poses_rel: (img_nums-1, 6)
            # poses_abs= {R, t, 0, 1}           poses_rel = (θx, θy, θz, ρx, ρy, ρz)
            poses_abs, poses_rel = read_pose_from_text(self.data_root / 'poses/{}.txt'.format(folder))

            # poses_abs_qua: (img_nums, 4) with (qw, qx, qy, qz)
            # poses_abs_qua = poses_SE3_to_quaternion(poses_abs)

            # imus: ((img_nums-1)*IMU_FREQ + 1, 6)
            imus = sio.loadmat(self.data_root / 'imus/{}.mat'.format(folder))['imu_data_interp']

            # fpaths: len(img_nums)
            fpaths = sorted((self.data_root / 'sequences/{}/image_2'.format(folder)).files("*.png"))

            for i in range(len(fpaths) - self.sequence_length):
                img_samples = fpaths[i:i + self.sequence_length]                    # img_samples: len(11)
                imu_samples = imus[i * IMU_FREQ:(i + self.sequence_length - 1)
                                   * IMU_FREQ + 1]                                  # imu_samples: (101, 6)

                # pose_abs_samples = poses_abs_qua[i:i + self.sequence_length]        # pose_abs_samples: (11, 4)
                pose_abs_samples = poses_abs[i:i + self.sequence_length]            # pose_abs_samples: (11, 4, 4)
                pose_rel_samples = poses_rel[i:i + self.sequence_length - 1]        # pose_rel_samples: (10, 6)

                pose_error = np.dot(np.linalg.inv(pose_abs_samples[0]), pose_abs_samples[-1])
                segment_rot = rotationError(pose_error)

                sample = {
                    'imgs': img_samples,                  # imgs: len(11)
                    'imus': imu_samples,                  # imus: (101, 6)
                    'abs_pose_gt': pose_abs_samples,      # abs_pose_gt: (11, 4, 4) or (11, 4)
                    'rel_pose_gt': pose_rel_samples,      # rel_pose_gt: (10, 6)
                    'rot': segment_rot                    # rot: a real number
                }
                sequence_set.append(sample)

        self.samples = sequence_set     # samples: len(17260) = len(4541-11) + len(1101-11) + ... + len(1591-11)

        # Generate weights based on the rotation of the training segments
        # Weights are calculated based on the histogram of rotations according to
        # the method in https://github.com/YyzHarry/imbalanced-regression
        rot_list = np.array([np.cbrt(item['rot'] * 180 / np.pi) for item in self.samples])
        rot_range = np.linspace(np.min(rot_list), np.max(rot_list), num=10)
        indexes = np.digitize(rot_list, rot_range, right=False)
        num_samples_of_bins = dict(Counter(indexes))
        emp_label_dist = [num_samples_of_bins.get(i, 0) for i in range(1, len(rot_range) + 1)]

        # Apply 1d convolution to get the smoothed effective label distribution
        lds_kernel_window = get_lds_kernel_window(kernel='gaussian', ks=7, sigma=5)
        eff_label_dist = convolve1d(np.array(emp_label_dist), weights=lds_kernel_window, mode='constant')

        self.weights = [np.float32(1 / eff_label_dist[bin_idx - 1]) for bin_idx in indexes]     # weights: len(17260)

    def __getitem__(self, index):
        '''
        return:
            imgs: (11, 3, H, W), or len(11): [img_path0, img_path1, ..., img_path_10]
            imus: (101, 6),
            abs_pose_gt:  (11, 4, 4), absolute poses in SE(3) from the first frame with {R, t, 0, 1}
                or
            abs_pose_gt:  (11, 4), absolute poses in quaternion from the first frame with [qw, qx, qy, qz]
            rel_pose_gt:  (10, 6), relative poses in 6DoF between every two frame with [θx, θy, θz, ρx, ρy, ρz]
            rot:  a real number,
            weights: a real number
        '''
        sample = self.samples[index]

        if self.load_cache:
            imgs = sample['imgs']   # only img_path
        else:
            imgs = [np.asarray(Image.open(img)) for img in sample['imgs']]
            if self.transform is not None:
                imgs = self.transform(np.asarray(imgs))

        imus = np.copy(sample['imus'])
        abs_pose_gt = np.copy(sample['abs_pose_gt']).astype(np.float32)
        rel_pose_gt = np.copy(sample['rel_pose_gt']).astype(np.float32)

        rot = sample['rot'].astype(np.float32)
        weight = self.weights[index]

        return imgs, imus, rel_pose_gt, rot, weight

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Training sequences: '
        for seq in self.train_seqs:
            fmt_str += '{} '.format(seq)
        fmt_str += '\n'
        fmt_str += '    Number of segments: {}\n'.format(self.__len__())
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))

        return fmt_str


def get_lds_kernel_window(kernel, ks, sigma):
    assert kernel in ['gaussian', 'triang', 'laplace']
    half_ks = (ks - 1) // 2
    if kernel == 'gaussian':
        base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
        kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / max(gaussian_filter1d(base_kernel, sigma=sigma))
    elif kernel == 'triang':
        kernel_window = triang(ks)
    else:
        def laplace(x): return np.exp(-abs(x) / sigma) / (2. * sigma)
        kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))
                             ) / max(map(laplace, np.arange(-half_ks, half_ks + 1)))

    return kernel_window
