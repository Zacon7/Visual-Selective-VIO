import sys

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from utils import custom_transform
from path import Path
from tqdm import tqdm
import pickle
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor


class KITTI(Dataset):
    def __init__(
        self,
        data_root,
        sequence_length=11,
        train_seqs=["00", "01", "02", "04", "06", "08", "09"],
        transform=None,
    ):

        self.data_root = Path(data_root)
        self.sequence_length = sequence_length
        self.transform = transform
        self.train_seqs = train_seqs
        self.image_cache = {}
        self.load_images()

    def load_images(self):
        with ThreadPoolExecutor() as executor:
            futures = []
            for folder in self.train_seqs:
                fpaths = sorted((self.data_root / "sequences/{}/image_2".format(folder)).files("*.png"))
                for img_path in tqdm(fpaths, desc=f'Loading images for sequence {folder}'):
                    if img_path not in self.image_cache:
                        # 提交图像加载任务到线程池
                        future = executor.submit(self.load_image, img_path)
                        futures.append(future)
                print('Train sequence {} image loading submitted.'.format(folder))
            # 等待所有任务完成
            concurrent.futures.wait(futures)
        print('Image load completed.\n')

        if self.transform is not None:
            with ThreadPoolExecutor() as executor:
                # 提交图像变换任务到线程池
                futures = [executor.submit(self.transform_image, path, img) for path, img in self.image_cache.items()]
                # 等待所有任务完成
                concurrent.futures.wait(futures)

            print('Image transformation completed.')

    def load_image(self, img_path):
        img = np.asarray(Image.open(img_path))
        self.image_cache[img_path] = img

    def transform_image(self, img_path, img):
        self.image_cache[img_path] = self.transform(img)


if __name__ == "__main__":
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        custom_transform.SubtractFloat(0.5),
        transforms.Resize((256, 512))
    ])

    # transform_train = [
    #     custom_transform.ToTensor(),
    #     custom_transform.Resize((256, 512)),
    # ]
    # transform_train = custom_transform.Compose(transform_train)

    train_dataset = KITTI(
        data_root="./data", sequence_length=11, transform=transform_train
    )

    # Save the dataset as a .pkl file
    with open('/root/autodl-tmp/kitti.pkl', 'wb') as f:
        pickle.dump(train_dataset.image_cache, f)
