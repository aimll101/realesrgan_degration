# coding=gbk
import cv2
import math
import numpy as np
import os
import os.path as osp
import random
import time
import torch
import torchvision
from torch.nn import functional as F
from degradations import circular_lowpass_kernel, random_mixed_kernels, random_add_gaussian_noise_pt, \
    random_add_poisson_noise_pt
import file_client
from logger import get_root_logger
from img_util import imfrombytes, img2tensor
from img_process_util import filter2D, USMSharp
from transforms import augment, paired_random_crop
from DiffJPEG import DiffJPEG
import PIL.Image as Image


class RealESRGANDatadegration():
    def __init__(self):
        super(RealESRGANDatadegration, self).__init__()
        self.scale = 1  # 缩放比例
        # self.gt_size = gt_size #输入图像的尺寸，需要h=w
        gpu_id = None  # 0
        device = None  #
        if gpu_id:
            self.device = torch.device(
                f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu') if device is None else device
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device

        # the first degradation process
        self.resize_prob = [0.1, 0.9, 0.0]  # up, down, keep
        self.resize_range = [0.15, 1.5]
        self.gaussian_noise_prob = 0.8
        self.noise_range = [1, 30]
        self.poisson_scale_range = [0.05, 3]
        self.gray_noise_prob = 1
        self.jpeg_range = [20, 30]

        # the second degradation process
        self.second_blur_prob = 0.8
        self.resize_prob2 = [0.1, 0.9, 0.0]  # up, down, keep
        self.resize_range2 = [0.3, 1.2]
        self.gaussian_noise_prob2 = 0.8
        self.noise_range2 = [1, 25]
        self.poisson_scale_range2 = [0.05, 2.5]
        self.gray_noise_prob2 = 1
        self.jpeg_range2 = [20, 30]

        # blur settings for the first degradation
        self.blur_kernel_size = 21
        self.kernel_list = ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso']
        self.kernel_prob = [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]  # a list for each kernel probability
        self.sinc_prob = 0.1
        self.blur_sigma = [0.2, 3]
        self.betag_range = [0.5, 4]  # betag used in generalized Gaussian blur kernels
        self.betap_range = [1, 2]  # betap used in plateau blur kernels

        ## blur settings for the second degradation
        self.blur_kernel_size2 = 21
        self.kernel_list2 = ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso']
        self.kernel_prob2 = [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]
        self.sinc_prob2 = 0.1
        self.blur_sigma2 = [0.2, 1.5]
        self.betag_range2 = [0.5, 4]
        self.betap_range2 = [1, 2]

        # a final sinc filter
        self.final_sinc_prob = 0.8
        ############################
        self.use_hflip = False
        self.use_rot = False
        ################
        self.kernel_range = [2 * v + 1 for v in range(3, 11)]  ## kernel size ranges from 7 to 21
        # TODO: kernel range is now hard-coded, should be in the configure file
        self.pulse_tensor = torch.zeros(21, 21).float()  # convolving with pulse tensor brings no blurry effect
        self.pulse_tensor[10, 10] = 1

        self.file_client = None
        self.usm_sharpener = USMSharp().to(self.device)  # do usm sharpening
        self.jpeger = DiffJPEG(differentiable=False).to(self.device)  # # simulate JPEG compression artifacts
        self.kernel_size = random.choice(self.kernel_range)
        self.kernel_1 = random_mixed_kernels(
            self.kernel_list,
            self.kernel_prob,
            self.kernel_size,
            self.blur_sigma,
            self.blur_sigma, [-math.pi, math.pi],
            self.betag_range,
            self.betap_range,
            noise_range=None)
        self.kernel_size_2 = random.choice(self.kernel_range)
        self.kernel_size_3 = random.choice(self.kernel_range)
        # random resize
        self.updown_type_1 = random.choices(['up', 'down', 'keep'], self.resize_prob)[0]
        if self.updown_type_1 == 'up':
            self.scale_1 = np.random.uniform(1, self.resize_range[1])
        elif self.updown_type_1 == 'down':
            self.scale_1 = np.random.uniform(self.resize_range[0], 1)
        else:
            self.scale_1 = 1
        self.mode_1 = random.choice(['area', 'bilinear', 'bicubic'])
        self.gaussian_1 = np.random.uniform()
        self.blur_1 = np.random.uniform()
        # random resize
        self.updown_type_2 = random.choices(['up', 'down', 'keep'], self.resize_prob2)[0]
        if self.updown_type_2 == 'up':
            self.scale_2 = np.random.uniform(1, self.resize_range2[1])
        elif self.updown_type_2 == 'down':
            self.scale_2 = np.random.uniform(self.resize_range2[0], 1)
        else:
            self.scale_2 = 1
        self.mode_2 = random.choice(['area', 'bilinear', 'bicubic'])
        self.gaussian_2 = np.random.uniform()

        self.random_number = np.random.uniform()

        self.mode_3 = random.choice(['area', 'bilinear', 'bicubic'])

    @torch.no_grad()
    def kernel(self):
        if self.file_client is None:
            self.file_client = file_client.FileClient()

        # -------------------------------- Load gt images -------------------------------- #
        # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
        gt_path = self.paths
        # avoid errors caused by high latency in reading files
        retry = 3
        while retry > 0:
            try:
                img_bytes = self.file_client.get(gt_path, 'gt')
            except (IOError, OSError) as e:
                logger = get_root_logger()
                logger.warn(f'File client error: {e}, remaining retry times: {retry - 1}')
                # change another file to read
                index = random.randint(0, self.__len__())
                gt_path = self.paths[index]
                time.sleep(1)  # sleep 1s for occasional server congestion
            else:
                break
            finally:
                retry -= 1

        img_gt = imfrombytes(img_bytes, float32=True)
        #
        # # -------------------- Do augmentation for training: flip, rotation -------------------- #
        # img_gt = augment(img_gt, self.use_hflip, self.use_rot)
        #
        # # crop or pad to 400
        # # TODO: 400 is hard-coded. You may change it accordingly
        # h, w = img_gt.shape[0:2]
        # crop_pad_size = self.gt_size
        # # pad
        # if h < crop_pad_size or w < crop_pad_size:
        #     pad_h = max(0, crop_pad_size - h)
        #     pad_w = max(0, crop_pad_size - w)
        #     img_gt = cv2.copyMakeBorder(img_gt, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
        # # crop
        # if img_gt.shape[0] > crop_pad_size or img_gt.shape[1] > crop_pad_size:
        #     h, w = img_gt.shape[0:2]
        #     # randomly choose top and left coordinates
        #     top = random.randint(0, h - crop_pad_size)
        #     left = random.randint(0, w - crop_pad_size)
        #     img_gt = img_gt[top:top + crop_pad_size, left:left + crop_pad_size, ...]

        # ------------------------ Generate kernels (used in the first degradation) ------------------------ #
        if np.random.uniform() < self.sinc_prob:
            # this sinc filter setting is for kernels ranging from [7, 21]
            if self.kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, self.kernel_size, pad_to=False)
        else:
            kernel = self.kernel_1
        # pad kernel
        pad_size = (21 - self.kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------ Generate kernels (used in the second degradation) ------------------------ #
        if np.random.uniform() < self.sinc_prob2:
            if self.kernel_size_2 < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel2 = circular_lowpass_kernel(omega_c, self.kernel_size_2, pad_to=False)
        else:
            kernel2 = random_mixed_kernels(
                self.kernel_list2,
                self.kernel_prob2,
                self.kernel_size_2,
                self.blur_sigma2,
                self.blur_sigma2, [-math.pi, math.pi],
                self.betag_range2,
                self.betap_range2,
                noise_range=None)

        # pad kernel
        pad_size = (21 - self.kernel_size_2) // 2
        kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------------------- the final sinc kernel ------------------------------------- #
        if np.random.uniform() < self.final_sinc_prob:
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(omega_c, self.kernel_size_3, pad_to=21)
            sinc_kernel = torch.FloatTensor(sinc_kernel)
        else:
            sinc_kernel = self.pulse_tensor

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt = img2tensor([img_gt], bgr2rgb=True, float32=True)[0]
        # img_gt = img_gt.unsqueeze(4)
        # print(img_gt.shape)
        kernel = torch.FloatTensor(kernel)
        kernel2 = torch.FloatTensor(kernel2)

        return_d = {'gt': img_gt, 'kernel1': kernel, 'kernel2': kernel2, 'sinc_kernel': sinc_kernel, 'gt_path': gt_path}
        return return_d

    @torch.no_grad()
    def synthesis(self, img_paths):
        """Accept data from dataloader, and then add two-order degradations to obtain LQ images.
        """
        # training data synthesis
        self.paths = img_paths
        data = self.kernel()
        gt = data['gt'].to(self.device)
        self.gt = gt.unsqueeze(0)
        self.gt_usm = self.usm_sharpener(self.gt)
        self.kernel1 = data['kernel1'].to(self.device)
        self.kernel2 = data['kernel2'].to(self.device)
        self.sinc_kernel = data['sinc_kernel'].to(self.device)

        ori_h, ori_w = self.gt.size()[2:4]

        # ----------------------- The first degradation process ----------------------- #
        # blur
        out = filter2D(self.gt_usm, self.kernel1)

        out = F.interpolate(out, scale_factor=self.scale_1, mode=self.mode_1)
        # add noise
        gray_noise_prob = self.gray_noise_prob
        if self.gaussian_1 < self.gaussian_noise_prob:
            out = random_add_gaussian_noise_pt(
                out, sigma_range=self.noise_range, clip=True, rounds=False, gray_prob=gray_noise_prob)
        else:
            out = random_add_poisson_noise_pt(
                out,
                scale_range=self.poisson_scale_range,
                gray_prob=gray_noise_prob,
                clip=True,
                rounds=False)
        # JPEG compression
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.jpeg_range)
        out = torch.clamp(out, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
        out = self.jpeger(out, quality=jpeg_p)

        # ----------------------- The second degradation process ----------------------- #
        # blur
        if self.blur_1 < self.second_blur_prob:
            out = filter2D(out, self.kernel2)

        out = F.interpolate(
            out, size=(int(ori_h / self.scale * self.scale_2), int(ori_w / self.scale * self.scale_2)),
            mode=self.mode_2)
        # add noise
        gray_noise_prob = self.gray_noise_prob2
        if self.gaussian_2 < self.gaussian_noise_prob2:
            out = random_add_gaussian_noise_pt(
                out, sigma_range=self.noise_range2, clip=True, rounds=False, gray_prob=gray_noise_prob)
        else:
            out = random_add_poisson_noise_pt(
                out,
                scale_range=self.poisson_scale_range2,
                gray_prob=gray_noise_prob,
                clip=True,
                rounds=False)

        # JPEG compression + the final sinc filter
        # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
        # as one operation.
        # We consider two orders:
        #   1. [resize back + sinc filter] + JPEG compression
        #   2. JPEG compression + [resize back + sinc filter]
        # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
        if self.random_number < 0.5:
            # resize back + the final sinc filter
            out = F.interpolate(out, size=(ori_h // self.scale, ori_w // self.scale), mode=self.mode_3)
            out = filter2D(out, self.sinc_kernel)
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.jpeg_range2)
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)
        else:
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.jpeg_range2)
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)
            # resize back + the final sinc filter
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, size=(ori_h // self.scale, ori_w // self.scale), mode=mode)
            out = filter2D(out, self.sinc_kernel)

        # clamp and round
        self.lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.

        # # random crop
        # gt_size = self.gt_size
        # (self.gt, self.gt_usm), self.lq = paired_random_crop([self.gt, self.gt_usm], self.lq, gt_size,
        #                                                      self.scale)

        # training pair pool
        # self._dequeue_and_enqueue()
        # sharpen self.gt again, as we have changed the self.gt with self._dequeue_and_enqueue
        self.gt_usm = self.usm_sharpener(self.gt)
        lq = self.lq.contiguous()
        lq = lq.squeeze(0)
        # lq = lq.cpu().numpy()
        # #lq = 255 * (1.0 - lq)
        # lq = Image.fromarray(lq.astype(np.uint8), mode='RGB')
        return lq


if __name__ == '__main__':

    def collect_image_paths(root_dir, extensions=None):
        if extensions is None:
            extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')  # 将列表转换为元组
        image_paths = []  # 用于存储所有图片路径的列表
        folders_images = []  # 二维列表，存储每个文件夹的图片路径列表

        # 遍历根目录及其所有子目录
        for root, dirs, files in os.walk(root_dir):
            # 过滤出当前目录中的图片文件
            # 使用tuple(extensions)确保endswith()方法接收到的是字符串或字符串元组
            images = [os.path.join(root, file) for file in files if file.lower().endswith(tuple(extensions))]
            # 如果当前目录中有图片，将其路径列表添加到image_paths中
            if images:
                image_paths.extend(images)
                # 将图片路径列表添加到二维列表中，表示当前文件夹的图片
                folders_images.append(images)

        return folders_images


    # 指定你的根目录路径
    root_directory = "/home/aimll/mot/data/Dataset/mot/DanceTrack"
    # 调用函数，获取按子文件夹分组的图片路径二维列表
    grouped_image_paths = collect_image_paths(root_directory)

    for i in grouped_image_paths:
        t = RealESRGANDatadegration()
        print(i[0])
        for j in i:
            result = t.synthesis(j)
            torchvision.utils.save_image(result, j)







