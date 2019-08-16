import os
import numpy as np
from PIL import Image
import torch
import torchvision
from torch.utils.data import Dataset
import cv2


class Network(torch.nn.Module):
    """IM-Net implementation"""

    def __init__(self):
        super(Network, self).__init__()

        def feature_extractor_block(int_input, int_output):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=int_input, out_channels=16, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=16, out_channels=int_output, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=int_output, out_channels=int_output, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False)
            )

        def basic_conv(int_input, int_output):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=int_input, out_channels=int_output, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=int_output, out_channels=int_output, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=int_output, out_channels=int_output, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False)
            )

        def up_sample(int_input, int_output):
            return torch.nn.Sequential(
                torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                torch.nn.Conv2d(in_channels=int_input, out_channels=int_output, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False)
            )

        def soft_max(int_input, int_output):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=int_input, out_channels=int_output, kernel_size=3, stride=1, padding=1),
                torch.nn.Softmax(dim=1)
            )

        def horizontal__sep_conv(int_input, int_output):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=int_input, out_channels=int_input, kernel_size=(1, 3), stride=1,
                                padding=(0, 1), groups=int_input),
                torch.nn.Conv2d(in_channels=int_input, out_channels=int_output, kernel_size=1),
            )

        def vertical__sep_conv(int_input, int_output):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=int_input, out_channels=int_input, kernel_size=(3, 1), stride=1,
                                padding=(1, 0), groups=int_input),
                torch.nn.Conv2d(in_channels=int_input, out_channels=int_output, kernel_size=1),
            )

            # FEATURE EXTRACTION (Siamese for different resolution input)

        self.feature_extract = feature_extractor_block(3, 25)
        self.pool = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        # ENCODER-DECODER
        # Hi-resolution
        self.enc_hi_1 = basic_conv(50, 64)
        self.enc_hi_2 = basic_conv(64, 80)
        self.enc_hi_3 = basic_conv(80, 128)
        self.enc_hi_4 = basic_conv(128, 200)
        self.enc_hi_5 = basic_conv(200, 128)
        self.upsample_hi_5 = up_sample(128, 128)
        self.upsample_hi_4 = up_sample(80, 80)
        self.upsample_hi_3 = up_sample(64, 64)
        self.dec_hi_5 = basic_conv(128, 80)
        self.dec_hi_4 = basic_conv(80, 64)
        self.dec_hi_3 = basic_conv(64, 50)

        # Middle-resolution
        self.enc_mi_1 = basic_conv(50, 64)
        self.enc_mi_2 = basic_conv(64, 80)
        self.enc_mi_3 = basic_conv(80, 64)
        self.upsample_mi_3 = up_sample(64, 64)
        self.upsample_mi_2 = up_sample(64, 64)
        self.dec_mi_3 = basic_conv(64, 64)
        self.dec_mi_2 = basic_conv(64, 50)

        # Low-resolution
        self.enc_lo_1 = basic_conv(50, 64)
        self.enc_lo_2 = basic_conv(64, 80)
        self.enc_lo_3 = basic_conv(80, 64)
        self.upsample_lo_3 = up_sample(64, 64)
        self.upsample_lo_2 = up_sample(64, 64)
        self.dec_lo_3 = basic_conv(64, 64)
        self.dec_lo_2 = basic_conv(64, 50)

        # MERGING BRANCHES LAYERS
        self.merge_conv = basic_conv(150, 50)
        self.merge = soft_max(50, 3)

        # HORIZONTAL MOTION
        self.horizontal_conv_p = horizontal__sep_conv(50, 50)
        self.horizontal_softmax_p = soft_max(50, 25)

        self.horizontal_conv_n = horizontal__sep_conv(50, 50)
        self.horizontal_softmax_n = soft_max(50, 25)

        # VERTICAL MOTION
        self.vertical_conv_p = vertical__sep_conv(50, 50)
        self.vertical_softmax_p = soft_max(50, 25)

        self.vertical_conv_n = vertical__sep_conv(50, 50)
        self.vertical_softmax_n = soft_max(50, 25)

        # OCCLUSION MAP
        self.occlusion_conv = basic_conv(50, 50)
        self.occlusion_softmax = soft_max(50, 2)

        # For Hi-resolution warp map upsample
        self.upscale_8x = torch.nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

    @staticmethod
    def warp(img_tensor, horizontal_flow, vertical_flow, device):
        w = img_tensor.size(3)
        h = img_tensor.size(2)
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        grid_x = torch.tensor(grid_x, requires_grad=False, device=device)
        grid_y = torch.tensor(grid_y, requires_grad=False, device=device)
        x = grid_x.unsqueeze(0).expand_as(horizontal_flow).float() + horizontal_flow
        y = grid_y.unsqueeze(0).expand_as(vertical_flow).float() + vertical_flow
        # range -1 to 1
        x = 2 * (x / w - 0.5)
        y = 2 * (y / h - 0.5)
        grid = torch.stack((x, y), dim=3)
        img_out = torch.nn.functional.grid_sample(img_tensor, grid)
        return img_out

    def forward(self, prev_tensor_hi, next_tensor_hi, prev_tensor_mi, next_tensor_mi,
                prev_tensor_lo, next_tensor_lo, device=torch.device('cuda')):
        # FEATURE EXTRACTION
        # Hi-resolution branch
        x_prev_hi = self.pool(self.feature_extract(self.pool(prev_tensor_hi)))
        x_next_hi = self.pool(self.feature_extract(self.pool(next_tensor_hi)))

        # Middle-resolution branch
        x_prev_mi = self.pool(self.feature_extract(self.pool(prev_tensor_mi)))
        x_next_mi = self.pool(self.feature_extract(self.pool(next_tensor_mi)))

        # Low-resolution branch
        x_prev_lo = self.pool(self.feature_extract(self.pool(prev_tensor_lo)))
        x_next_lo = self.pool(self.feature_extract(self.pool(next_tensor_lo)))

        # ENCODER-DECODER
        # Hi-resolution branch
        x_hi = torch.cat([x_prev_hi, x_next_hi], dim=1)
        x_enc_hi_1 = self.enc_hi_1(x_hi)
        x_enc_hi_1_avg = self.pool(x_enc_hi_1)
        x_enc_hi_2 = self.enc_hi_2(x_enc_hi_1_avg)
        x_enc_hi_2_avg = self.pool(x_enc_hi_2)
        x_enc_hi_3 = self.enc_hi_3(x_enc_hi_2_avg)
        x_enc_hi_3_avg = self.pool(x_enc_hi_3)
        x_enc_hi_4 = self.enc_hi_4(x_enc_hi_3_avg)
        x_enc_hi_4_avg = self.pool(x_enc_hi_4)
        x_enc_hi_5 = self.enc_hi_5(x_enc_hi_4_avg)

        x_dec_hi_4 = self.dec_hi_5(self.upsample_hi_5(x_enc_hi_5) + x_enc_hi_3_avg)
        x_dec_hi_3 = self.dec_hi_4(self.upsample_hi_4(x_dec_hi_4) + x_enc_hi_2_avg)
        x_dec_hi_2 = self.dec_hi_3(self.upsample_hi_3(x_dec_hi_3) + x_enc_hi_1_avg)

        # Middle-resolution branch
        x_mi = torch.cat([x_prev_mi, x_next_mi], dim=1)
        x_enc_mi_1 = self.enc_mi_1(x_mi)
        x_enc_mi_1_avg = self.pool(x_enc_mi_1)
        x_enc_mi_2 = self.enc_mi_2(x_enc_mi_1_avg)
        x_enc_mi_2_avg = self.pool(x_enc_mi_2)
        x_enc_mi_3 = self.enc_mi_3(x_enc_mi_2_avg)

        x_dec_mi_2 = self.dec_mi_3(self.upsample_mi_3(x_enc_mi_3) + x_enc_mi_1_avg)
        x_dec_mi_1 = self.dec_mi_2(self.upsample_mi_2(x_dec_mi_2))

        # Low-resolution branch
        x_lo = torch.cat([x_prev_lo, x_next_lo], dim=1)
        x_enc_lo_1 = self.enc_lo_1(x_lo)
        x_enc_lo_1_avg = self.pool(x_enc_lo_1)
        x_enc_lo_2 = self.enc_lo_2(x_enc_lo_1_avg)
        x_enc_lo_3 = self.enc_lo_3(x_enc_lo_2)

        x_dec_lo_2 = self.dec_lo_3(self.upsample_lo_3(x_enc_lo_3 + x_enc_lo_1_avg))
        x_dec_lo_1 = self.dec_lo_2(self.upsample_lo_2(x_dec_lo_2))

        # MERGING BRANCHES LAYERS
        x_merge = torch.cat([x_dec_hi_2, x_dec_mi_1, x_dec_lo_1], dim=1)
        x_merge = self.merge_conv(x_merge)
        x_merge = self.merge(x_merge)

        x_merge = (x_dec_hi_2 * x_merge[:, 0, :, :].unsqueeze(1)) + (x_dec_mi_1 * x_merge[:, 1, :, :].unsqueeze(1)) + (
                    x_dec_lo_1 * x_merge[:, 2, :, :].unsqueeze(1))

        # CENTER-OF-MASS FILTER
        patch = list()
        for i in range(-12, 13):
            patch.append(torch.ones((x_dec_hi_2.size(2), x_dec_hi_2.size(3))) * i)
        support_layer = torch.stack(patch).to(device)
        support_layer.requires_grad = False

        # HORIZONTAL MOTION
        x_horizontal_p = self.horizontal_conv_p(x_merge)
        x_horizontal_p = self.horizontal_softmax_p(x_horizontal_p)
        x_horizontal_p = (x_horizontal_p * support_layer).sum(dim=1)  # calculate horizontal center-of-mass

        x_horizontal_n = self.horizontal_conv_n(x_merge)
        x_horizontal_n = self.horizontal_softmax_n(x_horizontal_n)
        x_horizontal_n = (x_horizontal_n * support_layer).sum(dim=1)  # calculate horizontal center-of-mass

        # VERTICAL MOTION
        x_vertical_p = self.vertical_conv_p(x_merge)
        x_vertical_p = self.vertical_softmax_p(x_vertical_p)
        x_vertical_p = (x_vertical_p * support_layer).sum(dim=1)  # calculate vertical center-of-mass

        x_vertical_n = self.vertical_conv_n(x_merge)
        x_vertical_n = self.vertical_softmax_n(x_vertical_n)
        x_vertical_n = (x_vertical_n * support_layer).sum(dim=1)  # calculate vertical center-of-mass

        # OCCLUSION MAP
        x_occlusion = self.occlusion_conv(x_merge)
        x_occlusion = self.occlusion_softmax(x_occlusion)
        occlusion_prev = x_occlusion[:, 0, :, :].unsqueeze(1)
        occlusion_next = x_occlusion[:, 1, :, :].unsqueeze(1)

        # GENERATE MIDDLE FRAME
        # low resolution
        i_warp_low_0 = self.warp(self.pool(prev_tensor_lo), x_horizontal_p, x_vertical_p, device)
        i_warp_low_1 = self.warp(self.pool(next_tensor_lo), x_horizontal_n, x_vertical_n, device)
        mid_frame_low = i_warp_low_0 * occlusion_prev + i_warp_low_1 * occlusion_next

        # hi resolution
        i_warp_hi_0 = self.warp(prev_tensor_hi, self.upscale_8x(x_horizontal_p.unsqueeze(1)).squeeze(1),
                                self.upscale_8x(x_vertical_p.unsqueeze(1)).squeeze(1), device)
        i_warp_hi_1 = self.warp(next_tensor_hi, self.upscale_8x(x_horizontal_n.unsqueeze(1)).squeeze(1),
                                self.upscale_8x(x_vertical_n.unsqueeze(1)).squeeze(1), device)
        mid_frame_hi = i_warp_hi_0 * self.upscale_8x(occlusion_prev) + i_warp_hi_1 * self.upscale_8x(occlusion_next)

        return mid_frame_low, mid_frame_hi, (i_warp_low_0 , i_warp_low_1 ,
                                             x_horizontal_p, x_horizontal_n, x_vertical_p, x_vertical_n,
                                             self.upscale_8x(occlusion_prev))


class TripletDataset(Dataset):
    """for VIMEO dataset."""

    def __init__(self, root_dir, transform=None):

        self.root_dir = root_dir
        self.transform = transform
        self.len_ = 0
        self.path_list = []
        for dir_1 in os.listdir(self.root_dir):
            for dir_2 in os.listdir(self.root_dir + f'/{dir_1}'):
                self.len_ += 1
                self.path_list.append(f'/{dir_1}/{dir_2}/')

    def __len__(self):
        return self.len_

    def __getitem__(self, idx):
        img_path = self.root_dir + self.path_list[idx]

        image_prev = np.array(Image.open(img_path + 'im1.png')) / 255
        image_mid = np.array(Image.open(img_path + 'im2.png')) / 255
        image_next = np.array(Image.open(img_path + 'im3.png')) / 255

        w = image_prev.shape[0]
        h = image_prev.shape[1]

        patch_img_hi_resolution = (image_prev, image_next)
        patch_img_mid_resolution = (cv2.resize(image_prev, (h // 2, w // 2)),
                                    cv2.resize(image_next, (h // 2, w // 2)))
        patch_img_low_resolution = (cv2.resize(image_prev, (h // 4, w // 4)),
                                    cv2.resize(image_next, (h // 4, w // 4)))
        patch_mid_img = (image_mid, cv2.resize(image_mid, (h // 8, w // 8)))  # target middle images

        sample = {'img_hi_res': patch_img_hi_resolution, 'img_mid_res': patch_img_mid_resolution,
                  'img_low_res': patch_img_low_resolution, 'mid_img': patch_mid_img}
        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):
    def __call__(self, sample):
        for key, value in sample.items():
            sample[key] = [torch.Tensor(image.transpose((2, 0, 1))) for image in value]
        return sample


class ImNetLoss(object):
    def __init__(self, device):
        self.device = device
        self.L1_loss = torch.nn.SmoothL1Loss()
        self.MSE_Loss = torch.nn.MSELoss()
        vgg16 = torchvision.models.vgg16()
        self.vgg16_conv_4_3 = torch.nn.Sequential(*list(vgg16.children())[0][:22])
        self.vgg16_conv_4_3.to(device)
        for param in self.vgg16_conv_4_3.parameters():
            param.requires_grad = False

    def loss_calculation(self, low_predicted, hi_predicted, for_loss, batch):
        l1_loss_low = self.L1_loss(low_predicted, batch['mid_img'][1].to(self.device))
        l1_loss_hi = self.L1_loss(hi_predicted, batch['mid_img'][0].to(self.device))
        prcp_loss = self.MSE_Loss(self.vgg16_conv_4_3(hi_predicted),
                                  self.vgg16_conv_4_3(batch['mid_img'][0].to(self.device)))  # add from myself
        # TODO: add WARP loss, REG loss, SIM loss

        return l1_loss_low + 1.5 * l1_loss_hi + 10e2 * prcp_loss


class Utils(object):
    @staticmethod
    def batch_to_model(batch, device):
        return (batch['img_hi_res'][0].to(device), batch['img_hi_res'][1].to(device),
                batch['img_mid_res'][0].to(device), batch['img_mid_res'][1].to(device),
                batch['img_low_res'][0].to(device), batch['img_low_res'][1].to(device), device)
