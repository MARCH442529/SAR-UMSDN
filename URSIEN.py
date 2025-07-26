import torch
import torch.nn as nn
from loss import LossFunction
import functools
import torch
import torch.nn as nn
from loss import LossFunction
import functools
import cv2

import numpy as np
import matplotlib.pyplot as plt
import imageio
import scipy, scipy.misc, scipy.signal
import cv2
import sys
from torchvision.transforms import ToPILImage


def image_toIP(img):
    # 假设 tensor 是形状为 [1, 3, 640, 640] 的张量

    # 去掉批量维度（第 0 维），形状变为 [3, 640, 640]
    tensor_squeezed = img.squeeze(0)

    # 使用 ToPILImage 转换为 PIL 图像
    pil_image = ToPILImage()(tensor_squeezed)

    # 如果需要转换为 NumPy 数组，可以进一步使用 numpy() 方法
    array = np.array(pil_image)
    return array


def image_totTS(img):
    # 转换为 PyTorch 张量，形状为 [640, 640, 3]
    tensor = torch.from_numpy(img)

    # 调整维度顺序为 [3, 640, 640]（CHW 格式）
    tensor = tensor.permute(2, 0, 1)

    # 添加批量维度，形状变为 [1, 3, 640, 640]
    tensor = tensor.unsqueeze(0)
    return tensor


def computeTextureWeights(fin, sigma, sharpness):
    #计算水平和垂直方向的梯度
    # np.diff(fin, n=1, axis=0) 计算输入图像 fin 在垂直方向（行方向）的一阶差分。
    #np.vstack((..., fin[0, :] - fin[-1, :])) 将差分结果与首尾行的差值进行垂直堆叠，以处理图像边界。
    #类似地，水平方向的梯度计算使用 np.diff(fin, n=1, axis=1)，然后进行转置操作以对齐维度。
    dt0_v = np.vstack((np.diff(fin, n=1, axis=0), fin[0, :] - fin[-1, :]))
    dt0_h = np.vstack((np.diff(fin, n=1, axis=1).conj().T, fin[:, 0].conj().T - fin[:, -1].conj().T)).conj().T
    #高斯模糊：
    # 对梯度图进行二维卷积。
    # np.ones((1, sigma)) 和np.ones((sigma, 1))分别用于创建水平和垂直方向的高斯核。
    # mode = 'same'
    # 确保输出图像的尺寸与输入图像相同。
    gauker_h = scipy.signal.convolve2d(dt0_h, np.ones((1, sigma)), mode='same')
    gauker_v = scipy.signal.convolve2d(dt0_v, np.ones((sigma, 1)), mode='same')
    #计算权重矩阵：权重矩阵通过高斯模糊后的梯度图和原始梯度图的绝对值相乘，并加上锐化参数 sharpness 来计算。
    # 权重矩阵的值反映了图像中每个像素的纹理强度，用于后续的图像平滑处理。
    W_h = 1 / (np.abs(gauker_h) * np.abs(dt0_h) + sharpness)
    W_v = 1 / (np.abs(gauker_v) * np.abs(dt0_v) + sharpness)

    return W_h, W_v


def solveLinearEquation(IN, wx, wy, lamda):
    #IN.shape求IN的形状
    [r, c] = IN.shape
    k = r * c
    #构造稀疏矩阵的元素：
    #wx.flatten('F') 和 wy.flatten('F') 将权重矩阵展平为一维数组，'F' 表示按列优先顺序展平。
    # np.roll 用于将权重矩阵循环移动一个位置，以构造相邻像素之间的关系。
    # dxa 和 dya 分别是循环移动后的权重矩阵的展平版本。

    dx = -lamda * wx.flatten('F')
    dy = -lamda * wy.flatten('F')
    tempx = np.roll(wx, 1, axis=1)
    tempy = np.roll(wy, 1, axis=0)
    dxa = -lamda * tempx.flatten('F')
    dya = -lamda * tempy.flatten('F')
    #处理边界条件：
    #wx[:, -1]表示最后一列
    #处理图像边界处的权重矩阵，将边界处的权重设置为零，避免边界效应。
    tmp = wx[:, -1]
    tempx = np.concatenate((tmp[:, None], np.zeros((r, c - 1))), axis=1)
    tmp = wy[-1, :]
    tempy = np.concatenate((tmp[None, :], np.zeros((r - 1, c))), axis=0)
    dxd1 = -lamda * tempx.flatten('F')
    dyd1 = -lamda * tempy.flatten('F')
    #更新权重矩阵：
    wx[:, -1] = 0
    wy[-1, :] = 0
    dxd2 = -lamda * wx.flatten('F')
    dyd2 = -lamda * wy.flatten('F')
    # 构造稀疏矩阵：
   # 使用scipy.sparse.spdiags构造稀疏矩阵Ax和Ay。D是一个对角矩阵，包含稀疏矩阵的对角线元素 A是最终的稀疏矩阵，用于求解线性方程组。
    Ax = scipy.sparse.spdiags(np.concatenate((dxd1[:, None], dxd2[:, None]), axis=1).T, np.array([-k + r, -r]), k, k)
    Ay = scipy.sparse.spdiags(np.concatenate((dyd1[None, :], dyd2[None, :]), axis=0), np.array([-r + 1, -1]), k, k)
    D = 1 - (dx + dy + dxa + dya)
    A = ((Ax + Ay) + (Ax + Ay).conj().T + scipy.sparse.spdiags(D, 0, k, k)).T
    #将输入图像展平为一维数组，并使用 scipy.sparse.linalg.spsolve 求解线性方程组。将求解结果重新reshape为原始图像的形状。
    tin = IN[:, :]
    tout = scipy.sparse.linalg.spsolve(A, tin.flatten('F'))
    OUT = np.reshape(tout, (r, c), order='F')

    return OUT


def tsmooth(img, lamda=0.01, sigma=3.0, sharpness=0.001):
    I = cv2.normalize(img.astype('float64'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    x = np.copy(I)
    wx, wy = computeTextureWeights(x, sigma, sharpness)
    #这个函数用于求解线性方程组，以实现图像的平滑处理。lamda是正则化参数，用于控制平滑程度
    S = solveLinearEquation(I, wx, wy, lamda)
    return S


def rgb2gm(I):
    if (I.shape[2] == 3):
        I = cv2.normalize(I.astype('float64'), None, 0.0, 1.0, cv2.NORM_MINMAX)
        I = np.abs((I[:, :, 0] * I[:, :, 1] * I[:, :, 2])) ** (1 / 3)

    return I


def applyK(I, k, a=-0.3293, b=1.1258):
    f = lambda x: np.exp((1 - x ** a) * b)
    beta = f(k)
    gamma = k ** a
    J = (I ** gamma) * beta
    return J


def entropy(X):
    tmp = X * 255
    tmp[tmp > 255] = 255
    tmp[tmp < 0] = 0
    tmp = tmp.astype(np.uint8)
    _, counts = np.unique(tmp, return_counts=True)
    pk = np.asarray(counts)
    pk = 1.0 * pk / np.sum(pk, axis=0)
    S = -np.sum(pk * np.log2(pk), axis=0)
    return S


def maxEntropyEnhance(I, isBad, a=-0.3293, b=1.1258):

    tmp = cv2.resize(I, (50, 50), interpolation=cv2.INTER_AREA)
    tmp[tmp < 0] = 0
    tmp = tmp.real
    Y = rgb2gm(tmp)

    isBad = (isBad * 1).astype(np.uint8)
    isBad = cv2.resize(isBad, (50, 50), interpolation=cv2.INTER_CUBIC)

    isBad[isBad < 0.5] = 0
    isBad[isBad >= 0.5] = 1
    Y = Y[isBad == 1]

    if Y.size == 0:
        J = I
        return J

    f = lambda k: -entropy(applyK(Y, k))
    opt_k = scipy.optimize.fminbound(f, 1, 7)


    J = applyK(I, opt_k, a, b) - 0.01
    return J


def Enhance(img, mu=0.5, a=-0.3293, b=1.1258):
    lamda = 0.5
    sigma = 5
    #对图像进行处理，将其进行归一化，由0-255归一化到0-1
    I =  cv2.normalize(img.astype('float64'), None, 0.0, 1.0, cv2.NORM_MINMAX)

    #I的形状为W，H，C，这个操作是在C这个维度求为大值，得到的t_b形状为W，H
    t_b = np.max(I, axis=2)
    #OpenCV 的 resize 函数对图像 t_b 进行缩放cv2.resize(t_b, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)缩小一半
    t_our = cv2.resize(tsmooth(cv2.resize(t_b, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC), lamda, sigma),
                       (t_b.shape[1], t_b.shape[0]), interpolation=cv2.INTER_AREA)


    isBad = t_our < 0.5
    #过曝光的图像J
    J = maxEntropyEnhance(I, isBad)


    t = np.zeros((t_our.shape[0], t_our.shape[1], I.shape[2]))
    for i in range(I.shape[2]):
        t[:, :, i] = t_our
    #W为欠曝光的图像
    W = t ** mu
    #原图和W（欠曝光）相乘
    I2 = I * W
    # 过曝光和（1-W）相乘
    J2 = J * (1 - W)

    result = I2 + J2
    result[result > 1] = 1
    result[result < 0] = 0
    return result


class Embed(nn.Module):
    def __init__(self, layers, channels):
        super(Embed, self).__init__()

        kernel_size = 3
        dilation = 1
        padding = int((kernel_size - 1) / 2) * dilation
        #使用k=3,s=1,p=1的卷积，操作后图片形状不变
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.ReLU()
        )
        # 使用k=3,s=1,p=1的卷积，操作后图片形状不变
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        # blocks由n个k=3,s=1,p=1的卷积组成，操作后图片形状不变
        self.blocks = nn.ModuleList()
        for i in range(layers):
            self.blocks.append(self.conv)

        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, input):
        fea = self.in_conv(input)#[1,3,640,960]->[1,3,640,960]
        for conv in self.blocks:
            fea = fea + conv(fea)
        fea = self.out_conv(fea)

        illu = fea + input#[1,3,640,960]->[1,3,640,960]
        #torch.clamp(illu, 0.0001, 1)将值限制在【0，1】之间
        illu = torch.clamp(illu, 0.0001, 1)

        return illu


class CalibrateNetwork(nn.Module):
    def __init__(self, layers, channels):
        super(CalibrateNetwork, self).__init__()
        kernel_size = 3
        dilation = 1
        padding = int((kernel_size - 1) / 2) * dilation
        self.layers = layers

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        self.blocks = nn.ModuleList()
        for i in range(layers):
            self.blocks.append(self.convs)

        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, input):
        fea = self.in_conv(input)
        for conv in self.blocks:
            fea = fea + conv(fea)

        fea = self.out_conv(fea)
        delta = input - fea

        return delta



class Network(nn.Module):

    def __init__(self, stage=3):
        super(Network, self).__init__()
        self.stage = stage
        self.enhance = Embed(layers=1, channels=3)
        self.calibrate = CalibrateNetwork(layers=3, channels=16)
        self._criterion = LossFunction()

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()

        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1., 0.02)

    def forward(self, input):

        ilist, rlist, inlist, attlist = [], [], [], []

        input_op=input

        for i in range(self.stage):
            inlist.append(input_op)
            i = self.enhance(input_op)# #[1,3,640,960]->[1,3,640,960]
            r = input / i
            r = torch.clamp(r, 0, 1)
            r = image_toIP(r)
            r = Enhance(r)
            r = image_totTS(r).to('cuda').to(torch.float32)
            att = self.calibrate(r)
            input_op = input + att
            ilist.append(i)
            rlist.append(r)
            attlist.append(torch.abs(att))

        return ilist, rlist, inlist, attlist

    def _loss(self, input):
        i_list, en_list, in_list, _ = self(input)
        loss = 0
        for i in range(self.stage):
            loss += self._criterion(in_list[i], i_list[i])
        return loss
    #损失函数由两部分组成，一个是L2范数



class Finetunemodel(nn.Module):

    def __init__(self, weights):
        super(Finetunemodel, self).__init__()
        self.enhance = Embed(layers=1, channels=3)
        self.calibrate = CalibrateNetwork(layers=3, channels=16)
        self._criterion = LossFunction()

        base_weights = torch.load(weights)
        pretrained_dict = base_weights
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()

        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1., 0.02)

    def forward(self, input):

        i = self.enhance(input)
        r = input / i
        r = torch.clamp(r, 0, 1)
        input_op=image_toIP(input)
        input_op=Enhance(input_op)
        input_op=image_totTS(input_op).to('cuda').to(torch.float32)
        # att = self.calibrate(r)
        # input_op = input - att
        return i, input_op


    def _loss(self, input):
        i, r = self(input)
        loss = self._criterion(input, i)
        return loss

