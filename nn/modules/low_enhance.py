import torch
import torch.nn as nn


# ---- 模型部分：low_light_enhance ----
class Low_enhance_net(nn.Module):
    def __init__(self, in_channels):
        super(Low_enhance_net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 4, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, padding=1, stride=1)
        self.conv5 = nn.Conv2d(8, 8, kernel_size=3, padding=1, stride=1)


        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.leaky_relu((self.conv1(x)))
        x = self.leaky_relu((self.conv2(x)))
        x = self.leaky_relu((self.conv5(x)))

        r1, r2, r3, r4, r5, r6, r7, r8 = torch.split(x, 1, dim=1)
        return [r1, r2, r3, r4, r5, r6, r7, r8]





def low_enhance_feature(low_light_image, r):
    # 遍历 r 中的每个元素并逐步增强
    for r_it in r:
        # 将 r_it 通过 sigmoid 压缩到 0 到 1 的范围内，防止其值过大或过小
        r_it = torch.sigmoid(r_it)

        # 增强操作，添加一个很小的常数 1e-6 来提高数值稳定性，避免零除或下溢
        low_light_image = low_light_image + r_it * (torch.pow(low_light_image, 2) - low_light_image + 1e-6)

        # 对每次迭代的结果进行裁剪，避免值过大或过小
        low_light_image = torch.clamp(low_light_image, min=0.0, max=1.0)

    # 将原始图像加回结果，作为增强后的最终输出
    return low_light_image

