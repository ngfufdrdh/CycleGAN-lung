import numpy as np
from matplotlib import pyplot as plt
from Attention_module.nam import Att
import torch
import torch.nn as nn
# from Attention_module.ECA import eca_block
#
# class ResBlock(nn.Module):
#
#     def __init__(self, in_features, norm=False):
#         super(ResBlock, self).__init__()
#
#         block = [nn.ReflectionPad2d(1),
#                  nn.Conv2d(in_features, in_features, 3),
#                  # nn.InstanceNorm2d(in_features),
#                  nn.ReLU(inplace=True),
#                  nn.ReflectionPad2d(1),
#                  nn.Conv2d(in_features, in_features, 3),
#                  # nn.InstanceNorm2d(in_features)
#                  ]
#
#         if norm:
#             block.insert(2, nn.InstanceNorm2d(in_features))
#             block.insert(6, nn.InstanceNorm2d(in_features))
#
#         self.model = nn.Sequential(*block)
#
#     def forward(self, x):
#         return x + self.model(x)
#
#
# class Attn(nn.Module):
#     def __init__(self, input_nc=1):
#         super(Attn, self).__init__()
#
#         self.model01 = nn.Sequential(
#                  nn.Conv2d(input_nc, 32, 7, stride=1, padding=3),
#                  nn.InstanceNorm2d(32),
#                  nn.ReLU(inplace=True))
#
#         self.model02 = nn.Sequential(
#                   nn.Conv2d(32, 64, 3, stride=2, padding=1),
#                   nn.InstanceNorm2d(64),
#                   nn.ReLU(inplace=True))
#         self.nam = Att(64)
#         # self.eca = eca_block(64)
#         # self.model03 = nn.Sequential(ResBlock(64, norm=True))
#
#         self.model04 = nn.Sequential(nn.UpsamplingNearest2d(scale_factor=2))
#
#         # self.model05 = nn.Sequential(
#         #           nn.Conv2d(64, 64, 3, stride=1, padding=1),
#         #           nn.InstanceNorm2d(64),
#         #           nn.ReLU(inplace=True))
#
#
#         # model += [nn.UpsamplingNearest2d(scale_factor=2)]
#
#         self.model06 = nn.Sequential(nn.Conv2d(64, 32, 3, stride=1, padding=1),
#                   nn.InstanceNorm2d(32),
#                   nn.ReLU(inplace=True))
#
#         self.model07 = nn.Sequential(nn.Conv2d(32, 1, 7, stride=1, padding=3),
#                   nn.Sigmoid())
#
#
#     def forward(self, x):
#
#         out = self.model01(x)           # [1, 32, 320, 320]
#         out = self.model02(out)         # [1, 64, 160, 160]
#         out = self.nam(out)             # [1, 64, 160, 160]
#         #out = self.model03(out)         # [1, 64, 160, 160]
#         out = self.model04(out)         # [1, 64, 320, 320]
#         #out = self.model05(out)         # [1, 64, 320, 320]
#         out = self.model06(out)         # [1, 32, 320, 320]
#         out = self.model07(out)         # [1, 1, 320, 320]
#         return out
#
#
# if __name__ == '__main__':
#     # loss = torch.nn.L1Loss().cuda()
#     # ATT = Attn(1).cuda()
#     # target = torch.randn(1, 1, 32, 32).cuda()
#     # optimizer = torch.optim.Adam(ATT.parameters(), lr=0.01, betas=(0.5, 0.99))
#     # # print(out)
#     # for i in range(1000):
#     #     x = torch.randn(1, 1, 32, 32).cuda()
#     #     out = ATT(x)
#     #     totao_loss = loss(out, target)
#     #
#     #     optimizer.zero_grad()
#     #     totao_loss.backward()
#     #     optimizer.step()
#     #     print(totao_loss.item())
#
#     x = np.load(r'F:\Data\2D_CycleGAN\T_data_each_picture_cut_1_Mask\513501 yuetin.npy')
#     num_f = np.sum(x)
#     x = torch.from_numpy(x).unsqueeze(0)
#     x = x.unsqueeze(0)
#     ATT = Attn(1)
#     out = ATT(x)
#     out = out.squeeze(0)
#     out = out.squeeze(0).detach().numpy()
#     # out[out <= 0.5] = 0
#     # out[out > 0.5] = 1
#     num = np.sum(out)
#     print(num, num_f)
#     plt.imshow(out, cmap='Greys_r')
#     plt.show()
#
#     # print(out.shape)







class ResBlock(nn.Module):

    def __init__(self, in_features, norm=False):
        super(ResBlock, self).__init__()

        block = [nn.ReflectionPad2d(1),
                 nn.Conv2d(in_features, in_features, 3),
                 # nn.InstanceNorm2d(in_features),
                 nn.ReLU(inplace=True),
                 nn.ReflectionPad2d(1),
                 nn.Conv2d(in_features, in_features, 3),
                 # nn.InstanceNorm2d(in_features)
                 ]

        if norm:
            block.insert(2, nn.InstanceNorm2d(in_features))
            block.insert(6, nn.InstanceNorm2d(in_features))

        self.model = nn.Sequential(*block)

    def forward(self, x):
        return x + self.model(x)


class Attn(nn.Module):
    def __init__(self, input_nc=1):
        super(Attn, self).__init__()

        self.model01 = nn.Sequential(
                 nn.Conv2d(input_nc, 32, 7, stride=1, padding=3),
                 nn.InstanceNorm2d(32),
                 nn.ReLU(inplace=True))

        self.model02 = nn.Sequential(
                  nn.Conv2d(32, 64, 3, stride=2, padding=1),
                  nn.InstanceNorm2d(64),
                  nn.ReLU(inplace=True))
        self.nam = Att(64)
        self.model03 = nn.Sequential(ResBlock(64, norm=True))

        self.model04 = nn.Sequential(nn.UpsamplingNearest2d(scale_factor=2))

        self.model05 = nn.Sequential(
                  nn.Conv2d(64, 64, 3, stride=1, padding=1),
                  nn.InstanceNorm2d(64),
                  nn.ReLU(inplace=True))


        # model += [nn.UpsamplingNearest2d(scale_factor=2)]

        self.model06 = nn.Sequential(nn.Conv2d(64, 32, 3, stride=1, padding=1),
                  nn.InstanceNorm2d(32),
                  nn.ReLU(inplace=True))

        self.model07 = nn.Sequential(nn.Conv2d(32, 1, 7, stride=1, padding=3),
                  nn.Sigmoid())


    def forward(self, x):

        out = self.model01(x)    #[32, 320, 320]
        out = self.model02(out)  #[64, 160, 160]
        out = self.model03(out)  #[64, 160, 160]
        out = self.nam(out)      #[64, 160, 160]
        out = self.model04(out)  #[64, 320, 320]
        out = self.model05(out)  #[64, 320, 320]
        out = self.model06(out)  #[32, 320, 320]
        out = self.model07(out)  #[1, 320, 320]
        # out[out <= 0.5] = 0
        # out[out > 0.5] = 1
        return out



