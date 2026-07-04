import torch
import torch.nn as nn
import torch.nn.functional as F

class ODCA(nn.Module):
    def __init__(self, channel=64, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((1, None))
        self.pool_w = nn.AdaptiveAvgPool2d((None, 1))

        mip = max(8, channel // reduction)

        self.conv1 = nn.Sequential(
            nn.Conv2d(channel, mip, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(mip),
            nn.Hardswish()
        )

        self.conv_h = nn.Conv2d(mip, channel, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, channel, kernel_size=1, stride=1, padding=0)

        # 简单的对角注意力
        self.conv_d = nn.Conv2d(mip, channel, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        b, c, h, w = x.size()

        # 垂直和水平池化
        x_h = self.pool_h(x)  # (b, c, 1, w)
        x_w = self.pool_w(x)  # (b, c, h, 1)

        # 降维处理
        x_h_reduced = self.conv1(x_h)  # (b, mip, 1, w)
        x_w_reduced = self.conv1(x_w)  # (b, mip, h, 1)

        # 生成注意力图
        a_h = torch.sigmoid(self.conv_h(x_h_reduced))  # (b, c, 1, w)
        a_w = torch.sigmoid(self.conv_w(x_w_reduced))  # (b, c, h, 1)

        # 简单对角注意力（通过特征融合）
        # 将x_h_reduced和x_w_reduced扩展到相同的空间维度
        x_h_expanded = F.interpolate(x_h_reduced, size=(h, w), mode='bilinear', align_corners=False)  # (b, mip, h, w)
        x_w_expanded = F.interpolate(x_w_reduced, size=(h, w), mode='bilinear', align_corners=False)  # (b, mip, h, w)

        # 融合并降维
        x_diag = (x_h_expanded + x_w_expanded) / 2  # (b, mip, h, w)
        # 对融合后的特征进行全局平均池化以获得(b, mip, 1, 1)的表示
        x_diag_pooled = torch.mean(x_diag, dim=(2, 3), keepdim=True)  # (b, mip, 1, 1)

        # 生成对角注意力图并扩展到空间维度
        a_diag = torch.sigmoid(self.conv_d(x_diag_pooled))  # (b, c, 1, 1)
        a_diag = a_diag.expand(-1, -1, h, w)  # (b, c, h, w)

        # 扩展a_h和a_w到完整空间维度
        a_h_expanded = F.interpolate(a_h, size=(h, w), mode='bilinear', align_corners=False)  # (b, c, h, w)
        a_w_expanded = F.interpolate(a_w, size=(h, w), mode='bilinear', align_corners=False)  # (b, c, h, w)

        # 融合注意力（相乘方式）
        out = identity * a_w_expanded * a_h_expanded * a_diag

        return out