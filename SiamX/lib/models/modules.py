import torch
import torch.nn as nn
import torch.nn.functional as F
from .dcn import DeformConv, DeformConvPack


# ----------------------------------------
# neck module
# ----------------------------------------
class AdjustSingleLayer(nn.Module):
    def __init__(self, in_channels, out_channels, center_size=7):
        super(AdjustSingleLayer, self).__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.center_size = center_size

    def forward(self, feature):
        feature = self.downsample(feature)
        if feature.size(3) < 20:
            l = (feature.size(3) - self.center_size) // 2
            r = l + self.center_size
            feature = feature[:, :, l:r, l:r]
        return feature


class AdjustMultiLayer(nn.Module):
    def __init__(self, in_channels, out_channels, center_size=7):
        super(AdjustMultiLayer, self).__init__()
        self.num = len(out_channels)
        for i in range(self.num):
            self.add_module('downsample' + str(i),
                            AdjustSingleLayer(in_channels[i], out_channels[i], center_size))

    def forward(self, features):
        out = []
        for i in range(self.num):
            adj_layer = getattr(self, 'downsample' + str(i))
            out.append(adj_layer(features[i]))
        return out


# ----------------------------------------
# X module
# ----------------------------------------
class FeatureEncoder(nn.Module):
    """
    encode backbone feature
    """

    def __init__(self, in_channels, out_channels):
        super(FeatureEncoder, self).__init__()

        self.encoder_search = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.encoder_kernel = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, search, kernel):
        xf = self.encoder_search(search)
        zf = self.encoder_kernel(kernel)
        return xf, zf


def xcorr_depthwise(x, kernel):
    """
    depthwise cross correlation
    """
    batch = kernel.size(0)
    channel = kernel.size(1)
    x = x.view(1, batch * channel, x.size(2), x.size(3))
    kernel = kernel.view(batch * channel, 1, kernel.size(2), kernel.size(3))
    out = F.conv2d(x, kernel, groups=batch * channel)
    out = out.view(batch, channel, out.size(2), out.size(3))
    return out


class FeatureCorrelation(nn.Module):
    """
    compute feature correlation
    """

    def __init__(self, in_channels, out_channels):
        super(FeatureCorrelation, self).__init__()
        # encode backbone
        self.cls_encoder = FeatureEncoder(in_channels=in_channels, out_channels=out_channels)
        self.reg_encoder = FeatureEncoder(in_channels=in_channels, out_channels=out_channels)

    def forward(self, search, template):
        # encode first
        cls_xf, cls_zf = self.cls_encoder(search, template)
        reg_xf, reg_zf = self.reg_encoder(search, template)

        # cls and reg depthwise cross correlation
        cls_feature = xcorr_depthwise(cls_xf, cls_zf)
        reg_feature = xcorr_depthwise(reg_xf, reg_zf)

        return cls_feature, reg_feature


# ----------------------------------------
# regression and classification heads
# ----------------------------------------
class RegClsModuleX(nn.Module):

    def __init__(self, in_channels, towernum=1):
        super(RegClsModuleX, self).__init__()

        self.num = len(in_channels)
        for i in range(self.num):
            self.add_module('feature_correlation' + str(i),
                            FeatureCorrelation(in_channels=in_channels[i], out_channels=in_channels[i]))
        # self.cls_weight = nn.Parameter(torch.ones(self.num, in_channels[-1]))
        # self.reg_weight = nn.Parameter(torch.ones(self.num, in_channels[-1]))
        self.cls_weight = nn.Parameter(torch.ones(self.num))
        self.reg_weight = nn.Parameter(torch.ones(self.num))

        reg_tower = []
        cls_tower = []
        # cls head
        for i in range(towernum):
            if i == 0:
                cls_tower.append(nn.Conv2d(in_channels[-1], in_channels[-1], kernel_size=3, stride=1, padding=1))
            else:
                cls_tower.append(nn.Conv2d(in_channels[-1], in_channels[-1], kernel_size=3, stride=1, padding=1))

            cls_tower.append(nn.BatchNorm2d(in_channels[-1]))
            cls_tower.append(nn.ReLU())

        # box pred head
        for i in range(towernum):
            if i == 0:
                reg_tower.append(nn.Conv2d(in_channels[-1], in_channels[-1], kernel_size=3, stride=1, padding=1))
            else:
                reg_tower.append(nn.Conv2d(in_channels[-1], in_channels[-1], kernel_size=3, stride=1, padding=1))

            reg_tower.append(nn.BatchNorm2d(in_channels[-1]))
            reg_tower.append(nn.ReLU())

        self.add_module('cls_tower', nn.Sequential(*cls_tower))
        self.add_module('bbox_tower', nn.Sequential(*reg_tower))

        self.cls_predictor = nn.Conv2d(in_channels[-1], 1, kernel_size=3, stride=1, padding=1)
        self.bbox_predictor = nn.Conv2d(in_channels[-1], 4, kernel_size=3, stride=1, padding=1)

        # adjust scale
        self.adjust = nn.Parameter(0.1 * torch.ones(1))
        self.bias = nn.Parameter(torch.Tensor(1.0 * torch.ones(1, 4, 1, 1)).cuda())

    def forward(self, x_feats, z_feats):

        cls_feats = []
        reg_feats = []

        for i in range(self.num):
            feature_correlation = getattr(self, 'feature_correlation' + str(i))
            feature_cls, feature_reg = feature_correlation(x_feats[i], z_feats[i])
            cls_feats.append(feature_cls)
            reg_feats.append(feature_reg)

        cls_weight = F.softmax(self.cls_weight, 0)
        reg_weight = F.softmax(self.reg_weight, 0)

        def weighted_avg(vec, weight):
            mix = 0
            for i in range(self.num):
                # print (vec[i].shape, weight[i].shape)
                # batch, channel, w, h = vec[i].size()
                # print(weight[i].reshape(channel,1).shape)
                # local_weight = weight[i].view(channel,1).repeat(1,w*h).view(channel, w, h)
                # print(local_weight)
                # print(local_weight.shape)
                mix += vec[i] * weight[i]
            return mix

        cls_feature = weighted_avg(cls_feats, cls_weight)
        reg_feature = weighted_avg(reg_feats, reg_weight)

        cls_feat = self.cls_tower(cls_feature)
        cls_pred = 0.1 * self.cls_predictor(cls_feat)

        reg_feat = self.bbox_tower(reg_feature)
        reg_pred = self.adjust * self.bbox_predictor(reg_feat) + self.bias
        reg_pred = torch.exp(reg_pred)

        return cls_pred, reg_pred, cls_feat, reg_feat


class RegClsModule(nn.Module):

    def __init__(self, in_channels=256, towernum=1):
        super(RegClsModule, self).__init__()

        self.feature_correlation = FeatureCorrelation(in_channels=in_channels, out_channels=in_channels)
        reg_tower = []
        cls_tower = []
        # cls head
        for i in range(towernum):
            if i == 0:
                cls_tower.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
            else:
                cls_tower.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))

            cls_tower.append(nn.BatchNorm2d(in_channels))
            cls_tower.append(nn.ReLU())

        # box pred head
        for i in range(towernum):
            if i == 0:
                reg_tower.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
            else:
                reg_tower.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))

            reg_tower.append(nn.BatchNorm2d(in_channels))
            reg_tower.append(nn.ReLU())

        self.add_module('cls_tower', nn.Sequential(*cls_tower))
        self.add_module('bbox_tower', nn.Sequential(*reg_tower))

        self.cls_predictor = nn.Conv2d(in_channels, 1, kernel_size=3, stride=1, padding=1)
        self.bbox_predictor = nn.Conv2d(in_channels, 4, kernel_size=3, stride=1, padding=1)

        # adjust scale
        self.adjust = nn.Parameter(0.1 * torch.ones(1))
        self.bias = nn.Parameter(torch.Tensor(1.0 * torch.ones(1, 4, 1, 1)).cuda())

    def forward(self, search, template):

        cls_feature, reg_feature = self.feature_correlation(search, template)

        cls_feat = self.cls_tower(cls_feature)
        cls_pred = 0.1 * self.cls_predictor(cls_feat)

        reg_feat = self.bbox_tower(reg_feature)
        reg_pred = self.adjust * self.bbox_predictor(reg_feat) + self.bias
        reg_pred = torch.exp(reg_pred)

        return cls_pred, reg_pred, cls_feat, reg_feat


class MultiRegClsModules(nn.Module):

    def __init__(self, in_channels, towernum=1):
        super(MultiRegClsModules, self).__init__()
        self.num = len(in_channels)
        for i in range(self.num):
            self.add_module('reg_cls_head' + str(i), RegClsModule(in_channels[i], towernum))

        self.cls_weight = nn.Parameter(torch.ones(self.num))
        self.reg_weight = nn.Parameter(torch.ones(self.num))

    def forward(self, x_feats, z_feats):

        cls_preds = []
        reg_preds = []
        cls_feats = []
        reg_feats = []

        for i in range(self.num):
            reg_cls_head = getattr(self, 'reg_cls_head' + str(i))
            cls_pred, reg_pred, cls_feat, reg_feat = reg_cls_head(x_feats[i], z_feats[i])
            cls_preds.append(cls_pred)
            reg_preds.append(reg_pred)
            cls_feats.append(cls_feat)
            reg_feats.append(reg_feat)

        cls_weight = F.softmax(self.cls_weight, 0)
        reg_weight = F.softmax(self.reg_weight, 0)

        def weighted_avg(vec, weight):
            mix = 0
            for i in range(self.num):
                mix += vec[i] * weight[i]
            return mix

        cls_pred = weighted_avg(cls_preds, cls_weight)
        reg_pred = weighted_avg(reg_preds, reg_weight)
        cls_feat = weighted_avg(cls_feats, cls_weight)
        reg_feat = weighted_avg(reg_feats, reg_weight)

        return cls_pred, reg_pred, cls_feat, reg_feat


# ----------------------------------------
# Align module
# ----------------------------------------
class AdaptiveConv(nn.Module):
    """ Adaptive Conv is built based on Deformable Conv
    with precomputed offsets which derived from anchors

    modified from Cascaded RPN
    """

    def __init__(self, in_channels, out_channels):
        super(AdaptiveConv, self).__init__()
        self.conv = DeformConv(in_channels, out_channels, 3, padding=1)

    def forward(self, x, offset):
        N, _, H, W = x.shape
        assert offset is not None
        assert H * W == offset.shape[1]
        # reshape [N, NA, 18] to (N, 18, H, W)
        offset = offset.permute(0, 2, 1).reshape(N, -1, H, W)
        x = self.conv(x, offset)

        return x


class AlignHead(nn.Module):
    # align features and classification score

    def __init__(self, in_channels):
        super(AlignHead, self).__init__()
        self.rpn_conv = AdaptiveConv(in_channels, in_channels)
        self.rpn_cls = nn.Conv2d(in_channels, 1, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, offset):
        x = self.relu(self.rpn_conv(x, offset))
        cls_score = self.rpn_cls(x)
        return cls_score
