# ------------------------------------------------------------------------------
# 
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SiamX_(nn.Module):
    def __init__(self):
        super(SiamX_, self).__init__()
        self.features = None
        self.neck = None
        self.pred_head = None
        self.align_head = None
    
        self.zf = None
        self.score_size = 25
        self.search_size = 255
        self.xf_permutation = [0,1,1] #[1, 0, 1] [0, 1] [1] [0] [1]
        self.zf_permutation = [1,0,1] #[1, 1, 0] [1, 0] [1] [1] [0]

        self.batch = 32 if self.training else 1
        self.criterion = nn.BCEWithLogitsLoss()
        self.grids()

    def feature_extractor(self, x):
        return self.features(x)

    def _cls_loss(self, pred, label, select):
        if len(select.size()) == 0: return 0
        pred = torch.index_select(pred, 0, select)
        label = torch.index_select(label, 0, select)
        return self.criterion(pred, label)  # the same as tf version

    def _weighted_BCE(self, pred, label):
        pred = pred.view(-1)
        label = label.view(-1)
        pos = label.data.eq(1).nonzero().squeeze().cuda()
        neg = label.data.eq(0).nonzero().squeeze().cuda()

        loss_pos = self._cls_loss(pred, label, pos)
        loss_neg = self._cls_loss(pred, label, neg)
        return loss_pos * 0.5 + loss_neg * 0.5

    def _weighted_BCE_align(self, pred, label):
        pred = pred.view(-1)
        label = label.view(-1)
        pos = label.data.eq(1).nonzero().squeeze().cuda()
        neg = label.data.eq(0).nonzero().squeeze().cuda()

        loss_pos = self._cls_loss(pred, label, pos)
        loss_neg = self._cls_loss(pred, label, neg)

        return loss_pos * 0.5 + loss_neg * 0.5

    def _IOULoss(self, pred, target, weight=None):
        pred_left = pred[:, 0]
        pred_top = pred[:, 1]
        pred_right = pred[:, 2]
        pred_bottom = pred[:, 3]

        target_left = target[:, 0]
        target_top = target[:, 1]
        target_right = target[:, 2]
        target_bottom = target[:, 3]

        target_area = (target_left + target_right) * (target_top + target_bottom)
        pred_area = (pred_left + pred_right) * (pred_top + pred_bottom)

        w_intersect = torch.min(pred_left, target_left) + torch.min(pred_right, target_right)
        h_intersect = torch.min(pred_bottom, target_bottom) + torch.min(pred_top, target_top)

        area_intersect = w_intersect * h_intersect
        area_union = target_area + pred_area - area_intersect

        losses = -torch.log((area_intersect + 1.0) / (area_union + 1.0))

        if weight is not None and weight.sum() > 0:
            return (losses * weight).sum() / weight.sum()
        else:
            assert losses.numel() != 0
            return losses.mean()

    def add_iouloss(self, bbox_pred, reg_target, reg_weight):
        """
        :param bbox_pred:
        :param reg_target:
        :param reg_weight:
        :param grid_x:  used to get real target bbox
        :param grid_y:  used to get real target bbox
        :return:
        """

        bbox_pred_flatten = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        reg_target_flatten = reg_target.reshape(-1, 4)
        reg_weight_flatten = reg_weight.reshape(-1)
        pos_inds = torch.nonzero(reg_weight_flatten > 0).squeeze(1)

        bbox_pred_flatten = bbox_pred_flatten[pos_inds]
        reg_target_flatten = reg_target_flatten[pos_inds]

        loss = self._IOULoss(bbox_pred_flatten, reg_target_flatten)

        return loss

    # ---------------------
    # classification align
    # ---------------------
    def grids(self):
        """
        each element of feature map on input search image
        :return: H*W*2 (position for each element)
        """
        sz = self.score_size
        stride = 8

        sz_x = sz // 2
        sz_y = sz // 2

        x, y = np.meshgrid(np.arange(0, sz) - np.floor(float(sz_x)),
                           np.arange(0, sz) - np.floor(float(sz_y)))

        self.grid_to_search = {}
        self.grid_to_search_x = x * stride + self.search_size // 2
        self.grid_to_search_y = y * stride + self.search_size // 2

        self.grid_to_search_x = torch.Tensor(self.grid_to_search_x).unsqueeze(0).unsqueeze(0).cuda()
        self.grid_to_search_y = torch.Tensor(self.grid_to_search_y).unsqueeze(0).unsqueeze(0).cuda()

        self.grid_to_search_x = self.grid_to_search_x.repeat(self.batch, 1, 1, 1)
        self.grid_to_search_y = self.grid_to_search_y.repeat(self.batch, 1, 1, 1)

    def pred_to_image(self, bbox_pred):
        self.grid_to_search_x = self.grid_to_search_x.to(bbox_pred.device)
        self.grid_to_search_y = self.grid_to_search_y.to(bbox_pred.device)

        pred_x1 = self.grid_to_search_x - bbox_pred[:, 0, ...].unsqueeze(1)  # 25*25
        pred_y1 = self.grid_to_search_y - bbox_pred[:, 1, ...].unsqueeze(1)  # 
        pred_x2 = self.grid_to_search_x + bbox_pred[:, 2, ...].unsqueeze(1)  # 
        pred_y2 = self.grid_to_search_y + bbox_pred[:, 3, ...].unsqueeze(1)  # 

        pred = [pred_x1, pred_y1, pred_x2, pred_y2]
        pred = torch.cat(pred, dim=1)

        return pred


    def align_label(self, pred, target, weight):
        # calc predicted box iou (treat it as aligned label)

        pred = pred.permute(0, 2, 3, 1)  # [B, 25, 25, 4]
        pred_left = pred[..., 0]
        pred_top = pred[..., 1]
        pred_right = pred[..., 2]
        pred_bottom = pred[..., 3]

        target_left = target[..., 0]
        target_top = target[..., 1]
        target_right = target[..., 2]
        target_bottom = target[..., 3]

        target_area = (target_left + target_right) * (target_top + target_bottom)
        pred_area = (pred_left + pred_right) * (pred_top + pred_bottom)

        w_intersect = torch.min(pred_left, target_left) + torch.min(pred_right, target_right)
        h_intersect = torch.min(pred_bottom, target_bottom) + torch.min(pred_top, target_top)

        area_intersect = w_intersect * h_intersect
        area_union = target_area + pred_area - area_intersect

        ious = (area_intersect + 1.0) / (area_union + 1.0)

        ious = torch.abs(weight * ious)  # delete points out of object
       
        ious[ious < 0] = 0
        ious[ious >= 1] = 1

        return ious

    def offset(self, boxes, featmap_sizes):
        """
        refers to Cascade RPN
        Params:
        box_list: [N, 4]   [x1, y1, x2, y2] # predicted bbox
        """

        def _shape_offset(boxes, stride):
            ks = 3
            dilation = 1
            pad = (ks - 1) // 2
            idx = torch.arange(-pad, pad + 1, dtype=dtype, device=device)
            xx, yy = torch.meshgrid(idx, idx)
            xx = xx.reshape(-1)
            yy = yy.reshape(-1)

            pad = (ks - 1) // 2
            idx = torch.arange(-pad, pad + 1, dtype=dtype, device=device)
            yy, xx = torch.meshgrid(idx, idx)  # return order matters
            xx = xx.reshape(-1)
            yy = yy.reshape(-1)
            w = (boxes[:, 2] - boxes[:, 0] + 1) / stride
            h = (boxes[:, 3] - boxes[:, 1] + 1) / stride
            w = w / (ks - 1) - dilation
            h = h / (ks - 1) - dilation
            offset_x = w[:, None] * xx  # (NA, ks**2)
            offset_y = h[:, None] * yy  # (NA, ks**2)
            return offset_x, offset_y

        def _ctr_offset(boxes, stride, featmap_size):
            feat_h, feat_w = featmap_size
            image_size = self.search_size

            assert len(boxes) == feat_h * feat_w

            x = (boxes[:, 0] + boxes[:, 2]) * 0.5
            y = (boxes[:, 1] + boxes[:, 3]) * 0.5

            # # compute centers on feature map
            # x = (x - (stride - 1) * 0.5) / stride
            # y = (y - (stride - 1) * 0.5) / stride

            # different here for Siamese
            # use center of image as coordinate origin
            x = (x - image_size * 0.5) / stride + feat_w // 2
            y = (y - image_size * 0.5) / stride + feat_h // 2

            # compute predefine centers
            # different here for Siamese
            xx = torch.arange(0, feat_w, device=boxes.device)
            yy = torch.arange(0, feat_h, device=boxes.device)
            yy, xx = torch.meshgrid(yy, xx)
            xx = xx.reshape(-1).type_as(x)
            yy = yy.reshape(-1).type_as(y)

            offset_x = x - xx  # (NA, )
            offset_y = y - yy  # (NA, )
            return offset_x, offset_y

        num_imgs = len(boxes)
        dtype = boxes[0].dtype
        device = boxes[0][0].device

        featmap_sizes = featmap_sizes[2:]

        offset_list = []
        for i in range(num_imgs):
            c_offset_x, c_offset_y = _ctr_offset(boxes[i], 8, featmap_sizes)
            s_offset_x, s_offset_y = _shape_offset(boxes[i], 8)

            # offset = ctr_offset + shape_offset
            offset_x = s_offset_x + c_offset_x[:, None]
            offset_y = s_offset_y + c_offset_y[:, None]

            # offset order (y0, x0, y1, x0, .., y9, x8, y9, x9)from torch.autograd import Variable
            offset = torch.stack([offset_y, offset_x], dim=-1)
            offset = offset.reshape(offset.size(0), -1).unsqueeze(0)  # [NA, 2*ks**2]
            offset_list.append(offset)

        offsets = torch.cat(offset_list, 0)
        return offsets
    


    def template(self, z):
        """
            initialize tracking
        """
        zfs_stages, zf = self.feature_extractor(z)
        self.zf = [zfs_stages[-1], zf]
        if self.neck is not None:
            self.zf = self.neck(self.zf)
        if self.align_head is not None:
            self.update_flag = True
        else:
            pass


    def track(self, x):
        """
            for tracking
        """
        xfs_stages, xf = self.feature_extractor(x)
        xf = [xfs_stages[-1], xf]
        #xf = self.feature_extractor(x)
        #print("model/ocean.py: self.xf", x.shape, xf.shape)
        if self.neck is not None:
            xf = self.neck(xf)
    
        zf = self.zf

        x_feats = []
        z_feats = []
        for idx in range(len(self.xf_permutation)):
            x_feats.append(xf[self.xf_permutation[idx]])
            z_feats.append(zf[self.zf_permutation[idx]])


        cls_pred, reg_pred, cls_feat, reg_feat = self.pred_head(x_feats, z_feats)

        cls_align = None
        if self.align_head is not None:
            if self.update_flag:
                self.batch = 1
                self.search_size = x.size(-1)
                self.score_size = (self.search_size - 127) // 8 + 1 + 8
                self.grids()
                self.update_flag = False

            bbox_pred_to_img = self.pred_to_image(reg_pred)
            offsets = self.offset(bbox_pred_to_img.permute(0, 2, 3, 1).reshape(bbox_pred_to_img.size(0), -1, 4), reg_pred.size())
            cls_align = self.align_head(reg_feat, offsets)

        return cls_pred, reg_pred, cls_align, xf


    def forward(self, template, search, label=None, reg_target=None, reg_weight=None):
        """
            for training
        """
        # [x_, p1, p2], p3
        xfs_stages, xf = self.feature_extractor(search)
        zfs_stages, zf = self.feature_extractor(template)
        xf = [xfs_stages[-1], xf]
        zf = [zfs_stages[-1], zf]

        if self.neck is not None:
            xf = self.neck(xf)
            zf = self.neck(zf)

        # depth-wise cross correlation -->  box and cls pred
        x_feats = []
        z_feats = []
        for idx in range(len(self.xf_permutation)):
            x_feats.append(xf[self.xf_permutation[idx]])
            z_feats.append(zf[self.zf_permutation[idx]])
        cls_pred, reg_pred, cls_feat, reg_feat = self.pred_head(x_feats, z_feats)

        # loss
        cls_loss = self._weighted_BCE(cls_pred, label)
        reg_loss = self.add_iouloss(reg_pred, reg_target, reg_weight)

        cls_loss_align = None
        if self.align_head is not None: 
            bbox_pred_to_img = self.pred_to_image(reg_pred)
            offsets = self.offset(bbox_pred_to_img.permute(0, 2, 3, 1).reshape(bbox_pred_to_img.size(0), -1, 4), reg_pred.size())
            cls_align = self.align_head(reg_feat, offsets)
            cls_label_align = self.align_label(reg_pred, reg_target, reg_weight)
            cls_loss_align = self.criterion(cls_align.squeeze(), cls_label_align)
            if torch.isnan(cls_loss_align):
                cls_loss_align = 0 * cls_loss

        return cls_loss, cls_loss_align, reg_loss











