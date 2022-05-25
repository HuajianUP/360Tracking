import os
import cv2
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import get_subwindow_tracking, imgLookAt


def get_tracker(tracker, info):
    return Tracker_dict[tracker](info)


def inv_maximum(r):
    return np.maximum(r, 1. / r)


def get_search_region_size(target_wh, context_amount=0.5):
    pad = (target_wh[0] + target_wh[1]) * context_amount
    sz2 = (target_wh[0] + pad) * (target_wh[1] + pad)
    return np.sqrt(sz2)


class BaseTracker(object):
    def __init__(self, info):
        super(BaseTracker, self).__init__()
        self.grid_to_search_y = None
        self.grid_to_search_x = None
        self.info = info  # model and benchmark info
        # self.stride = 8
        self.align = info.align

    def init(self, im, target_pos, target_sz, model):
        state = dict()
        state['im_h'] = im.shape[0]
        state['im_w'] = im.shape[1]
        p = LocalConfig()
        self.grids(p)  # self.grid_to_search_x, self.grid_to_search_y

        s_z = get_search_region_size(target_sz, p.context_amount)
        # s_z = round(s_z)

        avg_chans = np.mean(im, axis=(0, 1))
        z_crop, _ = get_subwindow_tracking(im, target_pos, p.exemplar_size, s_z, avg_chans)
        z = z_crop.unsqueeze(0)

        net = model
        net.template(z.cuda())

        if p.windowing == 'cosine':
            window = np.outer(np.hanning(p.score_size), np.hanning(p.score_size))  # [17,17]
        elif p.windowing == 'uniform':
            window = np.ones(int(p.score_size), int(p.score_size))

        state['p'] = p
        state['net'] = net
        state['avg_chans'] = avg_chans
        state['window'] = window
        state['target_pos'] = target_pos
        state['target_sz'] = target_sz

        return state

    def update(self, net, x_crops, target_pos, target_sz, window, scale_z, p):

        cls_score, bbox_pred, cls_align, _ = net.track(x_crops)
        cls_score = torch.sigmoid(cls_score).squeeze().cpu().data.numpy()
        if self.align and cls_align is not None:
            cls_align = torch.sigmoid(cls_align).squeeze().cpu().data.numpy()
            cls_score = p.ratio * cls_score + (1 - p.ratio) * cls_align

        # offset to bbox[x0,y0,x1,y1]
        bbox_pred = bbox_pred.squeeze().cpu().data.numpy()
        pred_x1 = self.grid_to_search_x - bbox_pred[0, ...]
        pred_y1 = self.grid_to_search_y - bbox_pred[1, ...]
        pred_x2 = self.grid_to_search_x + bbox_pred[2, ...]
        pred_y2 = self.grid_to_search_y + bbox_pred[3, ...]

        # size penalty
        s_c = inv_maximum(get_search_region_size([pred_x2 - pred_x1, pred_y2 - pred_y1]) / (get_search_region_size(target_sz)))  # scale penalty
        r_c = inv_maximum((target_sz[0] / target_sz[1]) / ((pred_x2 - pred_x1) / (pred_y2 - pred_y1)))  # ratio penalty

        penalty = np.exp(-(r_c * s_c - 1) * p.penalty_k)
        pscore = penalty * cls_score

        # window penalty
        pscore = pscore * (1 - p.window_influence) + window * p.window_influence
        # get max
        r_max, c_max = np.unravel_index(pscore.argmax(), pscore.shape)
        # to real size
        # bbox
        pred_x1 = pred_x1[r_max, c_max]
        pred_y1 = pred_y1[r_max, c_max]
        pred_x2 = pred_x2[r_max, c_max]
        pred_y2 = pred_y2[r_max, c_max]

        pred_xs = (pred_x1 + pred_x2) / 2  # predicted location in the searh patch
        pred_ys = (pred_y1 + pred_y2) / 2
        diff_xs = pred_xs - p.instance_size // 2  # distance to search center
        diff_ys = pred_ys - p.instance_size // 2
        diff_xs, diff_ys = diff_xs / scale_z, diff_ys / scale_z

        pred_w = (pred_x2 - pred_x1) / scale_z
        pred_h = (pred_y2 - pred_y1) / scale_z

        target_sz = target_sz / scale_z
        # size learning rate
        lr = penalty[r_max, c_max] * cls_score[r_max, c_max] * p.lr
        # size rate
        res_xs = target_pos[0] + diff_xs
        res_ys = target_pos[1] + diff_ys
        res_w = pred_w * lr + (1 - lr) * target_sz[0]
        res_h = pred_h * lr + (1 - lr) * target_sz[1]

        target_pos = np.array([res_xs, res_ys])
        target_sz = target_sz * (1 - lr) + lr * np.array([res_w, res_h])

        return target_pos, target_sz, cls_score[r_max, c_max]

    def track(self, state, im, gt=None):
        p = state['p']
        net = state['net']
        avg_chans = state['avg_chans']
        window = state['window']
        target_pos = state['target_pos']
        target_sz = state['target_sz']

        s_z = get_search_region_size(target_sz, p.context_amount)
        scale_z = p.exemplar_size / s_z
        s_x = s_z * (p.instance_size / p.exemplar_size)

        x_crop, _ = get_subwindow_tracking(im, target_pos, p.instance_size, s_x, avg_chans)
        x_crop = x_crop.unsqueeze(0)

        target_pos, target_sz, _ = self.update(net, x_crop.cuda(), target_pos, target_sz * scale_z, window, scale_z, p)

        target_pos[0] = max(0, min(state['im_w'], target_pos[0]))
        target_pos[1] = max(0, min(state['im_h'], target_pos[1]))
        target_sz[0] = max(10, min(state['im_w'], target_sz[0]))
        target_sz[1] = max(10, min(state['im_h'], target_sz[1]))
        state['target_pos'] = target_pos
        state['target_sz'] = target_sz
        state['p'] = p

        return state

    def grids(self, p):
        """
        each element of feature map on input search image
        :return: H*W*2 (position for each element)
        """
        sz = p.score_size
        # the real shift is -param['shifts']
        sz_x = sz // 2
        sz_y = sz // 2
        x, y = np.meshgrid(np.arange(0, sz) - np.floor(float(sz_x)),
                           np.arange(0, sz) - np.floor(float(sz_y)))

        self.grid_to_search_x = x * p.total_stride + p.instance_size // 2
        self.grid_to_search_y = y * p.total_stride + p.instance_size // 2


class OmniTracker(BaseTracker):
    def init(self, im, target_pos, target_sz, model, hp=None):
        state = dict()
        state['im_h'] = im.shape[0]
        state['im_w'] = im.shape[1]
        p = LocalConfig()
        self.grids(p)  # self.grid_to_search_x, self.grid_to_search_y

        s_z = get_search_region_size(target_sz, p.context_amount)
        # s_z = round(s_z)
        # z_crop = imgLookAt(im, target_pos[0], target_pos[1], 100, fov=np.pi/2)
        # depend on the fov, if fov is larger than 90,
        # then we should directly crop the image instead of rectifying
        avg_chans = np.mean(im, axis=(0, 1))
        if s_z > state['im_h'] * 0.8:
            z_crop, _ = get_subwindow_tracking(im, target_pos, p.exemplar_size, s_z, avg_chans)
        else:
            z_crop = imgLookAt(im, target_pos[0], target_pos[1], p.exemplar_size, region_size=s_z)
        z = z_crop.unsqueeze(0)

        net = model
        net.template(z.cuda())

        if p.windowing == 'cosine':
            window = np.outer(np.hanning(p.score_size), np.hanning(p.score_size))  # [17,17]
        elif p.windowing == 'uniform':
            window = np.ones(int(p.score_size), int(p.score_size))

        state['p'] = p
        state['net'] = net
        state['avg_chans'] = avg_chans
        state['window'] = window
        state['target_pos'] = target_pos
        state['target_sz'] = target_sz
        return state

    def track(self, state, im, gt=None):
        p = state['p']
        net = state['net']
        window = state['window']
        avg_chans = state['avg_chans']
        target_pos = state['target_pos']
        target_sz = state['target_sz']

        s_z = get_search_region_size(target_sz, p.context_amount)
        scale_z = p.exemplar_size / s_z
        s_x = s_z * (p.instance_size / p.exemplar_size)

        if s_x > state['im_h'] * 0.8:
            x_crop, _ = get_subwindow_tracking(im, target_pos, p.instance_size, s_x, avg_chans)
        else:
            x_crop = imgLookAt(im, target_pos[0], target_pos[1], p.instance_size, region_size=s_x)
        x_crop = x_crop.unsqueeze(0)

        target_pos, target_sz, _ = self.update(net, x_crop.cuda(), target_pos, target_sz * scale_z, window, scale_z, p)

        if target_pos[0] > state['im_w']:
            target_pos[0] -= state['im_w']
        elif target_pos[0] < 0:
            target_pos[0] += state['im_w']
        if target_pos[1] > state['im_h']:
            target_pos[1] -= state['im_h']
        elif target_pos[1] < 0:
            target_pos[1] += state['im_h']

        target_sz[0] = max(10, min(state['im_w'], target_sz[0]))
        target_sz[1] = max(10, min(state['im_h'], target_sz[1]))
        state['target_pos'] = target_pos
        state['target_sz'] = target_sz
        state['p'] = p

        return state

Tracker_dict = {"omni": OmniTracker, "base": BaseTracker}


class LocalConfig(object):
    penalty_k = 0.062
    window_influence = 0.38
    lr = 0.765
    windowing = 'cosine'
    exemplar_size = 127
    instance_size = 255
    total_stride = 8
    score_size = (instance_size - exemplar_size) // total_stride + 1 + 8  # 
    context_amount = 0.5
    ratio = 0.94

    def update(self, newparam=None):
        if newparam:
            for key, value in newparam.items():
                setattr(self, key, value)
            self.renew()

    def renew(self):
        self.score_size = (self.instance_size - self.exemplar_size) // self.total_stride + 1 + 8  #
