import cv2
import torch
import numpy as np


def get_subwindow_tracking(im, pos, model_sz, original_sz, avg_chans, out_mode='torch'):
    """
    SiamFC type cropping
    """
    crop_info = dict()

    if isinstance(pos, float):
        pos = [pos, pos]

    sz = original_sz
    im_sz = im.shape
    c = (original_sz+1) / 2
    context_xmin = round(pos[0] - c)
    context_xmax = context_xmin + sz - 1
    context_ymin = round(pos[1] - c)
    context_ymax = context_ymin + sz - 1
    left_pad = int(max(0., -context_xmin))
    top_pad = int(max(0., -context_ymin))
    right_pad = int(max(0., context_xmax - im_sz[1] + 1))
    bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))

    context_xmin = context_xmin + left_pad
    context_xmax = context_xmax + left_pad
    context_ymin = context_ymin + top_pad
    context_ymax = context_ymax + top_pad

    r, c, k = im.shape
    if any([top_pad, bottom_pad, left_pad, right_pad]):
        te_im = np.zeros((r + top_pad + bottom_pad, c + left_pad + right_pad, k), np.uint8)
        # for return mask
        tete_im = np.zeros((r + top_pad + bottom_pad, c + left_pad + right_pad))

        te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im
        if top_pad:
            te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
        if bottom_pad:
            te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
        if left_pad:
            te_im[:, 0:left_pad, :] = avg_chans
        if right_pad:
            te_im[:, c + left_pad:, :] = avg_chans
        im_patch_original = te_im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1), :]
    else:
        tete_im = np.zeros(im.shape[0:2])
        im_patch_original = im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1), :]

    if not np.array_equal(model_sz, original_sz):
        im_patch = cv2.resize(im_patch_original, (model_sz, model_sz))
    else:
        im_patch = im_patch_original

    crop_info['crop_cords'] = [context_xmin, context_xmax, context_ymin, context_ymax]
    crop_info['empty_mask'] = tete_im
    crop_info['pad_info'] = [top_pad, left_pad, r, c]

    if out_mode == "torch":
        return im_to_torch(im_patch.copy()), crop_info
    else:
        return im_patch, crop_info


def im_to_torch(img):
    img = np.transpose(img, (2, 0, 1))  # C*H*W
    img = torch.from_numpy(img).float()
    return img


def check_keys(model, pretrained_state_dict, print_unuse=True):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = list(ckpt_keys - model_keys)
    missing_keys = list(model_keys - ckpt_keys)

    # remove num_batches_tracked
    for k in sorted(missing_keys):
        if 'num_batches_tracked' in k:
            missing_keys.remove(k)

    print('missing keys:{}'.format(missing_keys))
    if print_unuse:
        print('unused checkpoint keys:{}'.format(unused_pretrained_keys))
    # print('used keys:{}'.format(used_pretrained_keys))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    """
    Old style model is stored with all names of parameters share common prefix 'module.'
    """
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_pretrain(model, pretrained_path, print_unuse=True, device = None):
    print('load pretrained model from {}'.format(pretrained_path))

    device = torch.cuda.current_device() if device is None else device
    pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))

    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
        pretrained_dict = remove_prefix(pretrained_dict, 'feature_extractor.')  # remove online train
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')  # remove multi-gpu label
        pretrained_dict = remove_prefix(pretrained_dict, 'feature_extractor.')   # remove online train

    check_keys(model, pretrained_dict, print_unuse=print_unuse)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


def cxy_wh_2_rect(pos, sz):
    return [float(max(float(0), pos[0]-sz[0]/2)), float(max(float(0), pos[1]-sz[1]/2)), float(sz[0]), float(sz[1])]  # 0-index

