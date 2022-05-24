import _init_paths
import os
import cv2
import glob
import argparse
import numpy as np
from easydict import EasyDict as edict

from tracker import get_tracker
from utils import load_pretrain, cxy_wh_2_rect
import models.models as models


def parse_args():
    """
    args for testing.
    """
    parser = argparse.ArgumentParser(description='PyTorch Tracking Test')
    parser.add_argument('--arch', default='SiamX', type=str, help='model architecture')
    parser.add_argument('--resume', default='SiamX/snapshot/SiamX.pth', type=str, help='pretrained model')
    parser.add_argument('--tracker', default='base', type=str, help='what tracker will be used depends on '
                                                                    'the camera model of input')
    parser.add_argument('--video', default='', type=str, help='video file path')
    parser.add_argument('--save_video_path', default=None, type=str, help='path for saving tracking results as videos')
    parser.add_argument('--save_image_path', default=None, type=str, help='path for saving tracking results as images')
    parser.add_argument('--init_bbox', default=None, help='bbox in the first frame None or [lx, ly, w, h]')
    parser.add_argument('--save_results_path', default=None, type=str, help='path for saving tracking results as txt')
    args = parser.parse_args()
    return args


def get_frames(video_name):
    if not video_name:
        cap = cv2.VideoCapture(0)
        # warmup
        for i in range(5):
            cap.read()
        if not cap.isOpened():
            print("Error opening video stream")
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    elif video_name.endswith('avi') or video_name.endswith('mp4'):
        cap = cv2.VideoCapture(video_name)
        if not cap.isOpened():
            print("Error opening video file")
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    else:
        images = glob.glob(os.path.join(video_name, '*.png*'))
        images = sorted(images, key=lambda x: int(x.split('/')[-1].split('.')[0]))
        for img in images:
            frame = cv2.imread(img)
            yield frame


def tracking_video(tracker, net, video_name, args=None):
    # check init box is list or not
    if not isinstance(args.init_bbox, list) and args.init_bbox is not None:
        args.init_bbox = list(eval(args.init_bbox))
    else:
        args.init_bbox = None
        print('===> please draw a box with your mouse <====')

    # prepare saving
    if args.save_video_path:
        video_writer = None
        save_video_path = os.path.join(args.save_video_path, video_name)
        if not os.path.exists(save_video_path):
            os.makedirs(save_video_path)

    if args.save_image_path:
        save_image_path = os.path.join(args.save_image_path, video_name)
        if not os.path.exists(save_image_path):
            os.makedirs(save_image_path)

    if args.save_results_path:
        save_results_path = os.path.join(args.save_results_path, video_name)
        if not os.path.exists(save_results_path):
            os.makedirs(save_results_path)
        result_txt_path = os.path.join(save_results_path, 'results.txt')

    display_name = 'Tracking: {}'.format(video_name)
    cv2.namedWindow(display_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    # cv2.resizeWindow(display_name, 960, 720)

    count = 0
    SN = 0
    regions = []

    for frame in get_frames(args.video):
        frame_disp = frame.copy()
        if args.save_video_path and video_writer is None:
            fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')  # (*'XVID')
            video_writer = cv2.VideoWriter(os.path.join(save_video_path, video_name + '.mp4'),
                                           fourcc, 30.0, (frame.shape[1], frame.shape[0]))

        if count < 0:
            cv2.putText(frame_disp, 'Tracking!', (10, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, font_color, 1)
            cv2.putText(frame_disp, 'Press r to reset', (10, 40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, font_color, 1)
            cv2.putText(frame_disp, 'Press q to quit', (10, 60), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, font_color, 1)
            cv2.imshow(display_name, frame_disp)
            key = cv2.waitKey(100)
            if key == ord('q'):
                break
            elif key == ord('r'):
                count = 0
                continue
            else:
                continue
        elif count == 0:
            if args.init_bbox:
                lx, ly, w, h = args.init_bbox
                args.init_bbox = None
            else:
                lx, ly, w, h = (0, 0, 0, 0)
                while w == 0 or h == 0:
                    cv2.rectangle(frame_disp, (0, 0), (460, 30), (255, 255, 255), -1)
                    cv2.putText(frame_disp, 'Select target ROI and press ENTER', (10, 20),
                                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
                    try:
                        lx, ly, w, h = cv2.selectROI(display_name, frame_disp, fromCenter=False)
                    except:
                        exit()

            target_pos = np.array([lx + w / 2, ly + h / 2])
            target_sz = np.array([w, h])
            regions.append([lx, ly, w, h])
            state = tracker.init(frame, target_pos, target_sz, net)  # init tracker
        else:
            state = tracker.track(state, frame)
            location = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
            regions.append(location)
            x1, y1, x2, y2 = int(location[0]), int(location[1]), int(location[0] + location[2]), int(
                location[1] + location[3])
            cv2.rectangle(frame_disp, (x1, y1), (x2, y2), (0, 255, 0), 5)
            if args.save_video_path and video_writer:
                video_writer.write(frame_disp)
            if args.save_image_path:
                if SN == count:
                    save_name = os.path.join(save_image_path, '{:04d}.jpg'.format(SN))
                else:
                    save_name = os.path.join(save_image_path, '{:04d}_{}.jpg'.format(SN, count))
                cv2.imwrite(save_name, frame_disp)

            font_color = (0, 0, 0)
            cv2.rectangle(frame_disp, (0, 0), (220, 70), (255, 255, 255), -1)
            cv2.putText(frame_disp, 'Tracking!', (10, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, font_color, 1)
            cv2.putText(frame_disp, 'Press r to reset', (10, 40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, font_color, 1)
            cv2.putText(frame_disp, 'Press q to quit', (10, 60), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, font_color, 1)
            cv2.imshow(display_name, frame_disp)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('r'):
                count = 0
                continue
            elif key == ord('e'):
                count = -1
                continue
        count += 1
        SN += 1

    if args.save_results_path:
        with open(result_txt_path, "w") as fin:
            for x in regions:
                if isinstance(x, int):
                    fin.write("{:d}\n".format(x))
                else:
                    p_bbox = x.copy()
                    fin.write(','.join([str(i) for i in p_bbox]) + '\n')

    if args.save_video_path and video_writer:
        video_writer.release()


def tracking_cam(tracker, net, video_name, args=None):
    # prepare saving
    if args.save_video_path:
        video_writer = None
        save_video_path = os.path.join(args.save_video_path, video_name)
        if not os.path.exists(save_video_path):
            os.makedirs(save_video_path)

    if args.save_image_path:
        save_image_path = os.path.join(args.save_image_path, video_name)
        if not os.path.exists(save_image_path):
            os.makedirs(save_image_path)

    display_name = 'Tracking: {}'.format(video_name)
    cv2.namedWindow(display_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

    init_frame = -1
    for frame in get_frames(args.video):
        frame_disp = frame.copy()
        if args.save_video_path and video_writer is None:
            fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')  # (*'XVID')
            video_writer = cv2.VideoWriter(os.path.join(save_video_path, video_name + '.mp4'),
                                           fourcc, 30.0, (frame.shape[1], frame.shape[0]))

        font_color = (0, 0, 0)
        cv2.rectangle(frame_disp, (0, 0), (460, 70), (255, 255, 255), -1)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('r'):
            init_frame = 0
        elif key == ord("e"):
            init_frame = -1

        if init_frame == 0:
            cv2.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
            try:
                lx, ly, w, h = cv2.selectROI(display_name, frame_disp, fromCenter=False)
            except:
                exit()
            if w > 0 and h > 0:
                target_pos = np.array([lx + w / 2, ly + h / 2])
                target_sz = np.array([w, h])
                state = tracker.init(frame, target_pos, target_sz, net)  # init tracker
                init_frame += 1
        elif init_frame > 0:
            state = tracker.track(state, frame)
            location = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
            x1, y1, x2, y2 = int(location[0]), int(location[1]), int(location[0] + location[2]), int(
                location[1] + location[3])
            cv2.rectangle(frame_disp, (x1, y1), (x2, y2), (0, 255, 0), 5)
            if args.save_video_path and video_writer:
                video_writer.write(frame_disp)
            if args.save_image_path:
                save_name = os.path.join(save_image_path, '{:04d}.jpg'.format(init_frame))
                cv2.imwrite(save_name, frame_disp)

            cv2.putText(frame_disp, 'Tracking!', (10, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, font_color, 1)
            init_frame += 1

        cv2.putText(frame_disp, 'Press r to select a target', (10, 40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, font_color,
                    1)
        cv2.putText(frame_disp, 'Press q to quit', (10, 60), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, font_color, 1)
        cv2.imshow(display_name, frame_disp)


def main():
    args = parse_args()

    info = edict()
    info.arch = args.arch
    info.epoch_test = True
    info.align = False

    tracker = get_tracker(args.tracker, info)

    net = models.__dict__[args.arch](align=info.align)
    net = load_pretrain(net, args.resume)
    net.eval()
    net = net.cuda()

    print('======= Track with {} ======='.format(args.arch))
    if args.video:
        video_name = args.video.split('/')[-1].split('.')[0]
        tracking_video(tracker, net, video_name, args)
    else:
        video_name = 'webcam'
        tracking_cam(tracker, net, video_name, args)


if __name__ == '__main__':
    main()
