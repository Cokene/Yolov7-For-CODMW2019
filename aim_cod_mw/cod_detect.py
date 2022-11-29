from threading import Thread
from pathlib import Path
import argparse
import time
import sys

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from numpy import random

import pynput
from pynput.mouse import Listener, Button
import pyautogui
import win32api
import win32con
from aim_cod_mw.getkeys import key_check
from aim_cod_mw.grabscreen import mss_grab, grab_screen, getWindowData

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import check_img_size, check_requirements, check_imshow, clean_str, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel


def load_model(weights, half, device):
    model = attempt_load(weights, map_location=device)  # load FP32 model
    if half:
        model.half()  # to FP16
    return model


class GameCapture:
    def __init__(self, img_size=640):
        self.region = getWindowData()
        self.cap_prop_frame_width = self.region[4]
        self.cap_prop_frame_height = self.region[5]
        self.img_size = img_size

    @staticmethod
    def isNotNone():
        img = mss_grab()
        if img.max() is None:
            return False
        return True

    @staticmethod
    def read():
        image = mss_grab()
        return image

    @staticmethod
    def retrieve():
        success = False
        image = mss_grab()
        if image.max() is not None:
            success = True
            return success, image
        return success, image


class GameStream:
    def __init__(self, sources='streams.txt', img_size=640, stride=32, vid_stride=1):
        self.mode = 'stream'
        self.img_size = img_size
        self.stride = stride
        self.vid_stride = vid_stride

        n = len(sources)
        self.imgs, self.threads = [None] * n, [None] * n
        self.sources = [clean_str(x) for x in sources]  # clean source names for later
        for i, s, in enumerate(sources):
            # Start the thread to read game frames from the stream
            print(f'{i + 1}/{n}: {s}... ', end='')
            cap = GameCapture(img_size)
            assert cap.isNotNone(), f'Failed to open {s}'
            w = int(cap.cap_prop_frame_width)
            h = int(cap.cap_prop_frame_height)
            self.fps = 30

            self.imgs[i] = cap.read()  # guarantee first frame
            self.threads[i] = Thread(target=self.update, args=([i, cap]), daemon=True)
            print(f' success ({w}x{h} at {self.fps:.2f} FPS).')
            self.threads[i].start()
        print('')  # newline

        # check for common shapes
        s = np.stack([letterbox(x, self.img_size, stride=self.stride)[0].shape for x in self.imgs], 0)  # shapes
        self.rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
        if not self.rect:
            print('WARNING: Different stream shapes detected. For optimal performance supply similarly-shaped streams.')

    def update(self, index, cap):
        # Read next stream frame in a darmon thread
        n = 0
        while cap.isNotNone():
            n += 1
            # self.imgs[index] = cap.read()
            if n % self.vid_stride == 0:  # read per vid_stride frame
                success, im = cap.retrieve()
                self.imgs[index] = im if success else self.imgs[index] * 0
                n = 0
            # time.sleep(1 / self.fps)
            time.sleep(0.0)  # wait time

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        img0 = self.imgs.copy()
        if cv2.waitKey(1) == ord('`'):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration

        # Letterbox
        img = [letterbox(x, self.img_size, auto=self.rect, stride=self.stride)[0] for x in img0]

        # Stack
        img = np.stack(img, 0)

        # Convert
        img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to bsx3x416x416
        img = np.ascontiguousarray(img)

        return self.sources, img, img0, None

    def __len__(self):
        return 0


class Mouse:
    def __init__(self, lock_mode=False, tag=1):
        self.lock_mode = lock_mode  # lock mode
        self.tag = tag  # tag value default locate body
        self.button = pynput.mouse.Button
        self.mouse = pynput.mouse.Controller()
        self.mouse_x, self.mouse_y = self.mouse.position  # mouse position
        self.reaction_t = 180  # millisecond

    @staticmethod
    def on_click(self, x, y, button, pressed):
        if pressed:
            print(f'mouse {button} pressed ({x, y})')
        else:
            print(f'mouse {button} released {x, y}')

    @staticmethod
    def on_scroll(self, x, y, dx, dy):
        print(f'mouse scrolled {x, y} motion {dx, dy}')

    def on_move(self, x, y):
        self.mouse_x, self.mouse_y = x, y  # guarantee for real-time
        # print(f'pointer position: {x, y}')

    def click(self, x, y, button, pressed):
        if pressed and button == self.button.x1:  # Side mouse button
            self.lock_mode = not self.lock_mode  # negation
            print('lock mode:', 'on' if self.lock_mode else 'off')

    def scroll(self, rel_motion):
        if rel_motion[0] == 0 and rel_motion[1] != 0:
            self.mouse.scroll(rel_motion[0], rel_motion[1])

    def move(self, t_pos):  # target position
        self.mouse.move(t_pos[0], t_pos[1])

    # not recommended
    def tartget_lock_rel(self, targets, g_size):
        min_d = torch.inf
        shoot = False
        for det in targets:
            t, x_c, y_c, h, w, _ = det
            t = int(t)
            t_x, t_y = g_size[4] * float(x_c) + g_size[0], g_size[5] * float(y_c) + g_size[1]
            r = min(float(w) * g_size[4], float(h) * g_size[5])
            d = (int(t_x) - self.mouse_x) ** 2 + (int(t_y) - self.mouse_y) ** 2  # distance pointer to target
            if min_d > d >= int(r ** 2) and t == self.tag:
                x_center, y_center = int(t_x), int(t_y)
                min_d = d
                shoot = True
            elif d < min_d and d < int(r ** 2) and t == self.tag:
                shoot = False
        if shoot:
            cur_d = min_d
            while cur_d > int(r ** 2):
                dx = 100 if (x_center - self.mouse_x) > 0 else -100
                dy = 50 if (y_center - self.mouse_y) > 0 else -50
                cur_d = (x_center - self.mouse_x) ** 2 + (y_center - self.mouse_y) ** 2  # current distance
                pyautogui.moveRel(xOffset=dx, yOffset=dy)

    # recommended
    def target_lock(self, targets, g_size):
        min_d = torch.inf
        shoot = False
        for det in targets:
            t, x_c, y_c, h, w, _ = det
            t = int(t)
            t_x, t_y = g_size[4] * float(x_c) + g_size[0], g_size[5] * float(y_c) + g_size[1]
            r = min(float(w) * g_size[4], float(h) * g_size[5])
            d = (int(t_x) - self.mouse_x) ** 2 + (int(t_y) - self.mouse_y) ** 2  # distance current pointer to target
            if min_d > d >= int(r ** 2) and t == self.tag:
                x_center, y_center = int(t_x), int(t_y)
                min_d = d
                shoot = True
            elif d < min_d and d < int(r ** 2) and t == self.tag:
                shoot = False
        if shoot:
            self.mouse.position = (x_center, y_center)

    # listen events on daemon thread
    def lock_listener(self):
        while True:
            # Collect events until released
            with Listener(on_click=self.click, on_move=self.on_move) as listener:
                listener.join()


if __name__ == '__main__':
    weights = '../models/yolov7-tiny-codmw-best.pt'  # model path
    source = '0'  # desktop session
    device = select_device('0')
    half = device.type != 'cpu'  # half precision only supported on CUDA

    view_img = True
    lock_mode = False

    vid_stride = 1  # video frame rate stride
    imgsz = 640  # detect image size
    conf_thres = 0.25  # Confidence threshold
    iou_thres = 0.45  # NMS IoU threshold
    width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)  # windows width
    height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)  # windows height
    w_size = (width, height)  # windows desktop size

    mouse = Mouse(lock_mode=lock_mode, tag=0)  # tag value 0 head 1 body
    capture = GameCapture()
    model = load_model(weights, half, device)
    stride = int(model.stride.max())
    imgsz = check_img_size(imgsz, s=stride)

    # view_img = check_imshow()
    cudnn.benchmark = True  # set True to speed up constant image size inference
    dataset = GameStream(source, img_size=imgsz, stride=stride, vid_stride=vid_stride)  # self definition stream

    # Get nams and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    # daemon thread
    thread_listen = Thread(target=mouse.lock_listener, daemon=True)
    thread_listen.start()

    # Launch Marksman!
    # ************ protected ************#
    # with pynput.mouse.Listener(on_click=mouse.click) as listener:
    # with pynput.mouse.Events() as events:
    # ************ protected ************#
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # normalization
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=False)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, agnostic=False)
        t3 = time_synchronized()

        targets = []  # enemy detection
        for i, det in enumerate(pred):  # detections per image
            p, s, im0, frame = path[i], '%g:' % i, im0s[i].copy(), dataset.count

            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalised xywh
                    line = (cls, *xywh, conf)
                    target = ('%g ' * len(line)).rstrip() % line  # str
                    target = target.split(' ')  # list
                    targets.append(target)

                    if view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

            # draw point on bbox center
            g_size = getWindowData()  # game session window size
            if len(targets):
                if mouse.lock_mode:
                    # mouse.aim_lock(targets, w_size=w_size, g_size=g_size)
                    mouse.target_lock(targets, g_size=g_size)

                for i, det in enumerate(targets):
                    x_c, y_c = det[1:3]
                    x_center, y_center = g_size[4] * float(x_c), g_size[5] * float(y_c)
                    center = (int(x_center), int(y_center))
                    color = (0, 0, 255)  # draw targets with color boxes
                    cv2.circle(im0, center=center, radius=1, color=color, thickness=2)

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

        # terminate session(forced)
        keys = key_check()
        if '£' in keys:  # keyboard right ctrl
            cv2.destroyAllWindows()
            sys.exit()

            # *********** protected ************#
            # mouse listen events
            # it = next(events)
            # while it is not None and not isinstance(it, pynput.mouse.Events.Click):
            #     it = next(events)
            # if it is not None and it.button == it.button.x1 and it.pressed:
            #     mouse.lock_mode = not mouse.lock_mode
            #     print('lock mode: ', 'on' if mouse.lock_mode else 'off')
            # *********** protected ************#

    # print(f'Done. ({time.time() - t0:.3f}s)')
