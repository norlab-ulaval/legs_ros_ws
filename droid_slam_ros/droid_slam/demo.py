import sys
sys.path.append('droid_slam')

from tqdm import tqdm
import numpy as np
import torch
import lietorch
import cv2
import os
import glob 
import time
import argparse

from torch.multiprocessing import Process
from droid import Droid
from droid_async import DroidAsync

import torch.nn.functional as F


def show_image(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(1)

def image_stream(imagedir, calib, stride, stereo=False):
    """ image generator """

    K_l = K_r = None
    D_l = D_r = None

    if stereo:
        import json
        calib_l_path = os.path.join(imagedir, 'calib', 'zedx_left.json')
        calib_r_path = os.path.join(imagedir, 'calib', 'zedx_right.json')

        if not os.path.exists(calib_l_path) or not os.path.exists(calib_r_path):
            print(f"Error: Stereo calibration files not found at {calib_l_path} or {calib_r_path}")
            sys.exit(1)

        with open(calib_l_path, 'r') as f:
            data_l = json.load(f)
            k_l = data_l['k']
            K_l = np.array(k_l).reshape(3, 3)
            D_l = np.array(data_l['d'])

        with open(calib_r_path, 'r') as f:
            data_r = json.load(f)
            k_r = data_r['k']
            K_r = np.array(k_r).reshape(3, 3)
            D_r = np.array(data_r['d'])

        fx, fy, cx, cy = K_l[0,0], K_l[1,1], K_l[0,2], K_l[1,2]

    else:
        calib = np.loadtxt(calib, delimiter=" ")
        fx, fy, cx, cy = calib[:4]

        K = np.eye(3)
        K[0,0] = fx
        K[0,2] = cx
        K[1,1] = fy
        K[1,2] = cy
        K_l = K

    if stereo:
        imagedir_left = os.path.join(imagedir, 'image_left')
        imagedir_right = os.path.join(imagedir, 'image_right')
        image_list_left = sorted(os.listdir(imagedir_left))[::stride]
        image_list_right = sorted(os.listdir(imagedir_right))[::stride]
        assert len(image_list_left) == len(image_list_right)
    else:
        image_list = sorted(os.listdir(imagedir))[::stride]

    for t, imfile in enumerate(image_list_left if stereo else image_list):
        if stereo:
            image_left = cv2.imread(os.path.join(imagedir_left, image_list_left[t]))
            image_right = cv2.imread(os.path.join(imagedir_right, image_list_right[t]))
            
            # undistort with specific params
            image_left = cv2.undistort(image_left, K_l, D_l)
            image_right = cv2.undistort(image_right, K_r, D_r)
            
            images = [image_left, image_right]
        else:
            image = cv2.imread(os.path.join(imagedir, imfile))
            if len(calib) > 4:
                image = cv2.undistort(image, K_l, calib[4:])
            images = [image]

        h0, w0, _ = images[0].shape
        h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
        w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))

        images = [cv2.resize(img, (w1, h1)) for img in images]
        images = [img[:h1-h1%8, :w1-w1%8] for img in images]
        images = [torch.as_tensor(img).permute(2, 0, 1) for img in images]
        
        images = torch.stack(images)

        intrinsics = torch.as_tensor([fx, fy, cx, cy])
        intrinsics[0::2] *= (w1 / w0)
        intrinsics[1::2] *= (h1 / h0)

        yield t, images, intrinsics


def save_reconstruction(droid, save_path):

    if hasattr(droid, "video2"):
        video = droid.video2
    else:
        video = droid.video

    t = video.counter.value
    save_data = {
        "tstamps": video.tstamp[:t].cpu(),
        "images": video.images[:t].cpu(),
        "disps": video.disps_up[:t].cpu(),
        "poses": video.poses[:t].cpu(),
        "intrinsics": video.intrinsics[:t].cpu()
    }

    torch.save(save_data, save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagedir", type=str, help="path to image directory")
    parser.add_argument("--calib", type=str, help="path to calibration file")
    parser.add_argument("--t0", default=0, type=int, help="starting frame")
    parser.add_argument("--stride", default=3, type=int, help="frame stride")

    parser.add_argument("--weights", default="droid.pth")
    parser.add_argument("--buffer", type=int, default=512)
    parser.add_argument("--image_size", default=[240, 320])
    parser.add_argument("--disable_vis", action="store_true")

    parser.add_argument("--beta", type=float, default=0.3, help="weight for translation / rotation components of flow")
    parser.add_argument("--filter_thresh", type=float, default=2.4, help="how much motion before considering new keyframe")
    parser.add_argument("--warmup", type=int, default=8, help="number of warmup frames")
    parser.add_argument("--keyframe_thresh", type=float, default=4.0, help="threshold to create a new keyframe")
    parser.add_argument("--frontend_thresh", type=float, default=16.0, help="add edges between frames whithin this distance")
    parser.add_argument("--frontend_window", type=int, default=25, help="frontend optimization window")
    parser.add_argument("--frontend_radius", type=int, default=2, help="force edges between frames within radius")
    parser.add_argument("--frontend_nms", type=int, default=1, help="non-maximal supression of edges")

    parser.add_argument("--backend_thresh", type=float, default=22.0)
    parser.add_argument("--backend_radius", type=int, default=2)
    parser.add_argument("--backend_nms", type=int, default=3)
    parser.add_argument("--upsample", action="store_true")
    parser.add_argument("--asynchronous", action="store_true")
    parser.add_argument("--frontend_device", type=str, default="cuda")
    parser.add_argument("--backend_device", type=str, default="cuda")
    
    parser.add_argument("--reconstruction_path", help="path to saved reconstruction")
    parser.add_argument("--stereo", action="store_true")
    args = parser.parse_args()

    torch.multiprocessing.set_start_method('spawn')

    droid = None

    # need high resolution depths
    if args.reconstruction_path is not None:
        args.upsample = True

    tstamps = []
    for (t, image, intrinsics) in tqdm(image_stream(args.imagedir, args.calib, args.stride, args.stereo)):
        if t < args.t0:
            continue

        if not args.disable_vis:
            show_image(image[0])

        if droid is None:
            args.image_size = [image.shape[2], image.shape[3]]
            droid = DroidAsync(args) if args.asynchronous else Droid(args)
        
        droid.track(t, image, intrinsics=intrinsics)

    traj_est = droid.terminate(image_stream(args.imagedir, args.calib, args.stride, args.stereo))
    
    if args.reconstruction_path is not None:
        save_reconstruction(droid, args.reconstruction_path)
