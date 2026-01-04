#!/usr/bin/env python3

from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from geometry_msgs.msg import Pose, Point, Quaternion
from nav_msgs.msg import Odometry
from lifelong_msgs.msg import ImagePose  # Make sure to import your custom ImagePose message
from rclpy.node import Node
from cv_bridge import CvBridge  # Needed for converting between ROS Image messages and OpenCV images
import sys
from scipy.spatial.transform import Rotation as R
import os
from ament_index_python.packages import get_package_share_directory

# Add droid_slam to path
try:
    share_dir = get_package_share_directory('droid_slam_ros')
    sys.path.append(share_dir)
    sys.path.append(os.path.join(share_dir, 'droid_slam'))
except Exception as e:
    print(f"Could not find share directory: {e}")
    sys.path.append('droid_slam') # Fallback for local run

from PIL import Image as PILImage

from tqdm import tqdm
import numpy as np
import torch
import lietorch
import cv2
import glob
import time
import argparse
import rclpy
import json
from sensor_msgs.msg import Image as ROSImage
import message_filters
from droid import Droid
import torch.nn.functional as F

import tf2_ros
from tf2_geometry_msgs import do_transform_pose
import droid_backends
from lietorch import SE3
import geom.projective_ops as pops


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


class DroidNode(Node):
    def __init__(self, args):
        super().__init__('droid_node')
        
        self.args = args
        self.args.weights = "/home/mbo/legs_ws/install/droid_slam_ros/share/droid_slam_ros/droid.pth" # TODO replace with ROS param

        self.args.image_size = [344, 560] # TODO replace with ROS param
        self.droid = Droid(self.args)

        self.cam_transform = np.diag([1, -1, -1, 1])
        
        # Initialize TF2 Buffer and Listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Initialize ROS2 Publisher and Subscriber
        self.publisher = self.create_publisher(ImagePose, '/camera/color/imagepose', 10)
        self.odom_publisher = self.create_publisher(Odometry, 'estimated_odom', 10)
        
        # Subscriptions
        # Use absolute paths for input topics to ignore node namespace
        self.left_rect_sub = message_filters.Subscriber(self, ROSImage, '/zedx/left/image_rect')
        self.right_rect_sub = message_filters.Subscriber(self, ROSImage, '/zedx/right/image_rect')
        
        # Exact Time Sync for stereo pairs
        self.ts = message_filters.TimeSynchronizer([self.left_rect_sub, self.right_rect_sub], 10)
        self.ts.registerCallback(self.image_callback)
        
        # TODO add right camera info subscription
        self.intr_sub_left = self.create_subscription(CameraInfo, '/zedx/left/camera_info', self.cam_intr_left_cb, 1)
        self.intr_sub_right = self.create_subscription(CameraInfo, '/zedx/right/camera_info', self.cam_intr_right_cb, 1)

        self.image_counter = 0
        self.output_folder_ = '/home/mbo/legs_ws/output_images'
        self.json_file_path_ = os.path.join(self.output_folder_, 'transforms.json')
        if not os.path.exists(self.output_folder_):
            os.makedirs(self.output_folder_)            

        self.cam_params = {}
        self.bridge = CvBridge()
        self.baseline = None

    def cam_intr_left_cb(self, msg):
        if 'w' in self.cam_params:
            return

        K = msg.k.reshape(3, 3)
        params = {
            "w": msg.width,
            "h": msg.height,
            "K": K,
            "D": np.array(msg.d),
        }
        self.frames = []
        self.cam_params["left"] = params

    def cam_intr_right_cb(self, msg):
        if 'w' in self.cam_params:
            return

        K = msg.k.reshape(3, 3)
        params = {
            "w": msg.width,
            "h": msg.height,
            "K": K,
            "D": np.array(msg.d),
        }
        self.cam_params["right"] = params

    def xyzquat2mat(self, vec):
        xyz = vec[:3]
        quat = vec[3:]
        matrix = np.eye(4)
        rotation = R.from_quat(quat)
        matrix[:3, :3] = rotation.as_matrix()
        matrix[:3, 3] = xyz
        matrix = np.linalg.inv(matrix) @ self.cam_transform
        return matrix

    def image_callback(self, left_msg, right_msg):
        # Get baseline from TF if not already set
        if self.baseline is None:
            try:
                trans = self.tf_buffer.lookup_transform('zedx_left', 'zedx_right', rclpy.time.Time())
                # Baseline is typically the Euclidean distance, mainly along -x in camera frame for right cam
                # But here we just need the magnitude if we are converting disparity/stereo
                # Wait, DroidSLAM expects stereo pairs. We need the intrinsics [fx, fy, cx, cy]
                # and usually assumes rectified stereo with horizontal baseline.
                self.baseline = abs(trans.transform.translation.x) 
                print(f"Baseline found: {self.baseline}")
            except Exception as e:
                print(f"Could not get baseline: {e}")
                return

        if self.cam_params.get("left") is None or self.cam_params.get("right") is None:
            print("Waiting for camera info")
            return

        # Convert ROS Image messages to OpenCV images
        cv_left = self.bridge.imgmsg_to_cv2(left_msg, desired_encoding='bgr8')
        cv_right = self.bridge.imgmsg_to_cv2(right_msg, desired_encoding='bgr8')
        
        print(cv_left.shape, cv_right.shape)
        t = left_msg.header.stamp.sec + left_msg.header.stamp.nanosec * 1e-9

        # Undistort images

        print(self.cam_params['left']['D'])
        cv_left = cv2.undistort(cv_left, self.cam_params['left']['K'], self.cam_params['left']['D'])
        cv_right = cv2.undistort(cv_right, self.cam_params['right']['K'], self.cam_params['right']['D'])

        # Prepare inputs for Droid

        h0, w0, _ = cv_left.shape
        h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
        w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))

        cv_left = cv2.resize(cv_left, (w1, h1))
        cv_left = cv_left[:h1-h1%8, :w1-w1%8]
        cv_right = cv2.resize(cv_right, (w1, h1))
        cv_right = cv_right[:h1-h1%8, :w1-w1%8]

        image_left_tensor = torch.as_tensor(cv_left).permute(2, 0, 1)
        image_right_tensor = torch.as_tensor(cv_right).permute(2, 0, 1)

        print(image_right_tensor.shape, image_left_tensor.shape)
        
        stereo_image = torch.stack([image_left_tensor, image_right_tensor])

        K_l = self.cam_params['left']['K']
        fx, fy, cx, cy = K_l[0,0], K_l[1,1], K_l[0,2], K_l[1,2]
        intrinsics = torch.as_tensor([
            fx, fy, cx, cy
        ])
        intrinsics[0::2] *= (w1 / w0)
        intrinsics[1::2] *= (h1 / h0)


        print(stereo_image.shape)

        self.droid.track(t, stereo_image, depth=None, intrinsics=intrinsics)

        if self.droid.video.counter.value == self.image_counter:
            return

        self.image_counter += 1
        
        # Get latest pose
        pose = self.droid.video.poses[self.droid.video.counter.value - 1].cpu().numpy()
        posemat = self.xyzquat2mat(pose)
        pos_xyz = posemat[:3, 3]
        orient = R.from_matrix(posemat[:3, :3]).as_quat()

        pose = Pose(
            position=Point(x=pos_xyz[0], y=pos_xyz[1], z=pos_xyz[2]),
            orientation=Quaternion(x=orient[0], y=orient[1], z=orient[2], w=orient[3])
        )
        
        # Publish Odometry
        odom_msg = Odometry()
        odom_msg.header = left_msg.header
        odom_msg.header.frame_id = "map" # or "odom"
        odom_msg.child_frame_id = "zedx_left" # or camera frame
        odom_msg.pose.pose = pose
        self.odom_publisher.publish(odom_msg)

        # Save for transforms.json
        filename = f"{self.output_folder_}/image{self.image_counter:06d}.jpg"
        PILImage.fromarray(image_left_tensor.squeeze().cpu().permute(1, 2, 0).numpy()[:, :, ::-1].astype(np.uint8)).save(filename)
        frame_dat = {'transform_matrix': posemat[:3, :].tolist(), 'file_path': filename}
        self.frames.append(frame_dat)
        print(f"Processed frame {self.image_counter}")

    def save_reconstruction(self, save_path):
        if hasattr(self.droid, "video2"):
            video = self.droid.video2
        else:
            video = self.droid.video

        t = video.counter.value
        save_data = {
            "tstamps": video.tstamp[:t].cpu(),
            "images": video.images[:t].cpu(),
            "disps": video.disps_up[:t].cpu(),
            "poses": video.poses[:t].cpu(),
            "intrinsics": video.intrinsics[:t].cpu()
        }

        torch.save(save_data, save_path)
        print(f"Saved .pth to {save_path}")

    def shutdown(self):
        print("Starting reconstruction save...")
        if self.droid is None:
            return

        # terminate droid
        del self.droid.frontend
        # Global Bundle Adjustment
        print("Performing Global BA...")
        torch.cuda.empty_cache()
        self.droid.backend(7)

        torch.cuda.empty_cache()
        self.droid.backend(12)
        
        # Update poses
        video = self.droid.video
        poses = video.poses[:video.counter.value].cpu().numpy()
        tstamps = video.tstamp[:video.counter.value].cpu().numpy()
        
        # Save Trajectory (TUM Format)
        traj_path = os.path.join(self.output_folder_, 'stamped_traj_estimate.txt')
        print(f"Saving trajectory to {traj_path}...")
        with open(traj_path, 'w') as f:
            for i in range(len(poses)):
                # pose is [tx, ty, tz, qx, qy, qz, qw]
                p = poses[i]
                timestamp = tstamps[i]
                f.write(f"{timestamp} {p[0]} {p[1]} {p[2]} {p[3]} {p[4]} {p[5]} {p[6]}\n")

        # Save transforms.json (updated with optimized poses)
        print("Saving transforms.json...")
        self.frames = []
        for i in range(len(poses)):
            # Need to re-compute matrix from optimized pose
            posemat = self.xyzquat2mat(poses[i])
            filename = f"{self.output_folder_}/image{i+1:06d}.jpg" # Approximation of filename
            frame_dat = {'transform_matrix': posemat[:3, :].tolist(), 'file_path': filename}
            self.frames.append(frame_dat)
            
        self.cam_params['frames'] = self.frames
        with open(self.json_file_path_, "w") as json_file:
            json.dump(self.cam_params, json_file, cls=NumpyEncoder, indent=4)

        # Save .pth
        self.save_reconstruction(os.path.join(self.output_folder_, 'reconstruction.pth'))

        # # TODO Save .ply
        # ply_path = os.path.join(self.output_folder_, 'reconstruction.ply')

        print("Save complete.")


def main(mainargs=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--t0", default=0, type=int, help="starting frame")
    parser.add_argument("--stride", default=1, type=int, help="frame stride")

    parser.add_argument("--weights", default="droid.pth")
    parser.add_argument("--buffer", type=int, default=512)
    parser.add_argument("--image_size", default=[240, 320])
    parser.add_argument("--disable_vis", action="store_true")

    parser.add_argument("--beta", type=float, default=0.3, help="weight for translation / rotation components of flow")
    parser.add_argument("--filter_thresh", type=float, default=2, help="how much motion before considering new keyframe")
    parser.add_argument("--warmup", type=int, default=4, help="number of warmup frames")
    parser.add_argument("--keyframe_thresh", type=float, default=3, help="threshold to create a new keyframe")
    parser.add_argument("--frontend_thresh", type=float, default=16.0, help="add edges between frames whithin this distance")
    parser.add_argument("--frontend_window", type=int, default=50, help="frontend optimization window")
    parser.add_argument("--frontend_radius", type=int, default=2, help="force edges between frames within radius")
    parser.add_argument("--frontend_nms", type=int, default=1, help="non-maximal supression of edges")

    parser.add_argument("--backend_thresh", type=float, default=22.0)
    parser.add_argument("--backend_radius", type=int, default=2)
    parser.add_argument("--backend_nms", type=int, default=3)
    parser.add_argument("--upsample", action="store_true")
    parser.add_argument("--reconstruction_path", help="path to saved reconstruction")
    parser.add_argument("--datapath", default="/tmp", help="This should be ignored")
    args = parser.parse_args()
    
    # Enable Stereo
    args.stereo = True

    torch.multiprocessing.set_start_method('spawn')
    rclpy.init(args=mainargs)

    node = DroidNode(args)
    try:
        rclpy.spin(node)  # Keep the node alive
    except KeyboardInterrupt:
        node.shutdown()
        print("Exiting...")
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
