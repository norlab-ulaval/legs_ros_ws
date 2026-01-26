#!/usr/bin/env python3

from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from geometry_msgs.msg import Pose, Point, Quaternion
from nav_msgs.msg import Odometry
from rclpy.node import Node
from cv_bridge import CvBridge  # Needed for converting between ROS Image messages and OpenCV images
import sys
from scipy.spatial.transform import Rotation as R
import os
from ament_index_python.packages import get_package_share_directory

# Define paths
DROID_SLAM_ROOT = '/opt/DROID-SLAM'
DROID_SLAM_LIB  = os.path.join(DROID_SLAM_ROOT, 'droid_slam')

if os.path.exists(DROID_SLAM_ROOT):
    # 1. Add ROOT to path (so 'droid_backends' .so file can be found)
    sys.path.insert(0, DROID_SLAM_ROOT)
    
    # 2. Add LIB folder to path (so 'droid_net', 'depth_video' etc. can be found)
    # This fixes the "No module named droid_net" error
    sys.path.insert(0, DROID_SLAM_LIB)
    
    print(f"Force-loaded DROID-SLAM from: {DROID_SLAM_ROOT}")
else:
    print(f"WARNING: {DROID_SLAM_ROOT} not found")
    # ... fallback logic ...

# ...

try:
    # 3. Import DIRECTLY from 'droid', not 'droid_slam.droid'
    # Because we added the inner folder to sys.path, 'droid' is now a top-level module.
    # This also prevents Python from accidentally grabbing the version in /ros2_ws/src/...
    from droid import Droid
    print("Successfully imported 'Droid' class")

except ImportError as e:
    print(f"Failed to import Droid: {e}")
    raise e

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
    def __init__(self):
        super().__init__('droid_node')
        
        # Declare parameters
        self.declare_parameter('weights', '')
        self.declare_parameter('image_size', [344, 560])
        self.declare_parameter('disable_vis', False)
        self.declare_parameter('buffer', 512)
        self.declare_parameter('upsample', True)
        self.declare_parameter('t0', 0)
        self.declare_parameter('stride', 1)
        self.declare_parameter('beta', 0.3)
        self.declare_parameter('filter_thresh', 2.0)
        self.declare_parameter('warmup', 4)
        self.declare_parameter('keyframe_thresh', 4.0)
        self.declare_parameter('frontend_thresh', 16.0)
        self.declare_parameter('frontend_window', 50)
        self.declare_parameter('frontend_radius', 2)
        self.declare_parameter('frontend_nms', 1)
        self.declare_parameter('backend_thresh', 22.0)
        self.declare_parameter('backend_radius', 2)
        self.declare_parameter('backend_nms', 3)
        self.declare_parameter('reconstruction_path', '')
        self.declare_parameter('datapath', '/tmp')
        self.declare_parameter('storage_path', '')

        # Create args object to mimic argparse namespace
        class DroidArgs:
            pass
        self.args = DroidArgs()

        # Retrieve parameters
        weights = self.get_parameter('weights').get_parameter_value().string_value
        if not weights:
            try:
                weights = os.path.join(get_package_share_directory('droid_slam_ros'), 'droid.pth')
            except Exception:
                weights = "droid.pth"
        
        self.args.weights = weights
        self.args.image_size = self.get_parameter('image_size').get_parameter_value().integer_array_value
        if hasattr(self.args.image_size, 'tolist'):
             self.args.image_size = self.args.image_size.tolist()

        self.args.disable_vis = self.get_parameter('disable_vis').get_parameter_value().bool_value
        self.args.buffer = self.get_parameter('buffer').get_parameter_value().integer_value
        self.args.upsample = self.get_parameter('upsample').get_parameter_value().bool_value
        self.args.t0 = self.get_parameter('t0').get_parameter_value().integer_value
        self.args.stride = self.get_parameter('stride').get_parameter_value().integer_value
        self.args.beta = self.get_parameter('beta').get_parameter_value().double_value
        self.args.filter_thresh = self.get_parameter('filter_thresh').get_parameter_value().double_value
        self.args.warmup = self.get_parameter('warmup').get_parameter_value().integer_value
        self.args.keyframe_thresh = self.get_parameter('keyframe_thresh').get_parameter_value().double_value
        self.args.frontend_thresh = self.get_parameter('frontend_thresh').get_parameter_value().double_value
        self.args.frontend_window = self.get_parameter('frontend_window').get_parameter_value().integer_value
        self.args.frontend_radius = self.get_parameter('frontend_radius').get_parameter_value().integer_value
        self.args.frontend_nms = self.get_parameter('frontend_nms').get_parameter_value().integer_value
        self.args.backend_thresh = self.get_parameter('backend_thresh').get_parameter_value().double_value
        self.args.backend_radius = self.get_parameter('backend_radius').get_parameter_value().integer_value
        self.args.backend_nms = self.get_parameter('backend_nms').get_parameter_value().integer_value
        self.args.reconstruction_path = self.get_parameter('reconstruction_path').get_parameter_value().string_value
        self.args.datapath = self.get_parameter('datapath').get_parameter_value().string_value
        self.output_folder = self.get_parameter('storage_path').get_parameter_value().string_value
        
        self.args.stereo = True
        
        self.droid = Droid(self.args)

        self.cam_transform = np.diag([1, -1, -1, 1])
        
        # Initialize TF2 Buffer and Listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Initialize ROS2 Publisher and Subscriber
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
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)            

        self.cam_params = {}
        self.bridge = CvBridge()
        self.baseline = None # TODO baseline is actually hardcoded in DROID-SLAM

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
        # TODO log here
        self.get_logger().info("Received Left Camera Info", once=True)

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
        # TODO log here
        self.get_logger().info("Received Right Camera Info", once=True)

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
        start_time = time.time()
        # # Get baseline from TF if not already set
        # if self.baseline is None:
        #     try:
        #         trans = self.tf_buffer.lookup_transform('zedx_left', 'zedx_right', rclpy.time.Time())
        #         # Baseline is typically the Euclidean distance, mainly along -x in camera frame for right cam
        #         # But here we just need the magnitude if we are converting disparity/stereo
        #         # Wait, DroidSLAM expects stereo pairs. We need the intrinsics [fx, fy, cx, cy]
        #         # and usually assumes rectified stereo with horizontal baseline.
        #         self.baseline = abs(trans.transform.translation.x) 
        #         self.get_logger().info(f"Baseline found: {self.baseline}")
        #     except Exception as e:
        #         self.get_logger().error(f"Could not get baseline: {e}")
        #         return

        if self.cam_params.get("left") is None or self.cam_params.get("right") is None:
            self.get_logger().info("Waiting for camera info", once=True)
            return

        # Convert ROS Image messages to OpenCV images
        cv_left = self.bridge.imgmsg_to_cv2(left_msg, desired_encoding='bgr8')
        cv_right = self.bridge.imgmsg_to_cv2(right_msg, desired_encoding='bgr8')
        
        t = left_msg.header.stamp.sec + left_msg.header.stamp.nanosec * 1e-9

        # Undistort images

        cv_left = cv2.undistort(cv_left, self.cam_params['left']['K'], self.cam_params['left']['D'])
        cv_right = cv2.undistort(cv_right, self.cam_params['right']['K'], self.cam_params['right']['D'])

        # Prepare inputs for Droid

        h0, w0, _ = cv_left.shape
        h1 = self.args.image_size[0]
        w1 = self.args.image_size[1]

        cv_left = cv2.resize(cv_left, (w1, h1))
        cv_left = cv_left[:h1-h1%8, :w1-w1%8]
        cv_right = cv2.resize(cv_right, (w1, h1))
        cv_right = cv_right[:h1-h1%8, :w1-w1%8]

        image_left_tensor = torch.as_tensor(cv_left).permute(2, 0, 1)
        image_right_tensor = torch.as_tensor(cv_right).permute(2, 0, 1)

        
        stereo_image = torch.stack([image_left_tensor, image_right_tensor])

        K_l = self.cam_params['left']['K']
        fx, fy, cx, cy = K_l[0,0], K_l[1,1], K_l[0,2], K_l[1,2]
        intrinsics = torch.as_tensor([
            fx, fy, cx, cy
        ])
        intrinsics[0::2] *= (w1 / w0)
        intrinsics[1::2] *= (h1 / h0)



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

        process_time = time.time() - start_time
        self.get_logger().info(f"Processed frame {self.image_counter} in {process_time:.4f}s")

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
        self.get_logger().info(f"Saved .pth to {save_path}")

    def shutdown(self):
        self.get_logger().info("Starting reconstruction save...")
        if self.droid is None:
            return

        # terminate droid
        del self.droid.frontend
        # Global Bundle Adjustment
        self.get_logger().info("Performing Global BA...")
        torch.cuda.empty_cache()
        self.droid.backend(7)

        torch.cuda.empty_cache()
        self.droid.backend(12)
        
        # Update poses
        video = self.droid.video
        poses_w2c = video.poses[:video.counter.value].cpu().numpy()
        tstamps = video.tstamp[:video.counter.value].cpu().numpy()
        
        # Calculate C2W poses for trajectory file (to match demo.py output)
        poses_c2w = SE3(video.poses[:video.counter.value]).inv().data.cpu().numpy()
        
        # Save Trajectory (TUM Format) using C2W poses
        traj_path = os.path.join(self.output_folder, 'trajectory.txt')
        self.get_logger().info(f"Saving trajectory to {traj_path}...")
        with open(traj_path, 'w') as f:
            for i in range(len(poses_c2w)):
                # pose is [tx, ty, tz, qx, qy, qz, qw]
                p = poses_c2w[i]
                timestamp = tstamps[i]
                f.write(f"{timestamp} {p[0]} {p[1]} {p[2]} {p[3]} {p[4]} {p[5]} {p[6]}\n")
        

        # Save .pth
        self.save_reconstruction(os.path.join(self.output_folder, 'reconstruction.pth'))

        # # TODO Save .ply
        # ply_path = os.path.join(self.output_folder, 'reconstruction.ply')

        self.get_logger().info("Save complete.")


def main(mainargs=None):
    torch.multiprocessing.set_start_method('spawn')
    rclpy.init(args=mainargs)

    node = DroidNode()
    try:
        rclpy.spin(node)  # Keep the node alive
    except KeyboardInterrupt:
        node.shutdown()
        node.get_logger().info("Exiting...")
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
