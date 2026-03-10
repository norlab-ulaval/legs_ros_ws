#!/usr/bin/env python3

import os
import signal
import sys

import numpy as np
from ament_index_python.packages import get_package_share_directory
from cv_bridge import (
    CvBridge,  # Needed for converting between ROS Image messages and OpenCV images
)
from geometry_msgs.msg import Point, Pose, Quaternion
from nav_msgs.msg import Odometry
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import CameraInfo, CompressedImage, Image

# Define paths
try:
    share_dir = get_package_share_directory("droid_slam_ros")
except Exception as e:
    print(f"Could not find share directory: {e}")
    share_dir = os.getcwd()

# Search for droid-slam directory (hyphen or underscore)
candidate_roots = [
    os.path.join(share_dir, "droid-slam"),  # Expected in installed share or local
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "droid-slam"
    ),  # Relative to script
    "droid-slam",  # Relative to CWD
    "droid_slam",
]

DROID_SLAM_ROOT = None
for root in candidate_roots:
    if os.path.isdir(root) and os.path.isdir(os.path.join(root, "droid_slam")):
        DROID_SLAM_ROOT = os.path.abspath(root)
        break

if DROID_SLAM_ROOT:
    DROID_SLAM_LIB = os.path.join(DROID_SLAM_ROOT, "droid_slam")

    # 1. Add ROOT to path (so 'droid_backends' .so file can be found)
    if DROID_SLAM_ROOT not in sys.path:
        sys.path.insert(0, DROID_SLAM_ROOT)

    # 2. Add LIB folder to path (so 'droid_net', 'depth_video' etc. can be found)
    if DROID_SLAM_LIB not in sys.path:
        sys.path.insert(0, DROID_SLAM_LIB)

    print(f"Force-loaded DROID-SLAM from: {DROID_SLAM_ROOT}")
else:
    print(f"WARNING: DROID-SLAM source not found. Checked: {candidate_roots}")

try:
    # 3. Import DIRECTLY from 'droid', not 'droid_slam.droid'
    # Because we added the inner folder to sys.path, 'droid' is now a top-level module.
    from droid import Droid

    print("Successfully imported 'Droid' class")
except (ModuleNotFoundError, ImportError) as e:
    print(f"Failed to import Droid: {e}")
    raise e

import argparse
import glob
import json
import time

import cv2
import droid_backends
import geom.projective_ops as pops
import lietorch
import message_filters
import numpy as np
import rclpy
import tf2_ros
import torch
import torch.nn.functional as F
from lietorch import SE3
from PIL import Image as PILImage
from sensor_msgs.msg import Image as ROSImage
from tf2_geometry_msgs import do_transform_pose
from tqdm import tqdm


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


class DroidNode(Node):
    def __init__(self):
        super().__init__("droid_node")

        # Declare parameters
        self.declare_parameter("weights", "")
        self.declare_parameter("image_size", [344, 560])
        self.declare_parameter("disable_vis", False)
        self.declare_parameter("buffer", 512)
        self.declare_parameter("upsample", True)
        self.declare_parameter("t0", 0)
        self.declare_parameter("stride", 1)
        self.declare_parameter("beta", 0.3)
        self.declare_parameter("filter_thresh", 2.0)
        self.declare_parameter("warmup", 4)
        self.declare_parameter("keyframe_thresh", 4.0)
        self.declare_parameter("frontend_thresh", 16.0)
        self.declare_parameter("frontend_window", 50)
        self.declare_parameter("frontend_radius", 2)
        self.declare_parameter("frontend_nms", 1)
        self.declare_parameter("backend_thresh", 22.0)
        self.declare_parameter("backend_radius", 2)
        self.declare_parameter("backend_nms", 3)
        self.declare_parameter("reconstruction_path", "")
        self.declare_parameter("datapath", "/tmp")
        self.declare_parameter("storage_path", "")
        self.declare_parameter("pgo", True)

        # Create args object to mimic argparse namespace
        class DroidArgs:
            pass

        self.args = DroidArgs()

        # Retrieve parameters
        weights = self.get_parameter("weights").get_parameter_value().string_value
        if not weights:
            try:
                weights = os.path.join(
                    get_package_share_directory("droid_slam_ros"), "droid.pth"
                )
            except Exception:
                weights = "droid.pth"

        self.args.weights = weights
        self.args.image_size = (
            self.get_parameter("image_size").get_parameter_value().integer_array_value
        )
        if hasattr(self.args.image_size, "tolist"):
            self.args.image_size = self.args.image_size.tolist()

        self.args.disable_vis = (
            self.get_parameter("disable_vis").get_parameter_value().bool_value
        )
        self.args.buffer = (
            self.get_parameter("buffer").get_parameter_value().integer_value
        )
        self.args.upsample = (
            self.get_parameter("upsample").get_parameter_value().bool_value
        )
        self.args.t0 = self.get_parameter("t0").get_parameter_value().integer_value
        self.args.stride = (
            self.get_parameter("stride").get_parameter_value().integer_value
        )
        self.args.beta = self.get_parameter("beta").get_parameter_value().double_value
        self.args.filter_thresh = (
            self.get_parameter("filter_thresh").get_parameter_value().double_value
        )
        self.args.warmup = (
            self.get_parameter("warmup").get_parameter_value().integer_value
        )
        self.args.keyframe_thresh = (
            self.get_parameter("keyframe_thresh").get_parameter_value().double_value
        )
        self.args.frontend_thresh = (
            self.get_parameter("frontend_thresh").get_parameter_value().double_value
        )
        self.args.frontend_window = (
            self.get_parameter("frontend_window").get_parameter_value().integer_value
        )
        self.args.frontend_radius = (
            self.get_parameter("frontend_radius").get_parameter_value().integer_value
        )
        self.args.frontend_nms = (
            self.get_parameter("frontend_nms").get_parameter_value().integer_value
        )
        self.args.backend_thresh = (
            self.get_parameter("backend_thresh").get_parameter_value().double_value
        )
        self.args.backend_radius = (
            self.get_parameter("backend_radius").get_parameter_value().integer_value
        )
        self.args.backend_nms = (
            self.get_parameter("backend_nms").get_parameter_value().integer_value
        )
        self.args.reconstruction_path = (
            self.get_parameter("reconstruction_path").get_parameter_value().string_value
        )
        self.args.datapath = (
            self.get_parameter("datapath").get_parameter_value().string_value
        )
        self.output_folder = (
            self.get_parameter("storage_path").get_parameter_value().string_value
        )

        self.pgo = self.get_parameter("pgo").get_parameter_value().bool_value

        self.get_logger().info("======Initializing DROID-SLAM=======", once=True)
        self.get_logger().info("Loading parameters...", once=True)
        self.get_logger().info(f"disable_vis: {self.args.disable_vis}", once=True)
        self.get_logger().info(f"buffer: {self.args.buffer}", once=True)
        self.get_logger().info(f"upsample: {self.args.upsample}", once=True)
        self.get_logger().info(f"t0: {self.args.t0}", once=True)
        self.get_logger().info(f"stride: {self.args.stride}", once=True)
        self.get_logger().info(f"beta: {self.args.beta}", once=True)
        self.get_logger().info(f"filter_thresh: {self.args.filter_thresh}", once=True)
        self.get_logger().info(f"warmup: {self.args.warmup}", once=True)
        self.get_logger().info(
            f"keyframe_thresh: {self.args.keyframe_thresh}", once=True
        )
        self.get_logger().info(
            f"frontend_thresh: {self.args.frontend_thresh}", once=True
        )
        self.get_logger().info(
            f"frontend_window: {self.args.frontend_window}", once=True
        )
        self.get_logger().info(
            f"frontend_radius: {self.args.frontend_radius}", once=True
        )
        self.get_logger().info(f"frontend_nms: {self.args.frontend_nms}", once=True)
        self.get_logger().info(f"backend_thresh: {self.args.backend_thresh}", once=True)
        self.get_logger().info(f"backend_radius: {self.args.backend_radius}", once=True)
        self.get_logger().info(f"backend_nms: {self.args.backend_nms}", once=True)
        self.get_logger().info(
            f"reconstruction_path: {self.args.reconstruction_path}", once=True
        )
        self.get_logger().info(f"datapath: {self.args.datapath}", once=True)
        self.get_logger().info(f"storage_path: {self.output_folder}", once=True)

        self.timestamps_path = os.path.join(self.output_folder, "timestamps.txt")

        self.args.stereo = True

        self.droid = Droid(self.args)

        self.cam_transform = np.diag([1, -1, -1, 1])
        self.stride_ctr = 0

        # Initialize TF2 Buffer and Listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Initialize ROS2 Publisher and Subscriber
        self.odom_publisher = self.create_publisher(Odometry, "estimated_odom", 10)

        # Subscriptions
        # Use absolute paths for input topics to ignore node namespace
        self.left_rect_sub = message_filters.Subscriber(
            self, ROSImage, "/zedx/left/image_rect"
        )
        self.right_rect_sub = message_filters.Subscriber(
            self, ROSImage, "/zedx/right/image_rect"
        )

        # Exact Time Sync for stereo pairs
        self.ts = message_filters.TimeSynchronizer(
            [self.left_rect_sub, self.right_rect_sub], 10
        )
        self.ts.registerCallback(self.image_callback)

        self.intr_sub_left = self.create_subscription(
            CameraInfo, "/zedx/left/camera_info", self.cam_intr_left_cb, 1
        )
        self.intr_sub_right = self.create_subscription(
            CameraInfo, "/zedx/right/camera_info", self.cam_intr_right_cb, 1
        )

        self.image_counter = 0
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        self.cam_params = {}
        self.bridge = CvBridge()
        self.baseline = None
        self.process_times = []

        # Open in write mode once to clear/create the file
        with open(self.timestamps_path, "w") as f:
            f.write("timestamp publish_pose video_counter image_counter\n")

    def cam_intr_left_cb(self, msg):
        if "w" in self.cam_params:
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
        self.get_logger().info("Received Left Camera Info", once=True)

    def cam_intr_right_cb(self, msg):
        if "w" in self.cam_params:
            return

        K = msg.k.reshape(3, 3)
        params = {
            "w": msg.width,
            "h": msg.height,
            "K": K,
            "D": np.array(msg.d),
        }
        self.cam_params["right"] = params
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
        # Get baseline from TF if not already set
        # We are getting better results with the default baseline (0.1 m)
        # and rescaling afterwards
        self.baseline = 0.1
        if self.baseline is None:
            try:
                from rclpy.duration import Duration

                trans = self.tf_buffer.lookup_transform(
                    "zedx_left",
                    "zedx_right",
                    left_msg.header.stamp,
                    timeout=Duration(seconds=0.1),
                )
                self.baseline = 0.1 / abs(trans.transform.translation.x)
                self.get_logger().info(f"Baseline found: {self.baseline}")
            except Exception as e:
                self.get_logger().error(f"Could not get baseline: {e}")
                return

        if self.cam_params.get("left") is None or self.cam_params.get("right") is None:
            self.get_logger().info("Waiting for camera info", once=True)
            return

        timestamp = left_msg.header.stamp.sec + left_msg.header.stamp.nanosec * 1e-9

        if self.stride_ctr % self.args.stride != 0:
            self.get_logger().info(f"Skipping image {timestamp} due to stride")
            self.stride_ctr += 1
            return
        self.stride_ctr += 1

        # Convert ROS Image messages to OpenCV images
        cv_left = self.bridge.imgmsg_to_cv2(left_msg, desired_encoding="bgr8")
        cv_right = self.bridge.imgmsg_to_cv2(right_msg, desired_encoding="bgr8")

        # Undistort images

        # cv_left = cv2.undistort(cv_left, self.cam_params['left']['K'], self.cam_params['left']['D'])
        # cv_right = cv2.undistort(cv_right, self.cam_params['right']['K'], self.cam_params['right']['D'])

        # Prepare inputs for Droid

        h0, w0, _ = cv_left.shape
        h1 = self.args.image_size[0]
        w1 = self.args.image_size[1]

        cv_left = cv2.resize(cv_left, (w1, h1))
        cv_left = cv_left[: h1 - h1 % 8, : w1 - w1 % 8]
        cv_right = cv2.resize(cv_right, (w1, h1))
        cv_right = cv_right[: h1 - h1 % 8, : w1 - w1 % 8]

        image_left_tensor = torch.as_tensor(cv_left).permute(2, 0, 1)
        image_right_tensor = torch.as_tensor(cv_right).permute(2, 0, 1)

        stereo_image = torch.stack([image_left_tensor, image_right_tensor])

        K_l = self.cam_params["left"]["K"]
        fx, fy, cx, cy = K_l[0, 0], K_l[1, 1], K_l[0, 2], K_l[1, 2]
        intrinsics = torch.as_tensor([fx, fy, cx, cy, self.baseline])
        intrinsics[0:4:2] *= w1 / w0
        intrinsics[1:4:2] *= h1 / h0

        self.droid.track(timestamp, stereo_image, depth=None, intrinsics=intrinsics)
        publish_pose = self.droid.video.counter.value != self.image_counter

        with open(self.timestamps_path, "a") as f:
            f.write(
                f"{timestamp} {int(publish_pose)} {self.droid.video.counter.value} {self.image_counter}\n"
            )

        if publish_pose:
            self.image_counter += 1

            # Get latest pose
            pose_vec = (
                self.droid.video.poses[self.droid.video.counter.value - 1].cpu().numpy()
            )
            pose_c2w = (
                SE3(torch.from_numpy(pose_vec).unsqueeze(0)).inv().data[0].cpu().numpy()
            )
            pose = Pose(
                position=Point(
                    x=float(pose_c2w[0]), y=float(pose_c2w[1]), z=float(pose_c2w[2])
                ),
                orientation=Quaternion(
                    x=float(pose_c2w[3]),
                    y=float(pose_c2w[4]),
                    z=float(pose_c2w[5]),
                    w=float(pose_c2w[6]),
                ),
            )

            # Publish Odometry
            odom_msg = Odometry()
            odom_msg.header = left_msg.header
            odom_msg.header.frame_id = "map"  # or "odom"
            odom_msg.child_frame_id = "zedx_left"  # or camera frame
            odom_msg.pose.pose = pose
            self.odom_publisher.publish(odom_msg)

        process_time = time.time() - start_time
        self.process_times.append(process_time)
        self.get_logger().info(
            f"Processed frame {self.stride_ctr} | Keyframe: {self.image_counter}\nDuration: {process_time:.2f}s | Mean: {np.mean(self.process_times)}s | Max: {np.max(self.process_times)}s"
        )

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
            "intrinsics": video.intrinsics[:t].cpu(),
        }

        torch.save(save_data, save_path)
        self.get_logger().info(f"Saved .pth to {save_path}")

    def shutdown(self):
        # Ignore SIGINT to ensure shutdown completes
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        if self.pgo:
            print("Starting reconstruction save... (SIGINT/Ctrl+C is now ignored)")
            if self.droid is None:
                return

            start = time.time()
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
            tstamps = video.tstamp[: video.counter.value].cpu().numpy()

            # Calculate C2W poses for trajectory file (to match demo.py output)
            poses_c2w = SE3(video.poses[: video.counter.value]).inv().data.cpu().numpy()

            # Save Trajectory (TUM Format) using C2W poses
            traj_path = os.path.join(self.output_folder, "trajectory.txt")
            print(f"Saving trajectory to {traj_path}...")
            with open(traj_path, "w") as f:
                for i in range(len(poses_c2w)):
                    # pose is [tx, ty, tz, qx, qy, qz, qw]
                    p = poses_c2w[i]
                    timestamp = tstamps[i]
                    f.write(
                        f"{timestamp} {p[0]} {p[1]} {p[2]} {p[3]} {p[4]} {p[5]} {p[6]}\n"
                    )

            print(f"PGO completed in {time.time() - start:.2f}s")

            # # TODO Save .ply
            # ply_path = os.path.join(self.output_folder, 'reconstruction.ply')

            print("Save complete.")


def main(mainargs=None):
    torch.multiprocessing.set_start_method("spawn")
    rclpy.init(args=mainargs)

    node = DroidNode()
    try:
        rclpy.spin(node)  # Keep the node alive
    except KeyboardInterrupt:
        node.shutdown()
        node.get_logger().info("Exiting...")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
