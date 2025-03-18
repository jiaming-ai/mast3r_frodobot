import os
import time
import threading
import pickle
import shutil
import tempfile

import numpy as np
from mast3r.cloud_opt.sparse_ga import sparse_global_alignment

from mast3r.model import AsymmetricMASt3R
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images
import copy
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
import cv2
import logging
import torch
logging.basicConfig(level=logging.INFO)

def est_pose(ride_path, start_idx=0, end_idx=None, interval=1, 
             visualize=True, device="cuda", overwrite=False, model=None, show_scene=False):
    """
    Load a ride from a directory and reconstruct the scene.
    
    Args:
        ride_path: Path to the ride directory
        start_idx: Starting index for image selection
        end_idx: Ending index for image selection (None means all images)
        interval: Interval between selected images
        visualize: Whether to visualize odometry comparison
        
    Returns:
        scene: Reconstructed scene
    """
    # Device and model settings
    image_size = 512
    silent = False

    # Load images
    image_dir = os.path.join(ride_path, 'img')

    if not os.path.exists(image_dir):
        logging.error(f"Image directory {image_dir} does not exist")
        return
    
    all_img_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) 
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if end_idx is None:
        end_idx = len(all_img_files)
    end_idx = min(end_idx, len(all_img_files))
    recon_idxs = np.arange(start_idx, end_idx, interval)

    filelist = []
    for i, idx in enumerate(recon_idxs):
        filelist.append(os.path.join(ride_path, 'img', f'{idx}.jpg'))
    
    odom_file = os.path.join(ride_path, 'traj_data.pkl')
    with open(odom_file, 'rb') as f:
        odom_data = pickle.load(f)
   
    # Get delta odom
    delta_odom = []
    for i in range(len(recon_idxs) - 1):  # Exclude the last index
        idx_current = recon_idxs[i]
        idx_next = recon_idxs[i + 1]
        
        # Current pose
        x_current = odom_data['pos'][idx_current][0]
        y_current = odom_data['pos'][idx_current][1]
        theta_current = odom_data['yaw'][idx_current]
        
        # Next pose in world frame
        x_next = odom_data['pos'][idx_next][0]
        y_next = odom_data['pos'][idx_next][1]
        theta_next = odom_data['yaw'][idx_next]
        
        # Global displacement
        dx_global = x_next - x_current
        dy_global = y_next - y_current
        
        # Convert global displacement to local frame
        cos_theta = np.cos(theta_current)
        sin_theta = np.sin(theta_current)
        
        # Rotation matrix from world to local frame
        dx_local = cos_theta * dx_global + sin_theta * dy_global
        dy_local = -sin_theta * dx_global + cos_theta * dy_global
        
        # Angular change (normalized to [-pi, pi))
        dtheta = theta_next - theta_current
        dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi
        
        delta_odom.append({
            "from": idx_current,
            "to": idx_next,
            "dx": dx_local,
            "dy": dy_local,
            "dtheta": dtheta,
            "timestamp_from": odom_data['timestamps'][idx_current],
            "timestamp_to": odom_data['timestamps'][idx_next],
        })
        
        logging.debug(f"From {idx_current} to {idx_next}")
        logging.debug(f"delta_odom: {delta_odom[-1]['dx']:.2f}, {delta_odom[-1]['dy']:.2f}, {delta_odom[-1]['dtheta']:.2f}")
        logging.debug(f"distance: {np.sqrt(delta_odom[-1]['dx']**2 + delta_odom[-1]['dy']**2):.2f}")
        logging.debug("--------------------------------")
    
     # get compass data
    logging.debug(f"Getting compass data")
    compass_file = os.path.join(ride_path, 'compass_calibrated.pkl')
    with open(compass_file, 'rb') as f:
        compass_data = pickle.load(f)
    
    for delta_odom_item in delta_odom:
        timestamp_from = delta_odom_item['timestamp_from']
        timestamp_to = delta_odom_item['timestamp_to']

        head_idx_from = np.argmin(np.abs(compass_data[:,1] - timestamp_from))
        head_idx_to = np.argmin(np.abs(compass_data[:,1] - timestamp_to))

        # replace the dtheta with delta heading from compass
        d_heading = compass_data[head_idx_to, 0] - compass_data[head_idx_from, 0]
        # 3. normalize to [-pi, pi)
        d_heading = (d_heading + np.pi) % (2 * np.pi) - np.pi

        delta_odom_item['dtheta'] = d_heading
        logging.debug(f"d_heading from {delta_odom_item['from']} to {delta_odom_item['to']}: {d_heading:.2f}")
    
    # Set parameters
    outdir = os.path.join(ride_path, 'reconstruction')
    os.makedirs(outdir, exist_ok=True)
    
    # Optimization parameters
    optim_level = 'refine+depth'
    lr1 = 0.07
    niter1 = 500
    lr2 = 0.014
    niter2 = 200
    min_conf_thr = 1.5
    matching_conf_thr = 5.0
    
    # Scene graph parameters
    scenegraph_type = 'swin'
    winsize = 2
    win_cyclic = False
    refid = 0
    
    # Visualization parameters
    as_pointcloud = True
    mask_sky = False
    clean_depth = True
    transparent_cams = False
    cam_size = 0.2
    TSDF_thresh = 0.0
    shared_intrinsics = True 
    
    # Load and process images
    imgs = load_images(filelist, size=image_size, verbose=not silent)
    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]['idx'] = 1
        filelist = [filelist[0], filelist[0] + '_2']

    # Create scene graph
    scene_graph_params = [scenegraph_type]
    if scenegraph_type in ["swin", "logwin"]:
        scene_graph_params.append(str(winsize))
    elif scenegraph_type == "oneref":
        scene_graph_params.append(str(refid))
    if scenegraph_type in ["swin", "logwin"] and not win_cyclic:
        scene_graph_params.append('noncyclic')
    scene_graph = '-'.join(scene_graph_params)
    
    # Create image pairs
    pairs = make_pairs(imgs, scene_graph=scene_graph, prefilter=None, symmetrize=True)
    if optim_level == 'coarse':
        niter2 = 0
        
    # Set up cache directory
    cache_dir = os.path.join(outdir, 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    
    cam2w_file = os.path.join(ride_path, f'cam2w_{start_idx}_{end_idx}.npy')
    if overwrite or not os.path.exists(cam2w_file):
        logging.info(f"Estimating pose for {ride_path}, {start_idx} to {end_idx} on {device}")
    
        # Run sparse global alignment
        # Load model
        if model is None:
            model_name = "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
            weights_path = "naver/" + model_name
            model = AsymmetricMASt3R.from_pretrained(weights_path).to(device)
        scene = sparse_global_alignment(filelist, pairs, cache_dir,
                                        model, lr1=lr1, niter1=niter1, lr2=lr2, niter2=niter2, device=device,
                                        opt_depth='depth' in optim_level, shared_intrinsics=shared_intrinsics,
                                        matching_conf_thr=matching_conf_thr, odometry_data=delta_odom, lora_depth=dict(k=96, gamma=15, min_norm=.5),
                                        odometry_weight=0.4,scale_weight=1)
        if show_scene:
            scene.show()

        # save cam2w
        cam2w = scene.cam2w.cpu().numpy()
        np.save(cam2w_file, cam2w)

        shutil.rmtree(outdir)

    else:
        logging.info(f"Loading pose for {ride_path}, {start_idx} to {end_idx} from {cam2w_file}")

        # Visualize odometry comparison if requested
        if visualize:
            cam2w = np.load(cam2w_file)
            recon_odom = extract_odometry_from_cam2w(cam2w)
            print(recon_odom)

            selected_pos = odom_data['pos'][recon_idxs]
            selected_yaw = odom_data['yaw'][recon_idxs]
            selected_odom_data = {
                'pos': selected_pos,
                'yaw': selected_yaw
            }
            vis_outdir = os.path.join(outdir, 'odometry_visualization')
            visualize_odometry_comparison(
                original_odom_data=selected_odom_data,
                recon_odom_data=recon_odom,
                image_files=filelist,
                output_dir=vis_outdir
            )
        
    
    return cam2w

def extract_odometry_from_cam2w(cam2w_matrices):
    """
    Extract odometry from a list of camera-to-world transformation matrices.
    
    Args:
        cam2w_matrices: List of Nx4x4 camera-to-world transformation matrices
        z_weight: Weight for the z component (default: 1.0)
        
    Returns:
        Dictionary containing:
            - pos: List of [x, y] positions in odometry frame
            - yaw: List of yaw angles in odometry frame
    """
    if len(cam2w_matrices) == 0:
        return {'pos': [], 'yaw': []}
    
    # Initialize lists to store positions and orientations
    positions = []
    yaws = []
    
    # Process each matrix to get positions and orientations in world frame
    for matrix in cam2w_matrices:
        # Extract rotation and translation
        R = matrix[:3, :3]
        t = matrix[:3, 3]
        
        # Convert translation to odometry coordinate system
        # Forward (odom x) is z in OpenCV
        # Left (odom y) is -x in OpenCV
        x = t[2]       # z is forward in OpenCV
        y = -t[0]      # -x is left in OpenCV
        
        # Store position (x, y) in odometry frame
        positions.append([x, y])
        
        # Calculate yaw (rotation around vertical axis)
        # In odometry frame, yaw is rotation from x to y
        # In OpenCV, this corresponds to rotation from z to -x
        
        yaw = np.arctan2(R[0, 2], R[2, 2])
        yaws.append(yaw)
    
    # Make positions and orientations relative to the first frame
    origin_pos = positions[0]
    origin_yaw = yaws[0]
    
    # Create transformation matrix for the origin (first frame)
    cos_origin = np.cos(origin_yaw)
    sin_origin = np.sin(origin_yaw)
    
    # Initialize relative positions and yaws
    relative_positions = []
    relative_yaws = []
    
    for i, (pos, yaw) in enumerate(zip(positions, yaws)):
        # Calculate position relative to origin
        dx = pos[0] - origin_pos[0]
        dy = pos[1] - origin_pos[1]
        
        # Rotate the relative position to the origin's frame
        rel_x = cos_origin * dx + sin_origin * dy
        rel_y = -sin_origin * dx + cos_origin * dy
        
        relative_positions.append([rel_x, rel_y])
        
        # Calculate relative yaw (difference in orientation)
        rel_yaw = yaw - origin_yaw
        rel_yaw = (rel_yaw + np.pi) % (2 * np.pi) - np.pi
        relative_yaws.append(rel_yaw)
    
    return {
        'pos': relative_positions,
        'yaw': relative_yaws
    }

def visualize_odometry_comparison(
    original_odom_data, 
    recon_odom_data, 
    image_files,
    output_dir="./odometry_visualization",
    camera_height=0.561,  # meters above ground
):
    """
    Visualize comparison between original odometry and reconstructed odometry.
    
    Args:
        scene: Reconstructed scene from est_pose
        original_odom_data: Original odometry data dictionary with 'pos' and 'yaw'
        recon_odom_data: Reconstructed odometry data dictionary with 'pos' and 'yaw'
        image_files: List of image file paths
        output_dir: Directory to save visualizations
        camera_height: Height of camera above ground in meters
    """
    # Transform original odometry data to make first frame the origin
    orig_positions = np.array(original_odom_data['pos'])
    orig_yaws = np.array(original_odom_data['yaw'])
    
    # Get origin position and orientation
    origin_pos = orig_positions[0].copy()
    origin_yaw = orig_yaws[0]
    
    # Create transformation matrix for the origin
    cos_origin = np.cos(origin_yaw)
    sin_origin = np.sin(origin_yaw)
    
    # Initialize transformed positions and yaws
    transformed_positions = []
    transformed_yaws = []
    
    for i, (pos, yaw) in enumerate(zip(orig_positions, orig_yaws)):
        # Calculate position relative to origin
        dx = pos[0] - origin_pos[0]
        dy = pos[1] - origin_pos[1]
        
        # Rotate the relative position to the origin's frame
        rel_x = cos_origin * dx + sin_origin * dy
        rel_y = -sin_origin * dx + cos_origin * dy
        
        transformed_positions.append([rel_x, rel_y])
        
        # Calculate relative yaw (difference in orientation)
        rel_yaw = yaw - origin_yaw
        rel_yaw = (rel_yaw + np.pi) % (2 * np.pi) - np.pi
        transformed_yaws.append(rel_yaw)
    
    # Replace original data with transformed data
    transformed_odom_data = {
        'pos': transformed_positions,
        'yaw': transformed_yaws
    }
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Camera intrinsic matrix
    K = np.array([
        [203.93, 0, 192], 
        [0, 203.933, 144], 
        [0, 0, 1]
    ])
    
    # Load images
    images = []
    for img_path in image_files:
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (384, 288))  # Resize to match K matrix
            images.append(img)
    
    # Get positions from both odometry sources
    orig_positions = np.array(transformed_odom_data['pos'])
    recon_positions = np.array(recon_odom_data['pos'])
    
    # Create visualizations for each frame
    n_waypoints = 10
    for i in range(min(len(images), len(orig_positions), len(recon_positions))):
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # Get current image
        current_image = images[i].copy()
        
        # Project future odometry points onto the image
        for j in range(i+1, min(i+n_waypoints, len(orig_positions))):
            # Calculate relative position from current frame
            orig_rel_pos = orig_positions[j] - orig_positions[i]
            recon_rel_pos = recon_positions[j] - recon_positions[i]
            
            # Rotate to current frame's orientation
            orig_yaw = transformed_odom_data['yaw'][i]
            recon_yaw = recon_odom_data['yaw'][i]
            
            # Rotation matrices
            orig_rot = np.array([
                [np.cos(orig_yaw), -np.sin(orig_yaw)],
                [np.sin(orig_yaw), np.cos(orig_yaw)]
            ])
            recon_rot = np.array([
                [np.cos(recon_yaw), -np.sin(recon_yaw)],
                [np.sin(recon_yaw), np.cos(recon_yaw)]
            ])
            
            # Transform to local coordinates
            orig_local = orig_rot.T @ orig_rel_pos
            recon_local = recon_rot.T @ recon_rel_pos
            
            # Project original odometry point
            orig_camera = np.array([-orig_local[1], camera_height, orig_local[0]])
            if orig_camera[2] > 0:  # Only project points in front of camera
                orig_image = K @ orig_camera
                orig_image = orig_image / orig_camera[2]
                x, y = int(orig_image[0]), int(orig_image[1])
                if 0 <= x < current_image.shape[1] and 0 <= y < current_image.shape[0]:
                    # Original odometry in blue
                    cv2.circle(current_image, (x, y), 5, (0, 0, 255), -1)
            
            # Project reconstructed odometry point
            recon_camera = np.array([-recon_local[1], camera_height, recon_local[0]])
            if recon_camera[2] > 0:  # Only project points in front of camera
                recon_image = K @ recon_camera
                recon_image = recon_image / recon_camera[2]
                x, y = int(recon_image[0]), int(recon_image[1])
                if 0 <= x < current_image.shape[1] and 0 <= y < current_image.shape[0]:
                    # Reconstructed odometry in green
                    cv2.circle(current_image, (x, y), 5, (0, 255, 0), -1)
        
        # Display image with projected waypoints
        ax1.imshow(current_image)
        ax1.set_title(f"Frame {i} with Projected Odometry")
        ax1.axis('off')
        
        # Right subplot: Bird's-eye view of trajectory
        # Plot original odometry trajectory
        ax2.plot(orig_positions[:, 0], orig_positions[:, 1], 'b-', label='Original Odometry')
        ax2.scatter(orig_positions[i, 0], orig_positions[i, 1], color='blue', marker='o', s=100)
        
        # Plot reconstructed odometry trajectory
        ax2.plot(recon_positions[:, 0], recon_positions[:, 1], 'g-', label='Reconstructed Odometry')
        ax2.scatter(recon_positions[i, 0], recon_positions[i, 1], color='green', marker='o', s=100)
        
        # Draw orientation arrows for current position
        orig_arrow_len = 1.0
        recon_arrow_len = 1.0
        
        # Original odometry orientation
        orig_arrow_dx = orig_arrow_len * np.cos(transformed_odom_data['yaw'][i])
        orig_arrow_dy = orig_arrow_len * np.sin(transformed_odom_data['yaw'][i])
        ax2.arrow(orig_positions[i, 0], orig_positions[i, 1], 
                 orig_arrow_dx, orig_arrow_dy, 
                 head_width=0.2, head_length=0.3, fc='blue', ec='blue')
        
        # Reconstructed odometry orientation
        recon_arrow_dx = recon_arrow_len * np.cos(recon_odom_data['yaw'][i])
        recon_arrow_dy = recon_arrow_len * np.sin(recon_odom_data['yaw'][i])
        ax2.arrow(recon_positions[i, 0], recon_positions[i, 1], 
                 recon_arrow_dx, recon_arrow_dy, 
                 head_width=0.2, head_length=0.3, fc='green', ec='green')
        
        # Set equal aspect ratio and add grid
        ax2.set_aspect('equal')
        ax2.grid(True)
        ax2.set_title("Odometry Comparison (Global Frame)")
        ax2.set_xlabel("x (meters)")
        ax2.set_ylabel("y (meters)")
        ax2.legend()
        
        # Set reasonable limits for the plot
        all_x = np.concatenate([orig_positions[:, 0], recon_positions[:, 0]])
        all_y = np.concatenate([orig_positions[:, 1], recon_positions[:, 1]])
        x_min, x_max = all_x.min(), all_x.max()
        y_min, y_max = all_y.min(), all_y.max()
        
        # Add some margin
        x_margin = (x_max - x_min) * 0.1
        y_margin = (y_max - y_min) * 0.1
        ax2.set_xlim(x_min - x_margin, x_max + x_margin)
        ax2.set_ylim(y_min - y_margin, y_max + y_margin)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"odometry_comparison_frame_{i:03d}.png"), dpi=150)
        plt.close(fig)
    
    logging.info(f"Visualization complete. {len(images)} images saved to {output_dir}")

# Define worker function for multiprocessing
def worker_process(dirs, gpu_id, overwrite=False):
    time.sleep(gpu_id * 0.5)
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)  # Explicitly set the device
    
    # Empty cache before starting work
    torch.cuda.empty_cache()
    
    logging.info(f"Process {gpu_id} processing {len(dirs)} directories on {device}")
    # Load model once per process
    model_name = "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    weights_path = "naver/" + model_name
    model = AsymmetricMASt3R.from_pretrained(weights_path).to(device)
    
    for dir_path in dirs:
        # try:
        logging.info(f"Processing {dir_path} on {device}")
        # list all images files
        image_files = os.listdir(os.path.join(dir_path, 'img'))
        n_segments = len(image_files) // 100
        for i in range(n_segments-1):
            start_idx = i * 100
            end_idx = start_idx + 110 # overlap 10 frames
            est_pose(dir_path, start_idx, end_idx, device=device, overwrite=overwrite, model=model)

        # last two segments
        start_idx = (n_segments - 1) * 100
        est_pose(dir_path, start_idx, device=device, overwrite=overwrite)
        # except Exception as e:
        #     logging.error(f"Error processing {dir_path}: {e}")

def main(ride_path, num_process_per_gpu=4, overwrite=False):
    """
    Process multiple ride directories in parallel using multiple GPUs.
    
    Args:
        ride_path: Path containing multiple ride directories
        num_threads: Number of threads/GPUs to use
    """
    total_gpus = torch.cuda.device_count()
    total_process = total_gpus * num_process_per_gpu
    logging.info(f"Total GPUs: {total_gpus}")
    logging.info(f"Total processes: {total_process}")
    
    # Get all ride directories
    ride_dirs = []
    for dir in os.listdir(ride_path):
        if os.path.isdir(os.path.join(ride_path, dir)):
            ride_dirs.append(os.path.join(ride_path, dir))
    
    # Split directories into chunks for each process
    chunks = [[] for _ in range(total_process)]
    for i, dir_path in enumerate(ride_dirs):
        chunks[i % total_process].append(dir_path)
    
    # Create and start processes
    processes = []
    mp.set_start_method('spawn', force=True)  # Use spawn method for better CUDA compatibility
    
    for i in range(total_process):
        if len(chunks[i]) > 0:  # Only create processes for non-empty chunks
            p = mp.Process(target=worker_process, args=(chunks[i], i % total_gpus, overwrite))
            processes.append(p)
            p.start()
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    logging.info("All processing complete")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ride_path", type=str, default="data/filtered_2k")
    parser.add_argument("--num_process_per_gpu", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    print(args)
    
    if args.test:
        est_pose("data/frodobot_8k_1/ride_58248_e7808b_20240617121334", 0, 100, visualize=True, show_scene=True)
    else:
        main(args.ride_path, args.num_process_per_gpu, args.overwrite)


