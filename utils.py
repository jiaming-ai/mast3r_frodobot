import os
import logging
import numpy as np

logging.basicConfig(
    handlers=[logging.StreamHandler()],
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def check_progress(dir_path):
    """
    Check the progress of pose estimation across all ride directories.
    
    Args:
        dir_path: Path containing multiple ride directories
        
    Returns:
        Dictionary with progress statistics
    """
    
    logging.info(f"Checking progress in {dir_path}")
    
    # Statistics to track
    total_rides = 0
    total_segments = 0
    completed_segments = 0
    start_idx = 0
    end_idx = None
    rides_stats = []
    
    # Get all ride directories
    ride_dirs = []
    for dir_name in os.listdir(dir_path):
        full_dir = os.path.join(dir_path, dir_name)
        if os.path.isdir(full_dir):
            ride_dirs.append(full_dir)
    
    # Process each ride directory
    for ride_dir in ride_dirs:
        total_rides += 1
        ride_name = os.path.basename(ride_dir)
        
        # Check if image directory exists
        img_dir = os.path.join(ride_dir, 'img')
        if not os.path.exists(img_dir):
            logging.warning(f"No image directory found in {ride_dir}")
            continue
        
        # Count image files
        image_files = [f for f in os.listdir(img_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        actual_end_idx = len(image_files)
        
        # Calculate number of segments (following the logic in worker_process)
        n_segments = len(image_files) // 100
        ride_segments = 0
        ride_completed = 0
        
        # Check each segment
        for i in range(n_segments-1):
            segment_start_idx = i * 100
            segment_end_idx = segment_start_idx + 110  # overlap 10 frames
            
            # Skip if segment is outside the requested range
            if segment_end_idx < start_idx or segment_start_idx >= actual_end_idx:
                continue
                
            ride_segments += 1
            total_segments += 1
            
            # Check if this segment has been processed
            cam2w_file = os.path.join(ride_dir, f'cam2w_{segment_start_idx}_{segment_end_idx}.npy')
            if os.path.exists(cam2w_file):
                print(f"ride: {ride_name}, segment: {segment_start_idx} - {segment_end_idx}")
                ride_completed += 1
                completed_segments += 1
        
        # Check last segment
        if n_segments > 0:
            segment_start_idx = (n_segments - 1) * 100
            
            # Skip if segment is outside the requested range
            if segment_start_idx < actual_end_idx and segment_start_idx >= start_idx:
                ride_segments += 1
                total_segments += 1
                
                # Check if this segment has been processed
                cam2w_file = os.path.join(ride_dir, f'cam2w_{segment_start_idx}_{actual_end_idx}.npy')
                if os.path.exists(cam2w_file):
                    ride_completed += 1
                    completed_segments += 1
        
        # Store statistics for this ride
        completion_pct = 0 if ride_segments == 0 else (ride_completed / ride_segments) * 100
        rides_stats.append({
            'ride_name': ride_name,
            'total_segments': ride_segments,
            'completed_segments': ride_completed,
            'completion_percentage': completion_pct
        })
    
    # Calculate overall completion percentage
    overall_completion_pct = 0 if total_segments == 0 else (completed_segments / total_segments) * 100
    
    # Sort rides by completion percentage
    rides_stats.sort(key=lambda x: x['completion_percentage'])
    
    
    # # Print details for incomplete rides (less than 100% complete)
    incomplete_rides = [r for r in rides_stats if r['completion_percentage'] < 100]
    # if incomplete_rides:
    #     logging.info(f"Incomplete rides ({len(incomplete_rides)}):")
    #     for ride in incomplete_rides:
    #         logging.info(f"  {ride['ride_name']}: {ride['completed_segments']}/{ride['total_segments']} "
    #                     f"({ride['completion_percentage']:.2f}%)")
    # Print summary
    logging.info(f"Overall progress: {completed_segments}/{total_segments} segments completed ({overall_completion_pct:.2f}%)")
    logging.info(f"Total rides: {total_rides}")
    
    return {
        'total_rides': total_rides,
        'total_segments': total_segments,
        'completed_segments': completed_segments,
        'completion_percentage': overall_completion_pct,
        'rides_stats': rides_stats,
        'incomplete_rides': incomplete_rides
    }

def fix_nested_ride_directories(dir_path, correct_structure=False):
    """
    Identifies and optionally fixes incorrectly nested ride directories.
    
    Looks for structure like:
    data/ride_ID/ride_ID (where the inner folder contains the actual ride data)
    
    Args:
        dir_path: Path containing ride directories
        correct_structure: If True, fixes the structure by moving inner directories up
                          If False, only reports issues without making changes
    
    Returns:
        List of directories that were fixed or need fixing
    """
    logging.info(f"Checking for nested ride directories in {dir_path}")
    
    issues_found = []
    
    # Get all directories in the path
    for dir_name in os.listdir(dir_path):
        parent_dir_path = os.path.join(dir_path, dir_name)
        
        # Skip if not a directory
        if not os.path.isdir(parent_dir_path):
            continue
            
        # Check if this directory has no img folder but contains a subdirectory with the same name
        if not os.path.exists(os.path.join(parent_dir_path, 'img')):
            # Check if there's a subdirectory with the same name
            sub_dir_path = os.path.join(parent_dir_path, dir_name)
            if os.path.isdir(sub_dir_path) and os.path.exists(os.path.join(sub_dir_path, 'img')):
                issues_found.append(parent_dir_path)
                
                if correct_structure:
                    logging.info(f"Fixing nested structure for {dir_name}")
                    
                    # Create a temporary name for the parent directory
                    temp_parent_dir = f"{parent_dir_path}_temp"
                    
                    try:
                        # 1. Rename parent directory to temporary name
                        os.rename(parent_dir_path, temp_parent_dir)
                        
                        # 2. Move the inner directory to the original location
                        os.rename(os.path.join(temp_parent_dir, dir_name), parent_dir_path)
                        
                        # 3. Remove the now-empty temporary parent directory
                        os.rmdir(temp_parent_dir)
                        
                        logging.info(f"Successfully fixed structure for {dir_name}")
                    except Exception as e:
                        logging.error(f"Error fixing structure for {dir_name}: {str(e)}")
                else:
                    logging.info(f"Found nested structure issue: {dir_name} (use --correct_structure to fix)")
    
    return issues_found

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_path", type=str, default="data/filtered_2k")
    parser.add_argument("--correct_structure", action="store_true", 
                        help="Fix nested ride directory structure if found")
    args = parser.parse_args()
    
    # Check progress
    res = check_progress(args.dir_path)
    
    # Check and optionally fix nested directories
    fixed_dirs = fix_nested_ride_directories(args.dir_path, args.correct_structure)
    if fixed_dirs:
        logging.info(f"Found {len(fixed_dirs)} directories with nested structure issues")
    else:
        logging.info("No nested directory structure issues found")