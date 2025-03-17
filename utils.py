import os
import logging
import numpy as np

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
    
    # Print summary
    logging.info(f"Overall progress: {completed_segments}/{total_segments} segments completed ({overall_completion_pct:.2f}%)")
    logging.info(f"Total rides: {total_rides}")
    
    # Print details for incomplete rides (less than 100% complete)
    incomplete_rides = [r for r in rides_stats if r['completion_percentage'] < 100]
    if incomplete_rides:
        logging.info(f"Incomplete rides ({len(incomplete_rides)}):")
        for ride in incomplete_rides:
            logging.info(f"  {ride['ride_name']}: {ride['completed_segments']}/{ride['total_segments']} "
                        f"({ride['completion_percentage']:.2f}%)")
    
    return {
        'total_rides': total_rides,
        'total_segments': total_segments,
        'completed_segments': completed_segments,
        'completion_percentage': overall_completion_pct,
        'rides_stats': rides_stats,
        'incomplete_rides': incomplete_rides
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_path", type=str, default="data/filtered_2k")
    args = parser.parse_args()
    check_progress(args.dir_path)