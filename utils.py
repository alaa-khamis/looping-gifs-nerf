import json
import numpy as np
import cv2
import math
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

# Load json file
def load_transforms_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

# Write json file
def write_camera_path_json(camera_path_data, json_path):
    with open(json_path, 'w') as f:
        json.dump(camera_path_data, f, indent=2)

# Calculate FOV (horizontal)
def calculate_fov(fl_y, h):
    fov = 2 * math.degrees(math.atan(h / (2 * fl_y)))
    return fov

# Linear Interpolation
def linear_interpolate(value1, value2, factor):
    return factor * value1 + (1 - factor) * value2

def get_translation(frame):
    return frame[:,3] + 0.025

# 'transforms.json' format to ngp format
def normalize_transforms(transform):
    mat = np.copy(transform)
    mat = mat[:-1,:]
    mat[:,1] *= -1 # flip axis
    mat[:,2] *= -1
    mat[:,3] *= 0.33 #scale
    mat[:,3] += [0.5, 0.5, 0.5] #offset
    
    mat = mat[[1,2,0], :] # swap axis
    
    rm = R.from_matrix(mat[:,:3]) 
    
    # quaternion (x, y, z, w) and translation
    return rm.as_quat(), mat[:,3] + 0.025

# Find Crossing Frames
def find_cross_frames(data, fps, threshold=0.5):
    start_points = data[:fps]
    end_points = data[-3*fps:]
    
    min_distance = float('inf')
    closest_pair = None
    closest_indices = None

    for i, start_point in enumerate(start_points):
        for j, end_point in enumerate(end_points):
            distance = np.linalg.norm(start_point - end_point)

            if distance < min_distance:
                min_distance = distance
                closest_pair = (start_point, end_point)
                closest_indices = (i+1, len(data) - 3*fps + j + 1)

    if min_distance > threshold:
        return [], []

    return closest_pair, closest_indices


# Find most similar pair of images - ORB matching
def find_most_similar(images, fps):
    orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    best_match_pair = (None, None)
    best_match_idx = (None, None)
    max_matches = -float('inf')

    # Iterate through the first fps images
    for i in tqdm(range(fps), desc="Comparing images", ncols=100):
        reference_img = images[i]
        kp1, des1 = orb.detectAndCompute(reference_img, None)

        # Match with the last 2 * fps images
        for j in range(len(images) - 2*fps, len(images)):
            test_img = images[j]
            kp2, des2 = orb.detectAndCompute(test_img, None)

            matches = bf.match(des1, des2)
            matches = sorted(matches, key = lambda x:x.distance)

            # Choose best matched pair
            if len(matches) > max_matches and j - i > fps:
                max_matches = len(matches)
                best_match_pair = (reference_img, test_img)
                best_match_idx = (i+1, j + 1)

    return best_match_pair, best_match_idx
