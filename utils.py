import json
import numpy as np
import cv2
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

# Find frames that cross in the camera path
def find_cross_frames(data, threshold=0.2):
    hashmap = {}
    potential_pairs = []
    potential_pairs_idx = []

    # Populate the hashmap
    for idx, entry in enumerate(data):
        x, y = entry['transform_matrix'][0][3], entry['transform_matrix'][1][3]
        if (x, y) not in hashmap:
            hashmap[(x, y)] = [idx]
        else:
            hashmap[(x, y)].append(idx)

    # Check for close pairs
    for idx_i, entry_i in enumerate(data):
        x_i, y_i = entry_i['transform_matrix'][0][3], entry_i['transform_matrix'][1][3]

        for (x_j, y_j), indices in hashmap.items():
            if abs(x_i - x_j) <= threshold and abs(y_i - y_j) <= threshold:
                for idx_j in indices:
                    if idx_i != idx_j:
                        closeness = abs(x_i - x_j) + abs(y_i - y_j)
                        potential_pairs.append(((data[idx_i]['file_path'], data[idx_j]['file_path']), closeness))
                        potential_pairs_idx.append((idx_i+1, idx_j+1))  # Store indices as well

    # Sort the pairs and indices by closeness
    sorted_pairs_and_indices = sorted(zip(potential_pairs, potential_pairs_idx), key=lambda x: x[0][1])
    sorted_pairs, sorted_indices = zip(*sorted_pairs_and_indices)

    return sorted_pairs, sorted_indices

# Find most similar pair of images
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

        for j in range(fps + 1, len(images)):
            test_img = images[j]
            kp2, des2 = orb.detectAndCompute(test_img, None)

            matches = bf.match(des1, des2)
            matches = sorted(matches, key = lambda x:x.distance)

            if len(matches) > max_matches and j - i > fps:
                max_matches = len(matches)
                best_match_pair = (reference_img, test_img)
                best_match_idx = (i+1, j + 1)

    return best_match_pair, best_match_idx
