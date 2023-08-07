import json
import numpy as np
import cv2

def load_transforms_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def write_camera_path_json(camera_path_data, json_path):
    with open(json_path, 'w') as f:
        json.dump(camera_path_data, f, indent=2)

def normalize_translaitons(frames):

    transform_matrices = [np.array(frame["transform_matrix"]) for frame in frames]

    # Find the minimum and maximum values for x, y, and z coordinates
    min_x = min(matrix[0, 3] for matrix in transform_matrices)
    max_x = max(matrix[0, 3] for matrix in transform_matrices)
    min_y = min(matrix[1, 3] for matrix in transform_matrices)
    max_y = max(matrix[1, 3] for matrix in transform_matrices)
    min_z = min(matrix[2, 3] for matrix in transform_matrices)
    max_z = max(matrix[2, 3] for matrix in transform_matrices)

    # Calculate the scaling factors for x, y, and z coordinates
    scale_x = 2.0 / (max_x - min_x)
    scale_y = 2.0 / (max_y - min_y)
    scale_z = 2.0 / (max_z - min_z)

    # Normalize the positions of the translation matrices to [-1, 1]^3 bounding box
    for matrix in transform_matrices:
        matrix[0, 3] = scale_x * (matrix[0, 3] - min_x) - 1.0
        matrix[1, 3] = scale_y * (matrix[1, 3] - min_y) - 1.0
        matrix[2, 3] = scale_z * (matrix[2, 3] - min_z) - 1.0

    return transform_matrices


def check_crossing(frame1, frame2):
    translation1 = frame1[:2, 3]
    translation2 = frame2[:2, 3]

    return np.array_equal(translation1, translation2)

def find_cross_frames(frames):

    transform_matrices = [np.array(frame["transform_matrix"]) for frame in frames]

    # Calculate all possible pairs of frames
    num_frames = transform_matrices.shape[0]
    indices = np.triu_indices(num_frames, k=1)

    # Find pairs of frames that cross each other in x and y axis
    crossing_pairs = [(i, j) for i, j in zip(*indices) if check_crossing(transform_matrices[i], transform_matrices[j])]

    return crossing_pairs

def find_most_similar(images):

    most_similar = (None, None)

    min_similarity = float('inf')

    for i in range(len(images)):
        image1 = images[i]

        for j in range(i+1, len(images)):
            image2 = images[j]

            # Calculate Structural Similarity Index
            similarity, _ = cv2.compareSSIM(image1, image2, full=True)

            # Calculate Mean Squared Error:
            # mse = np.mean((image1 - image2) ** 2)
            # similarity_score = -mse  # Invert the sign to get lowest MSE

            if similarity < min_similarity:
                min_similarity = similarity
                most_similar = (image1, image2)

    return most_similar