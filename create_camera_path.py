import argparse
import os
import numpy as np
import cv2
import torch
from utils import *
from model import *

def generate_full_path(data, duration):
    camera_path_data = {
        "loop":False,
        "path":[], 
        "time":duration
    }
    
    n_frames = len(data['frames'])

    frames_dict = {}
    for i in range(n_frames):
        file = int(data['frames'][i]['file_path'].split('/')[-1][:-4]) # Get image number from file name
        transform = data['frames'][i]['transform_matrix']
        frames_dict[file] = transform
        
    frames_dict = dict(sorted(frames_dict.items()))

    for ind in np.linspace(1, n_frames, n_frames, endpoint=True, dtype=int):
        q, t = normalize_transforms(np.array(frames_dict[ind]))
        
        camera_path_data['path'].append({
            "R": list(q),
            "T": list(t),
            "aperture_size": 0.0,
            "fov": 50.625,
            "glow_mode": 0,
            "glow_y_cutoff": 0.0,
            "scale": 1.6500000953674316,
            "slice": 0.0
        })

    return camera_path_data

def generate_path(data, images, fps, smoothness, duration):
    camera_path_data = {
        "loop":True,
        "path":[], 
        "time":duration
    }
    
    n_frames = len(data['frames'])

    frames_dict = {}
    for i in range(n_frames):
        file = int(data['frames'][i]['file_path'].split('/')[-1][:-4]) # Get image number from file name
        transform = data['frames'][i]['transform_matrix']
        frames_dict[file] = transform
        
    frames_dict = dict(sorted(frames_dict.items()))

    cross_frames, cross_frames_indices = find_cross_frames(data['frames'], 0.2)
    
    if cross_frames_indices and (cross_frames_indices[0][1] - cross_frames_indices[0][0] > fps):
        start_idx = cross_frames_indices[0][0]

        q, t = normalize_transforms(np.array(frames_dict[start_idx]))

        camera_path_data['path'].append({
            "R": list(q),
            "T": list(t),
            "aperture_size": 0.0,
            "fov": 50.625,
            "glow_mode": 0,
            "glow_y_cutoff": 0.0,
            "scale": 1.6500000953674316,
            "slice": 0.0
        })

        start_idx += 1
        end_idx = cross_frames_indices[1][0]
    
    else:
        # pair_images, pair_images_indices = find_most_similar(images, fps)

        pair_images_indices = (2, 120)

        samples = 15

        start_idx = pair_images_indices[0] + samples
        end_idx = pair_images_indices[1] - samples

        frames = [torch.tensor(frames_dict[frame]).view(-1).cuda() for frame in frames_dict.keys()]

        frames = frames[start_idx : end_idx + 1]

        sequence_length = 8

        data = []
        for i in range(len(frames) - sequence_length):
            sequence = torch.stack()
            data.append(sequence)

        model = LSTMModel().cuda()

        model = train_model(data, model)

        model.eval()

        path = predict_path(model, data[-1], data[0])
        
        for matrix in path:
            q, t = normalize_transforms(np.array(matrix))

            camera_path_data['path'].append({
                "R": list(q),
                "T": list(t),
                "aperture_size": 0.0,
                "fov": 50.625,
                "glow_mode": 0,
                "glow_y_cutoff": 0.0,
                "scale": 1.6500000953674316,
                "slice": 0.0
            })
        
        start_idx += 1
        end_idx -= 1
    
    n_frames = end_idx - start_idx + 1

    for ind in np.linspace(start_idx, end_idx, max(25, (1-smoothness) * n_frames), endpoint=True, dtype=int):
        q, t = normalize_transforms(np.array(frames_dict[ind]))
        
        camera_path_data['path'].append({
            "R": list(q),
            "T": list(t),
            "aperture_size": 0.0,
            "fov": 50.625,
            "glow_mode": 0,
            "glow_y_cutoff": 0.0,
            "scale": 1.6500000953674316,
            "slice": 0.0
        })

    return camera_path_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--duration', type=int, default=10)
    parser.add_argument('--smoothness', type=int, default=0)
    parser.add_argument('--fps', type=int, default=24)
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()

    # Get data from 'images' directory
    image_directory = str(args.data + '/images/')
    image_files = [f for f in os.listdir(image_directory) if f.endswith('.jpeg') or f.endswith('.jpg')]
    images = []

    # Read images in grayscale
    for image_file in image_files:
        image_path = os.path.join(image_directory, image_file)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE )
        if image is not None:
            images.append(image)

    # Get data from 'trasnforms.json'
    transforms_data = load_transforms_json(str(args.data + '/transforms.json'))

    # Generate camera path
    camera_path_data = generate_path(transforms_data, images, args.fps, args.smoothness, args.duration)

    # Create JSON file
    write_camera_path_json(camera_path_data, str(args.output_dir + '/camera_path2.json'))

    # Full camera path
    full_camera_path = generate_full_path(transforms_data, args.duration)
    write_camera_path_json(full_camera_path, str(args.output_dir + '/full_camera_path2.json'))

if __name__ == '__main__':
    main()
