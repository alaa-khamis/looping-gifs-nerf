import argparse
import json
import numpy as np

from utils import *

def generate_keyframes(frames, threshold):
    keyframes = []

    prev_matrix = np.array(frames[0]['transform_matrix'])

    for frame in frames:

        curr_matrix = np.array(frame["transform_matrix"])

        diff = np.linalg.norm(curr_matrix - prev_matrix)

        if diff > threshold:
            keyframes.append(curr_matrix)
            prev_matrix = curr_matrix
        
    return keyframes

def normalize_positions(positions, min_val, max_val):
    positions = np.array(positions)
    ranges = max_val - min_val
    ranges[ranges == 0] = 1
    normalized_positions = (positions - min_val) / ranges
    return normalized_positions.tolist()


def create_camera_path(transforms_data, fov, aspect, render_height, render_width, duration, smoothness, fps):

    if render_height == -1:
        render_height = transforms_data["h"]
    if render_width == -1:
        render_width = transforms_data["w"]

    frames = transforms_data["frames"]

    # Find min and max for all frames
    all_positions = [np.array(frame['transform_matrix'])[:, -1] for frame in frames]
    min_val = np.min(all_positions, axis=0)
    max_val = np.max(all_positions, axis=0)

    #Generate keyframes based on camera positions
    keyframes = generate_keyframes(frames, 1)

    # Normalize positions of the keyframes
    keyframe_positions = [np.array(frame)[:, -1] for frame in keyframes]
    normalized_positions = normalize_positions(keyframe_positions, min_val, max_val)

    # Update the keyframes with the normalized positions
    for i, frame in enumerate(keyframes):
        frame[:, -1] = normalized_positions[i]

    #Generate keyframes based on camera positions
    num_keyframes = len(keyframes)

    new_keyframes = []  # Separate list to prevent confusion with original keyframes

    for i in range(num_keyframes):
        frame = keyframes[i].transpose().tolist()  # Convert numpy array to list
        flat_matrix = [item for sublist in frame for item in sublist]

        properties = [
            ["FOV", fov],
            ["NAME", f"Camera {i}"],
            ["TIME", i / (num_keyframes-1)]
        ]
        keyframe = {
            "matrix": json.dumps(flat_matrix),
            "fov": fov,
            "aspect": aspect,
            "properties": json.dumps(properties)
        }

        new_keyframes.append(keyframe)


    # Make into JSON camera_path file format
    camera_path_data = {
        "keyframes": new_keyframes,
        "camera_type": "perspective",
        "render_height": render_height,
        "render_width": render_width,
        "camera_path": [],
        "fps": fps,
        "seconds": duration,
        "smoothness_value": smoothness,
        "is_cycle": True,
        "crop": None
    }

    return camera_path_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fov', type=int, default=50)
    parser.add_argument('--aspect-ratio', type=float, default=1)
    parser.add_argument('--height', type=int, default=-1)
    parser.add_argument('--width', type=int, default=-1)
    parser.add_argument('--duration', type=int, default=2)
    parser.add_argument('--smoothness', type=int, default=0.5)
    parser.add_argument('--fps', type=int, default=24)
    parser.add_argument('--data', type=str, default='./transforms.json')
    parser.add_argument('--output', type=str, default='.')
    args = parser.parse_args()

    transforms_data = load_transforms_json(args.data)
    camera_path_data = create_camera_path(transforms_data, args.fov, args.aspect_ratio, args.height, args.width, args.duration, args.smoothness, args.fps)
    write_camera_path_json(camera_path_data, str(args.output + '/camera_path.json'))

if __name__ == '__main__':
    main()
