import argparse
import json

def load_transforms_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def create_camera_path(transforms_data, fov, aspect, render_height, render_width, duration, smoothness):
    keyframes = []
    camera_path = []
    
    keyframes_num = 5
    for i in range(keyframes_num):
        frame = transforms_data["frames"][i]
        matrix = frame["transform_matrix"]
        flat_matrix = [item for sublist in matrix for item in sublist]
        

        properties = [
            ["FOV", fov],
            ["NAME", f"Camera {i}"],
            ["TIME", i/(keyframes_num - 1)]  # Assuming equal time intervals
        ]
        keyframe = {
            "matrix": str(flat_matrix),
            "fov": fov,
            "aspect": aspect,
            "properties": json.dumps(properties)
        }

        keyframes.append(keyframe)

    # Create camera path
    for i, frame in enumerate(transforms_data["frames"]):
        matrix = frame["transform_matrix"]
        flat_matrix = [item for sublist in matrix for item in sublist]
    
        camera_position = {
            "camera_to_world": flat_matrix,
            "fov": fov,
            "aspect": aspect
        }
        camera_path.append(camera_position)

    # Make into JSON camera_path file format
    camera_path_data = {
        "keyframes": keyframes,
        "camera_type": "perspective",
        "render_height": render_height,
        "render_width": render_width,
        "camera_path": camera_path,
        "fps": 24,
        "seconds": duration,
        "smoothness_value": smoothness,
        "is_cycle": True,
        "crop": None
    }
    
    return camera_path_data

def write_camera_path_json(camera_path_data, json_path):
    with open(json_path, 'w') as f:
        json.dump(camera_path_data, f, indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['landscape', 'portrait'], default='landscape')
    parser.add_argument('--fov', type=int, default=50)
    parser.add_argument('--aspect-ratio', type=float, default=1.66996699669967)
    parser.add_argument('--height', type=int, default=1080)
    parser.add_argument('--width', type=int, default=1920)
    parser.add_argument('--duration', type=int, default=5)
    parser.add_argument('--smoothness', type=int, default=0.5)
    parser.add_argument('--data', type=str, default='./transforms.json')
    parser.add_argument('--output', type=str, default='.')
    args = parser.parse_args()

    if args.mode == 'portrait':
        args.aspect_ratio = 1 / args.aspect_ratio  # Invert aspect ratio for portrait mode
        args.height, args.width = args.width, args.height  # Swap width and height for portrait mode

    transforms_data = load_transforms_json(args.data)
    camera_path_data = create_camera_path(transforms_data, args.fov, args.aspect_ratio, args.height, args.width, args.duration, args.smoothness)
    write_camera_path_json(camera_path_data, str(args.output + '/camera_path.json'))

if __name__ == '__main__':
    main()
