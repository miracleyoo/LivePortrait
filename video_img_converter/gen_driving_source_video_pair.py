import os
import os.path as osp
import glob
import numpy as np
import cv2

# Load a video into a numpy array
def load_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        if len(frames) == 100:
            break
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return np.array(frames)

# Write a numpy array to a video file
def write_video(video, video_path, fps=30):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, fps, (video.shape[2], video.shape[1]))
    for frame in video:
        out.write(frame)
    out.release()

# Load a source image into a numpy array
def load_source_image(image_path):
    image = cv2.imread(image_path)
    return image

# Resize the short side of the video to 512 pixels, and do center crop for the long side, the output size is 512x512
def center_crop_and_resize_video(video, rt=512):
    """
    Args:
        video: numpy array, shape (T, H, W, C)
        rt: int, the short side of the output video (resize to)
    Returns:
        cropped_video: numpy array, shape (T, 512, 512, C)
    """
    frame_num, height, width, _ = video.shape
    if height > width:
        new_width = rt
        new_height = int(height * rt / width)
    else:
        new_height = rt
        new_width = int(width * rt / height)
    # Resize the video
    resized_video = np.stack([cv2.resize(img, (new_width, new_height)) for img in video])
    # Crop the center of the video
    _, height, width, _ = resized_video.shape
    start_x = (width - rt) // 2
    start_y = (height - rt) // 2
    print(height, width, start_x, start_y)
    cropped_video = resized_video[:,start_y:start_y+rt, start_x:start_x+rt]
    return cropped_video

if __name__ == '__main__':
    driving_video_root = '/prj/qct/mmrd-cv/esper/Misc0002/dataset/liveportrait-augmentation/driving_videos'
    # concat_video_root = '/prj/qct/mmrd-cv/esper/Misc0002/dataset/liveportrait-augmentation/augmented_videos/keli-driving'
    concat_video_root = '/prj/qct/mmrd-cv/esper/Misc0002/dataset/liveportrait-augmentation/augmented_videos'
    out_root = '/prj/qct/mmrd-cv/esper/Misc0002/dataset/liveportrait-augmentation/verify_videos'

    driving_video_paths = glob.glob(os.path.join(driving_video_root, '*.mp4'))
    drivers = [osp.splitext(osp.split(p)[1])[0] for p in driving_video_paths]
    driver_to_path = {dn: path for dn, path in zip(drivers, driving_video_paths)}

    source_names = ['ardavanm', 'duc']
    print(drivers, source_names)

    for dn in drivers:
        driving_video = load_video(driver_to_path[dn])
        for sn in source_names:
            concat_video_path = osp.join(concat_video_root, f'{dn}-driving',f'{sn}--{dn}_concat.mp4')
            concat_video = load_video(concat_video_path)
            cropped_driving_video = center_crop_and_resize_video(driving_video)
            combination = np.concatenate([cropped_driving_video, concat_video], axis=2)
            print(cropped_driving_video.shape, combination.shape)
            out_path = osp.join(out_root, f'{dn}-driving', f'{sn}--{dn}_concat.mp4')
            os.makedirs(osp.dirname(out_path), exist_ok=True)
            write_video(combination, out_path, fps=30)
        #     break
        # break
