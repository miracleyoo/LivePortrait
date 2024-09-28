import os
import cv2
import os.path as osp
import numpy as np
import pandas as pd
from multiprocessing import Pool

def video_to_images(args):
    """
    Split a video to images
    :param input_video_path: path to the input video
    :param split_image_root: path to the output folder
    :param driving_video_ver: version of the driving video
    :return: None
    """
    input_video_path, split_image_root, ver = args

    print("Now processing: ", input_video_path, " Version:", ver)
    # input_video_path = '/prj/qct/mmrd-cv/wonderland_data/3DFR_Data/3dMD/PAC/live-portrait-data/llp_results/atalie-driving/aanbunat--atalie.mp4'
    # split_image_root = '/prj/qct/mmrd-cv/wonderland_data/3DFR_Data/3dMD/PAC/live-portrait-data/augmented_images'
    # ver = 1

    os.makedirs(split_image_root, exist_ok=True)

    if ver == 1:
        image_list_record_root = '/prj/qct/mmrd-cv/wonderland_data/3DFR_Data_wrap4d/Pair_man/PAC/meta/train'
    elif ver == 2:
        image_list_record_root = '/mnt/QPFA-LV/dataset/LightCage_Process_MV_Data/PAC_Add_Nose/meta'

    stem = osp.basename(input_video_path).split('.')[0]
    source_name = stem.split('--')[0]
    driving_name = stem.split('--')[1].replace('_concat', '')
    # print(stem, source_name, driving_name)
    meta_file = osp.join(image_list_record_root, f'data_parsed_List_{driving_name}_man_process_list_train.csv')
    if ver == 1:
        image_filename_list = pd.read_csv(meta_file)['tar_bg'].tolist()
    elif ver == 2:
        image_filename_list = pd.read_csv(meta_file)['tar'].tolist()

    out_driving_folder = osp.join(split_image_root, driving_name, source_name)
    os.makedirs(out_driving_folder, exist_ok=True)

    cap = cv2.VideoCapture(input_video_path)
    # split the video into images, name them using the name from image_filename_list
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        image_filename = image_filename_list[frame_count]
        _, subfolder, image_name = image_filename.split('/')
        if ver == 2:
            image_name = image_name + '.jpg'
        subfolder = osp.join(out_driving_folder, subfolder)
        os.makedirs(subfolder, exist_ok=True)
        image_path = osp.join(subfolder, image_name)
        cv2.imwrite(image_path, frame)
        frame_count += 1

# augmented_video_root = '/prj/qct/mmrd-cv/wonderland_data/3DFR_Data/3dMD/PAC/live-portrait-data/llp_results'
# split_image_root='/prj/qct/mmrd-cv/wonderland_data/3DFR_Data/3dMD/PAC/live-portrait-data/augmented_images'
# driver = ['keli', 'matt'] #'atalie',
# driver = ['subject5_cap', 'subject5_tied_hair', 'subject25_cap', 'subject25_tied_hair'] #'atalie',
# driving_video_ver=2


augmented_video_root = '/prj/qct/mmrd-cv/esper/Misc0002/dataset/liveportrait-augmentation/augmented_videos'
split_image_root = '/prj/qct/mmrd-cv/esper/Misc0002/dataset/liveportrait-augmentation/augmented_images'
# driver = ['subject5_cap', 'subject5_tied_hair', 'subject25_cap', 'subject25_tied_hair'] #'atalie',


driver_ver_dct ={
    'atalie': 1,
    'keli': 1,
    'matt': 1,
    'subject5_cap': 2,
    'subject5_tied_hair': 2,
    'subject25_cap': 2,
    'subject25_tied_hair': 2,
}


input_pairs = []
for d,ver in driver_ver_dct.items():
    augmented_video_subroot = osp.join(augmented_video_root, f'{d}-driving')
    video_paths = os.listdir(augmented_video_subroot)
    # video_paths = [p for p in video_paths if '_concat' not in p
    video_paths = [p for p in video_paths if '_concat' in p]
    for video_path in video_paths:
        video_path = osp.join(augmented_video_subroot, video_path)
        input_pairs.append((video_path, split_image_root, ver))
        # print("Now processing: ", video_path, " Version:", ver)
        # print("Saving to: ", split_image_root)
        # print("-----------------------------------------------")
        # video_to_images(video_path, split_image_root, driving_video_ver=ver)

with Pool() as pool:
    pool.map(video_to_images, input_pairs)
