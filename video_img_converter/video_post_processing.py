import os
import cv2
import os.path as osp
import numpy as np
import pandas as pd

def video_to_images(input_video_path,
                    split_image_root='/prj/qct/mmrd-cv/wonderland_data/3DFR_Data/3dMD/PAC/live-portrait-data/augmented_images',
                    driving_video_ver=1):
    # input_video_path = '/prj/qct/mmrd-cv/wonderland_data/3DFR_Data/3dMD/PAC/live-portrait-data/llp_results/atalie-driving/aanbunat--atalie.mp4'
    # split_image_root = '/prj/qct/mmrd-cv/wonderland_data/3DFR_Data/3dMD/PAC/live-portrait-data/augmented_images'
    # driving_video_ver = 1

    os.makedirs(split_image_root, exist_ok=True)

    if driving_video_ver == 1:
        image_list_record_root = '/prj/qct/mmrd-cv/wonderland_data/3DFR_Data_wrap4d/Pair_man/PAC/meta/train'
    elif driving_video_ver == 2:
        image_list_record_root = '/mnt/QPFA-LV/dataset/LightCage_Process_MV_Data/PAC_Add_Nose/meta'

    stem = osp.basename(input_video_path).split('.')[0]
    source_name = stem.split('--')[0]
    driving_name = stem.split('--')[1]
    # print(stem, source_name, driving_name)
    meta_file = osp.join(image_list_record_root, f'data_parsed_List_{driving_name}_man_process_list_train.csv')
    if driving_video_ver == 1:
        image_filename_list = pd.read_csv(meta_file)['tar_bg'].tolist()
    elif driving_video_ver == 2:
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
        if driving_video_ver == 2:
            image_name = image_name + '.jpg'
        subfolder = osp.join(out_driving_folder, subfolder)
        os.makedirs(subfolder, exist_ok=True)
        image_path = osp.join(subfolder, image_name)
        cv2.imwrite(image_path, frame)
        frame_count += 1

augmented_video_root = '/prj/qct/mmrd-cv/wonderland_data/3DFR_Data/3dMD/PAC/live-portrait-data/llp_results'
split_image_root='/prj/qct/mmrd-cv/wonderland_data/3DFR_Data/3dMD/PAC/live-portrait-data/augmented_images'
# driver = ['keli', 'matt'] #'atalie',
driver = ['subject5_cap', 'subject5_tied_hair', 'subject25_cap', 'subject25_tied_hair'] #'atalie',

driving_video_ver=2

for d in driver:
    augmented_video_subroot = osp.join(augmented_video_root, f'{d}-driving')
    video_paths = os.listdir(augmented_video_subroot)
    video_paths = [p for p in video_paths if '_concat' not in p]
    for video_path in video_paths:
        video_path = osp.join(augmented_video_subroot, video_path)
        print("Now processing: ", video_path)
        print("Saving to: ", split_image_root)
        print("-----------------------------------------------")
        video_to_images(video_path, split_image_root, driving_video_ver=driving_video_ver)
