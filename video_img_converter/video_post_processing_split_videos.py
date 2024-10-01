import os
import cv2
import os.path as osp
import numpy as np
import pandas as pd
from multiprocessing import Pool

# input_video_path,
# split_image_root='/prj/qct/mmrd-cv/wonderland_data/3DFR_Data/3dMD/PAC/live-portrait-data/augmented_images',
# driving_video_ver=1,
# start_frame_idx=0,
# driving_name='ning'

def video_to_images(args):
    input_video_path = args['input_video_path']
    split_image_root = args['split_image_root']
    driving_video_ver = args['driving_video_ver']
    start_frame_idx = args['start_frame_idx']
    driving_name = args['driving_name']

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
    # driving_name = stem.split('--')[1]
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
        image_filename = image_filename_list[frame_count+start_frame_idx]
        _, subfolder, image_name = image_filename.split('/')
        if driving_video_ver == 2:
            image_name = image_name + '.jpg'
        image_name = image_name.replace('.05_C', '')
        subfolder = osp.join(out_driving_folder, subfolder)
        os.makedirs(subfolder, exist_ok=True)
        image_path = osp.join(subfolder, image_name)
        cv2.imwrite(image_path, frame)
        frame_count += 1
    return frame_count

def split_video_to_images(args):
    sn = args['source_name']
    dn = args['driving_name']
    augmented_video_subroots = args['augmented_video_subroots']
    split_num = args['split_num']
    total_frames = 0
    for idx in range(split_num):
        augmented_video_subroot = augmented_video_subroots[idx]
        file_name = f'{sn}--{dn}_{idx+1}_concat.mp4'
        video_path = osp.join(augmented_video_subroot, file_name)
        args['input_video_path'] = video_path
        args['start_frame_idx'] = total_frames
        print("Now processing: ", video_path)
        print("-----------------------------------------------")
        frame_count = video_to_images(args)
        total_frames += frame_count
    print("Total frames: ", total_frames)

# augmented_video_root = '/prj/qct/mmrd-cv/wonderland_data/3DFR_Data/3dMD/PAC/live-portrait-data/llp_results'
# split_image_root='/prj/qct/mmrd-cv/wonderland_data/3DFR_Data/3dMD/PAC/live-portrait-data/augmented_images'

augmented_video_root = '/prj/qct/mmrd-cv/esper/Misc0002/dataset/liveportrait-augmentation/augmented_videos'
split_image_root='/prj/qct/mmrd-cv/esper/Misc0002/dataset/liveportrait-augmentation/augmented_images'

driver = ['ning'] #'atalie',
driving_video_ver=1

# total_frames = 0
input_pairs = []
for dn in driver:
    augmented_video_subroots = [p for p in os.listdir(augmented_video_root) if p.startswith(dn+"_")]
    # sort the augmented_video_subroots by the number in the name
    augmented_video_subroots.sort(key=lambda x: int(x.lstrip(dn+"_").rstrip('-driving')))
    print(augmented_video_subroots)
    # break
    augmented_video_subroots = [osp.join(augmented_video_root, p) for p in augmented_video_subroots]
    split_num = len(augmented_video_subroots)

    subroot = augmented_video_subroots[0]
    source_names = [p.split('--')[0] for p in os.listdir(subroot) if p.endswith('.mp4') and '_concat' in p]

    for sn in source_names:
        args = {'source_name': sn,
                'driving_name': dn,
                'split_num': split_num,
                'augmented_video_subroots': augmented_video_subroots,
                'driving_video_ver': driving_video_ver,
                'split_image_root': split_image_root,
                }
        input_pairs.append(args)

        # total_frames = 0
        # for idx in range(split_num):
        #     augmented_video_subroot = augmented_video_subroots[idx]
        #     file_name = f'{sn}--{dn}_{idx+1}_concat.mp4'
        #     video_path = osp.join(augmented_video_subroot, file_name)
        #     print("Now processing: ", video_path)
        #     print("-----------------------------------------------")
        #     frame_count = video_to_images(video_path, split_image_root, driving_video_ver=driving_video_ver, start_frame_idx=total_frames, driving_name=dn)
        #     total_frames += frame_count
        # print("Total frames: ", total_frames)

with Pool() as pool:
    pool.map(split_video_to_images, input_pairs)
