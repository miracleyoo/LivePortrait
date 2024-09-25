# import os.path as osp
# import cv2
# import pandas as pd

# def images_to_video(image_paths, output_video_path, fps=30):
#     # Read the first image to get the dimensions
#     frame = cv2.imread(image_paths[0])
#     height, width, layers = frame.shape

#     # Define the codec and create VideoWriter object
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

#     for image_path in image_paths:
#         video.write(cv2.imread(image_path))

#     # Release the video writer object
#     video.release()

# image_list_record_root = '/prj/qct/mmrd-cv/wonderland_data/3DFR_Data_wrap4d/Pair_man/PAC/meta/train'
# image_lists_record_files = [
#     'data_parsed_List_atalie_man_process_list_train.csv',
#     'data_parsed_List_keli_man_process_list_train.csv',
#     'data_parsed_List_matt_man_process_list_train.csv',
#     'data_parsed_List_ning_man_process_list_train.csv'
# ]
# image_lists_record_files = [osp.join(image_list_record_root, p) for p in image_lists_record_files]
# for f in image_lists_record_files:
#     pd.read_csv(f)[]


# # Example usage
# image_paths = ['image1.jpg', 'image2.jpg', 'image3.jpg']  # Replace with your list of image paths
# output_video_path = 'output_video.mp4'
# images_to_video(image_paths, output_video_path)

# print(f"Video saved as {output_video_path}")
