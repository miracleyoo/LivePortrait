import os
import os.path as osp

driving_root = '/prj/qct/mmrd-cv/wonderland_data/3DFR_Data/3dMD/PAC/processed_videos'
source_root = '/local/mnt2/workspace2/zhongyan/projects/LivePortrait/source_images'
script_name = '/local/mnt2/workspace2/zhongyan/projects/LivePortrait/inference.py'
out_root = '/prj/qct/mmrd-cv/wonderland_data/3DFR_Data/3dMD/PAC/live-portrait-data'

driving_videos = [osp.join(driving_root, f) for f in os.listdir(driving_root) if f.endswith('.mp4')]
source_images = [osp.join(source_root, f) for f in os.listdir(source_root) if f.endswith('.jpg') or f.endswith('.png')]

# Loop through all possible combinations of the two arguments
for d in driving_videos:
    stem = osp.splitext(osp.split(d)[1])[0]
    for s in source_images:
        # Construct the command to execute func.py with the current arguments
        command = f"python {script_name} -d {d} -s {s}"
        # Execute the command
        os.system(command)
