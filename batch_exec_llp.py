import os
import os.path as osp

driving_root = '/prj/qct/mmrd-cv/wonderland_data/3DFR_Data/3dMD/PAC/live-portrait-data/processed_videos'
# source_root = '/local/mnt2/workspace2/zhongyan/projects/LivePortrait/source_images'
source_root = '/prj/qct/mmrd-cv/wonderland_data/3DFR_Data/3dMD/PAC/live-portrait-data/source_images/09202024'
script_name = '/local/mnt2/workspace2/zhongyan/projects/LivePortrait/inference.py'
out_root = '/prj/qct/mmrd-cv/wonderland_data/3DFR_Data/3dMD/PAC/live-portrait-data/llp_results'

driving_videos = [osp.join(driving_root, f) for f in os.listdir(driving_root) if f.endswith('.mp4')]
source_images = [osp.join(source_root, f) for f in os.listdir(source_root) if f.endswith('.jpg') or f.endswith('.png')]

# print(driving_videos)
# print(source_images)

# Loop through all possible combinations of the two arguments
for d in driving_videos[1:]:
    stem = osp.splitext(osp.split(d)[1])[0]
    out_dir = osp.join(out_root, stem)
    driving_template_path = d.replace('.mp4', '.pkl')
    if osp.exists(driving_template_path):
        driving_content = driving_template_path
    else:
        driving_content = d
    os.makedirs(out_dir, exist_ok=True)
    for s in source_images:
        # Construct the command to execute func.py with the current arguments
        command = f"CUDA_VISIBLE_DEVICES=1 python {script_name} -d {driving_content} -s {s} -o {out_dir}"
        print("Now running command: ", command)
        # Execute the command
        os.system(command)
    # break
