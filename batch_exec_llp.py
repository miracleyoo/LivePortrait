import os
import os.path as osp

# driving_root = '/prj/qct/mmrd-cv/wonderland_data/3DFR_Data/3dMD/PAC/live-portrait-data/processed_videos'
driving_root = '/prj/qct/mmrd-cv/wonderland_data/3DFR_Data/3dMD/PAC/live-portrait-data/experimental/driving_videos'
# source_root = '/local/mnt2/workspace2/zhongyan/projects/LivePortrait/source_images'
source_root = '/prj/qct/mmrd-cv/wonderland_data/3DFR_Data/3dMD/PAC/live-portrait-data/source_images/09202024'
script_name = '/local/mnt2/workspace2/zhongyan/projects/LivePortrait/inference.py'

# out_root = '/prj/qct/mmrd-cv/wonderland_data/3DFR_Data/3dMD/PAC/live-portrait-data/llp_results'
out_root = '/prj/qct/mmrd-cv/wonderland_data/3DFR_Data/3dMD/PAC/live-portrait-data/experimental/llp_results'

driving_videos = [osp.join(driving_root, f) for f in os.listdir(driving_root) if f.endswith('.mp4')]
source_images = [osp.join(source_root, f) for f in os.listdir(source_root) if f.endswith('.jpg') or f.endswith('.png')]

# print(driving_videos)
# print(source_images)

driving_videos = driving_videos[2:]

# Loop through all possible combinations of the two arguments
for d in driving_videos:
    d_stem = osp.splitext(osp.split(d)[1])[0]
    out_dir = osp.join(out_root, f'{d_stem}-driving')
    driving_template_path = d.replace('.mp4', '.pkl')
    os.makedirs(out_dir, exist_ok=True)
    for s in source_images:
        driving_content = d #driving_template_path if osp.exists(driving_template_path) else d
        s_stem = osp.splitext(osp.split(s)[1])[0]

        # Check whether the corresponding file pair is processed already
        out_path = osp.join(out_dir, f'{s_stem}--{d_stem}.mp4')
        if osp.exists(out_path):
            print(f"{out_path} is already processed! Continue to the next...")
            continue

        # Construct the command to execute func.py with the current arguments
        command = f"CUDA_VISIBLE_DEVICES=0 python {script_name} -d {driving_content} -s {s} -o {out_dir}"
        print("Now running command: ", command)
        # Execute the command
        os.system(command)
    # break
