import os
import math
import os.path as osp
from multiprocessing import Pool

driving_roots = ['/prj/qct/mmrd-cv/wonderland_data/3DFR_Data/3dMD/PAC/live-portrait-data/processed_videos']
# source_root = '/local/mnt2/workspace2/zhongyan/projects/LivePortrait/source_images'

source_roots = ['/prj/qct/mmrd-cv/wonderland_data/3DFR_Data/3dMD/PAC/live-portrait-data/source_images/09202024',
                '/prj/qct/mmrd-cv/wonderland_data/3DFR_Data/3dMD/PAC/live-portrait-data/source_images/09232024',
                '/prj/qct/mmrd-cv/wonderland_data/3DFR_Data/3dMD/PAC/live-portrait-data/source_images/09242024']

script_name = '/local/mnt2/workspace2/zhongyan/projects/LivePortrait/inference.py'
out_root = '/prj/qct/mmrd-cv/wonderland_data/3DFR_Data/3dMD/PAC/live-portrait-data/llp_results'

driving_videos = []
source_images = []
for driving_root in driving_roots:
    driving_videos.extend([osp.join(driving_root, f) for f in os.listdir(driving_root) if f.endswith('.mp4')])
for source_root in source_roots:
    source_images.extend([osp.join(source_root, f) for f in os.listdir(source_root) if f.endswith('.jpg') or f.endswith('.png')])

def exec_program(args):
    cuda_idx, pairs = args
    for pair in pairs:
        # print(cuda_idx, pair)
        command = f"CUDA_VISIBLE_DEVICES={cuda_idx} python {script_name} -d {pair[0]} -s {pair[1]} -o {pair[2]}"
        print("Now running command: ", command)
        # Execute the command
        os.system(command)

input_pairs = []

# Loop through all possible combinations of the two arguments
for d in driving_videos:#[3:]:
    # print(d)
    d_stem = osp.splitext(osp.split(d)[1])[0]
    out_dir = osp.join(out_root, f'{d_stem}-driving')
    driving_template_path = d.replace('.mp4', '.pkl')
    os.makedirs(out_dir, exist_ok=True)
    for s in source_images:
        driving_content = driving_template_path if osp.exists(driving_template_path) else d
        s_stem = osp.splitext(osp.split(s)[1])[0]

        # Check whether the corresponding file pair is processed already
        out_path = osp.join(out_dir, f'{s_stem}--{d_stem}.mp4')
        if osp.exists(out_path):
            print(f"{out_path} is already processed! Continue to the next...")
            continue

        input_pairs.append((driving_content, s, out_dir))
        # Construct the command to execute func.py with the current arguments

cuda_available_idxes = [1,3,4,5,6,7] #[2,3,4,5,6,7]
input_pairs_split = {i:[] for i in cuda_available_idxes}
thread_num = len(cuda_available_idxes)
pairs_num = len(input_pairs)
tasks_num_per_cuda = math.ceil(pairs_num / thread_num)
for i, p in enumerate(input_pairs):
    cuda_idx_order = i // tasks_num_per_cuda
    cuda_idx_real = cuda_available_idxes[cuda_idx_order]
    input_pairs_split[cuda_idx_real].append(p)
    # print(input_args[-1])

with Pool() as pool:
    pool.map(exec_program, input_pairs_split.items())
