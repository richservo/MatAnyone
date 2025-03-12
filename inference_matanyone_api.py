import os
import cv2
import tqdm
import random
import imageio
import numpy as np
from PIL import Image

import torch
import torchvision
import torch.nn.functional as F

from matanyone.model.matanyone import MatAnyone
from matanyone.inference.inference_core import InferenceCore

import warnings
warnings.filterwarnings("ignore")

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
VIDEO_EXTENSIONS = ('.mp4', '.mov', '.avi', '.MP4', '.MOV', '.AVI')

def read_frame_from_videos(frame_root):
    if frame_root.endswith(VIDEO_EXTENSIONS):  # Video file path
        video_name = os.path.basename(frame_root)[:-4]
        frames, _, info = torchvision.io.read_video(filename=frame_root, pts_unit='sec', output_format='TCHW') # RGB
        fps = info['video_fps']
    else:
        video_name = os.path.basename(frame_root)
        frames = []
        fr_lst = sorted(os.listdir(frame_root))
        for fr in fr_lst:
            frame = cv2.imread(os.path.join(frame_root, fr))[...,[2,1,0]] # RGB, HWC
            frames.append(frame)
        fps = 24  # default
        frames = torch.Tensor(np.array(frames)).permute(0, 3, 1, 2).contiguous() # TCHW
    
    length = frames.shape[0]

    return frames, fps, length, video_name

def gen_dilate(alpha, min_kernel_size, max_kernel_size): 
    kernel_size = random.randint(min_kernel_size, max_kernel_size)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size,kernel_size))
    fg_and_unknown = np.array(np.not_equal(alpha, 0).astype(np.float32))
    dilate = cv2.dilate(fg_and_unknown, kernel, iterations=1)*255
    return dilate.astype(np.float32)

def gen_erosion(alpha, min_kernel_size, max_kernel_size): 
    kernel_size = random.randint(min_kernel_size, max_kernel_size)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size,kernel_size))
    fg = np.array(np.equal(alpha, 255).astype(np.float32))
    erode = cv2.erode(fg, kernel, iterations=1)*255
    return erode.astype(np.float32)

@torch.inference_mode()
@torch.cuda.amp.autocast()
def main(input_path, mask_path, output_path, ckpt_path, n_warmup=10, r_erode=10, r_dilate=10, suffix="", save_image=False, max_size=-1):

    matanyone = MatAnyone.from_pretrained("PeiqingYang/MatAnyone").cuda().eval()
    processor = InferenceCore(matanyone, cfg=matanyone.cfg)

    # inference parameters
    r_erode = int(r_erode)
    r_dilate = int(r_dilate)
    n_warmup = int(n_warmup)
    max_size = int(max_size)

    # load input frames
    vframes, fps, length, video_name = read_frame_from_videos(input_path)
    repeated_frames = vframes[0].unsqueeze(0).repeat(n_warmup, 1, 1, 1) # repeat the first frame for warmup
    vframes = torch.cat([repeated_frames, vframes], dim=0).float()
    length += n_warmup  # update length

    # resize if needed
    if max_size > 0:
        h, w = vframes.shape[-2:]
        min_side = min(h, w)
        if min_side > max_size:
            new_h = int(h / min_side * max_size)
            new_w = int(w / min_side * max_size)

        vframes = F.interpolate(vframes, size=(new_h, new_w), mode="area")
        
    # set output paths
    os.makedirs(output_path, exist_ok=True)
    if suffix != "":
        video_name = f'{video_name}_{suffix}'
    if save_image:
        os.makedirs(f'{output_path}/{video_name}', exist_ok=True)
        os.makedirs(f'{output_path}/{video_name}/pha', exist_ok=True)
        os.makedirs(f'{output_path}/{video_name}/fgr', exist_ok=True)

    # load the first-frame mask
    mask = Image.open(mask_path).convert('L')
    mask = np.array(mask)

    bgr = (np.array([120, 255, 155], dtype=np.float32)/255).reshape((1, 1, 3)) # green screen to paste fgr
    objects = [1]

    # [optional] erode & dilate
    if r_dilate > 0:
        mask = gen_dilate(mask, r_dilate, r_dilate)
    if r_erode > 0:
        mask = gen_erosion(mask, r_erode, r_erode)

    mask = torch.from_numpy(mask).cuda()

    if max_size > 0:  # resize needed
        mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(new_h, new_w), mode="nearest")
        mask = mask[0,0]

    # inference start
    phas = []
    fgrs = []
    for ti in tqdm.tqdm(range(length)):
        # load the image as RGB; normalization is done within the model
        image = vframes[ti]

        image_np = np.array(image.permute(1,2,0))       # for output visualize
        image = (image / 255.).cuda().float()           # for network input

        if ti == 0:
            output_prob = processor.step(image, mask, objects=objects)      # encode given mask
            output_prob = processor.step(image, first_frame_pred=True)      # first frame for prediction
        else:
            if ti <= n_warmup:
                output_prob = processor.step(image, first_frame_pred=True)  # reinit as the first frame for prediction
            else:
                output_prob = processor.step(image)

        # convert output probabilities to alpha matte
        mask = processor.output_prob_to_mask(output_prob)

        # visualize prediction
        pha = mask.unsqueeze(2).cpu().numpy()
        com_np = image_np / 255. * pha + bgr * (1 - pha)
        
        # DONOT save the warmup frame
        if ti > (n_warmup-1):
            com_np = (com_np*255).astype(np.uint8)
            pha = (pha*255).astype(np.uint8)
            fgrs.append(com_np)
            phas.append(pha)
            if save_image:
                cv2.imwrite(f'{output_path}/{video_name}/pha/{str(ti-n_warmup).zfill(5)}.png', pha)
                cv2.imwrite(f'{output_path}/{video_name}/fgr/{str(ti-n_warmup).zfill(5)}.png', com_np[...,[2,1,0]])

    phas = np.array(phas)
    fgrs = np.array(fgrs)

    imageio.mimwrite(f'{output_path}/{video_name}_fgr.mp4', fgrs, fps=fps, quality=7)
    imageio.mimwrite(f'{output_path}/{video_name}_pha.mp4', phas, fps=fps, quality=7)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_path', type=str, default="inputs/video/test-sample1.mp4", help='Path of the input video or frame folder.')
    parser.add_argument('-m', '--mask_path', type=str, default="inputs/mask/test-sample1.png", help='Path of the first-frame segmentation mask.')
    parser.add_argument('-o', '--output_path', type=str, default="results/", help='Output folder. Default: results')
    parser.add_argument('-c', '--ckpt_path', type=str, default="pretrained_models/matanyone.pth", help='Path of the MatAnyone model.')
    parser.add_argument('-w', '--warmup', type=str, default="10", help='Number of warmup iterations for the first frame alpha prediction.')
    parser.add_argument('-e', '--erode_kernel', type=str, default="10", help='Erosion kernel on the input mask.')
    parser.add_argument('-d', '--dilate_kernel', type=str, default="10", help='Dilation kernel on the input mask.')
    parser.add_argument('--suffix', type=str, default="", help='Suffix to specify different target when saving, e.g., target1.')
    parser.add_argument('--save_image', action='store_true', default=False, help='Save output frames. Default: False')
    parser.add_argument('--max_size', type=str, default="-1", help='When positive, the video will be downsampled if min(w, h) exceeds. Default: -1 (means no limit)')

    
    args = parser.parse_args()

    main(input_path=args.input_path, \
         mask_path=args.mask_path, \
         output_path=args.output_path, \
         ckpt_path=args.ckpt_path, \
         n_warmup=args.warmup, \
         r_erode=args.erode_kernel, \
         r_dilate=args.dilate_kernel, \
         suffix=args.suffix, \
         save_image=args.save_image, \
         max_size=args.max_size)
