import os
import sys
import argparse
import glob
import copy

import zarr, numpy as np
import torch

import cv2
import scipy.ndimage

from utils.Poisson_blend_img import Poisson_blend_img
from tool.zget_flowNN_gradient import zget_flowNN_gradient
from tool.spatial_inpaint import spatial_inpaint
from tool.frame_inpaint import DeepFillv1

from utils.common_utils import gradient_mask
from utils.common_utils import binary_fill_holes

import psutil
import ray
from tqdm import trange, tqdm

DEVICE = 'cuda'

NUM_CPUS = psutil.cpu_count(logical=False)

ray.init(num_cpus = NUM_CPUS)

def rayProgressBar(refs, desc):
    bar = tqdm(total=len(refs), desc=desc)
    tdone = 0
    while True:
        done, waiting = ray.wait(refs)
        ray.get(done)
        
        tdone+= len(done)
        bar.update(len(done))
        
        refs = waiting
        if not refs:
            break

    bar.close()


def create_dir(dir):
    """Creates a directory if not exist.
    """
    if not os.path.exists(dir):
        os.makedirs(dir)

def find_fpb(args, num):
    if num <= args.fpb:
        return num
    for n in range(1,(num+1)):
        if (num % n == 0) and (num / n <= args.fpb):
            return args.fpb if num // n == 1 else num // n

def stepx(x, steps, total, margin):
    """Return slices with and without interpolation between batches.
    """
    start = 0 if x - margin <= 0 else x - margin
    end = total if steps + x + margin >= total else steps + x + margin
    #end = end if start > 0 else end + margin
    Interpolated = slice(start, end)
    nonInterpolated = slice(x, x + steps) if (x + steps) < total else slice(x, total)

    return Interpolated, nonInterpolated

@ray.remote
def rayPoisson_blend_img(args, dataset, iter, frame):

    zvideo, zmasks, zmask_gradient, zgradient_x, zgradient_y = \
    dataset['video'], dataset['masks'], dataset['masks_dilated'], dataset['gradient_x'], dataset['gradient_y']

    zmask_gradient[:, :, frame] = binary_fill_holes(zmask_gradient[:, :, frame])

    # After one gradient propagation iteration
    img = zvideo[:, :, :, frame] / 255.
    imgH, imgW, _ = img.shape
    if zmasks[:, :, frame].sum() > 0:
        try:
            frameBlend, UnfilledMask = \
            Poisson_blend_img(img,
                              zgradient_x[:, 0 : imgW - 1, :, frame],
                              zgradient_y[0 : imgH - 1, :, :, frame],
                              zmasks[:, :, frame],
                              zmask_gradient[:, :, frame])
            # UnfilledMask = scipy.ndimage.binary_fill_holes(UnfilledMask).astype(np.bool)
        except:
            frameBlend, UnfilledMask = img, zmasks[:, :, frame]

        frameBlend = np.clip(frameBlend, 0, 1.0)
        tmp = cv2.inpaint((frameBlend * 255).astype(np.uint8), UnfilledMask.astype(np.uint8), 3, cv2.INPAINT_TELEA).astype(np.float32) / 255.
        frameBlend[UnfilledMask, :] = tmp[UnfilledMask, :]

        zvideo[:, :, :, frame] = (frameBlend * 255).astype(np.uint8)
        zmasks[:, :, frame] = UnfilledMask

        frameBlend_ = copy.deepcopy(frameBlend)
        # Green indicates the regions that are not filled yet.
        frameBlend_[zmasks[:, :, frame], :] = [0, 1., 0]
    else:
        frameBlend_ = img

    cv2.imwrite(os.path.join(args.outroot, 'frame_seamless_comp_' + str(iter), '%05d.png'%frame), frameBlend_ * 255.)

@ray.remote
def recalculate_gradients(dataset, frame):
    zvideo, zmasks, zmask_gradient, zgradient_x, zgradient_y = \
    dataset['video'], dataset['masks'], dataset['masks_dilated'], dataset['gradient_x'], dataset['gradient_y']

    img = zvideo[:, :, :, frame] / 255.
    imgH, imgW, _ = img.shape

    zmask_gradient[:, :, frame] = gradient_mask(zmasks[:, :, frame])

    gradient_x = np.concatenate((np.diff(img, axis=1), np.zeros((imgH, 1, 3), dtype=np.float32)), axis=1)
    gradient_x[zmask_gradient[:, :, frame], :] = 0
    gradient_y = np.concatenate((np.diff(img, axis=0), np.zeros((1, imgW, 3), dtype=np.float32)), axis=0)
    gradient_y[zmask_gradient[:, :, frame], :] = 0

    zgradient_x[:, :, :, frame] = gradient_x
    zgradient_y[:, :, :, frame] = gradient_y

def video_completion_seamless(args):
    rargs = ray.put(args)
    iter = 0
    zdata = zarr.open_group(os.path.join(args.outroot, 'datasets.zarr'), mode='r+')

    zvideo = zdata['video']
    zmasks = zdata['masks']
    zgradient_x = zdata['gradient_x']
    zgradient_y = zdata['gradient_y']
    zmask_gradient = zdata['masks_dilated']
    zvideo_flow = zdata['flow_comp']

    imgH, imgW, _, nFrame = zvideo.shape
    # Image inpainting model.
    deepfill = DeepFillv1(pretrained_model=args.deepfill_model, image_shape=[imgH, imgW])

    # We iteratively complete the video.
    while(np.sum(zmasks) > 0):
        create_dir(os.path.join(args.outroot, 'frame_seamless_comp_' + str(iter)))
        #TODO: Find a better way to calculate flowNN_gradient on large frameset
        framesBatch = find_fpb(args, nFrame)
        for x in range(0, nFrame, framesBatch):
            _, rFrames = stepx(x, framesBatch, nFrame, 50)
            # Gradient propagation.
            zget_flowNN_gradient(args,
                                zgradient_x,
                                zgradient_y,
                                zmask_gradient,
                                zvideo_flow,
                                None,
                                None,
                                rFrames)

        rzdata = ray.put(zdata)
        rayProgressBar(
            [rayPoisson_blend_img.options(name='Poisson_blend({})'.format(frame)).remote(rargs, rzdata, iter, frame) for frame in range(nFrame)],
            'Poisson blending'
        )

        #mask, video_comp = spatial_inpaint(deepfill, zmasks, zvideo)
        keyFrameInd = np.argmax(np.sum(np.sum(zmasks, axis=0), axis=0))
        keyFrame = zvideo[:, :, :, keyFrameInd]
        print('Spartial inpainting on frame {0}'.format(keyFrameInd))
        with torch.no_grad():
            img_res = deepfill.forward(zvideo[:, :, :, keyFrameInd], zmasks[:, :, keyFrameInd])

        keyFrame[zmasks[:, :, keyFrameInd], :] = img_res[zmasks[:, :, keyFrameInd], :]
        zvideo[:, :, :, keyFrameInd] = keyFrame
        zmasks[:, :, keyFrameInd] = False

        # Re-calculate gradient_x/y_filled and mask_gradient
        rayProgressBar(
            [recalculate_gradients.remote(rzdata, frame) for frame in range(nFrame)],
            'Re-calculating gradients'
        )
        iter += 1

    create_dir(os.path.join(args.outroot, 'frame_seamless_comp_' + 'final'))
    for i in range(nFrame):
        img = zvideo[:, :, :, i]
        cv2.imwrite(os.path.join(args.outroot, 'frame_seamless_comp_' + 'final', '%05d.png'%i), img)

def main(args):

    video_completion_seamless(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # video completion
    parser.add_argument('--outroot', default='./result/', help="output directory")
    parser.add_argument('--fpb', default=300, type=int, help="Frames per batch. Adjust according to RAM availability")
    
    #flowNN
    parser.add_argument('--consistencyThres', dest='consistencyThres', default=np.inf, type=float, help='flow consistency error threshold')
    parser.add_argument('--alpha', dest='alpha', default=0.1, type=float)
    parser.add_argument('--Nonlocal', dest='Nonlocal', default=False, type=bool)

    # Deepfill
    parser.add_argument('--deepfill_model', default='./weight/imagenet_deepfill.pth', help="restore checkpoint")

    args = parser.parse_args()

    main(args)
