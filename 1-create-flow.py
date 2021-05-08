import os
import sys
import argparse
import glob

import zarr, numpy as np
import torch

import cv2
from PIL import Image
import scipy.ndimage
from skimage.feature import canny
import torchvision.transforms.functional as F

from RAFT import utils
from RAFT import RAFT

import utils.region_fill as rf
from utils.Poisson_blend import Poisson_blend
from utils.common_utils import gradient_mask
from utils.common_utils import binary_fill_holes

#from utils.common_utils import flow_edge
from edgeconnect.networks import EdgeGenerator_

import psutil
import ray
from tqdm import trange, tqdm

from zlib import crc32

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

def to_tensor(img):
    img = Image.fromarray(img)
    img_t = F.to_tensor(img).float()
    return img_t

def to_torch(img, rgb=True):
    if rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def infer(EdgeGenerator, device, flow_img_gray, edge, mask):

    # Add a pytorch dataloader
    flow_img_gray_tensor = to_tensor(flow_img_gray)[None, :, :].float().to(device)
    edge_tensor = to_tensor(edge)[None, :, :].float().to(device)
    mask_tensor = torch.from_numpy(mask.astype(np.float64))[None, None, :, :].float().to(device)

    # Complete the edges
    edges_masked = (edge_tensor * (1 - mask_tensor))
    images_masked = (flow_img_gray_tensor * (1 - mask_tensor)) + mask_tensor
    inputs = torch.cat((images_masked, edges_masked, mask_tensor), dim=1)
    with torch.no_grad():
        edges_completed = EdgeGenerator(inputs) # in: [grayscale(1) + edge(1) + mask(1)]
    edges_completed = edges_completed * mask_tensor + edge_tensor * (1 - mask_tensor)
    edge_completed = edges_completed[0, 0].data.cpu().numpy()
    edge_completed[edge_completed < 0.5] = 0
    edge_completed[edge_completed >= 0.5] = 1

    return edge_completed

def create_dir(dir):
    """Creates a directory if not exist.
    """
    if not os.path.exists(dir):
        os.makedirs(dir)

def create_dirs(args):
    dirs=(os.path.join(args.outroot, 'flow', 'backward_png'),
          os.path.join(args.outroot, 'flow', 'forward_png'),
          os.path.join(args.outroot, 'flow_comp', 'backward_png'),
          os.path.join(args.outroot, 'flow_comp', 'forward_png'),
    )
    for path in dirs:
        create_dir(path)

def initialize_RAFT(args):
    """Initializes the RAFT model.
    """
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to('cuda')
    model.eval()

    return model

def save_checksum(dataset, np_arr, process_num, frame):
    dataset['checksum'][process_num, frame] = crc32(np_arr.tobytes()) if np_arr is not None else 0

def valid_checksum(dataset, np_arr, process_num, frame):
    if dataset['checksum'][process_num, frame] == crc32(np_arr.tobytes()) \
    and dataset['checksum'][process_num, frame] != 0:
        return True
    else:
        return False

def divisible_by(shape, dv):
    for value in shape:
        if (value % dv) != 0:
            return False
    return True

@ray.remote
def extrapolation(args, dataset, shape, frame):
    """Prepares the data for video extrapolation.
    """
    zvideo, zmasks, zmasks_dilated, zflow_masks = \
        dataset['video'], dataset['masks'], dataset['masks_dilated'], dataset['flow_masks']

    if not valid_checksum(dataset, zmasks[..., frame], 3, frame) \
    or not valid_checksum(dataset, zflow_masks[..., frame], 4, frame) \
    or not valid_checksum(dataset, zmasks_dilated[..., frame], 5, frame)\
    or not valid_checksum(dataset, zvideo[..., frame], 9, frame):

        rimgH, rimgW = shape
        imgH, imgW = zvideo.shape[:2]

        flow_mask = np.ones(((imgH, imgW)), dtype=np.bool)
        flow_mask[rimgH, rimgW] = 0
        mask_dilated = gradient_mask(flow_mask)

        zmasks[..., frame] = flow_mask
        zflow_masks[..., frame] = flow_mask
        zmasks_dilated[...,frame] = mask_dilated
      
        # Extrapolates the FOV for video.
        zvideo[..., frame] = \
        cv2.inpaint(zvideo[..., frame], flow_mask.astype(np.uint8), 3, cv2.INPAINT_TELEA)

        save_checksum(dataset, zmasks[..., frame], 3, frame)
        save_checksum(dataset, zflow_masks[..., frame], 4, frame)
        save_checksum(dataset, zmasks_dilated[..., frame], 5, frame)
        save_checksum(dataset, zvideo[..., frame], 9, frame)

@ray.remote
def prepare_masks(args, dataset, frame):

        zmasks, zflow_masks = dataset['masks'], dataset['flow_masks']
        if valid_checksum(dataset, zmasks[..., frame], 3, frame) and \
           valid_checksum(dataset, zflow_masks[..., frame], 4, frame):
            return

        # Loads masks.
        filename_list = sorted(glob.glob(os.path.join(args.path_mask, '*.png')))

        mask_img = cv2.imread(filename_list[frame], cv2.IMREAD_GRAYSCALE)

        # Dilate 15 pixel so that all known pixel is trustworthy
        flow_mask_img = scipy.ndimage.binary_dilation(mask_img, iterations=15)
        # Close the small holes inside the foreground objects
        flow_mask_img = cv2.morphologyEx(flow_mask_img.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((21, 21),np.uint8)).astype(np.bool)
        flow_mask_img = binary_fill_holes(flow_mask_img)
        zflow_masks[:, :, frame] = flow_mask_img

        mask_img = scipy.ndimage.binary_dilation(mask_img, iterations=5)
        mask_img = binary_fill_holes(mask_img)
        zmasks[:, :, frame] = mask_img

        save_checksum(dataset, zmasks[..., frame], 3, frame)
        save_checksum(dataset, zflow_masks[..., frame], 4, frame)

@ray.remote(num_gpus = 0.5)
def calculate_flow(args, model, dataset, shape, frame):
    """Calculates optical flow.
    """
    zvideo, zflow = dataset['video'], dataset['flow']
    if valid_checksum(dataset, dataset['flow'][..., frame], 1, frame):
        return

    rimgH, rimgW = shape
    with torch.no_grad():
        for mode in range(2):
            prevf = to_torch(zvideo[rimgH, rimgW, :, frame])
            nextf = to_torch(zvideo[rimgH, rimgW, :, frame + 1])

            if mode == 0:
                _, flow = model(prevf, nextf, iters=20, test_mode=True)
                path_mode = 'forward'
            else:
               _, flow = model(nextf, prevf, iters=20, test_mode=True)
               path_mode = 'backward'

            flow = flow[0].permute(1, 2, 0).cpu().numpy()
            zflow[rimgH, rimgW, :, mode, frame] = flow
            # Flow visualization.
            flow_img = utils.flow_viz.flow_to_image(zflow[..., mode, frame])
            cv2.imwrite(os.path.join(args.outroot, 'flow', path_mode + '_png', '%09d.png'%frame), flow_img)

        save_checksum(dataset, dataset['flow'][..., frame], 1, frame)

@ray.remote
def complete_flow(args, dataset, frame):
    """Completes flow.
    """
    zflow, zflow_comp, zflow_masks = dataset['flow'], dataset['flow_comp'], dataset['flow_masks']
    if valid_checksum(dataset, dataset['flow_comp'][..., frame], 2, frame):
        return

    imgH, imgW = zflow.shape[:2]

    #compFlow = np.zeros(((imgH, imgW, 2, nFrame)), dtype=np.float32)

    for mode in range(2):
        flow = zflow[:, :, :, mode, frame]
        flow_mask_img = zflow_masks[:, :, frame + mode]
        flow_mask_gradient_img = gradient_mask(flow_mask_img)

        if args.edge_guide:
            zedges = dataset['edges']
            # imgH x (imgW - 1 + 1) x 2
            gradient_x = np.concatenate((np.diff(flow, axis=1), np.zeros((imgH, 1, 2), dtype=np.float32)), axis=1)
            # (imgH - 1 + 1) x imgW x 2
            gradient_y = np.concatenate((np.diff(flow, axis=0), np.zeros((1, imgW, 2), dtype=np.float32)), axis=0)

            # concatenate gradient_x and gradient_y
            gradient = np.concatenate((gradient_x, gradient_y), axis=2)

            # We can trust the gradient outside of flow_mask_gradient_img
            # We assume the gradient within flow_mask_gradient_img is 0.
            gradient[flow_mask_gradient_img, :] = 0

            # Complete the flow
            imgSrc_gy = gradient[:, :, 2 : 4]
            imgSrc_gy = imgSrc_gy[0 : imgH - 1, :, :]
            imgSrc_gx = gradient[:, :, 0 : 2]
            imgSrc_gx = imgSrc_gx[:, 0 : imgW - 1, :]
            zflow_comp[:, :, :, mode, frame] = Poisson_blend(flow, imgSrc_gx, imgSrc_gy, flow_mask_img, zedges[:, :, mode, frame])

        else:
            flow[:, :, 0] = rf.regionfill(flow[:, :, 0], flow_mask_img)
            flow[:, :, 1] = rf.regionfill(flow[:, :, 1], flow_mask_img)
            zflow_comp[:, :, :, mode, frame] = flow

        # Flow visualization.
        flow_img = utils.flow_viz.flow_to_image(zflow_comp[:, :, :, mode, frame])
        #flow_img = Image.fromarray(flow_img)

        # Saves the flow and flow_img.
        path_mode = 'forward' if mode == 0 else 'backward'
        cv2.imwrite(os.path.join(args.outroot, 'flow_comp', path_mode + '_png', '%09d.png'%frame), flow_img)

    save_checksum(dataset, dataset['flow_comp'][..., frame], 2, frame)

@ray.remote(num_gpus = 0.5)
def edge_completion(EdgeGenerator, dataset, frame):   

    zedges, zflow, zflow_masks = dataset['edges'], dataset['flow'], dataset['flow_masks']
    if valid_checksum(dataset, zedges[..., frame], 8, frame):
        return

    for mode in range(2):

        flow_mask_img = zflow_masks[:, :, frame + mode]

        flow_img_gray = (zflow[:, :, 0, mode, frame] ** 2 + zflow[:, :, 1, mode, frame] ** 2) ** 0.5
        flow_img_gray = flow_img_gray / flow_img_gray.max()

        edge_corr = canny(flow_img_gray, sigma=2, mask=(1 - flow_mask_img).astype(np.bool))
        edge_completed = infer(EdgeGenerator, torch.device('cuda:0'), flow_img_gray, edge_corr, flow_mask_img)
        zedges[:, :, mode, frame] = edge_completed

    save_checksum(dataset, zedges[..., frame], 8, frame)
    save_checksum(dataset, None, 2, frame)

@ray.remote
def get_gradients(dataset, frame):
    zvideo, zmasks, zmasks_dilated, zgradient_x, zgradient_y = \
    dataset['video'], dataset['masks'], dataset['masks_dilated'], dataset['gradient_x'], dataset['gradient_y']
    
    if valid_checksum(dataset, zmasks_dilated[..., frame], 5, frame) and \
       valid_checksum(dataset, zgradient_x[..., frame], 6, frame) and \
       valid_checksum(dataset, zgradient_y[..., frame], 7, frame):
        return

    img = zvideo[:, :, :, frame]
    imgH, imgW, _ = img.shape
    img[zmasks[:, :, frame], :] = 0
    img = cv2.inpaint(img, zmasks[:, :, frame].astype(np.uint8), 3, cv2.INPAINT_TELEA).astype(np.float32)  / 255.
    
    zmasks_dilated[:, :, frame] = gradient_mask(zmasks[:, :, frame])

    gradient_x = np.concatenate((np.diff(img, axis=1), np.zeros((imgH, 1, 3), dtype=np.float32)), axis=1)
    gradient_y = np.concatenate((np.diff(img, axis=0), np.zeros((1, imgW, 3), dtype=np.float32)), axis=0)
    gradient_x[zmasks_dilated[:, :, frame], :] = 0
    gradient_y[zmasks_dilated[:, :, frame], :] = 0

    zgradient_x[:, :, :, frame] = gradient_x
    zgradient_y[:, :, :, frame] = gradient_y

    save_checksum(dataset, zmasks_dilated[..., frame], 5, frame)
    save_checksum(dataset, zgradient_x[..., frame], 6, frame)
    save_checksum(dataset, zgradient_y[..., frame], 7, frame)

def video_completion_seamless(args):


    if args.gpu_profile == 'high':
        num_gpus = 0.25
    elif args.gpu_profile == 'medium':
        num_gpus = 0.5
    else:
        num_gpus = 1

    create_dirs(args)
    zdata = zarr.open_group(os.path.join(args.outroot, 'datasets.zarr'), mode='a')

    rargs = ray.put(args)
    # Flow model.
    RAFT_model = initialize_RAFT(args)
    rRAFT_model = ray.put(RAFT_model)

    # Loads frames.
    filename_list = glob.glob(os.path.join(args.path, '*.png')) + \
                    glob.glob(os.path.join(args.path, '*.jpg'))
    filename_list = sorted(filename_list)

    # Obtains imgH, imgW and nFrame.
    imgH, imgW = cv2.imread(filename_list[0]).shape[:2]
    nFrame = len(filename_list)

    if args.mode == 'object_removal' and not divisible_by((imgH, imgW,), 8):
        raise Exception('Height and width must be divisible by 8')

    if args.mode == 'video_extrapolation':
        # Defines new FOV.
        imgH_orig, imgW_orig = imgH, imgW
        imgH = int(8 * round((args.H_scale * imgH) / 8))
        imgW = int(8 * round((args.W_scale * imgW) / 8))
        H_start = int((imgH - imgH_orig) / 2)
        W_start = int((imgW - imgW_orig) / 2)

        shape = (
            slice(H_start , H_start + imgH_orig), 
            slice(W_start , W_start + imgW_orig),
        )
    else:
        shape = (
            slice(None , None), 
            slice(None, None),
        )

    #Create datasets
    zchecksum = zdata.require_dataset('checksum', shape=(10, nFrame),
        chunks=(10, 1), fill_value=0, dtype=np.int)
    zvideo = zdata.require_dataset('video', shape=(imgH, imgW, 3, nFrame),
        chunks=(imgH, imgW, 3, 1), fill_value=0, dtype=np.uint8) #0
    zflow = zdata.require_dataset('flow', shape=(imgH, imgW, 2, 2, nFrame - 1),
        chunks=(imgH, imgW, 2, 2, 1), fill_value=0, dtype=np.float32) #1
    zmasks = zdata.require_dataset('masks', shape=(imgH, imgW, nFrame),
        chunks=(imgH, imgW, 1), fill_value=0, dtype=np.bool) #3
    zflow_masks = zdata.require_dataset('flow_masks', shape=(imgH, imgW, nFrame),
        chunks=(imgH, imgW, 1), fill_value=0, dtype=np.bool) #4
    zmasks_dilated = zdata.require_dataset('masks_dilated', shape=(imgH, imgW, nFrame),
        chunks=(imgH, imgW, 1), fill_value=0, dtype=np.bool) #5
    zflow_comp = zdata.require_dataset('flow_comp', shape=(imgH, imgW, 2, 2, nFrame - 1),
        chunks=(imgH, imgW, 2, 2, 1), fill_value=0, dtype=np.float32) #2
    zgradient_x = zdata.require_dataset('gradient_x', shape=(imgH, imgW, 3, nFrame),
        chunks=(imgH, imgW, 3, 1), fill_value=0, dtype=np.float32) #6
    zgradient_y = zdata.require_dataset('gradient_y', shape=(imgH, imgW, 3, nFrame),
        chunks=(imgH, imgW, 3, 1), fill_value=0, dtype=np.float32) #7
    zedges = zdata.require_dataset('edges', shape=(imgH, imgW, 2, nFrame - 1),
        chunks=(imgH, imgW, 2, 1), fill_value=0, dtype=np.float32) if args.edge_guide else None #8

    if args.force:
        zchecksum[:] = 0

    rzdataset = ray.put(zdata)
    # Import frames.
    for frame in trange(nFrame, desc="Importing frames"):
        if not valid_checksum(zdata, zvideo[..., frame], 0, frame):
            zvideo[shape[0], shape[1], :, frame] = cv2.imread(filename_list[frame])
            save_checksum(zdata, zvideo[..., frame], 0, frame)

    rayProgressBar(
        [calculate_flow.options(num_gpus=num_gpus, name='calculate_flow({})'.format(frame)).remote(rargs, rRAFT_model, rzdataset, shape, frame) for frame in range(nFrame - 1)],
        'Calculating flows'
    )

    if args.mode == 'video_extrapolation':
        # Creates video and flow where the extrapolated region are missing.
        rayProgressBar(
            [extrapolation.options(name='prepare_masks({})'.format(frame)).remote(rargs, rzdataset, shape, frame) for frame in range(nFrame)],
            'Processing masks'
        )
    else:
        rayProgressBar(
            [prepare_masks.options(name='prepare_masks({})'.format(frame)).remote(rargs, rzdataset, frame) for frame in range(nFrame)],
            'Processing masks'
        )

    if args.edge_guide:
        # Edge completion model.
        EdgeGenerator = EdgeGenerator_()
        EdgeComp_ckpt = torch.load(args.edge_completion_model)
        EdgeGenerator.load_state_dict(EdgeComp_ckpt['generator'])
        EdgeGenerator.to(torch.device('cuda:0'))
        EdgeGenerator.eval()
        rEdgeGenerator = ray.put(EdgeGenerator)
        #zedges = zarr.open(os.path.join(args.outroot, 'edges_comp.zarr'), mode='w', shape=(imgH, imgW, 2, nFrame),
        #chunks=(imgH, imgW, 2, 1), dtype=np.float32)

        # Edge completion.
        rayProgressBar(
            [edge_completion.options(num_gpus=num_gpus, name='edge_completion({})'.format(frame)).remote(rEdgeGenerator, rzdataset, frame) for frame in range(nFrame - 1)],
            'Completing edges'
        )

    rayProgressBar(
        [complete_flow.options(name='complete_flow({})'.format(frame)).remote(rargs, rzdataset, frame) for frame in range(nFrame - 1)],
        'Completing flows'
    )

    rayProgressBar(
        [get_gradients.options(name='get_gradients({})'.format(frame)).remote(rzdataset, frame) for frame in range(nFrame)],
        'Calculating gradients'
    )

    print('\nFlow completion finished.\nYou may run python 2-video-completion.py --outroot {}'.format(args.outroot))

def main(args):

    assert args.mode in ('object_removal', 'video_extrapolation'), (
        "Accepted modes: 'object_removal', 'video_extrapolation', but input is %s"
    ) % mode

    video_completion_seamless(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # video completion
    parser.add_argument('--edge_guide', action='store_true', help='Whether use edge as guidance to complete flow')
    parser.add_argument('--mode', default='object_removal', help="modes: object_removal / video_extrapolation")
    parser.add_argument('--path', default='./data/tennis', help="dataset for evaluation")
    parser.add_argument('--path_mask', default='./data/tennis_mask', help="mask for object removal")
    parser.add_argument('--outroot', default='./result/', help="output directory")
    parser.add_argument('--consistencyThres', dest='consistencyThres', default=np.inf, type=float, help='flow consistency error threshold')
    parser.add_argument('--alpha', dest='alpha', default=0.1, type=float)
    parser.add_argument('--Nonlocal', dest='Nonlocal', default=False, type=bool)
    parser.add_argument('--force', action='store_true', help="Ignore checksum.")

    # RAFT
    parser.add_argument('--model', default='./weight/raft-things.pth', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')

    # Edge completion
    parser.add_argument('--edge_completion_model', default='./weight/edge_completion.pth', help="restore checkpoint")

    # extrapolation
    parser.add_argument('--H_scale', dest='H_scale', default=2, type=float, help='H extrapolation scale')
    parser.add_argument('--W_scale', dest='W_scale', default=2, type=float, help='W extrapolation scale')

    #Ray options
    parser.add_argument('--gpu_profile', default='medium', help="GPU usage profile: low / medium / high")

    args = parser.parse_args()

    main(args)
