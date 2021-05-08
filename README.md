# [ECCV 2020] Flow-edge Guided Video Completion

### [[Original code](https://github.com/vt-vl-lab/FGVC)]
<p align='center'>
<img src='http://chengao.vision/FGVC/files/FGVC_teaser.png' width='900'/>
</p>

We present a new flow-based video completion algorithm. Previous flow completion methods are often unable to retain the sharpness of motion boundaries. Our method first extracts and completes motion edges, and then uses them to guide piecewise-smooth flow completion with sharp edges. Existing methods propagate colors among local flow connections between adjacent frames. However, not all missing regions in a video can be reached in this way because the motion boundaries form impenetrable barriers. Our method alleviates this problem by introducing non-local flow connections to temporally distant frames, enabling propagating video content over motion boundaries. We validate our approach on the DAVIS dataset. Both visual and quantitative results show that our method compares favorably against the state-of-the-art algorithms.
<br/>

**[ECCV 2020] Flow-edge Guided Video Completion**
<br/>
[Chen Gao](http://chengao.vision), [Ayush Saraf](#), [Jia-Bin Huang](https://filebox.ece.vt.edu/~jbhuang/), and [Johannes Kopf](https://johanneskopf.de/)
<br/>
In European Conference on Computer Vision (ECCV), 2020

## Ray and Zarr implementation

The original code allocates all frames into GPU memory, which may limit the use only on expensive hardware. To alleviate that, the code was tweaked and features added, like Ray for multiprocessing and Zarr for storage. Depending on how many frames are, this code may use a good amount of RAM in video completion due to unoptimized flowNN process. A per batch frame processing was implemented to get around that, but Consider it as experimental, as it may produce unexpected results.

## Prerequisites

- Linux (tested on CentOS Linux release 7.4.1708)
- Anaconda
- Python 3.8 (tested on 3.8.5)
- PyTorch 1.6.0

and the Python dependencies listed in requirements.txt

- To get started, please run the following commands:
  ```
  conda create -n FGVC
  conda activate FGVC
  conda install pytorch=1.6.0 torchvision=0.7.0 cudatoolkit=10.1 -c pytorch
  conda install matplotlib scipy
  pip install -r requirements.txt
  ```

- Next, please download the model weight and demo data using the following command:
  ```
  chmod +x download_data_weights.sh
  ./download_data_weights.sh
  ```

## Quick start

First step: Create flow, process mask and gradients

- Object removal:
```bash
python 1-create-flow.py \
       --mode object_removal \
       --path ./data/tennis \
       --path_mask ./data/tennis_mask \
       --outroot ./result/tennis_removal \
```

- FOV extrapolation:
```bash
python 1-create-flow.py \
       --mode video_extrapolation \
       --path ./data/tennis \
       --outroot ./result/tennis_extrapolation \
       --H_scale 2 \
       --W_scale 2 \
```

Second step: Video completion
```bash
python 2-video-completion.py \
       --outroot ./result/tennis_extrapolation \
```


## License
This work is licensed under MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments
- Paper and original code are from [Gao-ECCV-FGVC](https://github.com/vt-vl-lab/FGVC)
  If you find this code useful for your research, please consider citing the following paper:

    @article{Gao-ECCV-FGVC,
        author    = {Gao, Chen and Saraf, Ayush and Huang, Jia-Bin and Kopf, Johannes},
        title     = {Flow-edge Guided Video Completion},
        booktitle = {European Conference on Computer Vision},
        year      = {2020}
    }

- Our flow edge completion network builds upon [EdgeConnect](https://github.com/knazeri/edge-connect).
- Our image inpainting network is modified from [DFVI](https://github.com/nbei/Deep-Flow-Guided-Video-Inpainting).
