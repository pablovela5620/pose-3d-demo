# Pose3D RoboFlow Example

Use Roboflow Inferece + Supervision to perform 3D keypoint detection via triangulation on an EasyMocap Dataset
<p align="center">
  <img src="media/pose-triangulation.gif" alt="example output" width="720" />
</p>

TLDR
1. Estimates 2d keypoints for each camera
2. Filters keypoints based on confidence
3. Triangulates them in 3D using camera intrinsics + extrinsics
4. Profit??

Makes use of
- Roboflow Inference for estimating 2D keypoints
- Roboflow Supervision to allow consistent access of keypoints through different models
- Pixi for easy installation
- Rerun for visualizing in 2D + 3D
- Beartype + Jaxtyping for runtime type checking of tensor types and shapes

## Installation
Currently only works on linux with an Nvidia GPU
Make sure you have the [Pixi](https://pixi.sh/latest/#installation) package manager installed
```bash
git clone https://github.com/pablovela5620/pose-3d-demo.git
cd pose-3d-demo
pixi run post-install
```
this will install all deps and download example data


## Usage
### Quick start
```bash
pixi run triangulate-example
```
### Single Video Keypoints
using pixi tasks
```bash
pixi run video-example
```

using python through conda shell (drops you into an activated conda enviroment)
```bash
pixi shell
python tools/run_pose_from_video.py --video-path data/street_dance/videos/01.mp4
```

### Triangulation
using pixi tasks
```bash
pixi run triangulate-example
```

using python through conda shell (drops you into an activated conda enviroment)
```bash
pixi shell
python tools/run_triangulation.py --data-dir data/street_dance
```

## Downsides
Currently only works with easymocap datasets and requires calibrated cameras. Also not doing any sort of robust filtering or tracking.
Just naive triangulation with keypoints and some rudamentary filtering. Lots that can be done in a much better way
Also not optimized at all, one could find many ways to make this run alot faster