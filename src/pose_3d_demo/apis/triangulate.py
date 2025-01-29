from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from timeit import default_timer as timer
from typing import TypeVar

import mmcv
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import supervision as sv
from inference import get_model
from inference.core.models.base import Model
from jaxtyping import Float, UInt8
from mmcv.video import VideoReader
from numpy import ndarray
from simplecv.camera_parameters import Intrinsics, PinholeParameters, rescale_intri
from simplecv.data.easymocap import load_cameras
from simplecv.ops.triangulate import batch_triangulate
from simplecv.rerun_log_utils import RerunTyroConfig, log_pinhole
from tqdm import tqdm

from pose_3d_demo.coco_info import COCO_17_ID2NAME, COCO_17_IDS, COCO_17_LINKS

BgrImageType = TypeVar("BgrImageType", bound=UInt8[ndarray, "H W 3"])
RgbImageType = TypeVar("RgbImageType", bound=UInt8[ndarray, "H W 3"])
# stop numpy from using scientific notation
np.set_printoptions(suppress=True)


@dataclass
class ViewEasyMocapConfig:
    data_dir: Path
    rr_config: RerunTyroConfig
    camera_subset: list[int] | None = None
    confidence_threshold: float = 0.7
    resize_factor: int = 4


class MultiVideoReader:
    def __init__(self, video_paths: list[Path]) -> None:
        # check that all video_paths are valid
        for video_path in video_paths:
            assert video_path.exists(), f"{video_path} does not exist"

        self.video_readers: list[VideoReader] = [
            VideoReader(str(video_path)) for video_path in video_paths
        ]
        # make sure all video_readers have the same length
        assert all(
            len(reader) == len(self.video_readers[0]) for reader in self.video_readers
        )

    def __len__(self) -> int:
        # Use minimum length to ensure safe iteration
        return min(len(reader) for reader in self.video_readers)

    def __iter__(self) -> Generator[list[BgrImageType] | None, None, None]:
        while True:
            bgr_list: list[BgrImageType] = []
            for reader in self.video_readers:
                bgr_image: BgrImageType | None = reader.read()
                match bgr_image:
                    case _ if bgr_image is not None:
                        bgr_list.append(bgr_image)
                    case None:
                        return
            yield bgr_list


def set_pose_annotation_context() -> None:
    rr.log(
        "/",
        rr.AnnotationContext(
            rr.ClassDescription(
                info=rr.AnnotationInfo(id=1, label="Person"),
                keypoint_annotations=[
                    rr.AnnotationInfo(id=id, label=name)
                    for id, name in COCO_17_ID2NAME.items()
                ],
                keypoint_connections=COCO_17_LINKS,
            )
        ),
        timeless=True,
    )


def create_blueprint(
    parent_log_path: Path, cameras: list[PinholeParameters]
) -> rrb.Blueprint:
    # get only 4 cameras evenly spaced
    n_cameras = len(cameras)
    step = n_cameras // 4
    cameras = cameras[::step][:4]

    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(),
            rrb.Vertical(
                contents=[
                    rrb.Spatial2DView(
                        origin=f"{parent_log_path}/{cam.name}/pinhole/",
                        contents=[
                            "+ $origin/**",
                        ],
                    )
                    for cam in cameras
                ]
            ),
            column_shares=[4, 1],
        ),
        collapse_panels=True,
    )
    return blueprint


def triangulate_easymocap(config: ViewEasyMocapConfig) -> None:
    parent_log_path = Path("world")
    rr.log(f"{parent_log_path}", rr.ViewCoordinates.FLU, static=True)
    set_pose_annotation_context()
    cameras: list[PinholeParameters] = load_cameras(data_path=config.data_dir)
    video_paths: list[Path] = sorted((config.data_dir / "videos").glob("*.mp4"))

    # filter cameras and video_paths based on camera_subset
    if config.camera_subset is not None:
        cameras = [cameras[i] for i in config.camera_subset]
        video_paths = [video_paths[i] for i in config.camera_subset]

    blueprint = create_blueprint(parent_log_path=parent_log_path, cameras=cameras)
    rr.send_blueprint(blueprint)

    # match config.logging_method:
    multiview_reader = MultiVideoReader(video_paths)

    start_load: float = timer()
    keypoint_model: Model = get_model(model_id="yolov8x-pose-640")
    print(f"Model loaded in {timer() - start_load:.2f} seconds")

    for cam in cameras:
        camera_log_path: Path = parent_log_path / cam.name
        # resize the image to the camera resolution
        target_height: int = int(cam.intrinsics.height // config.resize_factor)
        target_width: int = int(cam.intrinsics.width // config.resize_factor)
        rescaled_intri: Intrinsics = rescale_intri(
            cam.intrinsics, target_width=target_width, target_height=target_height
        )
        # need to create new cameras in order to generate new projection matrices
        new_cam: PinholeParameters = PinholeParameters(
            name=cam.name,
            extrinsics=cam.extrinsics,
            intrinsics=rescaled_intri,
        )
        # replace the old camera with the new camera
        cameras[cameras.index(cam)] = new_cam
        # cameras are stationary, so we log the pinhole parameters only once
        log_pinhole(camera=new_cam, cam_log_path=camera_log_path, static=True)

    pbar = tqdm(multiview_reader, total=len(multiview_reader))
    bgr_list: list[BgrImageType]

    for idx, bgr_list in enumerate(pbar):
        rr.set_time_sequence("frame_idx", idx)
        bgr: BgrImageType
        cam: PinholeParameters
        joints_xyc_list: list[Float[ndarray, "num_keypoints 3"]] = []
        projection_all_list: list[Float[np.ndarray, "3 4"]] = []
        for bgr, cam in zip(bgr_list, cameras, strict=True):
            image_height, image_width, _ = bgr.shape
            camera_log_path: Path = parent_log_path / cam.name
            pinhole_log_path: Path = camera_log_path / "pinhole"
            # resize the image to the camera resolution
            target_height: int = int(image_height // config.resize_factor)
            target_width: int = int(image_width // config.resize_factor)
            bgr = mmcv.imresize(bgr, (target_width, target_height))
            rr.log(
                f"{pinhole_log_path / 'image'}",
                rr.Image(bgr, color_model=rr.ColorModel.BGR).compress(jpeg_quality=75),
            )

            # get projection matrix
            projection_matrix: Float[np.ndarray, "3 4"] = cam.projection_matrix
            projection_all_list.append(projection_matrix)
            # estimate pose using roboflow infernce and supervision
            rgb: RgbImageType = mmcv.bgr2rgb(bgr)
            results = keypoint_model.infer(rgb)[0]

            key_points: sv.KeyPoints = sv.KeyPoints.from_inference(results)
            # filter keypoints based on number of detections and stack into a xyc array for triangulation
            match key_points.xy.shape:
                case (0, *_):
                    # no keypoints detected,
                    joints_xyc: Float[ndarray, "num_keypoints 3"] = np.full(
                        (len(COCO_17_IDS), 3), 0.0
                    )
                case (1, n_keypoints, 2):
                    # only get the high confidence keypoints
                    # round final keypoints to int
                    joints_2d: Float[ndarray, "num_keypoints 2"] = np.round(
                        key_points.xy[0]
                    )
                    # get the confidence of the best detection
                    best_confidence: Float[ndarray, "num_keypoints"] = (  # noqa: UP037
                        key_points.confidence[0]
                    )
                    # make scores either 0 or 1 based on the threshold
                    best_confidence[best_confidence < config.confidence_threshold] = 0
                    best_confidence[best_confidence >= config.confidence_threshold] = 1

                    # for triangulation, need to have xyc meaning x, y, confidence
                    joints_xyc: Float[ndarray, "num_keypoints 3"] = np.concatenate(
                        [joints_2d, best_confidence[:, None]], axis=1
                    )
                case (n, *_):
                    confidence: Float[ndarray, "num_detection num_keypoints"] = (
                        key_points.confidence
                    )
                    # get the average confidence of all keypoints per detection
                    avg_confidence: Float[ndarray, "num_detection"] = (  # noqa: UP037
                        confidence.mean(axis=1)
                    )
                    # get idx of max confidence for each detection
                    best_detection_idx: int = int(np.argmax(avg_confidence))
                    joints_2d: Float[ndarray, "num_keypoints 2"] = key_points.xy[
                        best_detection_idx
                    ]
                    joints_2d: Float[ndarray, "num_keypoints 2"] = np.round(joints_2d)
                    # get the confidence of the best detection
                    best_confidence: Float[ndarray, "num_detection"] = confidence[  # noqa: UP037
                        best_detection_idx
                    ]
                    # make scores either 0 or 1 based on the threshold
                    best_confidence[best_confidence < config.confidence_threshold] = 0
                    best_confidence[best_confidence >= config.confidence_threshold] = 1

                    # for triangulation, need to have xyc meaning x, y, confidence
                    joints_xyc: Float[ndarray, "num_keypoints 3"] = np.concatenate(
                        [joints_2d, best_confidence[:, None]], axis=1
                    )

            # keep confidence as 0/1 for triangulation
            joints_xyc_list.append(joints_xyc.copy())
            # if the last dimension value is 0, then the keypoint is not detected, make it nan only for logging
            joints_xyc[joints_xyc[:, 2] == 0] = np.nan
            rr.log(
                f"{pinhole_log_path / 'image' / 'pose'}",
                rr.Points2D(
                    positions=joints_xyc[..., :2],
                    radii=2.5,
                    class_ids=1,
                    keypoint_ids=COCO_17_IDS,
                    show_labels=False,
                ),
            )

        multiview_joints_xyc: Float[ndarray, "nViews nKeypoints 3"] = np.stack(
            joints_xyc_list
        )
        P_all: Float[ndarray, "nViews 3 4"] = np.array([P for P in projection_all_list])
        joints_xyzc: Float[ndarray, "nKeypoints 4"] = batch_triangulate(
            keypoints_2d=multiview_joints_xyc,
            projection_matrices=P_all,
            min_views=2,
        )
        # filter keypoints_3d by confidence
        joints_xyzc[joints_xyzc[:, -1] < 0.5] = np.nan

        rr.log(
            f"{parent_log_path}/keypoints_3d",
            rr.Points3D(
                positions=joints_xyzc[:, :3],
                class_ids=1,
                keypoint_ids=COCO_17_IDS,
                show_labels=False,
            ),
        )
