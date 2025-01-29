from dataclasses import dataclass
from pathlib import Path
from timeit import default_timer as timer

import cv2
import rerun as rr
import supervision as sv
from inference import get_model
from inference.core.models.base import Model
from mmcv.video import VideoReader
from simplecv.rerun_log_utils import RerunTyroConfig
from tqdm import tqdm


@dataclass
class PoseVideoConfig:
    video_path: Path
    rr_config: RerunTyroConfig
    confidence_threshold: float = 0.7


def pose_from_video(config: PoseVideoConfig) -> None:
    parent_log_path = Path("world")

    start_load: float = timer()
    model: Model = get_model(model_id="yolov8x-pose-640")
    print(f"Model loaded in {timer() - start_load:.2f} seconds")

    video_reader = VideoReader(filename=str(config.video_path))

    # average time taken
    average_time = 0
    pbar = tqdm(video_reader, total=len(video_reader))
    for frame_idx, bgr in enumerate(pbar):
        rr.set_time_sequence("frame_idx", frame_idx)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        start_infer = timer()
        results = model.infer(rgb)[0]
        average_time += timer() - start_infer

        # Any results object would work, regardless of which inference API is used
        keypoints = sv.KeyPoints.from_inference(results)
        annotated_image = sv.EdgeAnnotator(color=sv.Color.GREEN, thickness=5).annotate(
            rgb, keypoints
        )
        rr.log(
            f"{parent_log_path}/annotated_image",
            rr.Image(annotated_image).compress(jpeg_quality=90),
        )

    average_time /= frame_idx
    print(f"Average inference time: {average_time:.2f} seconds")
