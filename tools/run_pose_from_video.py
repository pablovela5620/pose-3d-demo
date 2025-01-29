import tyro

from pose_3d_demo.apis.pose_from_video import PoseVideoConfig, pose_from_video

# Example usage
if __name__ == "__main__":
    pose_from_video(
        tyro.cli(
            PoseVideoConfig,
            description="Visualize EasyMocap Triangulation",
        )
    )
