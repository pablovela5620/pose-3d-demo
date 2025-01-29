import tyro

from pose_3d_demo.apis.triangulate import ViewEasyMocapConfig, triangulate_easymocap

# Example usage
if __name__ == "__main__":
    triangulate_easymocap(
        tyro.cli(
            ViewEasyMocapConfig,
            description="Visualize EasyMocap Triangulation",
        )
    )
