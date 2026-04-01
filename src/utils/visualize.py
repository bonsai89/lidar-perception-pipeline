import os
import numpy as np
import vispy
from vispy.scene import visuals
from vispy.scene.cameras import TurntableCamera
from vispy.scene import SceneCanvas
from typing import List


class Visualizer:
    def __init__(self, data_folder: str, num_attributes: int = 3):
        """
        Initialize the visualizer with the given data folder and number of attributes per point.

        :param data_folder: Path to the folder containing point cloud data files.
        :param num_attributes: Number of attributes per point (e.g., 3 for x, y, z).
        """
        self._frames = self._read_data(data_folder)
        self._frame_index = 0
        self._num_attributes = num_attributes

        # Create the canvas for visualization
        self.canvas = SceneCanvas(keys='interactive', 
                                  show=True, 
                                  size=(1600, 900))
        self.canvas.events.key_press.connect(self._key_press)
        self.canvas.events.draw.connect(self._draw)

        # Set up a grid layout and add a view for point cloud visualization
        self.grid = self.canvas.central_widget.add_grid()
        self.scan_view = vispy.scene.widgets.ViewBox(parent=self.canvas.scene,
                                                     camera=TurntableCamera(distance=30.0))
        self.grid.add_widget(self.scan_view)

        # Add markers for point cloud and an XYZ axis visual
        self.scan_vis = visuals.Markers()
        self.scan_view.add(self.scan_vis)
        visuals.XYZAxis(parent=self.scan_view.scene)

        # Load the first frame
        self._update_input()

    def _read_data(self, data_folder: str) -> List[str]:
        """
        Read and return a sorted list of file paths in the data folder.

        :param data_folder: Path to the folder containing data files.
        :return: Sorted list of file paths.
        """
        return [os.path.join(data_folder, f) for f in sorted(os.listdir(data_folder))]

    def _update_input(self):
        """
        Load the current frame's point cloud data and update the visualization.
        """
        file = self._frames[self._frame_index]
        points = np.fromfile(file, dtype=np.float32).reshape(-1, self._num_attributes)[:, :3]

        # Normalize z-values for color mapping
        max_z = min(np.max(points[:, 2]), 2)
        min_z = max(np.min(points[:, 2]), -3)
        colors = np.clip(points[:, 2], min_z, max_z)
        colors = (colors - min_z) / (max_z - min_z) * 0.8 + 0.1
        colors = np.stack([colors, 1 - colors, 1 - colors * colors, np.ones_like(colors, dtype=float) * 0.8], axis=1)

        # Update the canvas title and point cloud data
        self.canvas.title = f"Frame: {self._frame_index}"
        self.scan_vis.set_data(points,
                               face_color=colors,
                               edge_color=colors,
                               size=1.0)

    def _key_press(self, event):
        """
        Handle key press events to navigate through frames.

        - Press 'N' to move to the next frame.
        - Press 'B' to move to the previous frame.

        :param event: Key press event.
        """
        if event.key == 'N':
            self._frame_index += 1
            if self._frame_index == len(self._frames):
                self._frame_index = 0
            self._update_input()

        if event.key == 'B':
            self._frame_index -= 1
            if self._frame_index < 0:
                self._frame_index = len(self._frames) - 1
            self._update_input()

    def _draw(self, event):
        """
        Handle the draw event to ensure the canvas remains interactive.

        :param event: Draw event.
        """
        if self.canvas.events.key_press.blocked():
            self.canvas.events.key_press.unblock()

    def run(self):
        """
        Start the visualization application.
        """
        self.canvas.app.run()


if __name__ == "__main__":
    data_folder = "data/train_augmented/bicyclist"  # Path to the folder containing point cloud data
    num_attributes = 3  # Use 3 for x, y, z. Change to 5 if visualizing optional challenge data
    vis = Visualizer(data_folder, num_attributes)
    vis.run()

