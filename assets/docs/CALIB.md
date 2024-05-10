# 📷 Calibration Guide

Our experimental platform includes an inhand camera and several global cameras. Here are the recommended calibration steps.

1. **Hand-Eye Calibration**. The transformation matrix from the inhand camera to the robot's tool center point (tcp), denoted as `INHAND_CAM_TCP`, remains constant and is established through hand-eye calibration techniques. After hand-eye calibration, this matrix is stored in `dataset/constants.py`. Generally, this matrix remains unchanged, so hand-eye calibration is typically required only once. Subsequent recalibration is necessary only if the position of the inhand camera relative to the robotic arm is altered.

2. **Camera Calibration**. We utilize ArUco markers for camera calibration. The opencv-python package includes an ArUco detector, enabling us to derive the transformation matrix from the camera to the marker. To proceed, move the robot to a specific pose, print out the marker, and place it where all cameras can detect it. Save the robot's pose in `tcp.npy` within the calibration folder, and store the transformation matrices from all cameras to the marker as a dictionary in `extrinsics.npy` in the calibration folder. Whenever there's a change in the camera position, this step needs to be repeated for recalibration. Please refer to the calibration directory in our sample data for details.

3. **Fetch Camera Intrinsics**. Camera intrinsics usually can be obtained via the camera api. Store all camera intrinsics as a dict in `INTRINSICS` in `dataset/constants.py`. For deployment, please also modify the `intrinsic` property of `Agent` in `eval_agent.py`.
