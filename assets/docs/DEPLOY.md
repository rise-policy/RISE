# 🤖 Deployment

1. Adjust the constant parameters in `utils/constants.py` according to your own environment.
   - `IMG_MEAN` and `IMG_STD` are the image normalization constants. Here we use ImageNet normalization coefficients.
   - `TRANS_MIN` and `TRANS_MAX` are the tcp normalization range in the camera coordinate.
   - `MAX_GRIPPER_WIDTH` indicates the gripper width normalization range (in meter).
   - `WORKSPACE_MIN` and `WORKSPACE_MAX` are the workspace range in the camera coordinate and used for point cloud cropping. 
   - `SAFE_WORKSPACE_MIN` and `SAFE_WORKSPACE_MAX` are the safe workspace range in the base coordinate (used for evaluation).
   - `SAFE_EPS` denotes the safe epsilon of the safe workspace range. Therefore, the real range should be [min + eps, max - eps].
   - `GRIPPER_THRESHOLD` denotes the gripper moving threshold (in meter) to avoid gripper action too frequently during evaluation.

2. Add the device support libraries of your own devices in the `device` folder.

3. Modify `eval_agent.py` to accomodate your own device.
   - Implement the `__init__(...)` function to initialize the devices. Move the robot to the ready pose, activate the gripper, start the camera, and flush the camera stream.
   - Implement the `intrinsics` property to return the intrinsics of the camera.
   - (Optional) Implement the `ready_pose` property to define the ready pose of the robot.
   - Implement the `ready_rot_6d` property to define the 6d rotation representation of the ready pose's rotation part.
   - Implement the `get_observation()` function to get the RGB and depth observations from the camera.
   - Implement the `set_tcp_pose(...)` function to set the tcp pose of the robot. The pose can be defined in any rotation representation, and feel free to use the `xyz_rot_transform` function in `utils/transformation.py` for rotation transformations. Also, implement the `blocking` option to block the robot moving process.
   - Implement the `set_gripper_width(...)` function to set the width of the gripper. Notice you might need to encode the original width (in meter) to the corresponding signal following the documentations of the gripper.
   - Implement the `stop()` function to stop the devices.