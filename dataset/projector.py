import os
import numpy as np

from dataset.constants import *
from utils.transformation import xyz_rot_to_mat, mat_to_xyz_rot


class Projector:
    def __init__(self, calib_path):
        self.cam_to_markers = np.load(os.path.join(calib_path, "extrinsics.npy"), allow_pickle = True).item()
        self.calib_icam_to_markers = np.array(self.cam_to_markers[INHAND_CAM[0]]).squeeze() # calib icam to marker
        self.calib_tcp = xyz_rot_to_mat(np.load(os.path.join(calib_path, "tcp.npy")), "quaternion") # base to calib tcp
        # cam to base
        self.cam_to_base = {}
        for cam in self.cam_to_markers.keys():
            if cam in INHAND_CAM:
                continue
            self.cam_to_base[cam] = np.array(self.cam_to_markers[cam]).squeeze() @ np.linalg.inv(self.calib_icam_to_markers) @ INHAND_CAM_TCP @ np.linalg.inv(self.calib_tcp)
        
    def project_tcp_to_camera_coord(self, tcp, cam, rotation_rep = "quaternion", rotation_rep_convention = None):
        assert cam not in INHAND_CAM, "Cannot perform inhand camera projection."
        return mat_to_xyz_rot(
            self.cam_to_base[cam] @ xyz_rot_to_mat(
                tcp, 
                rotation_rep = rotation_rep,
                rotation_rep_convention = rotation_rep_convention
            ), 
            rotation_rep = rotation_rep,
            rotation_rep_convention = rotation_rep_convention
        )

    def project_tcp_to_base_coord(self, tcp, cam, rotation_rep = "quaternion", rotation_rep_convention = None):
        assert cam not in INHAND_CAM, "Cannot perform inhand camera projection."
        return mat_to_xyz_rot(
            np.linalg.inv(self.cam_to_base[cam]) @ xyz_rot_to_mat(
                tcp, 
                rotation_rep = rotation_rep,
                rotation_rep_convention = rotation_rep_convention
            ),
            rotation_rep = rotation_rep,
            rotation_rep_convention = rotation_rep_convention
        )
