import os
from matplotlib import pyplot as plt
from gaussian_splatting.utils.system_utils import mkdir_p
from utils.logging_utils import Log
from utils.rotation_utils import Rotation2Quaternion

def save_gaussians(gaussians, path, name):
    if name is None:
        return
    point_cloud_path = os.path.join(path, name)
    gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

def save_images(cameras, path, name="image"):
    Log("===== Saving camera poses only (skipping images to save space) =====", tag="GauSS-MI")
    # Only save pose file, skip jpg_folder to conserve storage
    dir_pose = os.path.join(path, "gt_pose.txt")
    file_pose = open(dir_pose, "w")

    for idx in range(len(cameras)):
        if idx == 0:
            continue
        
        pos = cameras[idx].T_gt_ori
        Rot = cameras[idx].R_gt_ori
        quat = Rotation2Quaternion(Rot)
        file_pose.write(str(pos[0])+" "+str(pos[1])+" "+str(pos[2])+" "+str(quat.x)+" "+str(quat.y)+" "+str(quat.z)+" "+str(quat.w)+"\n")
    
    file_pose.close()
    Log(f"===== Saved {len(cameras)-1} camera poses to gt_pose.txt =====", tag="GauSS-MI")
