import airsim
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
import test3D.utils as utils

# Enters all directories and creates 3d plots (saves them as pickle objects)
# and pickle object with the x,y,z,colors data (relative and absolute)

if __name__ == "__main__":

    parent_dir = os.path.join(os.getcwd(), "swarm_raw_output")
    detected_dir = os.path.join(os.getcwd(), "swarm_detected")

    dronesID = os.listdir(parent_dir)
    wayPointsID = os.listdir(os.path.join(parent_dir, dronesID[0]))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    for droneIdx, drone in enumerate(dronesID):
        print(f"\n[DRONE]: {drone}")

        out = cv2.VideoWriter(f"{drone}_perspective_3D.avi", fourcc, 1.0, (3*1024,1024))

        for positionIdx, position in enumerate(wayPointsID):
            print(f"{4*' '}[POSITION]: {position}")

            current_raw_dir = os.path.join(parent_dir, drone, position)
            current_detected_dir = os.path.join(detected_dir, drone, position)

            time_steps = os.listdir(current_detected_dir)

            for time,time_step in enumerate(time_steps):
                print(f"time:{time}")

                raw_image_path = os.path.join(current_raw_dir, f"scene_time_{time}.png")
                depth_image_path = os.path.join(current_raw_dir, f"depth_scene_time_{time}.png")

                file_coordinates = os.path.join(current_raw_dir, f"coordinates3D.pickle")
                coordinates = pickle.load(open(file_coordinates,"rb"))
                x,y,z,color = [coordinates[0], coordinates[1], coordinates[2], coordinates[6]]

                file_3d_scene = os.path.join(current_raw_dir, f"3D_scene_tim3_{time}.png")
                utils.plot3dColor(x,y,z,color,x_lim=[100,0],pose=[30,-30],
                                  y_lim=[-70,70],save_path=file_3d_scene)


                frame_raw = cv2.imread(raw_image_path)
                frame_depth = cv2.imread(depth_image_path)
                frame_3D = cv2.imread(file_3d_scene)
                frame_3D = cv2.resize(frame_3D,(1024,1024))

                h, w, _ = frame_raw.shape
                print(f"frame_raw width:{w} -- height:{h}")
                h, w, _ = frame_depth.shape
                print(f"frame_depth width:{w} -- height:{h}")
                h, w, _ = frame_3D.shape
                print(f"frame_3D width:{w} -- height:{h}\n")

                fram_vertical = np.concatenate((frame_depth,frame_raw,frame_3D), axis=1)

                out.write(fram_vertical)

        out.release()
