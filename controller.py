# import setup_path
import airsim
import os
import cv2
import numpy as np
import time

class controller:

    def __init__(self, clientIn, droneName):

        self.client = clientIn
        self.name = droneName

        self.client.enableApiControl(True, self.name)
        self.client.armDisarm(True, self.name)

        self.parentRaw = os.path.join(os.getcwd(), "swarm_raw_output")
        try:
            os.makedirs(self.parentRaw)
        except OSError:
            if not os.path.isdir(self.parentRaw):
                raise

        self.parentDetect = os.path.join(os.getcwd(), "swarm_detected")
        try:
            os.makedirs(self.parentDetect)
        except OSError:
            if not os.path.isdir(self.parentDetect):
                raise

    def takeOff(self):

        return self.client.takeoffAsync(vehicle_name = self.name)

    def moveToPostion(self, x, y, z, speed):

        return self.client.moveToPositionAsync(x,y,z,speed,vehicle_name=self.name)

    def setCameraOrientation(self, cam_yaw, cam_pitch, cam_roll):

        self.client.simSetCameraOrientation("0",
                                            airsim.to_quaternion(cam_yaw, cam_pitch, cam_roll),
                                            vehicle_name = self.name)

    def getName(self):

        return self.name

    def getImages(self, save_raw=None):

        responses = self.client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True),  #depth visualization image
            airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)],
            vehicle_name = self.name)  #scene vision image in uncompressed RGB array

        if save_raw != None:

            filenameDepth = os.path.join(self.raw_dir, f"depth_time_{self.timeStep}" )
            airsim.write_pfm(os.path.normpath(filenameDepth + '.pfm'), airsim.get_pfm_array(responses[0]))

            filenameScene = os.path.join(self.raw_dir, f"scene_time_{self.timeStep}" )
            img1d = np.frombuffer(responses[1].image_data_uint8, dtype=np.uint8) #get numpy array
            img_rgb = img1d.reshape(responses[1].height, responses[1].width, 3) #reshape array to 3 channel image array H X W X 3
            cv2.imwrite(os.path.normpath(filenameScene + '.png'), img_rgb) # write to png
            self.imageScene = img_rgb

        return responses



    def getDepthFront(self):

        responses = self.client.simGetImages([
            airsim.ImageRequest("1", airsim.ImageType.DepthPerspective, True)],
            vehicle_name = self.name)  #scene vision image in uncompressed RGB array

        self.imageDepthFront = responses[0]


    def stabilize(self):

        task = self.client.moveByVelocityAsync(0,0,0,4,vehicle_name=self.name)

        return task


    def moveToZ(self, targetZ, speedClim=3.0):

        task = self.client.moveToZAsync(targetZ,speedClim,vehicle_name=self.name)

        return task


    def randomMoveZ(self):

        minThreshold = 20
        axiZ = -10
        pixeSquare = 50
        speedScalar = 2

        while True:

            # yawRandom = np.random.ranf()*np.pi
            yawRandom = np.random.randint(0,360)
            # self.client.rotateToYawAsync(yawRandom,vehicle_name=self.name).join()
            self.client.rotateByYawRateAsync(yawRandom,1,vehicle_name=self.name).join()

            self.getDepthFront()
            imageDepth = airsim.list_to_2d_float_array(self.imageDepthFront.image_data_float,
                                                       self.imageDepthFront.width,
                                                       self.imageDepthFront.height)

            # midVertical = np.vsplit(imageDepth,3)[1]
            # mid = np.hsplit(midVertical,3)[1]
            midW = self.imageDepthFront.width/2
            midH = self.imageDepthFront.width/2
            imageDepthTarget = imageDepth[int(midW-pixeSquare):int(midW+pixeSquare),
                                          int(midH-pixeSquare):int(midH+pixeSquare)]

            current = np.min(imageDepthTarget)
            # print(f"\ndistance current:{current}")
            # print(f"yawRandom:{yawRandom}")
            if current>minThreshold:

                vx = np.cos(np.radians(yawRandom))
                vy = np.sin(np.radians(yawRandom))

                task = self.client.moveByVelocityZAsync(speedScalar*vx, speedScalar*vy,axiZ, 5,
                                            airsim.DrivetrainType.ForwardOnly,
                                            airsim.YawMode(False, 0),
                                            vehicle_name=self.name)

                break

        return task

    def updateState(self, posIdx, timeStep):

        self.posIdx = posIdx
        self.timeStep = timeStep

        self.raw_dir = os.path.join(self.parentRaw, self.name, f"position_{self.posIdx}")
        if not os.path.isdir(self.raw_dir):
            os.makedirs(self.raw_dir)

        self.detected_dir = os.path.join(self.parentDetect, self.name, f"position_{self.posIdx}")
        if not os.path.isdir(self.detected_dir):
            os.makedirs(self.detected_dir)


    def detectObjects(self, detector, save_detected=False):

        detected_file_name = None
        if save_detected:
            detected_file_name = os.path.join(self.detected_dir,
                                              f"detected_time_{self.timeStep}.png")

        detections = detector.detect(self.imageScene, display=False, save=detected_file_name)

        return detections

    def getPose(self):
        return self.client.simGetVehiclePose(vehicle_name=self.name)


    def getState(self):
        return self.client.getMultirotorState(vehicle_name=self.name)


    def quit(self):

        self.client.armDisarm(False, self.name)
        self.client.enableApiControl(False, self.name)
