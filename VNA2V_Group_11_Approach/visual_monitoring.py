
import setup_path
import airsim
#import airsimneurips
import os
import sys
import math
import time
import argparse
import numpy as np
#import matplotlib.pyplot as pyplot
from mpl_toolkits import mplot3d
import mpl_toolkits.mplot3d.art3d as art3d
from scipy.spatial.transform import Rotation
import copy
#import qpsolvers
from scipy.optimize import minimize
import scipy.optimize as opt

# l2 norm of a vector
from numpy import array
from numpy.linalg import norm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import glob
import random
import cv2
import darknet
import csv



#codes related to object detection using yolov3 darknet library
def parser():
    parser = argparse.ArgumentParser(description="YOLO Object Detection")
    parser.add_argument("--input", type=str, default="",
                        help="image source. It can be a single image, a"
                        "txt with paths to them, or a folder. Image valid"
                        " formats are jpg, jpeg or png."
                        "If no input is given, ")
    parser.add_argument("--batch_size", default=1, type=int,
                        help="number of images to be processed at the same time")
    parser.add_argument("--weights", default=os.path.join(os.environ.get('DARKNET_PATH', './'),"weights/yolov4.weights"),
                        help="yolo weights path")
    parser.add_argument("--dont_show", action='store_true',
                        help="windown inference display. For headless systems")
    parser.add_argument("--ext_output", action='store_true',
                        help="display bbox coordinates of detected objects")
    parser.add_argument("--save_labels", action='store_true',
                        help="save detections bbox for each image in yolo format")
    parser.add_argument("--config_file", default=os.path.join(os.environ.get('DARKNET_PATH', './'),"cfg/yolov4.cfg"),
                        help="path to config file")
    parser.add_argument("--data_file", default=os.path.join(os.environ.get('DARKNET_PATH', './'),"cfg/coco.data"),
                        help="path to data file")
    parser.add_argument("--thresh", type=float, default=.25,
                        help="remove detections with lower confidence")
    return parser.parse_args()


def check_arguments_errors(args):
    assert 0 < args.thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
    if not os.path.exists(args.config_file):
        raise(ValueError("Invalid config path {}".format(os.path.abspath(args.config_file))))
    if not os.path.exists(args.weights):
        raise(ValueError("Invalid weight path {}".format(os.path.abspath(args.weights))))
    if not os.path.exists(args.data_file):
        raise(ValueError("Invalid data file path {}".format(os.path.abspath(args.data_file))))
    if args.input and not os.path.exists(args.input):
        raise(ValueError("Invalid image path {}".format(os.path.abspath(args.input))))


def image_detection(image_path, network, class_names, class_colors, thresh):
    # Darknet doesn't accept numpy images.
    # Create one with image we reuse for each detect
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    darknet_image = darknet.make_image(width, height, 3)

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height),
                               interpolation=cv2.INTER_LINEAR)

    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)
    darknet.free_image(darknet_image)
    image = darknet.draw_boxes(detections, image_resized, class_colors)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), detections

def convert2relative(image, bbox):
    """
    YOLO format use relative coordinates for annotation
    """
    x, y, w, h = bbox
    height, width, _ = image.shape
    return x/width, y/height, w/width, h/height

def save_annotations(name, image, detections, class_names):
    """
    Files saved with image_name.txt and relative coordinates
    """
    file_name = name.split(".")[:-1][0] + ".txt"
    with open(file_name, "w") as f:
        for label, confidence, bbox in detections:
            x, y, w, h = convert2relative(image, bbox)
            label = class_names.index(label)
            f.write("{} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}\n".format(label, x, y, w, h, float(confidence)))


def pixels_to_3D_pos(pixelX, pixelY, camInfo, depthImage, color=[], maxDistView = None):
    """From image pixels (2D) to relative(!) 3D coordinates"""

    if type(depthImage) == str:
        depth,s = airsim.read_pfm(depthImage)
    else:
        depthData = depthImage.image_data_float
        depthArray = np.array(depthData)
        #print('CameraInfo={}'.format(camInfo))
        #print('depthData= {}, pixelX={},  pixelY={},  height={}'.format(len(depthArray),pixelX,pixelY,depthImage.height))
        depth = np.reshape(depthArray, (depthImage.height, depthImage.width))
        if isinstance(maxDistView, float) or isinstance(maxDistView, int):
            depth[depth>maxDistView] = maxDistView

    #print('Camear info = {}'.format(camInfo))
    # depth = depthImage.image_data_uint8

    height, width = depth.shape
    #print(f"Image size: width:{width} -- height:{height}")
    halfWidth = width/2
    halfHeight= height/2

    camPitch, camRoll,camYaw = airsim.to_eularian_angles(camInfo.pose.orientation)

    # to rads (its on degrees now)
    hFoV = np.radians(camInfo.fov)
    vFoV = float(height)/width*hFoV

    pointsH = np.array(pixelY, dtype=int)
    pointsW = np.array(pixelX, dtype=int)

    pixelPitch = ((pointsH-halfHeight)/halfHeight) * (vFoV/2)
    pixelYaw = ((pointsW-halfWidth)/halfWidth) * (hFoV/2)

    theta = (np.pi/2) - pixelPitch
    #theta = pixelPitch
    # turn
    phi = pixelYaw

    r = depth[ pointsH, pointsW]
    #print('r={}'.format(r))
    idx = np.where(r<=100)
    #print('idx={}'.format(idx))

    x = r*np.sin(theta)*np.cos(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z = r*np.cos(theta)
    #print('x={},y={},z={},phi={},theta={}, pointsH={}, halfHeight={}, pixelPitch={}, vFoV={}, hFoV={}, height={}, width={}, ratio={}'.format(x, y, z, phi, theta, pointsH, halfHeight, pixelPitch, vFoV, hFoV, height, width, float(height)/width))

    return x, y, z

    # if len(color) != 0:
    #     return x[idx],y[idx],z[idx],color[idx]
    # else:
    #     return x[idx],y[idx],z[idx]


#Codes for motion planning for monitoring
def cost_J(cost_c_ij, nDrones, nObjects, drone_id, cost_c_ij_prev, delta):
    #use delta norm to approximate the max function, as delta becomes very high the approximation approaches the true maximum value of the given set of numbers
    J = 0
    if drone_id:        
        for k in range(0,nObjects):
            temp_norm = 0
            temp_norm = temp_norm + pow(cost_c_ij_prev[drone_id-1][k], delta)
            for i in range(0,drone_id-1):
                temp_norm = temp_norm + pow(cost_c_ij[i][k], delta)
            for i in range(drone_id, nDrones):
                temp_norm = temp_norm + pow(cost_c_ij[i][k], delta)

            J = J + pow(temp_norm,float(1)/float(delta))
    else:
        for k in range(0,nObjects):
            temp_norm = 0
            for i in range(0,nDrones):
                temp_norm = temp_norm + pow(cost_c_ij[i][k], delta)

            J = J + pow(temp_norm,float(1)/float(delta))

    return J

def contribution_Delta_i(drone_state, cost_c_ij, drone_state_prev, cost_c_ij_prev, nDrones, nObjects, delta):
    J_current = cost_J(cost_c_ij, nDrones, nObjects, [], [], delta)
    #print('cost_c_ij={}, cost_c_ij_prev ={}, J_current={}, '.format(cost_c_ij, cost_c_ij_prev, J_current))
    #print('drone_state={}, drone_state_prev={}'.format(drone_state, drone_state_prev))
    Delta_i = np.zeros((nDrones,1))
    for i in range(0,nDrones):
        J_prev = cost_J(cost_c_ij, nDrones, nObjects, i+1, cost_c_ij_prev, delta)
        #print('J_prev={}, '.format(J_prev))
        state_err = norm(drone_state[:,i] - drone_state_prev[:,i])
        if state_err>0.001:
            Delta_i[i]=(J_current - J_prev)/state_err

    return Delta_i, J_current

def evaluate_phi(x, optWindowSize):
    Phi = np.zeros((3,optWindowSize))
    for j in range(0,optWindowSize):
        Phi[:,j] = x[:,j]

    return Phi


def evaluate_next_state(drone_ind, cost_J_i, state_history, currentState, nDrones, detected_objects, nObjects, optWindowSize, m, rho):
    J_vec = np.zeros((optWindowSize,1));
    #Phi = np.zeros((length_of_phi,optWindowSize));
    for j in range(0,optWindowSize):
        J_vec[j] = cost_J_i[j]

    #print('state_history={}'.format(state_history))
    Phi = evaluate_phi(state_history,optWindowSize)
    sol = np.linalg.lstsq(np.transpose(Phi), J_vec)
    theta_star = sol[0]
    #print('theta_star = {}'.format(theta_star))
    #print('Phi = {}'.format(Phi))
    #print('J_vec = {}'.format(J_vec))


    #generate m random perturbations 
    count = 0
    iter_num = 0
    flag_valid_random_point = 1
    random_points = np.zeros((3,m))
    while (count<m) and iter_num<5*m:
        iter_num = iter_num + 1
        random_point = np.random.random(3)
        temp_random = state_history[:,optWindowSize-1] + [rho*random_point[0], rho*random_point[1], random_point[2]]  #current state plus the random perturbation
        #print('temp_random={}'.format(temp_random))
        #check collisions with other drones
        for i in range(0,nDrones):
            if (norm(temp_random - currentState[:,i])<1) and (i != drone_ind):
                flag_valid_random_point = 0
                print('1')

        #check collisions with the objects
        #print('rand={}, obj_pos={}'.format(temp_random[0:2],np.array(detected_objects[i][2:3])))
        for i in range(0,nObjects):
            #print('detected objects={}'.format(detected_objects[i]))
            object_pos = detected_objects[i][2]
            #print('detected objects={}'.format(object_pos[0:2]))
            if norm(temp_random[0:2] - object_pos[0:2])<1:
                flag_valid_random_point = 0
                print('2')

        if flag_valid_random_point:
            random_points[:,count] = temp_random
            count = count + 1

    Phi_prime = evaluate_phi(random_points,m)

    opt_ind = np.argmax(np.dot(np.transpose(Phi_prime),theta_star))
    #print('nextState={}'.format(random_points[:,opt_ind]))

    return random_points[:,opt_ind]


def Rot_mat_from_euler(seq,angles):   #from body frame to earth frame
    ind=[0,1,2,0,1]
    
    rotMat=np.identity(3)
    for i in range(0,3):
        #print(seq[i])
        rotMat0=np.zeros((3,3))
        rotMat0[seq[i]-1,seq[i]-1]=1
        rotMat0[ind[seq[i]],ind[seq[i]]]=math.cos(angles[i])
        rotMat0[ind[seq[i]],ind[seq[i]+1]]=-math.sin(angles[i])
        rotMat0[ind[seq[i]+1],ind[seq[i]+1]]=math.cos(angles[i])
        rotMat0[ind[seq[i]+1],ind[seq[i]]]=math.sin(angles[i])
        #print(rotMat0)
        rotMat=rotMat.dot(rotMat0)  #the order of rotation matrices is for body to earth rotation

    return rotMat



if __name__ == "__main__":
    # args = sys.argv
    # args.pop(0)
    # arg_parser = argparse.ArgumentParser("Orbit.py makes drone fly in a circle with camera pointed at the given center vector")
    # arg_parser.add_argument("--radius", type=float, help="radius of the orbit", default=10)
    # arg_parser.add_argument("--altitude", type=float, help="altitude of orbit (in positive meters)", default=35)
    # arg_parser.add_argument("--speed", type=float, help="speed of orbit (in meters/second)", default=3)
    # arg_parser.add_argument("--center", help="x,y direction vector pointing to center of orbit from current starting position (default 1,0)", default="1,0")
    # arg_parser.add_argument("--iterations", type=float, help="number of 360 degree orbits (default 3)", default=1)
    # arg_parser.add_argument("--snapshots", type=float, help="number of FPV snapshots to take during orbit (default 0)", default=0)    
    # args = arg_parser.parse_args(args)    
    # nav = OrbitNavigator(args.radius, args.altitude, args.speed, args.iterations, args.center.split(','), args.snapshots)
    #nav.start()
    
    nDrones = 4
    altitude = 20
    # connect to the AirSim simulator
    client = airsim.MultirotorClient()
    client.confirmConnection()
    #Arm the drones and take off

    #Generate ids for the drones
    drone_id =[]
    for i in range(0,nDrones):
        id = i+1
        drone_id.append("Drone"+str(id))

    print('ids ={}'.format(drone_id))
    #Take-off the drones 
    f = []
    for i in range(0,nDrones):
        id = i+1
        client.enableApiControl(True, drone_id[i])
        flagArm = client.armDisarm(True, drone_id[i])
        print("Drone "+str(id)+" armed = {}".format(flagArm))
        f.append(client.moveToPositionAsync(0, 0, -altitude, 2, 30, vehicle_name=drone_id[i]))
        print("Drone "+str(id)+" taking off = {}".format(flagArm))

    for i in range(0,nDrones):
        f[i].join()
    print('f={}'.format(f))
    print("All drones took off = {}".format(flagArm))
    #client.simPause(True)
    #print('Simulation stopped')

    #Offsets of the drones from the player start location (local origin for the entire system)
    offset = np.zeros((3,nDrones))
    file_offset = os.path.join(os.getcwd(),f"offset.txt")
    with open(file_offset) as file:
        lc = 0;
        for line in file:
            data = line.split()
            offset[:,lc] = data
            lc = lc + 1

    print('offset ={}'.format(offset))
    file.close()

    #Load a yolo detector
    args = parser()
    check_arguments_errors(args)

    random.seed(3)  # deterministic bbox colors
    network, class_names, class_colors = darknet.load_network(
        args.config_file,
        args.data_file,
        args.weights,
        batch_size=args.batch_size
    )
    print("Network loaded")

    #
    folder_name = os.path.join(os.getcwd(),f"Data/Test_1")

    nTimeSteps = 50
    optWindowSize = 3
    pos_err_tol = 0.5
    nObjects = 0
    delta = 3 #parameter for delta-norm approximation of J function
    rho = 2
    dt_exec = 10
    frame_count = np.zeros((1,nDrones), int)
    currentState = np.zeros((3,nDrones))
    nextState = np.zeros((3,nDrones))
    prevState = np.zeros((3,nDrones))
    X_history = np.zeros((nDrones,nTimeSteps))
    Y_history = np.zeros((nDrones,nTimeSteps))
    Yaw_history = np.zeros((nDrones,nTimeSteps))
    CostJi_history = np.zeros((nDrones,nTimeSteps))
    cost_J_history = np.zeros((nTimeSteps,1))

    detected_object_3d_pos_from_drone = []
    detected_object_3d_pos =[]
    detected_objects_by_a_drone =[]
    detected_objects = []
    uniquely_detected_objects = []
    cost_c_ij = []
    cost_c_ij_prev = []
    cost_J_i = []
    Delta_i = []
    for i in range(0,nDrones):
        # state0 = client.getMultirotorState(vehicle_name="Drone"+str(i+1))
        # pitch, roll, yaw  = airsim.to_eularian_angles(state0.kinematics_estimated.orientation)
        # nextState[:,i] = [state0.kinematics_estimated.position.x_val, state0.kinematics_estimated.position.y_val, yaw]
        detected_objects_by_a_drone.append([])
        cost_c_ij.append([])
        cost_c_ij_prev.append([])
        cost_J_i.append([])
        Delta_i.append(0)

    #print('detected_obj_by_drone={}'.format(detected_objects_by_a_drone))

    for tk in range(0,nTimeSteps):
        print('tk = {}'.format(tk))
    
        for i in range(0,nDrones):
            id = i+1
            state0 = client.getMultirotorState(vehicle_name="Drone"+str(id))
            pitch, roll, yaw  = airsim.to_eularian_angles(state0.kinematics_estimated.orientation)
            #print("state, x = {}, y={}".format(state0.kinematics_estimated.position.x_val, state0.kinematics_estimated.position.y_val))
            X_history[i,tk] = state0.kinematics_estimated.position.x_val + offset[0,i]
            Y_history[i,tk] = state0.kinematics_estimated.position.y_val + offset[1,i]
            Yaw_history[i,tk] = yaw;
            currentState[:,i] = [X_history[i,tk], Y_history[i,tk], Yaw_history[i,tk]]
            currentState[:,i] = currentState[:,i]   #Current state in global frame
            # f = client.moveToPositionAsync(nextState[0,i] - offset[0,i], nextState[1,i] - offset[1,i], altitude - offset[2,i], 2, 8, airsim.DrivetrainType.MaxDegreeOfFreedom, airsim.YawMode(False, (nextState[2,i])*float(180)/math.pi), vehicle_name="drone"+str(id))
            # f.join()
            #print('drone = {}'.format(id))
            image_responses = client.simGetImages([airsim.ImageRequest("front_center_custom", airsim.ImageType.Scene)], vehicle_name="Drone"+str(id))  #scene vision image in uncompressed RGB array
            depth_responses = client.simGetImages([airsim.ImageRequest("front_center_custom", airsim.ImageType.DepthPerspective, True)], vehicle_name="Drone"+str(id))  #scene vision image in uncompressed RGB array
            image = image_responses[0]
            depth_image = depth_responses[0]
            frame_count[0,i] = frame_count[0,i] + 1;
            filename = os.path.join(folder_name, "Drone_"+str(id)+"_frame_"+str(frame_count[0,i]))
            #png format
            #print("Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))
            #print('response = {}'.format(image.image_type))
            airsim.write_file(os.path.normpath(filename + '.png'), image.image_data_uint8)
            filename=os.path.normpath(filename + '.png')
            #detect object in the image
            #print("filename: {}".format(filename))
            image_mat, detections = image_detection(filename, network, class_names, class_colors, args.thresh)

            #save_annotations(filename, image_mat, detections, class_names)
            #if len(detections):
                #print('detections={}'.format(detections[0]))
            #cv2.imshow('Inference', image_mat)
            cv2.imwrite(filename, image_mat)

            camInfo = client.simGetCameraInfo("front_center_custom",vehicle_name="Drone"+str(i+1))
            dimageHeight, dimageWidth, dimageChannels = image_mat.shape
            detected_objects_by_a_drone[i]=[]
            #cost_c_ij[i]=[]
            for label, confidence, bbox in detections:
                pixelX, pixelY, w, h = bbox
                #print('pixelX={}, pixelY={}'.format(pixelX,pixelY))
                #scale the pixels to the size of the depthImage
                pixelY = pixelY * depth_image.height/dimageHeight
                pixelX = pixelX * depth_image.width/dimageWidth
                #print('pixelX={}, pixelY={}'.format(pixelX,pixelY))
                detected_object_3d_pos_from_drone = pixels_to_3D_pos(pixelX, pixelY, camInfo, depth_image,[], 100)
                detected_object_3d_pos = detected_object_3d_pos_from_drone + offset[:,i]
                #print('detected_obj_pos_from_drone={}'.format(detected_object_3d_pos_from_drone))
                #print('detected_obj_pos={}'.format(detected_object_3d_pos))
                label = class_names.index(label)
                detected_objects_by_a_drone[i].append([label, float(confidence), detected_object_3d_pos])
                flag_new_object = 1;
                for k in range(0,len(detected_objects)):
                    if (detected_objects[k][0] == label) and (norm(detected_objects[k][2:5] -detected_object_3d_pos ) < pos_err_tol):
                        flag_new_object = 0
                        cost_c_ij[i][k]=float(confidence)
                        #break
                    else:
                        cost_c_ij[i][k]=0

                if flag_new_object:
                    nObjects = nObjects + 1
                    detected_objects.append([label, float(confidence), detected_object_3d_pos])
                    cost_c_ij[i].append(float(confidence))
                    cost_c_ij_prev[i].append(0)
                    for i1 in range(0,i):
                        cost_c_ij[i1].append(0)
                        cost_c_ij_prev[i1].append(0)
                    for i1 in range(i+1,nDrones):
                        cost_c_ij[i1].append(0)
                        cost_c_ij_prev[i1].append(0)


                #print('detected_obj_by_drone={}'.format(detected_objects_by_a_drone[i]))

            #print('detected_objects={}'.format(detected_objects))

            #print('cost_c_ij={}'.format(cost_c_ij))
        #calculate the contribution from each drone
        if tk>0:
            Delta_i, cost_J_history[tk] = contribution_Delta_i(currentState, cost_c_ij, prevState, cost_c_ij_prev, nDrones, nObjects, delta)

        for i in range(0,nDrones):
            if tk>0:
                if tk<optWindowSize:
                    cost_J_i[i].append(0)
                    cost_J_i[i][tk] = cost_J_i[i][tk-1] + Delta_i[i]
                    nextState[:,i] = np.random.random(3) #evaluate_next_state(cost_J_i[i], np.array([X_history[i,:], Y_history[i,:], Yaw_history[i,:]]), tk+1, 10)
                    nextState[0:2,i] = rho* nextState[0:2,i]
                    nextState[:,i] = currentState[:,i] + nextState[:,i]
                    CostJi_history[i,tk] = cost_J_i[i][tk]
                else:
                    cost_J_i[i].append(0)
                    cost_J_i[i][optWindowSize] = cost_J_i[i][optWindowSize-1] + Delta_i[i]
                    CostJi_history[i,tk] = cost_J_i[i][optWindowSize]
                    cost_J_i[i].pop(0) #remove the older entry from the list
                    ind1 = tk - optWindowSize+1;
                    nextState[:,i] = evaluate_next_state(i, cost_J_i[i], np.array([X_history[i,ind1:tk+1], Y_history[i,ind1:tk+1], Yaw_history[i,ind1:tk+1]]), currentState, nDrones, detected_objects, nObjects, optWindowSize, 7, rho)
            else:
                cost_J_i[i].append(Delta_i[i])
                nextState[:,i] = np.random.random(3) #evaluate_next_state(cost_J_i[i], np.array([X_history[i,:], Y_history[i,:], Yaw_history[i,:]]), tk+1, 10)
                nextState[0:2,i] = rho* nextState[0:2,i]
                nextState[:,i] = currentState[:,i] + nextState[:,i]
                CostJi_history[i,tk] = Delta_i[i]
                
            

        #if tk>0:
            #print('Delta_i={}, '.format(Delta_i))
            #print('Delta_i={}, cost_J_i={}'.format(Delta_i[i],cost_J_i))


        #send commands to the drones
        #print('currentState={}'.format(currentState))
        #print('cost_J_i={}'.format(cost_J_i))
        #print('nextState={}'.format(nextState))
        #print('Simulation started\n')
        #f = []
        #client.simPause(False)
        for i in range(0,nDrones):
            id = i+1
            #print('x={}, y={}, z={}'.format(nextState[0,i] - offset[0,i],nextState[1,i] - offset[1,i], - altitude - offset[2,i]))
            #client.enableApiControl(True, drone_id[i])
            f [i]= client.moveToPositionAsync(nextState[0,i] - offset[0,i], nextState[1,i] - offset[1,i], - altitude - offset[2,i], 2, dt_exec, airsim.DrivetrainType.MaxDegreeOfFreedom, airsim.YawMode(False, (nextState[2,i])*float(180)/math.pi), vehicle_name="Drone"+str(id))
            #print('f={}'.format(f))
        #check if the commands are completed
        for i in range(0,nDrones):
            f[i].join()
        #f.join()

        #client.simPause(True)
        #print('Simulation paused \n')

        prevState = copy.deepcopy(currentState)
        cost_c_ij_prev = copy.deepcopy(cost_c_ij)

    Xhistoryfile = os.path.join(folder_name,"Xhistory.csv")
    Yhistoryfile = os.path.join(folder_name,"Yhistory.csv")
    Yawhistoryfile = os.path.join(folder_name,"Yawhistory.csv")
    Jihistoryfile = os.path.join(folder_name,"Jihistory.csv")
    Jhistoryfile = os.path.join(folder_name,"Jhistory.csv")
    
    np.savetxt(Xhistoryfile, X_history, delimiter=",")
    np.savetxt(Yhistoryfile, Y_history, delimiter=",")
    np.savetxt(Yawhistoryfile, Yaw_history, delimiter=",")
    np.savetxt(Jihistoryfile, CostJi_history, delimiter=",")
    np.savetxt(Jhistoryfile, cost_J_history, delimiter=",")
    
    print('cost_J_history={}'.format(cost_J_history))
    plt.plot(range(0,nTimeSteps), cost_J_history)
    plt.xlabel('time step')
    plt.ylabel('cost J')
    plt.show()
            
            




