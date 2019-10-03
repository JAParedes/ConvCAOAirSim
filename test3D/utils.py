import airsim
import cv2
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle

plt.style.use('ggplot')


def to_absolute_coordinates(x,y,z,file_state):
    stateList = pickle.load(open(file_state,"rb"))

    camInfo = stateList[1]

    pitch,roll,yaw = airsim.to_eularian_angles(camInfo.pose.orientation)

    # in AirSim rotation to X-axis -> roll, Y-axis -> pitch, Z-axis -> yaw
    theta = [roll,pitch,yaw]

    xRotated, yRotated, zRotated = rotate(x,y,z,theta)

    pos = camInfo.pose.position
    t = [pos.x_val, pos.y_val, pos.z_val]
    xOut,yOut,zOut = translation(xRotated,yRotated,zRotated,t)

    return xOut, yOut, zOut


def translation(x,y,z,t):

    positions = np.stack((x,y,z), axis=1)
    out = positions + t
    return out[:,0], out[:,1], out[:,2]


def rotate(x,y,z,theta):

    tx,ty,tz = theta

    Rx = np.array([[1,0,0], [0, np.cos(tx), -np.sin(tx)], [0, np.sin(tx), np.cos(tx)]])
    # we are using different minus sign because we rotated counter-clockwise
    Ry = np.array([[np.cos(ty), 0, -np.sin(ty)], [0, 1, 0], [np.sin(ty), 0, np.cos(ty)]])
    # Ry = np.array([[np.cos(ty), 0, -np.sin(ty)], [0, 1, 0], [np.sin(ty), 0, np.cos(ty)]])
    Rz = np.array([[np.cos(tz), -np.sin(tz), 0], [np.sin(tz), np.cos(tz), 0], [0,0,1]])

    R = np.dot(Rx, np.dot(Ry, Rz))

    positions = np.stack((x,y,z),axis=1)

    rotated = positions.dot(R)

    return rotated[:,0],rotated[:,1],rotated[:,2]


def kickstart(random_points=[300,300,"square"],file_pfm="_",cam_pitch=0.0):

    d,s = airsim.read_pfm(file_pfm)

    height, width = d.shape
    print(f"Image size: width:{width} -- height:{height}")
    halfWidth = width/2
    halfHeight= height/2

    camPitch = cam_pitch
    camYaw = 0.0

    hFoV = (np.pi/2)
    vFoV = (height/width)*hFoV

    randomPointsSize = random_points[0]*random_points[1]
    if random_points[2] == "square":

        pointsH = np.random.randint(height,size=(randomPointsSize))
        pointsW = np.random.randint(width,size=(randomPointsSize))

    elif random_points[2] == "circle":

        r = np.random.uniform(0,halfHeight,randomPointsSize)
        thetas = np.random.uniform(0,2*np.pi,randomPointsSize)

        pointsH = r*np.sin(thetas)
        pointsW = r*np.cos(thetas)

        centerH = int(halfHeight)
        centerW = int(halfWidth)

        pointsH = centerH + pointsH.astype(int)
        pointsW = centerW + pointsW.astype(int)

    pixelPitch = ((pointsH-halfHeight)/halfHeight) * (vFoV/2)
    pixelYaw = ((pointsW-halfWidth)/halfWidth) * (hFoV/2)

    theta = (np.pi/2) - pixelPitch + camPitch
    # turn
    phi = pixelYaw + camYaw

    r = d[ pointsH , pointsW ]
    idx = np.where(r<100)

    x = r*np.sin(theta)*np.cos(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z = r*np.cos(theta)

    img = getColorPerPixel(file_pfm)
    colors = img[ pointsH , pointsW ]

    return x[idx],y[idx],z[idx],colors[idx]

def getColorPerPixel(file_name):
    file_name = file_name.replace("pfm","png")
    file_name = file_name.replace("depth","scene")
    print(f"loading image {file_name} ...")
    img = cv2.imread(file_name)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return imgRGB


def plot3d(x,y,z,size):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, z, s=size)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    ax.invert_zaxis()
    # ax.invert_xaxis()

    plt.show()
    plt.close()

def plot3dColor(x,y,z,colors,size=0.3,x_lim=None, show=False):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, z,c=colors/255.0, s=size)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    ax.invert_zaxis()
    ax.invert_yaxis()

    # ax.view_init(elev=0,azim=180)

    if x_lim!=None:
        ax.set_xlim(100,0)

    if show:
        plt.show()
        # plt.savefig("test.png")
        plt.close()
        return ax
    else:
        return ax
