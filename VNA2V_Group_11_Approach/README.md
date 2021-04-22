# Autonomous and cooperative design of the monitor positions for a team of UAVs to maximize the quantity and quality of detected objects (VAN2V: Team 11 Implementation) #
 
# Installation (VNA2V_Group_11_Approach) #

NOTE: These instructions have only been verified by the group in Ubuntu 18.

### 0. Create a conda virtual environment (optional)
A conda environment can be helpful when running this code, specially if you use different versions of python. Download and [install](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) conda and create and environent with python 3 installed, as mentioned [here](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). For all the next operations, make sure to activate the conda environment yuo create.

### 1. Dependencies
Install the required system packages
```
$ pip install airsim Shapely descartes opencv-contrib-python
```
Remember that the code requires python 3

### 2. Darknet
Clone the [Darknet](https://github.com/AlexeyAB/darknet) repository and run the "make" command to build darknet. Make sure to set "LIBSO=1" in the Makefile (it may be initially "LIBSO=1"). Then, create the "weights" directory in the cloned repo and download the [yolov4.weights](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights) inside it.

### 3. Environments
Download any of the available [AirSim Enviroments](https://github.com/microsoft/AirSim/releases). 

### 4. Run Example
To run the example, first set the absolute path to your Darknet repository in the "run_visual_monitoring.sh" script.

Then, you need a "settings.json" file in the ~/Documents/AirSim directory to set up the simulation settings. Some example settings files are available in the settings/VNA2V_Group_11_Approach_Settings directory in the current repository for flying 2, 3, and 4 drones simultaneously. You may copy any of these files to the ~/Documents/AirSim directory and rename them as "settings.json".

Next, make sure the "offset.txt" file contains the initial positions of all the drones you intend to fly, in the order in which they are declared in the settings file. Other offset text files are included, which correspond to the settings files settings/VNA2V_Group_11_Approach_Settings directory in the current repository for flying 2, 3, and 4 drones simultaneously.

Furthermore, within the in "visual_monitoring.py" script in line 299, set the nDrones variable to the number of drones you intend to use.

Go to the directory containing the environment you downloaded in 3. It should have a .sh script. For example, for the AirSimNH environment, there should be a AirSimNH.sh file. Open a terminal in that folder and run
```
$ ./AirSimNH.sh -ResX=640 -ResY=480 -windowed
```
which should run the environment in a 640 by 480 window. In case you encounter issues with Vulkan, you may need to run 
```
sudo apt install mesa-vulkan-drivers
```

Finally, in another terminal (after activating the conda environment if you created one) run the following script:
```
$ run_visual_monitoring.sh
```
Other options are available withing the python script. Your data will be saved in the Data_1 folder.
