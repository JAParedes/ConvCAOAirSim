# Autonomous and cooperative design of the monitor positions for a team of UAVs to maximize the quantity and quality of detected objects (VAN2V: Team 11 Implementation) #
 
# Installation (appConvergence) #

NOTE: These instructions have only been verified by the group in Ubuntu 18.

### 0. Create a conda virtual environment (optional)
A conda environment can be helpful when running this code, specially if you use different versions of python. Download and [install](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) conda and create and environent with python 3 installed, as mentioned [here](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). For all the next operations, make sure to activate the conda environment yuo create.

### 1. Dependencies
Install the required system packages
```
$ pip install airsim Shapely descartes opencv-contrib-python
```
Remember that the code requires python 3

### 2. Detector
Second, you have to define a detector capable of producing bounding boxes of objects along with the corresponding confidences levels from RGB images.

For the needs of our application we utilized YOLOv3 detector, trained on the [COCO dataset](http://cocodataset.org/#home). You can download this detector from [here](https://convcao.hopto.org/index.php/s/mh8WIDpprE70SO3). After downloading the file, extract the yolo-coco folder inside your local ConvCao_AirSim folder and put the files inside the yolo-coco folder (coco.names, yolov3.cfg and yolov3.weights).

Note that the code can be modified to work with other detectors.

Make sure to replace the "detector-path" entry in the appSettings.json file with the absolute path to the previously downloaded detector (should be inside the yolo-coco folder).

### 3. Environments
Download any of the available [AirSim Enviroments](https://github.com/microsoft/AirSim/releases). 

### 4. Run Example
To run the "MultiAgentMod.py" script, first, you need a "settings.json" file in the ~/Documents/AirSim directory to set up the simulation settings. Some example settings files are available in the settings/appConvergence_Settings file in the current repository for flying 2, 3, and 4 drones simultaneously. You may copy any of these files to the ~/Documents/AirSim directory and rename them as "settings.json".

Next, within the "MultiAgentMod.py" in line 741, make sure the OFFSETS dictionary has a key corresponding to the name of each drone in your "settings.json" file (don't include GlobalHawk) and that their corresponding initial positions array is included as well and matchs the one in "settings.json".

Go to the directory containing the environment you downloaded in 3. It should have a .sh script. For example, for the AirSimNH environment, there should be a AirSimNH.sh file. Open a terminal in that folder and run
```
$ ./AirSimNH.sh -ResX=640 -ResY=480 -windowed
```
which should run the environment in a 640 by 480 window. In case you encounter issues with Vulkan, you may need to run 
```
sudo apt install mesa-vulkan-drivers
```

Finally, in another terminal (after activating the conda environment if you created one) run the "MultiAgentMod.py" script:
```
$ python MultiAgentMod.py --waypoints 50
```
where waypoints determines the number of iterations of the program. Other options are available withing the python script.Your data will be saved in the results_1 folder.
