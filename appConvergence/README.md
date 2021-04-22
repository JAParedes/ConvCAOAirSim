# Autonomous and cooperative design of the monitor positions for a team of UAVs to maximize the quantity and quality of detected objects (VAN2V: Team 11 Implementation) #
 
# Installation (appConvergence) #

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

### 3. Enviroments
Download any of the available [AirSim Enviroments](https://github.com/microsoft/AirSim/releases). 

### 4. Run Example
To run the "MultiAgentMod.py" script, you need to replace the "detector-path" entry in the appSettings.json file with the path to the previously downloaded detector (should be inside the yolo-coco folder, as per 2.)

Finally, in another terminal run the "MultiAgentMod.py" script:
```
$ python MultiAgentMod.py --waypoints 50
```
Detailed instructions for running specific applications are inside every corresponding app folder
