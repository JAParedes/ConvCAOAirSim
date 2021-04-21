# Autonomous and cooperative design of the monitor positions for a team of UAVs to maximize the quantity and quality of detected objects (VAN2V: Team 11 Implementation)#

This code is a modification of the code in https://github.com/dimikout3/ConvCAOAirSim presented in for the VNA2V class. Only small modifications were made to the present code
 
# Installation (VNA2V: appConvergence only)#

### 1. Dependencies
First, install the required system packages
(NOTE: the majority of the experiments were conducted in a conda enviroment, therefore we stongly advise you to download and [install](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) a conda virtual enviroment) (VNA2V: Note that this code required python 3. Creating a conda environmanet is especially helpful if some of your own code uses python 2.7):
```
$ pip install airsim Shapely descartes opencv-contrib-python
```

### 2. Detector
Second, you have to define a detector capable of producing bounding boxes of objects along with the corresponding confidences levels from RGB images.

For the needs of our application we utilized YOLOv3 detector, trained on the [COCO dataset](http://cocodataset.org/#home). You can download this detector from [here](https://convcao.hopto.org/index.php/s/mh8WIDpprE70SO3). After downloading the file, extract the yolo-coco folder inside your local ConvCao_AirSim folder.

It is worth highlighting that, you could use a deifferent detector (tailored to the application needs), as the proposed methodology is agnostic as far the detector's choise is concerned.

### 3. Enviroments
Download any of the available [AirSim Enviroments](https://github.com/microsoft/AirSim/releases) (VNA2V: We have tested the AirSimNH environment)

### 4. Run Example
To run an example with the Convergence testbed you need to just replace the "detector-path" entry - inside this [file](https://github.com/dimikout3/ConvCAO_AirSim/blob/master/appConvergence/appSettings.json) - with your path to the previously downloaded detector. (VNA2V: The yolo-coco file is already set in the main directory)

Finally run the "MultiAgent.py" script:
```
$ python MultiAgent.py
```
Detailed instructions for running specific applications are inside every corresponding app folder

(VNA2V: run "python MultiAgentMod.py --waypoints 50" wittin the AppConvergence directory so that the code runs for 50 iterations. Other options are available within the file. Before running this python file, make sure to have your environment already running)
