# Object-detection-and-tracking-using-yolov3-and-deepsort
* Implemented object detection and tracking using YOLOv3 and Deep SORT. 
* YOLOv3 uses a convolutional neural network for object detection,and Deep SORT combines this with motion data for accurate tracking. 
* This can be used for various applications such as 
  * Drone
  * Robots
  * Plane
  * Self-driving cars
  * Sensor fusion
# Setting up environment to run on cpu  
  ## To set up the environment for this project,follow the steps  
  ### 1.Create Environment (Conda recommended)
    #Tensorflow CPU
    conda env create -f conda-cpu.yml
    conda activate tracker-cpu
  ### 2.Install dependencies 
     #Tensorflow CPU
     pip install -r requirements.txt
        
## References
* [AiGuysCode_YOLOv3](https://github.com/theAIGuysCode/yolov3_deepsort "AiGuysCode")

* [DeepSort_Repository](https://github.com/nwojke/deep_sort "DeepSort")


