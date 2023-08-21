# Strawberry count and detection using YOLOv7

**Note- This module is made using help from - https://github.com/WongKinYiu/yolov7**

[!Detection_image.png](https://github.com/amitamola/Strawberry-Counting-and-Ripeness-detection/blob/main/results/final_test/output/2236.png)

## 1. Here's description of each Notebook in this codebase:
-  **1. Dataset split NB**: This noteboook was used to split the original 3000 images into train, validation and test set in 0.8, 0.1 and 0.1 ratio. Instructions to perform same available inside the notebook.

- **2. Training_the_YOLO_model (Google Colab)**: Notebook that was used to perform the training on our set of images using Google Colab and it's GPUs. Steps to perform a YOLOv7 training on a custom dataset are mentioned at the end of this file.

- **3. Tfevent to plots.ipynb**: Notebook that uses tfevents file from tensorboard and creates beautiful plots.

- **4. Inference Notebook on original test set.ipynb**: Notebook to perform inference on test images and gives the final mAP score using mAP module.

- **5. Inference Notebook for Group 8 test images.ipynb**: Notebook to perform inference any unseen standalone test image or images.
    

## 2. Description of the folders in this codebase:

- **cfg** - Folder containing config files for YOLOv7 model to train

- **data** - Folder where the splitted data is kept

- **mAP** - This is the folder containing the mAP calculating code. More details available in the README.md inside of how to make use of it.

- **models** - Folder containing the YOLOv7 architecture.

- **plots** - Folder containing plots created by YOLO itself while training as well as plots created using **3. Tfevent to plots.ipynb**

- **results** - Folder containing all the outputs and result images containing the bounding boxes from detection and the csv containing the same information.

- **utils** - Utility files used by YOLO model for training and inferencing purpose.

## 3. Description of important files:

- *best.pt* - The best model we obtain after training on our custom dataset in Google Colab. If the file is corrupt or not available, one can download it from the drive link - https://tinyurl.com/uuc5rf4u

- *detect.py* - Python script to perform detection using GPU

- *events.out.tfevents.1671505893.1883a359b2f3.20572.0* - Events file from Tensorboard

- *requirements.txt* - Requirements file to install necessary libraries

- *requirements_gpu.txt* - To install necessary libraries

- *train.py* - Script to perform model training

- *yolov7.pt* - Pre-trained yolov7 weights on coco dataset.


# Training a custom YOLOv7 model:

1. Clone YOLOv7 Repo
Clone the yolov7 repository from GitHub by running the following command in the terminal. ```git clone https://github.com/WongKinYiu/yolov7.git```

2. Install requirements
    - Inside the cloned yolov7 folder you will find a file called **requirements.txt**.
    - Open this File.
    - Remove Line 11 and Line 12 which are torch and torchvision.
    - Create a new file called **requirements_gpu.txt** in the yolov7 folder
    - Open this new file and paste the following content into that file:  
    
    ```
       -i https://download.pytorch.org/whl/cu113
       torch==1.11.0+cu113
       torchvision==0.12.0+cu113```
3. Prepare the data as done in **1. Dataset split NB**

4. Editing Config Files
    - Now open the “coco.yaml” file from the data folder and delete the first 4 lines (till the download part).
    - Set ‘train: data/train‘
    - Set ‘val: data/val‘
    - Set ‘nc:3‘ (no of classes), since we have 3 types of strawberries
    - Set names:['unripe', 'partially_ripe', 'fully_ripe']
    
    Now open the yolov7/cfg/training
    - open the “yolov7.yaml” file
    - in line 2; change the nc=3

5. Download pre-trained yolov7 weights
    - Now we need to download pre-trained yolov7 weights.
    - Open the GitHub link https://github.com/WongKinYiu/yolov7#performance and click on YOLOv7
    - Move the downloaded yolov7.pt file to the yolov7 folder.

6. Let’s Train yolov7 on the custom dataset
    - Open the drive and upload this folder to the drive.
    - Open Colab and open the  **2. Training_the_YOLO_model (Google Colab)** notebook. Set the runtime to GPU. Runtime > Change runtime type > GPU.
    - And run the cells and give necessary permissions.
    - You can change the epochs according to your need.
    - Also, you can change the batch size according to your GPU. If the training gives a memory error, try reducing it. I eventually had to reduce mine to 4.

[!training.jpg](https://github.com/amitamola/Strawberry-Counting-and-Ripeness-detection/blob/main/training.jpg)

7. Inference
    - Just download the final best.pt file in the same folder in colab that you trained
    - Either perform infernce using the command provided in the notebook or
    - Copy the best.pt file in the local folder and use the **4. Inference Notebook on original test set.ipynb** file to run inference on test set.
    - Or the **5. Inference Notebook for Group 8 test images.ipynb** file to run inference on any image you want.

### NOTE- To perform GPU based inference, one can use detect.py file available in the codebase.

## Final performance of our model

[!eval.jpg](https://github.com/amitamola/Strawberry-Counting-and-Ripeness-detection/blob/main/evaluation.jpg)
