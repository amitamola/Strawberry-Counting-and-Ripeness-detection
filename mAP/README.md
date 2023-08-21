# mAP (mean Average Precision)
This code will evaluate the performance of your neural net for object recognition.

In practice, a **higher mAP** value indicates a **better performance** of your neural net, given your ground-truth and set of classes.

## Citation
This repo is for my own use and has been worked on using https://github.com/Cartucho/mAP. My addition here is that it results in a result.csv that contains label level PR values.


## Running the code

Step by step:

  1. [Create the ground-truth files](#create-the-ground-truth-files)
  2. Copy the ground-truth files into the folder **input/ground-truth/**
  3. [Create the detection-results files](#create-the-detection-results-files)
  4. Copy the detection-results files into the folder **input/detection-results/**
  5. Run the code:
         ```
         python main.py
         ```

This operation is also performed inside the notebook - **"4. Inference Notebook on original test set"** but this module can be used on your own as well using same steps above.

#### Create the ground-truth files

- Create a separate ground-truth text file for each image.
- Use **matching names** for the files (e.g. image: "image_1.jpg", ground-truth: "image_1.txt").
- In these files, each line should be in the following format:
    ```
    <class_name> <left> <top> <right> <bottom> [<difficult>]
    ```
- The `difficult` parameter is optional, use it if you want the calculation to ignore a specific detection.
- E.g. "image_1.txt":
    ```
    tvmonitor 2 10 173 238
    book 439 157 556 241
    book 437 246 518 351 difficult
    pottedplant 272 190 316 259
    ```

#### Create the detection-results files

- Create a separate detection-results text file for each image.
- Use **matching names** for the files (e.g. image: "image_1.jpg", detection-results: "image_1.txt").
- In these files, each line should be in the following format:
    ```
    <class_name> <confidence> <left> <top> <right> <bottom>
    ```
- E.g. "image_1.txt":
    ```
    tvmonitor 0.471781 0 13 174 244
    cup 0.414941 274 226 301 265
    book 0.460851 429 219 528 247
    chair 0.292345 0 199 88 436
    book 0.269833 433 260 506 336
    ```