# Mosquito Detection & Classification with YOLOv9 and RT-DETR Ensemble

This repository contains the code for the **"Low Light Object Detection and Classification of Mosquitos"** Kaggle competition. The goal is to develop a model that can accurately detect and classify mosquitoes into six distinct species from images, a critical task for preventing disease outbreaks.

This solution implements an ensemble of two powerful object detection models, **YOLOv9** and **RT-DETR**, to achieve robust performance.

## 📜 Project Overview

Mosquitoes are vectors for numerous diseases like Zika, Dengue, and Chikungunya, making their surveillance and control a global health priority. This project automates the identification of mosquito species from images, which is traditionally a labor-intensive process. By leveraging deep learning, we can create a more efficient and scalable solution.

The primary tasks are:

1.  **Object Detection:** Locate the mosquito within each image by predicting a bounding box.
2.  **Classification:** Identify the species of the detected mosquito from six possible classes.

The final model performance is evaluated based on the **mean Average Precision (mAP)** score.

## 🤖 Models & Methodology

To maximize accuracy, this project uses a simple yet effective ensemble approach. Two state-of-the-art models, YOLOv9 and RT-DETR, are trained independently. For the final prediction on each test image, the model with the highest confidence score is chosen.

### Workflow

The entire process is contained within the `rtdetr-yolo-ensemble.ipynb` notebook and follows these steps:

1.  **Data Preparation**:

      * The dataset is first copied into the working directory.
      * The training dataset is split into an 80% training set and a 20% validation set to monitor model performance.

2.  **Configuration**:

      * A `dataset.yaml` file is created to define the dataset paths and class names for the Ultralytics framework.

    <!-- end list -->

    ```yaml
    path: /kaggle/working/final_dlp_data
    train: train/images
    val: val/images
    test: test/images

    names:
        0: aegypti
        1: albopictus
        2: anopheles
        3: culex
        4: culiseta
        5: japonicus/koreicus
    ```

3.  **Model Training**:

      * **YOLOv9**: The `yolov9c.pt` pretrained model is fine-tuned on the mosquito dataset for 5 epochs.
      * **RT-DETR**: The `rtdetr-l.pt` pretrained model is also fine-tuned for 5 epochs.

4.  **Inference and Ensemble**:

      * The best-performing weights for both models are loaded.
      * The script iterates through each image in the test set.
      * Both YOLOv9 and RT-DETR generate a prediction (bounding box, class, and confidence score).
      * The prediction with the **highest confidence score** between the two models is selected as the final result for that image.
      * If one model fails to make a detection, the other model's prediction is used.
      * The results are compiled into a `submission.csv` file in the format required by the competition.

## 📊 Results

The models were evaluated on the validation set after 5 epochs of training.

| Model | mAP50-95 (Validation) |
| :--- | :---: |
| YOLOv9c | 0.269 |
| **RT-DETR-l** | **0.293** |

RT-DETR showed slightly better performance on the validation set, validating the ensemble approach which leverages the strengths of both architectures.

## 🚀 How to Run

To reproduce these results, follow these steps:

1.  **Clone the Repository**:

    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```

2.  **Install Dependencies**:
    The primary dependency is the `ultralytics` library.

    ```bash
    pip install -q ultralytics pandas numpy
    ```

3.  **Download Dataset**:

      * Download the dataset from the [Kaggle competition page](https://www.google.com/search?q=https://www.kaggle.com/competitions/dlp-object-detection-week-10/data).
      * Place the data in a directory structure that the notebook expects, for example: `/kaggle/input/dlp-object-detection-week-10/`.

4.  **Run the Notebook**:

      * Open `rtdetr-yolo-ensemble.ipynb` in a Jupyter environment (like Jupyter Lab or VS Code).
      * Ensure the file paths in the notebook match your dataset location.
      * Execute the cells sequentially from top to bottom. The notebook will handle data splitting, training, inference, and creating the final `dual_submission_final.csv` file.
