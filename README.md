# BDD100K Object Detection — YOLOv8, YOLOv9, RT-DETR

This repository provides a complete workflow for preprocessing, training, and evaluating YOLOv8, YOLOv9, and RT-DETR on the BDD100K dataset — a large-scale and diverse driving dataset widely used for autonomous driving research.

# 📌 Dataset: BDD100K

BDD100K is one of the largest and most diverse driving datasets, containing 100,000 images collected across various weather, lighting, and city environments.

Official Dataset Reference:
- https://bair.berkeley.edu/blog/2018/05/30/bdd/

## 📥 1. Downloading the Dataset

- Dataset Source 1: http://bdd-data.berkeley.edu/
- Dataset Source 2: https://www.kaggle.com/datasets/solesensei/solesensei_bdd100k

### Steps

1. Click Download Dataset.
2. Download the following: "100K Images" and "Labels"
3. (Optional) Download Detection 2020 Labels for additional experiments.

## 📁 Dataset Structure
Your downloaded dataset should look like:
```bash
bdd100k/
 ├── images/
 |    ├── 10k/
 │    ├── 100k/
 │    │     ├── train/   (70,000 images)
 │    │     ├── val/     (10,000 images)
 │    │     └── test/    (20,000 images)
 bdd100k_labels_release
 ├── labels/
 |    ├── bdd100k_labels_images_train.json
 │    ├── bdd100k_labels_images_val.json
```
# 🛠 2. Preprocessing

All preprocessing steps are included inside:

- Data_Preprocessing_Visualization.ipynb

These notebooks convert BDD100K annotations into YOLO format, clean corrupted labels, visualize bounding boxes, and generate the final dataset structure for training.

### Dataset Structure After Preprocessing
```bash
bdd100k/
 ├── images/
 |    ├── 10k/
 │    ├── 100k/
 │    │     ├── train/ (.jpg*  70,000 images)
 │    │     ├── val/   (.jpg*  10,000 images)
 │    │     └── test/  (.jpg*  20,000 images)
 ├── labels/
 |    ├── train/ (.txt* 70,000)
 │    ├── valid/ (.txt* 10,000)
```
# 🧠 3. Training Pipelines
For more details about the models, refer to the official Ultralytics documentation: https://docs.ultralytics.com/

**The models used in this project are Model	Notebook:**
1. **YOLOv8**:	Yolov8.ipynb
   
2. **YOLOv9**:	Yolov9.ipynb
   
3. **RT-DETR**:	RT_DETR.ipynb
   
**The notebooks include:**
### 1. Install ultralytics
```
bash

pip install ultralytics
```
### 2. YAML configuration
#### Created file: **bdd100k.yaml**
```
path: C:\Users\bdd100k
train: images\100k\train
val: images\100k\val
names:
  0: person
  1: rider
  2: car
  3: truck
  4: bus
  5: train
  6: motor
  7: bike
  8: traffic light
  9: traffic sign
```
### 3. Hyperparameter settings
```md
Table of Hyperparameters
```
<div align="center">
<table border="1" cellpadding="6" cellspacing="0">
  <tr>
    <th><b>Hyperparameter</b></th>
    <th><b>Values</b></th>
  </tr>
  <tr><td>epochs</td><td>50</td></tr>
  <tr><td>imgsz</td><td>640</td></tr>
  <tr><td>batch</td><td>64</td></tr>
  <tr><td>device</td><td>0</td></tr>
  <tr><td>workers</td><td>4</td></tr>
  <tr><td>cache</td><td>True</td></tr>
  <tr><td>pretrained</td><td>True</td></tr>
  <tr><td>close_mosaic</td><td>10</td></tr>
  <tr><td>deterministic</td><td>False</td></tr>
  <tr><td>save_period</td><td>5</td></tr>
  <tr><td>patience</td><td>10</td></tr>
</table>
</div>

### 4. Training
- #### 4.1 Using CLI
##### YOLOv8
```
yolo detect train 
    data=bdd100k.yaml 
    model=yolov8n.pt 
    epochs=50 imgsz=640 
    batch=auto device=0 
    cache=True workers=4 
    close_mosaic=10 save_period=5 
    seed=42 project=runs/detect/
    name=bdd100k_yolov8
```
##### YOLOv9
```
yolo detect train 
    data=bdd100k.yaml 
    model=yolov9c.pt 
    epochs=50 imgsz=640 
    batch=auto device=0 
    cache=True workers=4 
    close_mosaic=10 save_period=5 
    seed=42 project=runs/detect/
    name=bdd100k_yolov9
```
##### RT-DERT
```
yolo detect train 
    data=bdd100k.yaml 
    model=rtdetr-l.pt
    epochs=50 imgsz=640 
    batch=auto device=0 
    cache=True workers=4 
    close_mosaic=10 save_period=5 
    seed=42 project=runs/detect/
    name=bdd100k_rtdert
```

- #### 4.2 Using Python API
You can use it con 3 file model Yolov8.ipynb, Yolov9.ipynb and RT_DERT.ipynb
### 5. Evaluation (mAP50, mAP50-95, precision/recall)

**All training outputs are saved in the ```/runs``` directory for later inspection.**

```
run/
 ├── weights/
 │    ├── 100k/
 │    │     ├── best.pt
 │    │     ├── epoch*.pt
 │    │     └── last.pt
 |    ├── args.yaml
 │    ├── confusion.matrix.png
 │    ├── .....
```

# ⚙️ 4. Environment Setup

### For RT-DETR you must install NumPy < 2.0 is required.

Install the correct NumPy version (CLI)

```
# Restart your environment after installation.
pip install -U "numpy<2" --force-reinstall
```

# 📊 5. Output Folders

After training, results are saved in:
```
run/bdd100k_yolov8/

run/bdd100k_yolov9/

run/bdd100k_rtdetr/
```

Each folder contains:

1. Trained weights

2. Training logs

3. Validation predictions

4. Evaluation reports

# 📌 6. Project Structure
```
├── Data_Preprocessing_Yolo.ipynb
├── Data_Preprocessing_Visualization.ipynb
├── Yolov8.ipynb
├── Yolov9.ipynb
├── RT_DETR.ipynb
├── bdd100k_yolov8/
├── bdd100k_yolov9/
├── bdd100k_rtdetr/
├── requirements.txt
└── README.md
```
# 🚀 7. Models Used

- YOLOv8 (Ultralytics)

- YOLOv9 (Enhanced hybrid task model)

- RT-DETR (Real-Time DETR by PaddlePaddle / re-implementation by Ultralytics)

#### These models are selected for:

- Fast training

- Good small-object performance

- High accuracy for detection tasks

# 📌 8. Citation
```
@inproceedings{xia2018predicting,
    title={Predicting driver attention in critical situations},
    author={Xia, Ye and Zhang, Danqing and Kim, Jinkyu and Nakayama, Ken and Zipser, Karl and Whitney, David},
    booktitle={Asian conference on computer vision},
    pages={658--674},
    year={2018},
    organization={Springer}
}
```

