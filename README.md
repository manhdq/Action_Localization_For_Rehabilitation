# Action_Localization_For_Rehabilitation


## Introduction
The problems of rehabilitation are always necessary while the human resources and economic conditions, the current labors machines are still limited, and some of the problems can be partly solved by automatically supporting motion recognition and supervision in rehabilitation for patients. There are methods for recognizing different behaviors and movements, but they are not many and difficult to apply to the current problem, especially similar specific methods in healthcare are almost non-existent, not to mention data related to this problem is very rare. almost all of them are confidential and are really lacking (it can be said that they don't exist). Therefore, in this study, I propose a 3-stages model: **Human Heatmaps Extractor**, **Patch Features Extractor**, and **Temporal Action Localizor**, using reasonable data augmentation methods, the model can learn generally from only limited datasets.

<br>

<div align="center">
    <img src="./images/model_overview.png" width="600px" alt><br>
Model overview

</div>

<br>

## Quick start
1. Clone this repo
  ```
  git clone https://github.com/manhdqhe153129/Action_Localization_For_Rehabilitation.git
  ```
  
2. Set up NMS algorithm is implemented in C++ for Temporal Action Localizor stage
  ```
  cd ./ActionFormer/libs/utils
  python setup.py install --user
  cd ../../..
  ```
  The code should be recompiled every time you update PyTorch
  
3. Install dependencies
  ```
  pip install -r requirements.txt
  ```
  
## Main results on rehabilitation custom data
| Arch                                                        | mAP@tIoU0.5 | mAP@tIoU0.6 | mAP@tIoU0.7 | mAP@tIoU0.8 | mAP@tIoU0.9 | mAP@tIoU.3-.9 |
|-------------------------------------------------------------|-------------|-------------|-------------|-------------|-------------|---------------|
| HR-R(2+1)D18-ActionFormer                                   | 0.998       |     0.985   |  0.965      |  0.846      | 0.35        | 0.877         |
| HR-R(2+1)D12-ActionFormer                                   | 0.999       |     0.978   |  0.883      |  0.737      | 0.260       | 0.837         |
| **HR-R(2+1)D12-ActionFormer**<br><sup>(expansion ratio 1.5) | 0.999       |     0.995   |  0.917      |  0.758      | 0.499       | 0.881         |
| LiteHR-R(2+1)D12-ActionFormer<br><sup>(expansion ratio 1.5) | -           |     -       |  -          |  -          | -           | -             |
 
Model checkpoints

| Arch                                                        | Heatmaps Extractor     | Patch Features Extractor | Temporal Action Localizor |
|-------------------------------------------------------------|------------------------|--------------------------|---------------------------|
| HR-R(2+1)D18-ActionFormer                                   | OneDrive / GoogleDrive | OneDrive / GoogleDrive   | OneDrive / GoogleDrive    |
| HR-R(2+1)D12-ActionFormer                                   | OneDrive / GoogleDrive | OneDrive / GoogleDrive   | OneDrive / GoogleDrive    |
| **HR-R(2+1)D12-ActionFormer**<br><sup>(expansion ratio 1.5) | OneDrive / GoogleDrive | OneDrive / GoogleDrive   | OneDrive / GoogleDrive    |
| LiteHR-R(2+1)D12-ActionFormer<br><sup>(expansion ratio 1.5) | OneDrive / GoogleDrive | OneDrive / GoogleDrive   | OneDrive / GoogleDrive    |
  
## Testing
  

<br>

<div align="center">
    <img src="./images/output_demo.gif" width="600px" alt><br>
Output demo

</div>

<br>