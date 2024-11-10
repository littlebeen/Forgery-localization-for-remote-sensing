# Forgery-localization-for-remote-sensing

🔥 Congradualations! This paper has been accepted by IEEE Transactions on Geoscience and Remote Sensing! It is my second TGRS. Hope I can achieve a ISPRS in the near future!

The code for FLDCF: A Collaborative Framework for Forgery Localization and Detection in Satellite Imagery.

# Data preprocessing 🔨

Our dataset could be obtained from 

```
├── mix3d
│   ├── main_instance_segmentation.py <- the main file
│   ├── conf                          <- hydra configuration files
│   ├── datasets
│   │   ├── preprocessing             <- folder with preprocessing scripts
│   │   ├── semseg.py                 <- indoor dataset
│   │   └── utils.py        
│   ├── models                        <- Mask3D modules
│   ├── trainer
│   │   ├── __init__.py
│   │   └── trainer.py                <- train loop
│   └── utils
├── data
│   ├── processed                     <- folder for preprocessed datasets
│   └── raw                           <- folder for raw datasets
├── scripts                           <- train scripts
├── docs
├── README.md
└── saved                             <- folder that stores models and logs
```

# Training and testing 🚆

Train
```
python src/main.py
```
Inference
```
python src/test.py
```

# Trained checkpoints 💾

Our pretrain model could be obtained from 


## BibTeX 🙏

If you have any questions, be free to contact with me! I promise I will reply as soon as possible.
```
loading
```
