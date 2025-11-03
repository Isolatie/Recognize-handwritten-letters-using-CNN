# Recognize handwritten letters using CNN

A TensorFlow-based deep learning project that classifies **handwritten letters** from the **EMNIST dataset** using a **Convolutional Neural Network (CNN)**.  
Includes data preprocessing, visualization, and prediction display utilities.

---

## 🚀 Overview

This project is to showcase how to build, train, and visualize a **CNN** for letter recognition using the **EMNIST dataset**. 
The notebook or script:
- Downloads and preprocesses the EMNIST data  
- Constructs a convolutional neural network  
- Trains it to classify handwritten letters (A–Z)  
- Visualizes model predictions in a 3×3 grid  

---

## 🧠 Model Architecture

| Layer | Type | Parameters | Activation |  
|--------|------|-------------|-------------|  
| 1 | Conv2D | 32 filters (5×5), same padding | ReLU |  
| 2 | MaxPooling2D | 2×2 pool size | — |  
| 3 | Conv2D | 64 filters (5×5), same padding | ReLU |  
| 4 | MaxPooling2D | 2×2 pool size | — |  
| 5 | Flatten | — | — |  
| 6 | Dense | Output = `num_classes + 1` | Softmax |  

Loss function: **Categorical Crossentropy**  
Optimizer: **Adam**  
Metric: **Accuracy**

---

## ⚙️ Requirements

```
pip install -r requirements.txt
```

# Python 3.10  
numpy>=1.23.0  
torch>=2.0.0  
torchvision>=0.15.0  
tensorflow>=2.12.0  
matplotlib>=3.7.0  
idx2numpy>=1.3.0  

run simulation by  
```
python featuremap.py
```



