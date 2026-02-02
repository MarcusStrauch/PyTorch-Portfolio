# PyTorch Portfolio Project by Marcus Strauch

This project contains an ablation study using a residual neural network architecture with and without convolutional block attention modules (CBAM) as proposed by Woo, Sanghyun, et al. (2018) [[1]](#cbam). Both models are trained on the plant disease image dataset published by Mohanty et. al. (2016), using the same training loop, loss function, optimizer and scheduler. The aim was to investigate whether the inclusion of CBAM can markedly increase the multiclassification performance of the neural network model for an unbalanced, moderately sized dataset of images.

## Technologies Used
<p align="center">
  <img width="652" height="196" alt="Image" src="https://github.com/user-attachments/assets/c0cdd35e-dfc8-491f-a398-e83fd91eb955" />
</p>

## Data Augmentation

Prior to model training, the data was split into training, validation, and testing sets. To reduce overfitting, the images data in the training dataset was augmented using transformations frequently used in computer vision, especially in medical multiclass classification tasks. This was achieved using a variety of random geometric and color transformations, as well as random blurring. 

All images, including validation and test images were resized to 224x224 pixel formats, and normalized prior to model training.

## Neural Network Architectures

To perform the ablation study, a shared ResNet model architecture class was established as shown in [fig. 1](#fig1). For the model without CBAM, the corresponding channel and spatial attention modules of the ResNet blocks were turned off.

### Overall structure

<a id="fig1">
  <p align="center">
    <img width="388" height="774" alt="Image" src="https://github.com/user-attachments/assets/a209964e-f5e7-4370-b960-05df5499bafc" /><br>
    <em>Figure 1: Shared ResNet model architecture with ablation study changes split into two paths.</em>
  </p>
</a>

The implementations of the ResNet blocks used in this study are shown in [fig. 2](#fig2), illustrating the data flow through the individual ResNet blocks and in the case of the CBAM-enabled version, the channel and spatial attention modules.

### ResNet Blocks variants without and with CBAM
<a id="fig2">
  <p align="center">
    <img width="1012" height="953" alt="Image" src="https://github.com/user-attachments/assets/916d307e-caf4-481d-b5e9-8c0669568f7b" /><br>
    <em>Figure 2: The two ResNet block architectures used in this study.</em>
  </p>
</a>

## Results

| Model | Accuracy | Macro F1 | Weighted F1 |
|------|---------|----------|-------------|
| ResNet | 0.91 | 0.88 | 0.91 |
| ResNet + CBAM | **0.95** | **0.93** | **0.95** |

<details>
<summary><strong> Classification Report, ResNet (no CBAM)</strong></summary>

```text
precision    recall  f1-score   support

Apple___Apple_scab       0.86      0.80      0.83        94
Apple___Black_rot        0.84      0.94      0.88        93
Apple___Cedar_apple_rust 0.95      1.00      0.98        41
Apple___healthy          0.91      0.93      0.92       246


```
</details>

<img width="1153" height="796" alt="Image" src="https://github.com/user-attachments/assets/5f78eda7-1418-4204-9e91-990cd116da3d" />

## References
<a id="cbam">[1]</a>
Woo, Sanghyun, et al. 
"Cbam: Convolutional block attention module." 
Proceedings of the European conference on computer vision (ECCV). 2018.

<a id="plantvillage">[2]</a>
Mohanty, Sharada P., David P. Hughes, and Marcel Salathé. 
"Using deep learning for image-based plant disease detection." 
Frontiers in plant science 7 (2016): 215232.

