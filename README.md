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

To perform the ablation study, a shared ResNet model architecture class was established (), which differed only 

### Overall structure

<figure>
  <img width="388" height="774" alt="Image" src="https://github.com/user-attachments/assets/a209964e-f5e7-4370-b960-05df5499bafc" />
  <figcaption><em>Figure 1: Shared ResNet model architecture with ablation study changes split into two paths.</em></figcaption>
</figure>




### ResNet Blocks variants without and with CBAM

<img width="1012" height="953" alt="Image" src="https://github.com/user-attachments/assets/916d307e-caf4-481d-b5e9-8c0669568f7b" />

## Results

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

