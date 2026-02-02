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

                                 Apple, Apple scab       0.86      0.80      0.83        94
                                  Apple, Black rot       0.84      0.94      0.88        93
                           Apple, Cedar apple rust       0.95      1.00      0.98        41
                                    Apple, healthy       0.91      0.93      0.92       246
                                Blueberry, healthy       0.94      0.97      0.95       226
            Cherry(including sour), Powdery mildew       0.94      0.96      0.95       158
                   Cherry(including sour), healthy       0.78      0.99      0.87       128
  Corn(maize), Cercospora leaf spot Gray leaf spot       0.79      0.88      0.83        77
                         Corn(maize), Common rust        1.00      0.97      0.99       179
                 Corn(maize), Northern Leaf Blight       0.91      0.87      0.89       147
                              Corn(maize), healthy       0.99      0.99      0.99       174
                                  Grape, Black rot       0.87      0.79      0.83       177
                       Grape, Esca (Black Measles)       0.98      0.95      0.97       207
         Grape, Leaf blight (Isariopsis Leaf Spot)       0.88      0.97      0.92       162
                                    Grape, healthy       0.77      1.00      0.87        64
           Orange, Haunglongbing (Citrus greening)       0.98      0.99      0.98       826
                             Peach, Bacterial spot       0.98      0.85      0.91       344
                                    Peach, healthy       0.95      0.98      0.96        54
                       Bell Pepper, Bacterial spot       0.97      0.82      0.89       150
                              Bell Pepper, healthy       0.90      0.96      0.93       222
                              Potato, Early blight       0.95      0.96      0.95       150
                               Potato, Late blight       0.71      0.80      0.75       150
                                   Potato, healthy       0.22      1.00      0.37        23
                                Raspberry, healthy       0.53      0.98      0.69        55
                                  Soybean, healthy       0.96      0.91      0.94       764
                            Squash, Powdery mildew       0.96      0.99      0.98       275
                           Strawberry, Leaf scorch       0.97      0.99      0.98       167
                               Strawberry, healthy       0.87      0.91      0.89        68
                            Tomato, Bacterial spot       0.97      0.88      0.92       319
                              Tomato, Early blight       0.92      0.62      0.74       150
                               Tomato, Late blight       0.86      0.71      0.78       287
                                 Tomato, Leaf Mold       0.80      0.92      0.86       143
                        Tomato, Septoria leaf spot       0.98      0.84      0.90       265
      Tomato, Spider mites Two-spotted spider mite       0.82      0.90      0.86       252
                               Tomato, Target Spot       0.87      0.79      0.83       211
             Tomato, Tomato Yellow Leaf Curl Virus       0.97      0.91      0.94       803
                       Tomato, Tomato mosaic virus       0.77      1.00      0.87        56
                                   Tomato, healthy       0.93      1.00      0.96       239

                                          accuracy                           0.91      8146
                                         macro avg       0.88      0.91      0.88      8146
                                      weighted avg       0.92      0.91      0.91      8146


```
</details>

<details>
<summary><strong> Classification Report, ResNet + CBAM</strong></summary>

```text
                                                    precision    recall  f1-score   support

                                Apple, Apple scab       0.95      0.89      0.92        94
                                 Apple, Black rot       0.96      0.97      0.96        93
                          Apple, Cedar apple rust       0.95      1.00      0.98        41
                                   Apple, healthy       1.00      0.87      0.93       246
                               Blueberry, healthy       0.94      0.97      0.95       226
           Cherry(including sour), Powdery mildew       0.96      0.98      0.97       158
                  Cherry(including sour), healthy       0.97      0.95      0.96       128
 Corn(maize), Cercospora leaf spot Gray leaf spot       0.72      0.87      0.79        77
                        Corn(maize), Common rust        0.99      1.00      1.00       179
                Corn(maize), Northern Leaf Blight       0.88      0.83      0.86       147
                             Corn(maize), healthy       0.99      1.00      1.00       174
                                 Grape, Black rot       0.96      0.87      0.91       177
                      Grape, Esca (Black Measles)       0.97      0.99      0.98       207
        Grape, Leaf blight (Isariopsis Leaf Spot)       0.93      1.00      0.96       162
                                   Grape, healthy       0.94      1.00      0.97        64
          Orange, Haunglongbing (Citrus greening)       1.00      0.94      0.97       826
                            Peach, Bacterial spot       0.96      0.94      0.95       344
                                   Peach, healthy       0.84      1.00      0.92        54
                     Pepper, bell, Bacterial spot       0.88      1.00      0.93       150
                            Pepper, bell, healthy       0.95      0.98      0.96       222
                             Potato, Early blight       0.96      0.94      0.95       150
                              Potato, Late blight       0.91      0.89      0.90       150
                                  Potato, healthy       0.66      0.91      0.76        23
                               Raspberry, healthy       0.95      0.96      0.95        55
                                 Soybean, healthy       0.99      0.96      0.97       764
                           Squash, Powdery mildew       1.00      0.99      0.99       275
                          Strawberry, Leaf scorch       0.96      0.99      0.97       167
                              Strawberry, healthy       0.91      1.00      0.95        68
                           Tomato, Bacterial spot       0.89      0.96      0.92       319
                             Tomato, Early blight       0.82      0.90      0.86       150
                              Tomato, Late blight       0.94      0.80      0.87       287
                                Tomato, Leaf Mold       0.96      0.97      0.96       143
                       Tomato, Septoria leaf spot       0.96      0.97      0.97       265
     Tomato, Spider mites Two-spotted spider mite       0.90      0.92      0.91       252
                              Tomato, Target Spot       0.89      0.94      0.92       211
            Tomato, Tomato Yellow Leaf Curl Virus       0.96      0.96      0.96       803
                      Tomato, Tomato mosaic virus       0.75      1.00      0.85        56
                                  Tomato, healthy       0.95      1.00      0.98       239

                                          accuracy                           0.95      8146
                                         macro avg       0.92      0.95      0.93      8146
                                      weighted avg       0.95      0.95      0.95      8146

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

