# PyTorch Portfolio Project by Marcus Strauch

This project contains an ablation study using a residual neural network architecture with and without convolutional block attention modules (CBAM) as proposed by Woo, Sanghyun, et al. (2018) [[1]](#cbam). Both models are trained on the plant disease image dataset published by Mohanty et. al. (2016), using the same training loop, loss function, optimizer and scheduler. The aim was to investigate whether the inclusion of CBAM can markedly increase the multiclassification performance of the neural network model for an unbalanced, moderately sized dataset of images.

## Technologies Used
<p align="center">
  <img width="652" height="196" alt="Image" src="https://github.com/user-attachments/assets/c0cdd35e-dfc8-491f-a398-e83fd91eb955" />
</p>

## Neural Network Architectures

<img width="388" height="774" alt="Image" src="https://github.com/user-attachments/assets/a209964e-f5e7-4370-b960-05df5499bafc" />

test

<img width="389" height="451" alt="Image" src="https://github.com/user-attachments/assets/69948214-8d93-4995-b1a1-8e02e19353ed" />
<img width="505" height="900" alt="Image" src="https://github.com/user-attachments/assets/1ab4c849-d367-4462-9e6b-4f7e480c167b" />

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

