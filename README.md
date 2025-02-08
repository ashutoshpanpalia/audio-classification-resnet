# ResNet-50 for Accelerometer-Based Fault Classification

## Overview
This project explores the use of **pre-trained Convolutional Neural Networks (CNNs)  defect classification using acclerometer data** by transforming time-series sensor data into spectrogram-like images and applying transfer learning.

The paper **"Rethinking CNN Models for Audio Classification"** by Palanisamy et al. (2020), which demonstrated that pre-trained CNNs (originally trained on images) can be strong baselines for **audio classification**, I investigated whether a similar approach could be applied to **accelerometer data**.

To test this hypothesis, I used a publicly available dataset from **Iantovics et al. (2022)** that contains accelerometer readings for different electric motor defects. The project compares different training approaches on a **ResNet-50** model.

---

## What This Project Does

### **1. Data Preprocessing**
- Converted **raw accelerometer time-series data (X, Y, Z axes)** into **Mel spectrograms** to create image-like representations.
- This transformation allows us to leverage CNN architectures that excel at image recognition.

### **2. Transfer Learning Approach**
The project applies **three different training strategies** to a **ResNet-50 model**:
1. **Fine-tuning the entire model** starting from pre-trained ImageNet weights.
2. **Freezing all layers except the first and last**, retraining only these.
3. **Training from scratch**, randomly initializing the network weights.

![image](https://github.com/user-attachments/assets/41c7f5ec-72fc-4e5c-89cc-8158ea0197c3)


### **3. Model Training**
- **Framework:** PyTorch
- **Optimization:** Adam optimizer, Cross-Entropy Loss
- **Hardware:** Supports both CPU & GPU acceleration

### **4. Model Evaluation**
![image](https://github.com/user-attachments/assets/5f52f914-4fd2-49ac-9717-15a3d1f1ee7b)

---

## Results & Key Takeaways
- **Pre-trained ResNet-50 models are highly effective for accelerometer-based classification tasks** when the input is converted into a spectrogram representation.
- **Fine-tuning only the first and last layers** has training efficiency but comes with a massive trade-off in performance.
- **Training from scratch** performs better than fine-tuning only the first and last layer, but never out performs the training which starts with pre-trained weights. 

This work supports the idea that **CNNs trained on images can be successfully applied to non-visual domains** like accelerometer data, broadening the scope of transfer learning in deep learning applications. The results also re-iterates the findings by Palanisamy et al. that for a given standard model using pretrained
weights is better than using randomly initialized weights.

---

## References
- **Palanisamy, K., Singhania, D., & Yao, A. (2020).** *Rethinking CNN Models for Audio Classification.* arXiv preprint arXiv:2007.11154. [https://arxiv.org/abs/2007.11154](https://arxiv.org/abs/2007.11154)
- **Iantovics, L. B., Gligor, A., Montequín, V. R., Balogh, Z., Budinská, I., Gatial, E., ... & Dreyer, J. (2022).** *SOON: Social Network of Machines Solution for Predictive Maintenance of Electrical Drive in Industry 4.0.* Acta Marisiensis. Seria Technologica, 19(2), 12-19. [https://doi.org/10.2478/amset-2022-0012]

---

### **Acknowledgments**
Special thanks to the authors of the referenced research papers for their foundational work in transfer learning and predictive maintenance. 🚀

