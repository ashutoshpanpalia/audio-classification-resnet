# ResNet-50 for Accelerometer-Based Fault Classification

## Overview
This project explores the use of **pre-trained Convolutional Neural Networks (CNNs) for accelerometer data classification** by transforming time-series sensor data into spectrogram-like images and applying transfer learning.

Inspired by the paper **"Rethinking CNN Models for Audio Classification"** by Palanisamy et al. (2020), which demonstrated that pre-trained CNNs (originally trained on images) can be strong baselines for **audio classification**, I investigated whether a similar approach could be applied to **accelerometer data**.

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

### **3. Model Training**
- **Framework:** PyTorch
- **Optimization:** Adam optimizer, Cross-Entropy Loss
- **Hardware:** Supports both CPU & GPU acceleration

### **4. Model Evaluation**
- Evaluated trained models on the test set to compare performance under the three training strategies.
- The results aligned with prior research, showing that transfer learning can effectively be applied to accelerometer-based classification tasks.

---

## Results & Key Takeaways
- **Pre-trained ResNet-50 models are highly effective for accelerometer-based classification tasks** when the input is converted into a spectrogram representation.
- **Fine-tuning only the first and last layers** provided a good balance between training efficiency and accuracy.
- **Training from scratch** performed the worst due to the limited dataset size, highlighting the value of transfer learning.

This work supports the idea that **CNNs trained on images can be successfully applied to non-visual domains** like accelerometer data, broadening the scope of transfer learning in deep learning applications.

---

## References
- **Palanisamy, K., Singhania, D., & Yao, A. (2020).** *Rethinking CNN Models for Audio Classification.* arXiv preprint arXiv:2007.11154. [https://arxiv.org/abs/2007.11154](https://arxiv.org/abs/2007.11154)
- **Iantovics, L. B., Gligor, A., Montequín, V. R., Balogh, Z., Budinská, I., Gatial, E., ... & Dreyer, J. (2022).** *SOON: Social Network of Machines Solution for Predictive Maintenance of Electrical Drive in Industry 4.0.* Acta Marisiensis. Seria Technologica, 19(2), 12-19. [DOI Link (if available)]

---

## How to Use
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/resnet-spectrogram-classification.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the training script:
   ```bash
   python train.py
   ```
4. Evaluate the trained model:
   ```bash
   python evaluate.py
   ```

---

## Future Work
- Experiment with other CNN architectures (e.g., EfficientNet, MobileNet) for comparison.
- Test different spectrogram transformation techniques to optimize feature extraction.
- Apply the method to real-world predictive maintenance applications.

---

### **Acknowledgments**
Special thanks to the authors of the referenced research papers for their foundational work in transfer learning and predictive maintenance. 🚀

