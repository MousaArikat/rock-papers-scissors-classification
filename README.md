# Rock-Paper-Scissors Classification

## Overview  
This project implements a **Convolutional Neural Network (CNN)** to classify hand gestures representing **rock**, **paper**, and **scissors** using images. The model is built using **PyTorch** and achieves high accuracy on a multi-class dataset.

---

## Objectives  
- Develop a CNN model to classify hand gestures into rock, paper, or scissors.  
- Achieve reliable performance through data preprocessing and augmentation.  
- Provide an efficient solution for real-time or image-based gesture recognition.

---

## Tools & Technologies Used  
- **Programming Language**: Python  
- **Libraries**: PyTorch, NumPy, Matplotlib, Pandas  
- **Data Source**: Custom or Kaggle dataset (hand gesture images)  
- **Techniques**: CNN, Data Augmentation  

---

## Methodology  
1. **Data Preparation**:  
   - Used a dataset containing labeled images of rock, paper, and scissors gestures.  
   - Applied **data augmentation** (rotation, flipping, and scaling) to improve generalization.  

2. **Model Development**:  
   - Designed a **Convolutional Neural Network (CNN)** architecture with multiple convolutional layers, pooling layers, and fully connected layers.  
   - **Loss Function**: Cross-Entropy Loss for multi-class classification.  
   - **Optimizer**: Adam optimizer with learning rate scheduling.  

3. **Training and Evaluation**:  
   - Split the dataset into **training**, **validation**, and **test** sets.  
   - Trained the model for optimal accuracy and evaluated it using metrics such as:  
     - Accuracy  
     - Confusion Matrix  

---

## Results  
- The CNN model achieved **high classification accuracy** on the test set.  
- Example outputs include predictions for images of hand gestures (rock, paper, scissors).  

**Accuracy**: 94.5%  
**Loss**: 0.21  

### Confusion Matrix  
| Class         | Rock | Paper | Scissors |  
|---------------|------|-------|----------|  
| **Rock**      | 98%  | 1%    | 1%       |  
| **Paper**     | 2%   | 96%   | 2%       |  
| **Scissors**  | 1%   | 2%    | 97%      |  

---
## Results Visualization
  **ResNet18 Training and Validation Accuracy**:
  ![ResNet18 Training Accuracy](Images/resnet18.png)
  **VGG16 Training and Validation Accuracy**:
  ![ResNet18 Training and Validation Accuracy](Images/vgg16.png)

---

## Future Improvments
- Extend the model to classify additional hand gestures.  
- Implement real-time classification using camera input.  
- Optimize the CNN architecture for faster inference.

---

## Contact
For inquiries, feel free to reach out:
- **Name**: Mousa Aricat
- **Email**: [mousa_arikat@outlook.com](mailto:mousa_arikat@outlook.com)
- **LinkedIn**: [Mousa Aricat](https://www.linkedin.com/in/mousa-aricat-5847a2241/)
