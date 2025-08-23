# 📚 Deep Learning for Book Cover Classification  

This project applies **deep learning and transfer learning** to classify book covers into **10 genres** using the [Amazon Books Review dataset](https://jmcauley.ucsd.edu/data/amazon/).  

I designed a **scalable training pipeline** with caching, subsampling, and a two-stage fine-tuning strategy using **MobileNetV2**.  

---

## 🛠️ Skills & Tools
- **Python** (data pipelines, automation, caching)
- **Deep Learning**: TensorFlow / Keras (CNNs, Transfer Learning, Fine-Tuning)
- **Data Science**: NumPy, Pandas, scikit-learn
- **Visualization**: Matplotlib, Seaborn (confusion matrix, training curves)
- **ML Practices**: regularization, augmentation, class balancing, experiment tracking

---

## 📊 Results
- **Subsampled dataset (5,000 covers)**:  
  - Test Accuracy: **30.2%** | Macro F1: **0.28**  
  - Strongest categories: *Juvenile Fiction (F1=0.44)*, *Biography (F1=0.42)*  
  - Weakest categories: *Education (F1=0.10)*, *Social Science (F1=0.07)*  

- **Full dataset (~73,000 covers)**:  
  - Test Accuracy: **44.7%** | Macro F1: **0.39**  
  - Better overall generalization, but class imbalance remains a challenge.  

---

## 🚀 Key Takeaways
- Built a **scalable image classification pipeline** (caching, subsampling, reproducible splits).  
- Applied **transfer learning** with MobileNetV2 and fine-tuned top layers.  
- Identified **semantic/visual overlaps** between categories (e.g., *Education vs Business & Economics*).  

---
  

 
