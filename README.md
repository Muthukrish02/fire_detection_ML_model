# 🔥 Fire Detection Using Machine Learning

This project uses a **Machine Learning model** to detect fire in images or videos. The model is trained and tested in a **Google Colab notebook** and can be reused for real-time or batch fire detection systems.

---

## 📌 Features

- 📁 Uses image/video data for fire detection
- 🔍 Logistic Regression (or other ML model as per notebook)
- 🧪 Train/Test split with evaluation metrics
- 💾 Model saving and loading using `joblib` or `pickle`
- 📊 Visualization using Matplotlib and Seaborn
- ☁️ Fully run on Google Colab (GPU/TPU optional)

---

## 📂 Files Included

- `fire_detection.ipynb` – Main notebook for training/testing the model
- `fire_model.pkl` – Trained ML model (can be used for prediction)
- `README.md` – Project overview and usage instructions

---

## 🚀 Try on Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Muthukrish02/fire_detection_ML_model/blob/main/fire_detection.ipynb)

---

## 🛠 How to Use

1. Clone or download the repo
2. Open `fire_detection.ipynb` in Google Colab
3. Run the cells to:
   - Load data
   - Train the model
   - Save the model
4. Predict using the saved model:

```python
import joblib
model = joblib.load("fire_model.pkl")
result = model.predict([your_input_data])
print(result)
