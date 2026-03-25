# 🏠 Loan Prediction App (KNN)

A simple and interactive **Machine Learning web application** built using **Streamlit** that predicts loan-related outcomes using the **K-Nearest Neighbors (KNN) algorithm**.

---

## 🚀 Live Demo

https://loanpredictionknn-nikhilbontha.streamlit.app/

---

## 📌 Features

* 📊 Dataset preprocessing (missing values, encoding, scaling)
* 🤖 KNN Regression model
* 🎯 Adjustable **K value** using slider
* 📈 Model performance metrics (MSE, R² Score)
* 🧾 User-friendly input form (dropdowns + numeric inputs)
* 🔮 Real-time prediction

---

## 🛠️ Tech Stack

* Python 🐍
* Streamlit 🌐
* Pandas & NumPy 📊
* Scikit-learn 🤖
* Matplotlib 📈

---

## 📂 Project Structure

```
├── app.py                # Main Streamlit app
├── loan_data.csv         # Dataset
├── requirements.txt      # Dependencies
└── README.md             # Project documentation
```

---


## 📊 How It Works

1. Dataset is loaded and preprocessed
2. Categorical features are encoded using one-hot encoding
3. Features are scaled using MinMaxScaler
4. KNN model is trained based on selected K value
5. User inputs are transformed and passed to the model
6. Prediction is displayed instantly

---

## 🎯 Example Inputs

* City: Hyderabad / Chennai / Bangalore / Mumbai
* Employment Type: Salaried / Self-Employed / Unemployed
* Loan Type: Home / Personal
* Age, Income, Loan Amount, Credit Score

---

## 📈 Model Used

* **K-Nearest Neighbors Regressor**
* Distance-based learning algorithm
* Performance evaluated using:

  * Mean Squared Error (MSE)
  * R² Score

---

## 🔥 Future Improvements

* 📌 Add classification (Loan Approved / Rejected)
* 📊 Visualization graphs (Actual vs Predicted)
* 💾 Save/load trained model (.pkl)
* 🌐 Deploy with custom domain
* 🤖 Add explainable AI (why prediction?)

---

## 👨‍💻 Author

**Nikhil Bontha**
🔗 LinkedIn: https://www.linkedin.com/in/nikhil-bontha-159872288

---

## ⭐ Support

If you like this project, give it a ⭐ on GitHub!

---
