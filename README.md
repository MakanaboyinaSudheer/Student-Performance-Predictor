# Student Performance & Risk Prediction System

This is a complete end-to-end Machine Learning web application designed to predict a student's exam score based on various performance factors and provide personalized recommendations. 

## 📁 Project Structure

```
student_project/
│
├── app.py                # Flask Web Application Backend
├── train_model.py        # Python script for Data Preprocessing, EDA, and Model Training
├── templates/
│   └── index.html        # Frontend HTML
├── static/
│   └── style.css         # Frontend Styling
└── README.md             # Project Instructions
```

## 🚀 How to Run the Project

### 1. Requirements Install
Make sure you have all required libraries installed. You can install them using pip:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn flask
```

### 2. Dataset Setup
Ensure that the `StudentPerformanceFactors.csv` file is placed directly in the `student_project` root directory.

### 3. Train the Model
Run the training script to preprocess data, evaluate models, and save the final Random Forest model (`model.pkl`).

```bash
python train_model.py
```
*(This script will also generate Exploratory Data Analysis (EDA) visualizations as PNG images in your folder.)*

*(Note: The `train_model.py` mirrors the steps usually taken in a Jupyter Notebook `notebook.ipynb`. You can run its code inside a Jupyter Notebook interactively to observe output cell by cell.)*

### 4. Run the Flask Application
Once the `model.pkl` is successfully created, start the web app:

```bash
python app.py
```

### 5. Access the Web App
Open your browser and navigate to:
[http://127.0.0.1:5000](http://127.0.0.1:5000)

## 📊 Features Included
- **Exploratory Data Analysis:** Data correlations, visualizations, missing value handling.
- **Machine Learning Pipeline:** One-Hot Encoding, StandardScaler scaling.
- **Model Comparison:** Evaluates Linear Regression, Decision Tree, and Random Forest.
- **Feature Importance:** Computes and plots feature significance.
- **Web App:** Clean, modern frontend for end-users.
- **Recommendation Engine:** Personalised feedback system based on input rules.
