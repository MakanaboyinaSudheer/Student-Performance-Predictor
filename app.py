from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model pipeline spanning preprocessing and ML
try:
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    model = None
    print("WARNING: model.pkl not found! Please run train_model.py first.")

# RECOMMENDATION SYSTEM LOGIC based on prompt requirements
def get_recommendations(features):
    suggestions = []
    
    # 1. Low study_hours -> suggest increasing study time
    if float(features.get('study_hours', 0)) < 15:
        suggestions.append("💡 Suggestion: Start increasing your study time to at least 15-20 hours per week for better preparation.")
        
    # 2. Low attendance -> suggest attending classes
    if float(features.get('attendance', 0)) < 85:
        suggestions.append("⚠️ Suggestion: Your attendance is a low risk factor. Make an effort to attend all classes.")
        
    # 3. Poor sleep -> suggest better sleep habits
    if float(features.get('sleep_hours', 0)) < 7:
        suggestions.append("🛌 Suggestion: Ensure better sleep habits! 7-8 hours of sleep per night drastically improves memory retention.")
        
    # 4. Low motivation -> suggest engagement tips
    if features.get('motivation_level', '').lower() in ['low', 'poor', 'medium']:
        suggestions.append("🚀 Suggestion: Building consistency can help with motivation. Break down tasks and actively engage in class or study groups.")
        
    if not suggestions:
        suggestions.append("🌟 Amazing! You have very strong foundational habits for high performance. Keep it up!")
        
    return suggestions

def get_grade_and_risk(score):
    # Grades: A (>=85), B (70–84), C (50–69), Fail (<50)
    # Risk Level: Low, Medium, High
    if score >= 85:
        return 'A', 'Low'
    elif score >= 70:
        return 'B', 'Low'
    elif score >= 50:
        return 'C', 'Medium'
    else:
        return 'Fail', 'High'

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    grade = None
    risk = None
    suggestions = []
    
    if request.method == 'POST':
        if model is None:
            return render_template('index.html', error="Model not found. Please train the model (train_model.py) first and save model.pkl.")
        
        try:
            # Capture user input from the HTML form
            features = {
                'study_hours': float(request.form['study_hours']),
                'attendance': float(request.form['attendance']),
                'motivation_level': request.form['motivation_level'],
                'learning_resources': request.form['learning_resources'],
                'parental_participation': request.form['parental_participation'],
                'teacher_quality': request.form['teacher_quality'],
                'peer_influence': request.form['peer_influence'],
                'family_background': request.form['family_background'],
                'internet_availability': request.form['internet_availability'],
                'sleep_hours': float(request.form['sleep_hours']),
                'school_type': request.form['school_type']
            }
            
            # Predict
            df_features = pd.DataFrame([features])
            pred_score = model.predict(df_features)[0]
            
            # Constraints: ensure score stays within expected exam bounds [0, 100]
            pred_score = max(0, min(100, pred_score))
            
            # Get Grades, Risk and Recommendations feature
            grade, risk = get_grade_and_risk(pred_score)
            suggestions = get_recommendations(features)
            
            prediction = round(pred_score, 2)
            
        except Exception as e:
            # Output handling for input errors
            return render_template('index.html', error=f"An error occurred predicting your score: {str(e)}")

    # Display variables efficiently across frontend
    return render_template('index.html', prediction=prediction, grade=grade, risk=risk, suggestions=suggestions)

if __name__ == "__main__":
    app.run(debug=True)
