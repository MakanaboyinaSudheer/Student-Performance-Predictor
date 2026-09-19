from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np
import os

app = Flask(__name__)

# Model loading logic
MODEL_PATH = 'model.pkl'

def load_trained_model():
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, 'rb') as file:
                return pickle.load(file)
        except Exception as e:
            print(f"ERROR: Failed to load model: {e}")
            return None
    else:
        print("WARNING: model.pkl not found! Please run train_model.py first.")
        return None

model = load_trained_model()

def get_recommendations(features):
    """
    Generate personalized academic suggestions based on input metrics.
    """
    suggestions = []
    
    # 1. Study Hours
    study_hours = float(features.get('study_hours', 0))
    if study_hours < 15:
        suggestions.append("📚 Study Commitment: Increase dedicated weekly study time to 15-20 hours for optimal retention.")
    elif study_hours > 35:
        suggestions.append("⚖️ Study-Life Balance: High study hours detected. Ensure break intervals to avoid academic burnout.")

    # 2. Attendance
    attendance = float(features.get('attendance', 0))
    if attendance < 80:
        suggestions.append("⚠️ Attendance Warning: Current attendance rate is below 80%. Prioritize attending lectures to stay aligned with key course concepts.")
    elif attendance < 90:
        suggestions.append("📈 Attendance Tip: Aim for above 90% attendance to maximize interactive learning and instructor feedback.")

    # 3. Sleep Habits
    sleep_hours = float(features.get('sleep_hours', 0))
    if sleep_hours < 7:
        suggestions.append("🛌 Sleep Hygiene: Ensure 7 to 8 hours of sleep per night. Adequate rest significantly boosts memory consolidation and focus.")

    # 4. Motivation Level
    motivation = str(features.get('motivation_level', '')).lower()
    if motivation in ['low', 'medium']:
        suggestions.append("🚀 Goal Setting: Break down large assignments into daily manageable goals to build consistent motivation and momentum.")

    # 5. Resource Access
    resources = str(features.get('learning_resources', '')).lower()
    if resources == 'low':
        suggestions.append("📖 Learning Assets: Leverage digital open-access libraries, campus study labs, and peer resource-sharing groups.")

    # 6. Teacher Quality & Tutoring
    teacher = str(features.get('teacher_quality', '')).lower()
    if teacher == 'low':
        suggestions.append("💡 Academic Support: Seek supplemental online tutorials or form peer study groups to reinforce challenging topics.")

    # 7. Peer Influence
    peer = str(features.get('peer_influence', '')).lower()
    if peer == 'negative':
        suggestions.append("👥 Positive Environment: Engage with collaborative study partners who encourage academic discipline and accountability.")

    if not suggestions:
        suggestions.append("🌟 Stellar Academic Profile! Your current study habits, sleep, and environment provide an outstanding foundation for top performance.")
        
    return suggestions

def get_grade_and_risk(score):
    """
    Map score to grade tier and risk category.
    """
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
    error = None
    form_data = None

    # Reload model if missing initially
    global model
    if model is None:
        model = load_trained_model()

    if request.method == 'POST':
        if model is None:
            error = "Model (model.pkl) is not available. Please execute 'python train_model.py' to build and save the model pipeline."
            return render_template('index.html', error=error, form_data=request.form)

        try:
            # Parse inputs from form
            form_data = {
                'study_hours': float(request.form.get('study_hours', 0)),
                'attendance': float(request.form.get('attendance', 0)),
                'motivation_level': request.form.get('motivation_level', 'Medium'),
                'learning_resources': request.form.get('learning_resources', 'Medium'),
                'parental_participation': request.form.get('parental_participation', 'Medium'),
                'teacher_quality': request.form.get('teacher_quality', 'Medium'),
                'peer_influence': request.form.get('peer_influence', 'Neutral'),
                'family_background': request.form.get('family_background', 'Medium'),
                'internet_availability': request.form.get('internet_availability', 'Yes'),
                'sleep_hours': float(request.form.get('sleep_hours', 7)),
                'school_type': request.form.get('school_type', 'Public')
            }

            # Convert to DataFrame matching model input format
            df_features = pd.DataFrame([form_data])
            
            # Predict score using fitted Pipeline
            pred_score = model.predict(df_features)[0]
            
            # Clamp exam score within realistic exam bounds [0, 100]
            pred_score = float(np.clip(pred_score, 0, 100))
            
            # Compute Grade, Risk Tier and Recommendations
            grade, risk = get_grade_and_risk(pred_score)
            suggestions = get_recommendations(form_data)
            prediction = round(pred_score, 1)

        except Exception as e:
            error = f"Error processing prediction request: {str(e)}"

    return render_template('index.html', 
                           prediction=prediction, 
                           grade=grade, 
                           risk=risk, 
                           suggestions=suggestions, 
                           error=error, 
                           form_data=form_data)

if __name__ == "__main__":
    print("Starting Student Performance Predictor App on http://127.0.0.1:5000 ...")
    app.run(debug=True)
