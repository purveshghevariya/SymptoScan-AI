from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Global variables
model = None
label_encoder = None
scaler = None
symptom_columns = None
disease_info = None
metadata = None
medications_db = None
workout_plans_db = None
diet_plans_db = None

def load_models():
    """Load all trained models and medical databases"""
    global model, label_encoder, scaler, symptom_columns, disease_info, metadata
    global medications_db, workout_plans_db, diet_plans_db
    
    try:
        model = joblib.load('models/disease_predictor.pkl')
        label_encoder = joblib.load('models/label_encoder.pkl')
        scaler = joblib.load('models/scaler.pkl')
        symptom_columns = joblib.load('models/symptom_columns.pkl')
        disease_info = joblib.load('models/disease_info.pkl')
        metadata = joblib.load('models/metadata.pkl')
        
        # Load medical databases
        medications_db = joblib.load('models/medications.pkl')
        workout_plans_db = joblib.load('models/workout_plans.pkl')
        diet_plans_db = joblib.load('models/diet_plans.pkl')
        
        print("✅ All models and medical databases loaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return False

def get_medication(disease):
    """Get medication from trained database"""
    # Try exact match first
    if disease in medications_db:
        return medications_db[disease]
    
    # Try partial match
    for key in medications_db:
        if key.lower() in disease.lower() or disease.lower() in key.lower():
            return medications_db[key]
    
    # Return default
    return medications_db.get('default', {
        'medications': ['Consult healthcare provider'],
        'dosage': 'As prescribed',
        'duration': 'As prescribed'
    })

def get_workout(disease):
    """Get workout plan from trained database"""
    if disease in workout_plans_db:
        return workout_plans_db[disease]
    
    for key in workout_plans_db:
        if key.lower() in disease.lower() or disease.lower() in key.lower():
            return workout_plans_db[key]
    
    return workout_plans_db.get('default', {
        'exercises': [
            {'name': 'Walking', 'duration': '30 min', 'frequency': '5x/week'},
            {'name': 'Stretching', 'duration': '15 min', 'frequency': 'Daily'}
        ],
        'intensity': 'Moderate'
    })

def get_diet(disease):
    """Get diet plan from trained database"""
    if disease in diet_plans_db:
        return diet_plans_db[disease]
    
    for key in diet_plans_db:
        if key.lower() in disease.lower() or disease.lower() in key.lower():
            return diet_plans_db[key]
    
    return diet_plans_db.get('default', {
        'foods_to_eat': ['Fruits', 'Vegetables', 'Whole grains', 'Lean proteins'],
        'foods_to_avoid': ['Processed foods', 'Excessive sugar', 'Trans fats']
    })

def generate_pdf(prediction_data, user_data):
    """Generate PDF report with selected symptoms"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    story = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = styles['Title']
    heading_style = styles['Heading2']
    normal_style = styles['Normal']
    
    # Title
    title = Paragraph("<b>SymptoScan AI - Health Report</b>", title_style)
    story.append(title)
    story.append(Spacer(1, 0.3*inch))
    
    # Patient Information Section
    story.append(Paragraph("<b>Patient Information</b>", heading_style))
    story.append(Spacer(1, 0.1*inch))
    
    patient_data = [
        ['Name:', user_data.get('name', 'N/A')],
        ['Age:', str(user_data.get('age', 'N/A'))],
        ['Gender:', user_data.get('gender', 'N/A')],
        ['Contact:', user_data.get('contact', 'N/A')],
        ['Date:', datetime.now().strftime('%B %d, %Y - %I:%M %p')]
    ]
    
    patient_table = Table(patient_data, colWidths=[2*inch, 4*inch])
    patient_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('BACKGROUND', (1, 0), (1, -1), colors.whitesmoke),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(patient_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Selected Symptoms Section
    story.append(Paragraph("<b>Selected Symptoms</b>", heading_style))
    story.append(Spacer(1, 0.1*inch))
    
    symptoms = prediction_data.get('symptoms', [])
    if symptoms:
        # Format symptoms nicely
        symptoms_text = ", ".join([symptom.replace('_', ' ').title() for symptom in symptoms])
        story.append(Paragraph(f"<i>Total Symptoms: {len(symptoms)}</i>", normal_style))
        story.append(Spacer(1, 0.05*inch))
        story.append(Paragraph(symptoms_text, normal_style))
    else:
        story.append(Paragraph("No symptoms recorded", normal_style))
    
    story.append(Spacer(1, 0.3*inch))
    
    # Diagnosis Results Section
    story.append(Paragraph("<b>Diagnosis Results</b>", heading_style))
    story.append(Spacer(1, 0.1*inch))
    
    results_data = [
        ['Predicted Disease:', prediction_data.get('disease', 'N/A')],
        ['Confidence Score:', f"{prediction_data.get('confidence', 0)}%"]
    ]
    
    results_table = Table(results_data, colWidths=[2*inch, 4*inch])
    results_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.lightblue),
        ('BACKGROUND', (1, 0), (1, -1), colors.lightcyan),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(results_table)
    story.append(Spacer(1, 0.2*inch))
    
    # Description
    story.append(Paragraph("<b>Disease Description</b>", heading_style))
    story.append(Spacer(1, 0.1*inch))
    description = prediction_data.get('description', 'No description available')
    story.append(Paragraph(description, normal_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Precautions
    story.append(Paragraph("<b>Safety Precautions</b>", heading_style))
    story.append(Spacer(1, 0.1*inch))
    precautions = prediction_data.get('precautions', [])
    for i, precaution in enumerate(precautions, 1):
        story.append(Paragraph(f"{i}. {precaution}", normal_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Medications
    story.append(Paragraph("<b>Medication Recommendations</b>", heading_style))
    story.append(Spacer(1, 0.1*inch))
    medications = prediction_data.get('medications', {})
    
    med_data = [
        ['Medications:', ', '.join(medications.get('medications', ['N/A']))],
        ['Dosage:', medications.get('dosage', 'As prescribed')],
        ['Duration:', medications.get('duration', 'As prescribed')]
    ]
    
    med_table = Table(med_data, colWidths=[2*inch, 4*inch])
    med_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgreen),
        ('BACKGROUND', (1, 0), (1, -1), colors.honeydew),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(med_table)
    story.append(Spacer(1, 0.2*inch))
    
    # Workout Plan
    story.append(Paragraph("<b>Exercise Recommendations</b>", heading_style))
    story.append(Spacer(1, 0.1*inch))
    workout = prediction_data.get('workout_plan', {})
    story.append(Paragraph(f"<b>Intensity Level:</b> {workout.get('intensity', 'Moderate')}", normal_style))
    story.append(Spacer(1, 0.05*inch))
    
    exercises = workout.get('exercises', [])
    for exercise in exercises:
        ex_text = f"• {exercise.get('name', 'N/A')} - {exercise.get('duration', 'N/A')} ({exercise.get('frequency', 'N/A')})"
        story.append(Paragraph(ex_text, normal_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Diet Plan
    story.append(Paragraph("<b>Dietary Recommendations</b>", heading_style))
    story.append(Spacer(1, 0.1*inch))
    diet = prediction_data.get('diet_plan', {})
    
    story.append(Paragraph("<b>Foods to Eat:</b>", normal_style))
    foods_to_eat = ', '.join(diet.get('foods_to_eat', ['N/A']))
    story.append(Paragraph(foods_to_eat, normal_style))
    story.append(Spacer(1, 0.1*inch))
    
    story.append(Paragraph("<b>Foods to Avoid:</b>", normal_style))
    foods_to_avoid = ', '.join(diet.get('foods_to_avoid', ['N/A']))
    story.append(Paragraph(foods_to_avoid, normal_style))
    story.append(Spacer(1, 0.3*inch))
    
    # Disclaimer
    disclaimer_text = """
    <b>IMPORTANT DISCLAIMER:</b><br/>
    This report is generated by an AI system for educational purposes only. 
    It is NOT a substitute for professional medical advice, diagnosis, or treatment. 
    Always seek the advice of qualified healthcare professionals with any questions 
    regarding medical conditions. Never disregard professional medical advice or 
    delay seeking it because of information in this report.
    """
    story.append(Paragraph(disclaimer_text, normal_style))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer

# Routes
@app.route('/')
def home():
    return render_template('home.html', metadata=metadata)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/blog')
def blog():
    return render_template('blog.html')

@app.route('/symptoms')
def symptoms():
    if model is None:
        return render_template('error.html', error="Models not loaded. Run: python train_model.py")
    return render_template('symptoms.html', symptoms=symptom_columns, metadata=metadata)

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        symptoms = data.get('symptoms', [])
        
        if not symptoms:
            return jsonify({'error': 'No symptoms provided'}), 400
        
        # Create input vector
        input_vector = np.zeros(len(symptom_columns))
        for symptom in symptoms:
            if symptom in symptom_columns:
                idx = symptom_columns.index(symptom)
                input_vector[idx] = 1
        
        # Add symptom count
        symptom_count = np.sum(input_vector)
        input_vector = np.append(input_vector, symptom_count)
        
        # Scale and predict
        input_scaled = scaler.transform([input_vector])
        prediction = model.predict(input_scaled)[0]
        disease = label_encoder.inverse_transform([prediction])[0]
        
        # Get probability
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(input_scaled)[0]
            confidence = float(max(proba) * 100)
        else:
            confidence = 95.0
        
        # Get information from databases
        description = disease_info['descriptions'].get(disease, 'Consult a healthcare provider.')
        precautions = disease_info['precautions'].get(disease, ['Consult a doctor'])
        medications = get_medication(disease)
        workout = get_workout(disease)
        diet = get_diet(disease)
        
        return jsonify({
            'disease': disease,
            'confidence': round(confidence, 2),
            'description': description,
            'precautions': precautions,
            'medications': medications,
            'workout_plan': workout,
            'diet_plan': diet,
            'symptoms': symptoms
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate-report', methods=['POST'])
def generate_report():
    try:
        data = request.get_json()
        prediction_data = data.get('prediction_data')
        user_data = data.get('user_data')
        
        pdf_buffer = generate_pdf(prediction_data, user_data)
        filename = f"Report_{user_data.get('name')}_{datetime.now().strftime('%Y%m%d')}.pdf"
        
        return send_file(pdf_buffer, mimetype='application/pdf', 
                        as_attachment=True, download_name=filename)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("=" * 60)
    print("   🏥 SymptoScan AI - Health Prediction System")
    print("=" * 60)
    
    if load_models():
        print("\n🚀 Server starting at http://localhost:5001")
        print("=" * 60)
        app.run(debug=True, host='0.0.0.0', port=5001)
    else:
        print("\n❌ Please train models first: python train_model.py")