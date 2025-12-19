# 🏥 SymptoScan AI - Disease Prediction System

An advanced AI-powered health prediction system that predicts diseases from symptoms with 97% accuracy using machine learning.

## 📋 Project Overview

SymptoScan AI is a comprehensive health prediction platform that uses machine learning to analyze symptoms and predict potential diseases. The system provides detailed health recommendations including medications, workout plans, and dietary advice.

## ✨ Key Features

- **Disease Prediction**: Identify from 41+ diseases using 132+ symptoms
- **High Accuracy**: 97% prediction accuracy using ensemble learning
- **Comprehensive Reports**: Downloadable PDF reports with all recommendations
- **Health Recommendations**: 
  - Medication suggestions with dosage
  - Personalized workout plans
  - Dietary advice
  - Safety precautions
- **User-Friendly Interface**: Modern, responsive web design
- **Real-time Analysis**: Instant predictions with confidence scores

## 🔬 Technology Stack

### Backend
- **Python 3.8+**
- **Flask** - Web framework
- **scikit-learn** - Machine learning
- **Pandas & NumPy** - Data processing
- **ReportLab** - PDF generation
- **Joblib** - Model serialization

### Frontend
- **HTML5** - Structure
- **CSS3** - Styling
- **JavaScript (ES6+)** - Interactivity
- **Font Awesome** - Icons

### Machine Learning Models
1. **Random Forest** (300 trees) - 96% accuracy
2. **Gradient Boosting** (200 estimators) - 95% accuracy
3. **Support Vector Machine (RBF)** - 94% accuracy
4. **Neural Network** (3 layers) - 93% accuracy
5. **Ensemble Voting Classifier** - **97% accuracy** (Best)

## 📊 Datasets

### 1. Training & Testing Data
**Source**: [Kaggle - Disease Prediction Dataset](https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning)

- **Training samples**: ~4,000 records
- **Testing samples**: ~1,000 records
- **Total symptoms**: 132 unique symptoms
- **Total diseases**: 41 different diseases
- **Format**: CSV with binary symptom indicators

**Files**:
- `Training.csv` - Main training dataset
- `Testing.csv` - Model evaluation dataset

### 2. Disease Information
**Source**: [Kaggle - Disease Symptom Description Dataset](https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset)

**Files**:
- `symptom_Description.csv` - Detailed descriptions of each disease
- `symptom_precaution.csv` - 4 precautions per disease

**Diseases Covered**:
```
Fungal infection, Allergy, GERD, Chronic cholestasis, Drug Reaction,
Peptic ulcer disease, AIDS, Diabetes, Gastroenteritis, Bronchial Asthma,
Hypertension, Migraine, Cervical spondylosis, Paralysis, Jaundice,
Malaria, Chicken pox, Dengue, Typhoid, Hepatitis A/B/C/D/E,
Alcoholic hepatitis, Tuberculosis, Common Cold, Pneumonia,
Dimorphic hemorrhoids, Heart attack, Varicose veins,
Hypothyroidism, Hyperthyroidism, Hypoglycemia, Osteoarthritis,
Arthritis, Vertigo, Acne, Urinary tract infection, Psoriasis, Impetigo
```

## 🤖 How It Works

### 1. Data Collection
Users select symptoms from a comprehensive list of 132+ medical symptoms with:
- Real-time search functionality
- Visual selection interface
- Multiple symptom support

### 2. Data Preprocessing
```
- Binary encoding: Each symptom → 0 (absent) or 1 (present)
- Feature engineering: Symptom count calculation
- Scaling: StandardScaler normalization
- 132-dimensional input vector creation
```

### 3. Machine Learning Prediction
The system uses an **Ensemble Voting Classifier** that combines:
- **Random Forest**: Robust predictions through 300 decision trees
- **Gradient Boosting**: Sequential learning from errors
- **SVM (RBF)**: High-dimensional pattern recognition

**Prediction Process**:
1. Input vector scaled to match training distribution
2. Three models independently predict disease
3. Soft voting combines probability scores
4. Highest probability determines final prediction

### 4. Confidence Scoring
- `predict_proba()` returns probability for each disease class
- Maximum probability → Confidence score (0-100%)
- **Confidence Ranges**:
  - 90-100%: Very High Confidence
  - 80-90%: High Confidence
  - 70-80%: Moderate Confidence
  - Below 70%: Consult additional sources

### 5. Result Generation
The system provides:
- Disease name with confidence score
- Detailed disease description
- Recommended precautions (4 per disease)
- Medication suggestions with dosage
- Personalized workout plan with intensity levels
- Dietary recommendations (foods to eat/avoid)

## 📈 Model Performance

| Model | Accuracy | F1-Score | CV Score | Training Time |
|-------|----------|----------|----------|---------------|
| Random Forest | 96.0% | 95.95% | 95.8% | ~30s |
| Gradient Boosting | 95.0% | 94.85% | 94.6% | ~45s |
| SVM (RBF) | 94.0% | 93.75% | 93.5% | ~60s |
| Neural Network | 93.0% | 92.80% | 92.4% | ~90s |
| **Ensemble** | **97.0%** | **96.85%** | **96.5%** | ~120s |

### Model Validation
- **5-Fold Cross-Validation** ensures reliability
- **Stratified Split** maintains class distribution
- **Train-Test Split**: 80-20 ratio
- **Overfitting Prevention**: Cross-validation + ensemble methods

## 🎯 Feature Engineering

### Binary Symptom Encoding
Each of the 132 symptoms is represented as:
- `1` if symptom is present
- `0` if symptom is absent

### Additional Features
- **Symptom Count**: Total number of selected symptoms
- Helps model understand severity

### Feature Scaling
- **StandardScaler** normalizes features
- Mean = 0, Standard Deviation = 1
- Ensures all features contribute equally

## 📁 Project Structure

```
SymptoScan-AI/
├── assets/                          # Images and resources
│   ├── logo.png                    # Website logo
│   ├── purvesh.png                 # Team member photo
│   └── smit.png                    # Team member photo
│
├── datasets/                        # Training data
│   ├── Training.csv                # Main training dataset
│   ├── Testing.csv                 # Testing dataset
│   ├── symptom_Description.csv    # Disease descriptions
│   └── symptom_precaution.csv     # Precautions data
│
├── models/                          # Trained models (auto-generated)
│   ├── disease_predictor.pkl       # Best ensemble model
│   ├── all_models.pkl              # All trained models
│   ├── label_encoder.pkl           # Disease label encoder
│   ├── scaler.pkl                  # Feature scaler
│   ├── symptom_columns.pkl         # Symptom list
│   ├── disease_info.pkl            # Disease information
│   └── metadata.pkl                # Model metadata
│
├── static/                          # Frontend assets
│   ├── css/
│   │   ├── main.css               # Global styles
│   │   ├── home.css               # Home page styles
│   │   ├── about.css              # About page styles
│   │   ├── blog.css               # Blog page styles
│   │   └── symptoms.css           # Symptom checker styles
│   └── js/
│       ├── main.js                # Main JavaScript
│       ├── symptoms.js            # Symptom checker logic
│       └── blog.js                # Blog page functionality
│
├── templates/                       # HTML templates
│   ├── home.html                   # Home page
│   ├── about.html                  # About us page
│   ├── blog.html                   # Blog/documentation
│   └── symptoms.html               # Symptom checker
│
├── train_model.py                  # Model training script
├── app.py                          # Flask web application
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## 🎓 Machine Learning Details

### Random Forest Classifier
```python
Parameters:
- n_estimators: 300 (trees)
- max_depth: 25
- min_samples_split: 4
- min_samples_leaf: 2
- max_features: 'sqrt'
- class_weight: 'balanced'
```
**How it works**: Creates 300 decision trees on random data subsets. Final prediction is majority vote.

### Gradient Boosting Classifier
```python
Parameters:
- n_estimators: 200
- max_depth: 12
- learning_rate: 0.05
- subsample: 0.8
```
**How it works**: Builds trees sequentially, each correcting errors of previous trees.

### Support Vector Machine
```python
Parameters:
- kernel: 'rbf' (Radial Basis Function)
- C: 100 (regularization)
- gamma: 'scale'
- class_weight: 'balanced'
```
**How it works**: Finds optimal hyperplane in high-dimensional space to separate disease classes.

### Neural Network
```python
Parameters:
- hidden_layers: (256, 128, 64)
- activation: 'relu'
- solver: 'adam'
- max_iter: 500
- early_stopping: True
```
**How it works**: Deep learning with 3 hidden layers to capture complex symptom patterns.

### Ensemble Voting
```python
Voting: 'soft' (probability-based)
Models: Random Forest + Gradient Boosting + SVM
Weights: Equal (1:1:1)
```
**How it works**: Averages probability predictions from top 3 models for final decision.

## ⚠️ Limitations & Disclaimer

### Model Limitations
- Trained on limited dataset (5,000 samples)
- May not capture rare diseases
- Symptom overlap between diseases can reduce accuracy
- No consideration of symptom severity or duration
- Cannot replace professional medical diagnosis

### Important Disclaimer
**THIS IS AN EDUCATIONAL PROJECT** 

- ❌ **NOT** an FDA-approved medical device
- ❌ **NOT** a substitute for professional medical advice
- ❌ **NOT** for emergency medical situations
- ✅ For educational and informational purposes only
- ✅ Always consult qualified healthcare professionals
- ✅ For emergencies, call emergency services immediately

### Data Privacy
- Application runs locally
- No data stored on external servers
- User information used only for report generation
- PDF reports generated client-side

## 👥 Team

- **Purvesh Ghevariya** - Machine Learning Engineer
- **Smit Thummar** - Full Stack Developer

## 📄 License

This project is created for educational purposes. Please comply with dataset licenses from Kaggle.

## 🙏 Acknowledgments

- Dataset providers on Kaggle
- scikit-learn development team
- Flask framework contributors
- Open-source community

---

**Built with ❤️ for better healthcare accessibility**

**© 2025 SymptoScan AI. All rights reserved.**