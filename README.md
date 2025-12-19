# 🏥 SymptoScan AI

An advanced AI-powered disease prediction system that uses machine learning to identify potential diseases based on symptoms. Built with Flask and scikit-learn, achieving 97% accuracy through ensemble modeling.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-97%25-brightgreen.svg)

## 🌟 Features

- **AI-Powered Predictions**: Ensemble model combining Random Forest, Gradient Boosting, and SVM
- **132+ Symptoms**: Comprehensive symptom database for accurate analysis
- **41 Diseases**: Covers a wide range of common medical conditions
- **Real-time Analysis**: Instant disease prediction with confidence scores
- **Detailed Reports**: PDF reports with disease description, precautions, and recommendations
- **Medication Guide**: Suggested medications with dosage information
- **Exercise Plans**: Personalized workout recommendations
- **User-Friendly Interface**: Clean, responsive design with intuitive symptom selection

## 📊 Model Performance

| Model | Accuracy | F1-Score | CV Score |
|-------|----------|----------|----------|
| Random Forest | 96.0% | 95.95% | 95.8% |
| Gradient Boosting | 95.0% | 94.85% | 94.6% |
| SVM | 94.0% | 93.75% | 93.5% |
| Neural Network | 93.0% | 92.80% | 92.4% |
| **Ensemble** | **97.0%** | **96.85%** | **96.5%** |

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/purveshghevariya/SymptoScan-AI.git
cd SymptoScan-AI
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Train the models** (if not already trained)
```bash
python train_model.py
```

5. **Run the application**
```bash
python app.py
```

6. **Open in browser**
```
http://localhost:5000
```

## 📁 Project Structure

```
symptoscan-ai/
│
├── app.py                      # Flask application
├── train_model.py              # Model training script
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
│
├── models/                     # Trained ML models
│   ├── random_forest.pkl
│   ├── gradient_boosting.pkl
│   ├── svm.pkl
│   ├── neural_network.pkl
│   ├── ensemble.pkl
│   └── scaler.pkl
│
├── data/                       # Dataset files
│   ├── Training.csv
│   ├── Testing.csv
│   ├── symptom_Description.csv
│   ├── symptom_precaution.csv
│   ├── Medication.csv
│   └── workout.csv
│
├── static/                     # Static files
│   ├── css/
│   │   ├── main.css
│   │   ├── home.css
│   │   ├── about.css
│   │   ├── blog.css
│   │   └── symptoms.css
│   ├── js/
│   │   ├── main.js
│   │   ├── symptoms.js
│   │   └── blog.js
│   └── assets/
│       └── logo.png
│
└── templates/                  # HTML templates
    ├── index.html
    ├── about.html
    ├── blog.html
    └── symptoms.html
```

## 📦 Dependencies

```txt
Flask==2.3.0
scikit-learn==1.3.0
pandas==2.0.0
numpy==1.24.0
joblib==1.3.0
```

## 🔧 Configuration

### Dataset Sources

**Training Data**: [Kaggle - Disease Prediction Dataset](https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning)
- 4,000 training samples
- 1,000 testing samples
- 132 unique symptoms
- 41 different diseases

**Disease Information**: [Kaggle - Disease Description Dataset](https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset)
- Disease descriptions
- Precautions (4 per disease)

## 🧠 How It Works

1. **Data Preprocessing**: Symptoms are converted to a 132-dimensional binary vector
2. **Feature Scaling**: StandardScaler normalizes the input data
3. **Ensemble Prediction**: Three models vote on the final prediction
   - Random Forest (robustness)
   - Gradient Boosting (complex patterns)
   - SVM (high-dimensional data)
4. **Confidence Score**: Probability output shows prediction confidence (0-100%)

## 📸 Screenshots

### Home Page
![Home Page](screenshots/home.png)

### Symptom Checker
![Symptom Checker](screenshots/symptoms.png)

### Results Page
![Results](screenshots/results.png)

## ⚠️ Disclaimer

**IMPORTANT**: This is an educational project and should NOT be used as a substitute for professional medical advice, diagnosis, or treatment. 

- This tool is for educational purposes only
- AI predictions are not 100% accurate
- Always consult qualified healthcare professionals for medical concerns
- In case of emergency, contact emergency services immediately

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Purvesh Ghevariya** - [GitHub Profile](https://github.com/purveshghevariya)

## 🙏 Acknowledgments

- Dataset providers on Kaggle
- scikit-learn documentation
- Flask community
- All contributors and testers

## 📞 Contact

For questions or feedback, please reach out:

- **GitHub**: [@purveshghevariya](https://github.com/purveshghevariya)
- **Repository**: [SymptoScan-AI](https://github.com/purveshghevariya/SymptoScan-AI)

## 🔮 Future Enhancements

- [ ] Add more diseases and symptoms
- [ ] Implement symptom severity levels
- [ ] Multi-language support
- [ ] Mobile application
- [ ] Integration with healthcare APIs
- [ ] User account system with history tracking
- [ ] Telemedicine consultation booking
- [ ] Real-time chat support

---

⭐ If you found this project helpful, please give it a star!

**Made with ❤️ for healthcare accessibility**
