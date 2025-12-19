import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class EnhancedDiseasePredictor:
    def __init__(self):
        self.models = {}
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.symptom_columns = []
        self.feature_importance = {}
        
    def load_data(self):
        """Load training and testing datasets"""
        print("Loading datasets...")
        
        train_df = pd.read_csv('datasets/Training.csv')
        test_df = pd.read_csv('datasets/Testing.csv')
        
        df = pd.concat([train_df, test_df], ignore_index=True)
        
        print(f"✓ Total samples: {len(df)}")
        print(f"✓ Number of diseases: {df['prognosis'].nunique()}")
        print(f"✓ Number of symptoms: {len(df.columns) - 1}")
        
        return df
    
    def preprocess_data(self, df):
        """Advanced preprocessing with feature engineering"""
        print("\n" + "="*60)
        print("PREPROCESSING DATA")
        print("="*60)
        
        X = df.drop('prognosis', axis=1)
        y = df['prognosis']
        
        print("✓ Checking for missing values...")
        missing_count = X.isnull().sum().sum()
        if missing_count > 0:
            print(f"  Found {missing_count} missing values. Filling with 0...")
            X = X.fillna(0)
        
        X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
        self.symptom_columns = list(X.columns)
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Feature engineering
        X['symptom_count'] = X.sum(axis=1)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"✓ Training samples: {len(X_train)}")
        print(f"✓ Testing samples: {len(X_test)}")
        print(f"✓ Features: {X_train.shape[1]}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_models(self, X_train, X_test, y_train, y_test):
        """Train multiple optimized ML models"""
        print("\n" + "="*60)
        print("TRAINING ADVANCED MODELS")
        print("="*60)
        
        models_to_train = {
            'Random Forest': RandomForestClassifier(
                n_estimators=300,
                max_depth=25,
                min_samples_split=4,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=200,
                max_depth=12,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42
            ),
            'SVM (RBF)': SVC(
                kernel='rbf',
                C=100,
                gamma='scale',
                probability=True,
                random_state=42,
                class_weight='balanced'
            ),
            'Neural Network': MLPClassifier(
                hidden_layer_sizes=(256, 128, 64),
                activation='relu',
                solver='adam',
                max_iter=500,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1
            ),
            'Naive Bayes': GaussianNB()
        }
        
        best_accuracy = 0
        best_model_name = None
        best_f1 = 0
        
        for name, model in models_to_train.items():
            print(f"\n{'='*60}")
            print(f"🔹 Training: {name}")
            print(f"{'='*60}")
            
            try:
                model.fit(X_train, y_train)
                
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)
                
                train_accuracy = accuracy_score(y_train, y_pred_train)
                test_accuracy = accuracy_score(y_test, y_pred_test)
                f1 = f1_score(y_test, y_pred_test, average='weighted')
                
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
                
                print(f"  Train Accuracy: {train_accuracy:.4f}")
                print(f"  Test Accuracy:  {test_accuracy:.4f}")
                print(f"  F1 Score:       {f1:.4f}")
                print(f"  CV Score:       {cv_mean:.4f} (+/- {cv_std:.4f})")
                
                self.models[name] = {
                    'model': model,
                    'train_accuracy': train_accuracy,
                    'test_accuracy': test_accuracy,
                    'f1_score': f1,
                    'cv_score': cv_mean,
                    'cv_std': cv_std
                }
                
                if test_accuracy > best_accuracy or (test_accuracy == best_accuracy and f1 > best_f1):
                    best_accuracy = test_accuracy
                    best_f1 = f1
                    best_model_name = name
                
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[name] = model.feature_importances_
                    
            except Exception as e:
                print(f"  ⚠️ Error training {name}: {str(e)}")
                continue
        
        # Ensemble model
        print(f"\n{'='*60}")
        print("🔹 Creating Ensemble Model")
        print(f"{'='*60}")
        
        ensemble = VotingClassifier(
            estimators=[
                ('rf', self.models['Random Forest']['model']),
                ('gb', self.models['Gradient Boosting']['model']),
                ('svm', self.models['SVM (RBF)']['model'])
            ],
            voting='soft'
        )
        
        ensemble.fit(X_train, y_train)
        y_pred_ensemble = ensemble.predict(X_test)
        ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)
        ensemble_f1 = f1_score(y_test, y_pred_ensemble, average='weighted')
        
        print(f"  Test Accuracy: {ensemble_accuracy:.4f}")
        print(f"  F1 Score:      {ensemble_f1:.4f}")
        
        self.models['Ensemble'] = {
            'model': ensemble,
            'test_accuracy': ensemble_accuracy,
            'f1_score': ensemble_f1
        }
        
        if ensemble_accuracy > best_accuracy:
            best_accuracy = ensemble_accuracy
            best_model_name = 'Ensemble'
        
        print("\n" + "="*60)
        print("MODEL TRAINING SUMMARY")
        print("="*60)
        print(f"🏆 Best Model: {best_model_name}")
        print(f"🎯 Accuracy:   {best_accuracy:.4f}")
        print(f"📊 F1 Score:   {best_f1:.4f}")
        print("="*60)
        
        return best_model_name
    
    def load_medical_datasets(self):
        """Load medication, workout, and diet datasets"""
        print("\n" + "="*60)
        print("LOADING MEDICAL DATASETS")
        print("="*60)
        
        medications = {}
        workout_plans = {}
        diet_plans = {}
        
        try:
            # Load medications dataset
            med_df = pd.read_csv('datasets/medications.csv')
            for _, row in med_df.iterrows():
                medications[row['Disease']] = {
                    'medications': row['Medications'].split(','),
                    'dosage': row['Dosage'],
                    'duration': row['Duration']
                }
            print(f"✓ Loaded medications for {len(medications)} diseases")
        except Exception as e:
            print(f"⚠️ Could not load medications dataset: {e}")
            medications = self._get_default_medications()
        
        try:
            # Load workout plans dataset
            workout_df = pd.read_csv('datasets/workout_plans.csv')
            for disease in workout_df['Disease'].unique():
                disease_data = workout_df[workout_df['Disease'] == disease]
                workout_plans[disease] = {
                    'exercises': [
                        {
                            'name': row['Exercise'],
                            'duration': row['Duration'],
                            'frequency': row['Frequency']
                        }
                        for _, row in disease_data.iterrows()
                    ],
                    'intensity': disease_data.iloc[0]['Intensity']
                }
            print(f"✓ Loaded workout plans for {len(workout_plans)} diseases")
        except Exception as e:
            print(f"⚠️ Could not load workout plans dataset: {e}")
            workout_plans = self._get_default_workouts()
        
        try:
            # Load diet plans dataset
            diet_df = pd.read_csv('datasets/diet_plans.csv')
            for _, row in diet_df.iterrows():
                diet_plans[row['Disease']] = {
                    'foods_to_eat': row['Foods_to_Eat'].split(','),
                    'foods_to_avoid': row['Foods_to_Avoid'].split(',')
                }
            print(f"✓ Loaded diet plans for {len(diet_plans)} diseases")
        except Exception as e:
            print(f"⚠️ Could not load diet plans dataset: {e}")
            diet_plans = self._get_default_diets()
        
        return medications, workout_plans, diet_plans
    
    def _get_default_medications(self):
        """Default medication database"""
        return {
            'Fungal infection': {
                'medications': ['Clotrimazole', 'Fluconazole', 'Terbinafine'],
                'dosage': '200mg daily',
                'duration': '2-4 weeks'
            },
            'Allergy': {
                'medications': ['Cetirizine', 'Loratadine', 'Fexofenadine'],
                'dosage': '10mg once daily',
                'duration': 'As needed'
            },
            'default': {
                'medications': ['Consult healthcare provider'],
                'dosage': 'As prescribed',
                'duration': 'As prescribed'
            }
        }
    
    def _get_default_workouts(self):
        """Default workout plans"""
        return {
            'Diabetes': {
                'exercises': [
                    {'name': 'Walking', 'duration': '30 min', 'frequency': 'Daily'},
                    {'name': 'Cycling', 'duration': '30 min', 'frequency': '3x/week'}
                ],
                'intensity': 'Moderate'
            },
            'default': {
                'exercises': [
                    {'name': 'Walking', 'duration': '30 min', 'frequency': '5x/week'}
                ],
                'intensity': 'Moderate'
            }
        }
    
    def _get_default_diets(self):
        """Default diet plans"""
        return {
            'Diabetes': {
                'foods_to_eat': ['Whole grains', 'Leafy greens', 'Berries', 'Nuts'],
                'foods_to_avoid': ['Sugar', 'White bread', 'Fried foods']
            },
            'default': {
                'foods_to_eat': ['Fruits', 'Vegetables', 'Whole grains', 'Lean proteins'],
                'foods_to_avoid': ['Processed foods', 'Excessive sugar', 'Trans fats']
            }
        }
    
    def save_models(self, best_model_name, medications, workout_plans, diet_plans):
        """Save trained models and medical data"""
        print("\n" + "="*60)
        print("SAVING MODELS AND MEDICAL DATA")
        print("="*60)
        
        os.makedirs('models', exist_ok=True)
        
        # Save best model
        best_model = self.models[best_model_name]['model']
        joblib.dump(best_model, 'models/disease_predictor.pkl')
        print(f"✓ Best model saved: {best_model_name}")
        
        # Save all models
        joblib.dump(self.models, 'models/all_models.pkl')
        print("✓ All models saved")
        
        # Save encoders and scalers
        joblib.dump(self.label_encoder, 'models/label_encoder.pkl')
        joblib.dump(self.scaler, 'models/scaler.pkl')
        joblib.dump(self.symptom_columns, 'models/symptom_columns.pkl')
        print("✓ Encoders and scalers saved")
        
        # Save medical data
        joblib.dump(medications, 'models/medications.pkl')
        joblib.dump(workout_plans, 'models/workout_plans.pkl')
        joblib.dump(diet_plans, 'models/diet_plans.pkl')
        print("✓ Medical databases saved")
        
        # Save disease information
        try:
            desc_df = pd.read_csv('datasets/symptom_Description.csv')
            descriptions = dict(zip(desc_df['Disease'], desc_df['Description']))
            
            prec_df = pd.read_csv('datasets/symptom_precaution.csv')
            precautions = {}
            for _, row in prec_df.iterrows():
                disease = row['Disease']
                prec_list = [row[f'Precaution_{i}'] for i in range(1, 5) 
                            if pd.notna(row.get(f'Precaution_{i}', None))]
                precautions[disease] = prec_list
            
            disease_info = {
                'descriptions': descriptions,
                'precautions': precautions
            }
            joblib.dump(disease_info, 'models/disease_info.pkl')
            print("✓ Disease information saved")
        except Exception as e:
            print(f"⚠️ Could not load disease info: {e}")
        
        # Save metadata
        metadata = {
            'best_model': best_model_name,
            'accuracy': self.models[best_model_name].get('test_accuracy', 0),
            'f1_score': self.models[best_model_name].get('f1_score', 0),
            'num_diseases': len(self.label_encoder.classes_),
            'num_symptoms': len(self.symptom_columns)
        }
        joblib.dump(metadata, 'models/metadata.pkl')
        print("✓ Metadata saved")
        print("="*60)

def main():
    """Main training pipeline"""
    print("\n" + "="*70)
    print(" "*15 + "🏥 ADVANCED DISEASE PREDICTION SYSTEM")
    print(" "*20 + "Model Training Pipeline")
    print("="*70)
    
    predictor = EnhancedDiseasePredictor()
    
    # Load and preprocess data
    df = predictor.load_data()
    X_train, X_test, y_train, y_test = predictor.preprocess_data(df)
    
    # Train models
    best_model_name = predictor.train_models(X_train, X_test, y_train, y_test)
    
    # Load medical datasets
    medications, workout_plans, diet_plans = predictor.load_medical_datasets()
    
    # Save everything
    predictor.save_models(best_model_name, medications, workout_plans, diet_plans)
    
    print("\n" + "="*70)
    print(" "*20 + "✅ TRAINING COMPLETE!")
    print("="*70)
    print("\n📌 Next Steps:")
    print("   1. Run the Flask app: python app.py")
    print("   2. Open browser: http://localhost:5001")
    print("   3. Start predicting diseases!")
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    main()