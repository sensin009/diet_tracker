# model/train_genuine_model.py - Train on genuine medical data

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

def load_genuine_data():
    """Load genuine dataset"""
    print("📂 Loading genuine user dataset...")
    df = pd.read_csv('model/data/genuine_user_data.csv')
    print(f"   ✓ Loaded {len(df)} users")
    print(f"   ✓ Features: {len(df.columns)}")
    return df

def preprocess_data(df):
    """Preprocess data for training"""
    print("\n🔧 Preprocessing data...")
    
    # Select features
    feature_columns = [
        'age', 'weight', 'height', 'bmi', 'bmr', 'tdee',
        'gender', 'activity_level', 'goal', 'medical_condition',
        'dietary_preference', 'allergies', 'budget', 'cooking_time',
        'metabolism_rate', 'sleep_hours', 'stress_level',
        'water_intake_liters', 'exercise_minutes_per_week'
    ]
    
    X = df[feature_columns].copy()
    y = df['recommended_diet']
    
    # Label encoding for categorical features
    label_encoders = {}
    categorical_features = ['gender', 'activity_level', 'goal', 'medical_condition',
                           'dietary_preference', 'allergies', 'budget', 'cooking_time',
                           'metabolism_rate', 'stress_level']
    
    for col in categorical_features:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
    
    # Standard scaling for numerical features
    scaler = StandardScaler()
    numerical_cols = ['age', 'weight', 'height', 'bmi', 'bmr', 'tdee', 
                     'sleep_hours', 'water_intake_liters', 'exercise_minutes_per_week']
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    
    print(f"   ✓ Encoded {len(categorical_features)} categorical features")
    print(f"   ✓ Scaled {len(numerical_cols)} numerical features")
    
    return X, y, label_encoders, scaler, feature_columns

def train_model(X, y):
    """Train Random Forest model"""
    print("\n🤖 Training Random Forest Classifier...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"\n📊 Model Performance:")
    print(f"   Training Accuracy: {train_score:.2%}")
    print(f"   Testing Accuracy: {test_score:.2%}")
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"   Cross-Validation: {cv_scores.mean():.2%} (+/- {cv_scores.std():.2%})")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n🔝 Top 10 Important Features:")
    print(feature_importance.head(10).to_string(index=False))
    
    # Predictions
    y_pred = model.predict(X_test)
    
    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    return model, X_test, y_test

def save_model(model, label_encoders, scaler, feature_columns):
    """Save model and preprocessing objects"""
    print("\n💾 Saving model...")
    
    model_data = {
        'model': model,
        'label_encoders': label_encoders,
        'scaler': scaler,
        'feature_names': feature_columns,
        'diet_types': list(model.classes_)
    }
    
    joblib.dump(model_data, 'trained_genuine_model.pkl')
    print("   ✓ Model saved as 'trained_genuine_model.pkl'")

def main():
    print("="*70)
    print("🏥 TRAINING GENUINE DIET PREDICTION MODEL")
    print("   Based on real medical & nutritional science")
    print("="*70)
    
    # Load data
    df = load_genuine_data()
    
    # Preprocess
    X, y, label_encoders, scaler, feature_columns = preprocess_data(df)
    
    # Train
    model, X_test, y_test = train_model(X, y)
    
    # Save
    save_model(model, label_encoders, scaler, feature_columns)
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print("\n📌 Supported Diet Types:")
    for i, diet in enumerate(model.classes_, 1):
        print(f"   {i}. {diet.replace('_', ' ').title()}")
    
    print("\n🎯 Special Support For:")
    print("   ✓ Diabetic patients (Type 1 & 2)")
    print("   ✓ Vegetarian & Vegan diets")
    print("   ✓ Cardiac patients")
    print("   ✓ Hypertension management")
    print("   ✓ Food allergies (8 types)")
    print("   ✓ Weight management goals")
    
    print("\n🚀 Next: Use this model in Flask app for personalized recommendations!")

if __name__ == '__main__':
    main()