# train_diet_model.py - Random Forest model for diet recommendation

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Generate synthetic training data
def generate_training_data(n_samples=1000):
    np.random.seed(42)
    
    data = {
        'age': np.random.randint(18, 70, n_samples),
        'weight': np.random.uniform(50, 120, n_samples),
        'height': np.random.uniform(150, 200, n_samples),
        'gender': np.random.choice(['male', 'female'], n_samples),
        'activity_level': np.random.choice(['sedentary', 'light', 'moderate', 'active', 'very active'], n_samples),
        'goal': np.random.choice(['weight_loss', 'muscle_gain', 'maintenance', 'endurance'], n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Calculate BMI
    df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
    
    # Encode categorical variables
    df['gender_encoded'] = df['gender'].map({'male': 1, 'female': 0})
    activity_map = {'sedentary': 0, 'light': 1, 'moderate': 2, 'active': 3, 'very active': 4}
    df['activity_encoded'] = df['activity_level'].map(activity_map)
    
    # Logic-based diet assignment for realistic training
    diet_type = []
    for idx, row in df.iterrows():
        bmi = row['bmi']
        goal = row['goal']
        activity = row['activity_encoded']
        
        if goal == 'weight_loss' or bmi > 25:
            diet_type.append('weight_loss')
        elif goal == 'muscle_gain' and activity >= 3:
            diet_type.append('muscle_gain')
        elif goal == 'endurance' and activity >= 3:
            diet_type.append('endurance')
        elif bmi < 18.5:
            diet_type.append('muscle_gain')
        else:
            diet_type.append('maintenance')
    
    df['diet_type'] = diet_type
    
    return df

# Train Random Forest model
def train_model():
    print("🤖 Generating training data...")
    df = generate_training_data(1000)
    
    # Features for model
    features = ['age', 'weight', 'height', 'bmi', 'gender_encoded', 'activity_encoded']
    X = df[features]
    y = df['diet_type']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest
    print("🌲 Training Random Forest model...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    rf_model.fit(X_train, y_train)
    
    # Evaluate
    train_score = rf_model.score(X_train, y_train)
    test_score = rf_model.score(X_test, y_test)
    
    print(f"\n✓ Training Accuracy: {train_score:.2%}")
    print(f"✓ Testing Accuracy: {test_score:.2%}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n📊 Feature Importance:")
    print(feature_importance.to_string(index=False))
    
    # Save model
    joblib.dump(rf_model, 'diet_rf_model.pkl')
    print("\n💾 Model saved as 'diet_rf_model.pkl'")
    print("✅ Training complete! You can now run your Flask app.")
    
    return rf_model

if __name__ == '__main__':
    model = train_model()