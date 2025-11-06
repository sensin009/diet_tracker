# model/predictor.py - Prediction interface for Flask app
import joblib
import pandas as pd
import numpy as np
import os

class DietPredictor:
    def __init__(self, model_path=None):
        self.model_path = model_path
        self.model_data = None
        self.load_model()
        
    def load_model(self):
        """Load trained model (try multiple known RF artifacts)"""
        candidate_paths = []
        if self.model_path:
            candidate_paths.append(self.model_path)
        # Try common files in both project root and module dir
        module_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(module_dir, '..'))
        candidate_paths.extend([
            os.path.join(project_root, 'diet_rf_model.pkl'),
            os.path.join(project_root, 'trained_genuine_model.pkl'),
            os.path.join(project_root, 'trained_diet_model.pkl'),
            os.path.join(module_dir, 'diet_rf_model.pkl'),
            os.path.join(module_dir, 'trained_genuine_model.pkl'),
            os.path.join(module_dir, 'trained_diet_model.pkl')
        ])
        for path in candidate_paths:
            try:
                if os.path.exists(path):
                    self.model_data = joblib.load(path)
                    self.model_path = path
                    print(f"✓ Model loaded from {path}")
                    return
            except Exception as e:
                print(f"⚠ Could not load model at {path}: {e}")
        print("⚠ No RandomForest model found; using fallback rules")
        self.model_data = None
    
    def calculate_bmi(self, weight, height):
        """Calculate BMI"""
        return weight / ((height / 100) ** 2)
    
    def calculate_bmr(self, weight, height, age, gender):
        """Calculate Basal Metabolic Rate"""
        if gender == 'male':
            return 10 * weight + 6.25 * height - 5 * age + 5
        else:
            return 10 * weight + 6.25 * height - 5 * age - 161
    
    def calculate_tdee(self, bmr, activity_level):
        """Calculate Total Daily Energy Expenditure"""
        multipliers = {
            'sedentary': 1.2,
            'light': 1.375,
            'moderate': 1.55,
            'active': 1.725,
            'very_active': 1.9
        }
        return bmr * multipliers.get(activity_level, 1.2)
    
    def predict_diet(self, user_info):
        """
        Predict diet type for user
        
        Args:
            user_info (dict): Dictionary with user information
                - age, weight, height, gender, activity_level, goal
                - medical_condition, dietary_preference, allergies
                - budget, cooking_time, metabolism_rate, sleep_hours, stress_level
        
        Returns:
            dict: Prediction results with diet type, confidence, and recommendations
        """
        if self.model_data is None:
            return self._fallback_prediction(user_info)
        
        try:
            # Calculate derived features
            bmi = self.calculate_bmi(user_info['weight'], user_info['height'])
            bmr = self.calculate_bmr(
                user_info['weight'],
                user_info['height'],
                user_info['age'],
                user_info['gender']
            )
            tdee = self.calculate_tdee(bmr, user_info['activity_level'])
            
            # Prepare input data
            input_data = pd.DataFrame([{
                'age': user_info['age'],
                'weight': user_info['weight'],
                'height': user_info['height'],
                'bmi': bmi,
                'bmr': bmr,
                'tdee': tdee,
                'gender': user_info['gender'],
                'activity_level': user_info['activity_level'],
                'goal': user_info['goal'],
                'medical_condition': user_info.get('medical_condition', 'none'),
                'dietary_preference': user_info.get('dietary_preference', 'none'),
                'allergies': user_info.get('allergies', 'none'),
                'budget': user_info.get('budget', 'medium'),
                'cooking_time': user_info.get('cooking_time', 'medium'),
                'metabolism_rate': user_info.get('metabolism_rate', 'normal'),
                'sleep_hours': user_info.get('sleep_hours', 7.0),
                'stress_level': user_info.get('stress_level', 'medium')
            }])
            
            # Encode categorical features
            model = self.model_data['model']
            label_encoders = self.model_data['label_encoders']
            scaler = self.model_data['scaler']
            feature_names = self.model_data['feature_names']
            
            # Encode
            input_encoded = input_data.copy()
            for col, le in label_encoders.items():
                if col in input_encoded.columns:
                    try:
                        input_encoded[col] = le.transform([str(input_encoded[col].iloc[0])])
                    except:
                        input_encoded[col] = 0  # Default encoding for unknown values
            
            # Scale numerical features
            numerical_cols = ['age', 'weight', 'height', 'bmi', 'bmr', 'tdee', 'sleep_hours']
            input_encoded[numerical_cols] = scaler.transform(input_encoded[numerical_cols])
            
            # Predict
            prediction = model.predict(input_encoded[feature_names])[0]
            probabilities = model.predict_proba(input_encoded[feature_names])[0]
            
            # Get top 3 predictions
            top_3_idx = np.argsort(probabilities)[-3:][::-1]
            alternatives = [
                {
                    'diet_type': model.classes_[idx],
                    'confidence': float(probabilities[idx])
                }
                for idx in top_3_idx
            ]
            
            # Calculate target calories
            target_calories = self._calculate_target_calories(tdee, user_info['goal'])
            
            # Get recommendations based on allergies and medical conditions
            recommendations = self._get_recommendations(
                prediction,
                user_info.get('allergies', 'none'),
                user_info.get('medical_condition', 'none')
            )
            
            return {
                'success': True,
                'diet_type': prediction,
                'confidence': float(probabilities[np.argmax(probabilities)]),
                'alternatives': alternatives,
                'bmi': round(bmi, 1),
                'bmr': round(bmr, 0),
                'tdee': round(tdee, 0),
                'target_calories': round(target_calories, 0),
                'recommendations': recommendations
            }
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return self._fallback_prediction(user_info)
    
    def _calculate_target_calories(self, tdee, goal):
        """Calculate target daily calories based on goal"""
        if goal == 'weight_loss':
            return tdee - 500
        elif goal == 'muscle_gain':
            return tdee + 400
        elif goal == 'endurance':
            return tdee + 300
        else:
            return tdee
    
    def _get_recommendations(self, diet_type, allergies, medical_condition):
        """Get personalized recommendations"""
        recommendations = []
        
        # Allergy-based recommendations
        if allergies != 'none':
            allergy_map = {
                'dairy': 'Use plant-based milk alternatives (almond, oat, soy)',
                'nuts': 'Replace with seeds (sunflower, pumpkin) for healthy fats',
                'gluten': 'Choose rice, quinoa, and gluten-free grains',
                'shellfish': 'Get protein from chicken, fish, or plant sources',
                'eggs': 'Use flax eggs or chia seeds as binding agents',
                'soy': 'Opt for other plant proteins like lentils and chickpeas'
            }
            if allergies in allergy_map:
                recommendations.append(f"🚫 Allergy Alert: {allergy_map[allergies]}")
        
        # Medical condition recommendations
        if medical_condition != 'none':
            medical_map = {
                'diabetes': 'Focus on low-GI foods, monitor carb intake, include fiber',
                'hypertension': 'Limit sodium to 1500mg/day, eat potassium-rich foods',
                'heart_disease': 'Choose lean proteins, limit saturated fats, eat omega-3s'
            }
            if medical_condition in medical_map:
                recommendations.append(f"💊 Medical: {medical_map[medical_condition]}")
        
        # Diet-specific tips
        diet_tips = {
            'weight_loss': '🔥 Stay in calorie deficit, drink plenty of water',
            'weight_loss_active': '🏃 High protein to preserve muscle, post-workout meals important',
            'muscle_gain': '💪 Eat every 3-4 hours, focus on protein timing',
            'endurance': '⚡ Carb-loading before long sessions, hydration is key',
            'keto': '🥑 Keep carbs under 50g, increase healthy fats',
            'vegetarian_balanced': '🥗 Combine plant proteins, supplement B12',
            'vegan_balanced': '🌱 Ensure adequate protein, consider B12 and iron supplements'
        }
        
        if diet_type in diet_tips:
            recommendations.append(diet_tips[diet_type])
        
        return recommendations
    
    def _fallback_prediction(self, user_info):
        """Fallback logic when model is not available"""
        bmi = self.calculate_bmi(user_info['weight'], user_info['height'])
        bmr = self.calculate_bmr(
            user_info['weight'],
            user_info['height'],
            user_info['age'],
            user_info['gender']
        )
        tdee = self.calculate_tdee(bmr, user_info['activity_level'])
        
        # Simple rule-based prediction
        goal = user_info['goal']
        activity = user_info['activity_level']
        medical = user_info.get('medical_condition', 'none')
        
        if medical == 'diabetes':
            diet_type = 'low_carb_balanced'
        elif medical == 'heart_disease':
            diet_type = 'heart_healthy'
        elif goal == 'weight_loss' or bmi > 27:
            diet_type = 'weight_loss_active' if activity in ['active', 'very_active'] else 'weight_loss'
        elif goal == 'muscle_gain':
            diet_type = 'muscle_gain'
        elif goal == 'endurance':
            diet_type = 'endurance'
        else:
            diet_type = 'maintenance'
        
        return {
            'success': True,
            'diet_type': diet_type,
            'confidence': 0.75,
            'alternatives': [],
            'bmi': round(bmi, 1),
            'bmr': round(bmr, 0),
            'tdee': round(tdee, 0),
            'target_calories': round(self._calculate_target_calories(tdee, goal), 0),
            'recommendations': self._get_recommendations(diet_type, user_info.get('allergies', 'none'), medical)
        }

# Global predictor instance
_predictor = None

def get_predictor():
    """Get singleton predictor instance"""
    global _predictor
    if _predictor is None:
        _predictor = DietPredictor()
    return _predictor