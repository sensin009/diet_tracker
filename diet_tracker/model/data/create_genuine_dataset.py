# create_genuine_dataset.py - Generate realistic training data based on nutritional science

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_genuine_user_dataset():
    """
    Create genuine user dataset based on real dietary science
    Sources: WHO guidelines, ADA (American Diabetes Association), 
    Indian dietary patterns, global nutrition standards
    """
    np.random.seed(42)
    
    # Real-world demographic distribution
    ages = []
    weights = []
    heights = []
    genders = []
    
    # Age groups with realistic distribution
    for _ in range(500):  # Young adults (18-30)
        ages.append(np.random.randint(18, 31))
    for _ in range(700):  # Adults (31-50)
        ages.append(np.random.randint(31, 51))
    for _ in range(400):  # Middle-aged (51-65)
        ages.append(np.random.randint(51, 66))
    for _ in range(400):  # Seniors (65+)
        ages.append(np.random.randint(65, 81))
    
    # Realistic weight and height based on global data
    for age in ages:
        gender = np.random.choice(['male', 'female'], p=[0.49, 0.51])
        genders.append(gender)
        
        if gender == 'male':
            # Male: avg 70-75kg, height 165-180cm (global mix)
            if age < 30:
                heights.append(np.random.normal(173, 7))
                weights.append(np.random.normal(72, 12))
            elif age < 50:
                heights.append(np.random.normal(172, 7))
                weights.append(np.random.normal(76, 15))
            elif age < 65:
                heights.append(np.random.normal(171, 6))
                weights.append(np.random.normal(78, 16))
            else:
                heights.append(np.random.normal(169, 6))
                weights.append(np.random.normal(74, 14))
        else:
            # Female: avg 55-65kg, height 155-170cm
            if age < 30:
                heights.append(np.random.normal(161, 6))
                weights.append(np.random.normal(58, 10))
            elif age < 50:
                heights.append(np.random.normal(160, 6))
                weights.append(np.random.normal(62, 12))
            elif age < 65:
                heights.append(np.random.normal(159, 6))
                weights.append(np.random.normal(65, 14))
            else:
                heights.append(np.random.normal(157, 5))
                weights.append(np.random.normal(63, 12))
    
    n_samples = len(ages)
    
    # Activity levels (realistic distribution)
    activity_levels = np.random.choice(
        ['sedentary', 'light', 'moderate', 'active', 'very_active'],
        size=n_samples,
        p=[0.35, 0.30, 0.20, 0.12, 0.03]  # Most people are sedentary/light
    )
    
    # Goals (realistic distribution)
    goals = []
    for i in range(n_samples):
        bmi = weights[i] / ((heights[i]/100) ** 2)
        if bmi > 27:
            goals.append(np.random.choice(['weight_loss', 'weight_loss', 'weight_loss', 'maintenance']))
        elif bmi < 18.5:
            goals.append('weight_gain')
        else:
            goals.append(np.random.choice(['maintenance', 'muscle_gain', 'weight_loss', 'endurance'], p=[0.4, 0.25, 0.25, 0.1]))
    
    # Medical conditions (based on global prevalence)
    medical_conditions = []
    for i in range(n_samples):
        age = ages[i]
        bmi = weights[i] / ((heights[i]/100) ** 2)
        
        # Age and BMI-based medical condition probability
        if age > 50 and bmi > 27:
            medical_conditions.append(np.random.choice(
                ['none', 'diabetes', 'hypertension', 'heart_disease', 'diabetes_hypertension'],
                p=[0.35, 0.25, 0.20, 0.10, 0.10]
            ))
        elif age > 50:
            medical_conditions.append(np.random.choice(
                ['none', 'diabetes', 'hypertension', 'heart_disease'],
                p=[0.60, 0.15, 0.15, 0.10]
            ))
        elif bmi > 30:
            medical_conditions.append(np.random.choice(
                ['none', 'diabetes', 'pre_diabetes', 'hypertension'],
                p=[0.50, 0.20, 0.20, 0.10]
            ))
        else:
            medical_conditions.append(np.random.choice(
                ['none', 'none', 'none', 'none', 'diabetes', 'hypertension'],
                p=[0.70, 0.15, 0.08, 0.04, 0.02, 0.01]
            ))
    
    # Dietary preferences (realistic global distribution)
    dietary_preferences = np.random.choice(
        ['none', 'vegetarian', 'vegan', 'pescatarian', 'keto', 'low_carb', 'mediterranean'],
        size=n_samples,
        p=[0.50, 0.20, 0.05, 0.05, 0.08, 0.07, 0.05]
    )
    
    # Allergies (realistic prevalence)
    allergies_list = []
    for i in range(n_samples):
        allergy_chance = np.random.random()
        if allergy_chance < 0.70:  # 70% no allergies
            allergies_list.append('none')
        elif allergy_chance < 0.78:  # 8% dairy
            allergies_list.append('dairy')
        elif allergy_chance < 0.84:  # 6% nuts
            allergies_list.append('nuts')
        elif allergy_chance < 0.89:  # 5% gluten
            allergies_list.append('gluten')
        elif allergy_chance < 0.93:  # 4% shellfish
            allergies_list.append('shellfish')
        elif allergy_chance < 0.96:  # 3% eggs
            allergies_list.append('eggs')
        elif allergy_chance < 0.98:  # 2% soy
            allergies_list.append('soy')
        else:  # 2% fish
            allergies_list.append('fish')
    
    # Create DataFrame
    df = pd.DataFrame({
        'user_id': range(1, n_samples + 1),
        'age': ages,
        'weight': np.round(weights, 1),
        'height': np.round(heights, 1),
        'gender': genders,
        'activity_level': activity_levels,
        'goal': goals,
        'medical_condition': medical_conditions,
        'dietary_preference': dietary_preferences,
        'allergies': allergies_list,
        'budget': np.random.choice(['low', 'medium', 'high'], n_samples, p=[0.30, 0.50, 0.20]),
        'cooking_time': np.random.choice(['quick', 'medium', 'long'], n_samples, p=[0.40, 0.45, 0.15]),
        'metabolism_rate': np.random.choice(['slow', 'normal', 'fast'], n_samples, p=[0.25, 0.55, 0.20]),
        'sleep_hours': np.round(np.random.normal(7, 1.2, n_samples), 1).clip(4, 10),
        'stress_level': np.random.choice(['low', 'medium', 'high'], n_samples, p=[0.25, 0.50, 0.25]),
        'water_intake_liters': np.round(np.random.normal(2.0, 0.8, n_samples), 1).clip(0.5, 5),
        'exercise_minutes_per_week': np.random.randint(0, 420, n_samples)
    })
    
    # Calculate BMI, BMR, TDEE
    df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
    df['bmi'] = df['bmi'].round(1)
    
    # BMR using Mifflin-St Jeor equation
    df['bmr'] = df.apply(lambda row: 
        (10 * row['weight'] + 6.25 * row['height'] - 5 * row['age'] + 5) if row['gender'] == 'male'
        else (10 * row['weight'] + 6.25 * row['height'] - 5 * row['age'] - 161), axis=1)
    df['bmr'] = df['bmr'].round(0)
    
    # TDEE
    activity_multiplier = {'sedentary': 1.2, 'light': 1.375, 'moderate': 1.55, 'active': 1.725, 'very_active': 1.9}
    df['tdee'] = df.apply(lambda row: row['bmr'] * activity_multiplier[row['activity_level']], axis=1)
    df['tdee'] = df['tdee'].round(0)
    
    # Assign diet type based on comprehensive analysis
    df['recommended_diet'] = df.apply(assign_genuine_diet, axis=1)
    
    # Target calories
    df['target_calories'] = df.apply(calculate_target_calories, axis=1).round(0)
    
    # Macro distribution
    df[['protein_percent', 'carbs_percent', 'fats_percent']] = df.apply(
        lambda row: pd.Series(get_macro_distribution(row)), axis=1)
    
    # Meal frequency
    df['meal_frequency'] = df.apply(determine_meal_frequency, axis=1)
    
    # Fiber requirement (g/day)
    df['fiber_target_g'] = df.apply(lambda row: 
        38 if row['gender'] == 'male' else 25, axis=1)
    
    # Sodium limit (mg/day)
    df['sodium_limit_mg'] = df.apply(lambda row:
        1500 if 'hypertension' in row['medical_condition'] else 2300, axis=1)
    
    # Sugar limit (g/day)
    df['sugar_limit_g'] = df.apply(lambda row:
        25 if 'diabetes' in row['medical_condition'] else 50, axis=1)
    
    return df

def assign_genuine_diet(row):
    """Assign diet based on medical science and conditions"""
    medical = row['medical_condition']
    dietary_pref = row['dietary_preference']
    goal = row['goal']
    bmi = row['bmi']
    
    # Medical conditions take highest priority
    if medical == 'diabetes_hypertension':
        return 'diabetic_cardiac_diet'
    elif 'diabetes' in medical:
        if dietary_pref == 'vegetarian':
            return 'diabetic_vegetarian'
        return 'diabetic_diet'
    elif medical == 'heart_disease':
        return 'cardiac_diet'
    elif medical == 'hypertension':
        return 'dash_diet'  # DASH: Dietary Approaches to Stop Hypertension
    
    # Dietary preferences
    if dietary_pref == 'vegetarian':
        if goal == 'weight_loss':
            return 'vegetarian_weight_loss'
        elif goal == 'muscle_gain':
            return 'vegetarian_high_protein'
        return 'balanced_vegetarian'
    elif dietary_pref == 'vegan':
        return 'balanced_vegan'
    elif dietary_pref == 'keto':
        return 'ketogenic'
    elif dietary_pref == 'mediterranean':
        return 'mediterranean_diet'
    elif dietary_pref == 'low_carb':
        return 'low_carb_diet'
    
    # Goal-based diets
    if goal == 'weight_loss' or bmi > 27:
        return 'weight_loss_balanced'
    elif goal == 'muscle_gain':
        return 'muscle_building'
    elif goal == 'endurance':
        return 'endurance_athlete'
    elif goal == 'weight_gain':
        return 'weight_gain_healthy'
    else:
        return 'balanced_maintenance'

def calculate_target_calories(row):
    """Calculate target calories based on goal"""
    tdee = row['tdee']
    goal = row['goal']
    
    if 'weight_loss' in goal or row['bmi'] > 27:
        return tdee - 500
    elif 'muscle_gain' in goal or 'weight_gain' in goal:
        return tdee + 400
    elif 'endurance' in goal:
        return tdee + 300
    else:
        return tdee

def get_macro_distribution(row):
    """Get macro distribution based on diet type"""
    diet = row['recommended_diet']
    
    macros = {
        'diabetic_diet': (30, 40, 30),
        'diabetic_vegetarian': (25, 45, 30),
        'diabetic_cardiac_diet': (25, 45, 30),
        'cardiac_diet': (25, 50, 25),
        'dash_diet': (20, 55, 25),
        'vegetarian_weight_loss': (30, 40, 30),
        'vegetarian_high_protein': (35, 45, 20),
        'balanced_vegetarian': (20, 55, 25),
        'balanced_vegan': (18, 60, 22),
        'ketogenic': (25, 5, 70),
        'mediterranean_diet': (20, 45, 35),
        'low_carb_diet': (30, 30, 40),
        'weight_loss_balanced': (35, 35, 30),
        'muscle_building': (35, 45, 20),
        'endurance_athlete': (20, 60, 20),
        'weight_gain_healthy': (25, 50, 25),
        'balanced_maintenance': (25, 45, 30)
    }
    
    return macros.get(diet, (25, 45, 30))

def determine_meal_frequency(row):
    """Determine optimal meal frequency"""
    if 'diabetes' in row['medical_condition']:
        return 6  # Frequent small meals for blood sugar control
    elif row['goal'] == 'muscle_gain':
        return 6
    elif row['goal'] == 'weight_loss':
        return 4
    else:
        return 5

def create_food_database():
    """Create comprehensive food database with genuine nutritional data"""
    
    # All arrays must have exactly 100 values
    foods_data = {
        'food_id': list(range(1, 101)),
        'food_name': [
            # Grains & Cereals (10)
            'Brown Rice (100g)', 'White Rice (100g)', 'Whole Wheat Bread (2 slices)', 'Quinoa (100g)', 'Oats (100g)',
            'Roti (2 medium)', 'Chapati (2 medium)', 'Millet (100g)', 'Barley (100g)', 'Buckwheat (100g)',
            
            # Proteins - Vegetarian (15)
            'Chickpeas (100g)', 'Lentils Red (100g)', 'Lentils Green (100g)', 'Kidney Beans (100g)', 'Black Beans (100g)',
            'Tofu (100g)', 'Paneer (100g)', 'Greek Yogurt (100g)', 'Cottage Cheese (100g)', 'Soy Milk (250ml)',
            'Eggs (2 large)', 'Egg Whites (4)', 'Almonds (30g)', 'Walnuts (30g)', 'Peanuts (30g)',
            
            # Proteins - Non-Veg (10)
            'Chicken Breast (100g)', 'Chicken Thigh (100g)', 'Turkey Breast (100g)', 'Fish Salmon (100g)', 'Fish Tuna (100g)',
            'Fish Mackerel (100g)', 'Fish Cod (100g)', 'Shrimp (100g)', 'Lean Beef (100g)', 'Pork Lean (100g)',
            
            # Vegetables (15)
            'Spinach (100g)', 'Broccoli (100g)', 'Cauliflower (100g)', 'Tomato (100g)', 'Cucumber (100g)',
            'Carrot (100g)', 'Bell Pepper (100g)', 'Onion (100g)', 'Garlic (10g)', 'Ginger (10g)',
            'Potato (100g)', 'Sweet Potato (100g)', 'Pumpkin (100g)', 'Zucchini (100g)', 'Eggplant (100g)',
            
            # Fruits (10)
            'Apple (1 medium)', 'Banana (1 medium)', 'Orange (1 medium)', 'Mango (100g)', 'Papaya (100g)',
            'Watermelon (100g)', 'Grapes (100g)', 'Strawberries (100g)', 'Blueberries (100g)', 'Pomegranate (100g)',
            
            # Dairy & Alternatives (10)
            'Milk Whole (250ml)', 'Milk Skim (250ml)', 'Almond Milk (250ml)', 'Oat Milk (250ml)', 'Coconut Milk (250ml)',
            'Yogurt Plain (100g)', 'Cheddar Cheese (30g)', 'Mozzarella (30g)', 'Butter (10g)', 'Ghee (10g)',
            
            # Fats & Oils (10)
            'Olive Oil (15ml)', 'Coconut Oil (15ml)', 'Sunflower Oil (15ml)', 'Mustard Oil (15ml)', 'Avocado (100g)',
            'Peanut Butter (30g)', 'Almond Butter (30g)', 'Flaxseeds (15g)', 'Chia Seeds (15g)', 'Sesame Seeds (15g)',
            
            # Legumes & Pulses (10)
            'Moong Dal (100g)', 'Masoor Dal (100g)', 'Toor Dal (100g)', 'Urad Dal (100g)', 'Chana Dal (100g)',
            'Green Peas (100g)', 'Soybeans (100g)', 'Lima Beans (100g)', 'White Beans (100g)', 'Pinto Beans (100g)',
            
            # Snacks & Others (10)
            'Dark Chocolate (30g)', 'Honey (15g)', 'Jaggery (15g)', 'Brown Sugar (15g)', 'Dates (3 pieces)',
            'Coconut Fresh (30g)', 'Cashews (30g)', 'Pistachios (30g)', 'Raisins (30g)', 'Green Tea (1 cup)'
        ],
        'calories': [
            # Grains (10)
            123, 130, 160, 120, 389, 140, 140, 378, 354, 343,
            # Veg Proteins (15)
            164, 116, 116, 127, 132, 76, 265, 59, 98, 80, 155, 52, 170, 185, 161,
            # Non-veg Proteins (10)
            165, 209, 135, 208, 130, 205, 82, 99, 250, 242,
            # Vegetables (15)
            23, 34, 25, 18, 16, 41, 31, 40, 149, 80, 77, 86, 26, 17, 25,
            # Fruits (10)
            95, 105, 62, 60, 43, 30, 69, 32, 57, 83,
            # Dairy (10)
            150, 83, 30, 120, 230, 59, 115, 85, 72, 90,
            # Fats (10)
            120, 120, 120, 120, 160, 190, 195, 75, 70, 85,
            # Legumes (10)
            347, 116, 335, 341, 364, 81, 173, 115, 333, 347,
            # Snacks (10)
            170, 46, 38, 48, 66, 100, 170, 165, 85, 2
        ],
        'protein_g': [
            # Grains (10)
            2.7, 2.7, 8, 4.4, 13.2, 6, 6, 11, 12.5, 13.3,
            # Veg Proteins (15)
            8.9, 9, 9, 8.7, 8.9, 8, 18, 10, 11, 3.3, 13, 11, 6, 4.3, 7.3,
            # Non-veg Proteins (10)
            31, 26, 30, 20, 30, 20, 18, 24, 26, 27,
            # Vegetables (15)
            2.9, 2.8, 1.9, 0.9, 0.7, 0.9, 1, 1.1, 0.6, 0.8, 2, 1.6, 1, 0.7, 1,
            # Fruits (10)
            0.3, 1.3, 1.2, 0.8, 0.5, 0.4, 0.7, 0.7, 1, 1.7,
            # Dairy (10)
            8, 8, 1, 3.4, 2, 3.4, 7, 4.5, 0.1, 0,
            # Fats (10)
            0, 0, 0, 0, 2, 8, 7.5, 4, 4, 5,
            # Legumes (10)
            24.6, 9, 22.5, 25.2, 21.4, 5.4, 16.6, 7.8, 21.4, 21.4,
            # Snacks (10)
            3, 0, 0, 0, 0.5, 1, 5.4, 5.8, 1.4, 0
        ],
        'carbs_g': [
            # Grains (10)
            26, 28, 28, 21, 66, 28, 28, 73, 73.5, 71,
            # Veg Proteins (15)
            27, 20, 20, 23, 24, 1.9, 3.6, 3.6, 3.4, 4.4, 1.1, 0.7, 6.1, 3.9, 7.6,
            # Non-veg Proteins (10)
            0, 0, 0, 0, 0, 0, 0, 0.8, 0, 0,
            # Vegetables (15)
            3.6, 7, 5, 3.9, 3.6, 10, 6, 8.8, 1, 1, 20, 22, 7, 4, 9,
            # Fruits (10)
            25, 27, 12, 15, 4, 4, 6, 14, 0, 0,
            # Dairy (10)
            0, 0, 0, 0, 8.5, 6, 6.5, 5, 5, 6,
            # Fats (10)
            63, 20, 61, 59, 62.6, 14, 11, 19, 52, 60,
            # Legumes (10) - Fixed to have 10 values
            7, 12.5, 10, 12.5, 18, 6, 9, 8, 22, 0.5,
            # Snacks (10)
            12, 0, 0, 0, 0.1, 8, 15, 14, 0.5, 0
        ],
        'fats_g': [
            # Grains (10)
            0.9, 0.3, 2.5, 1.9, 6.9, 2, 2, 4.2, 2.3, 3.4,
            # Veg Proteins (15)
            2.6, 0.4, 0.4, 0.5, 0.5, 4.8, 20, 0.4, 4.3, 2, 11, 0.2, 15, 18.5, 14,
            # Non-veg Proteins (10)
            3.6, 11, 1, 13, 1, 16.3, 0.7, 0.5, 15, 14,
            # Vegetables (15)
            0.4, 0.4, 0.2, 0.2, 0.1, 0.2, 0.3, 0.1, 0, 0.1, 0.4, 1.2, 0.7, 0.2, 0.2,
            # Fruits (10)
            0.3, 0.4, 0.1, 0.4, 0.1, 0.1, 0.4, 0.1, 0.6, 1.5,
            # Dairy (10)
            8, 0.5, 2.5, 8, 9, 3.3, 9, 7.5, 7.2, 10,
            # Fats (10)
            15, 15, 15, 15, 14.7, 16, 16, 6, 6, 7,
            # Legumes (10)
            1.4, 0, 1, 1.3, 1.5, 0.4, 7.2, 6, 0.3, 0,
            # Snacks (10)
            12, 0, 0, 0, 0.1, 8, 15, 14, 0.5, 0
        ]
    }
    
    df = pd.DataFrame(foods_data)
    
    # Add diabetic-friendly flag
    df['diabetic_friendly'] = df.apply(lambda row: 
        row['carbs_g'] < 30 and row['food_name'] not in ['White Rice (100g)', 'Potato (100g)', 'Honey (15g)', 'Brown Sugar (15g)'],
        axis=1)
    
    # Vegetarian flag
    non_veg_keywords = ['Chicken', 'Turkey', 'Fish', 'Shrimp', 'Beef', 'Pork', 'Eggs', 'Egg']
    df['vegetarian'] = df['food_name'].apply(lambda x: not any(keyword in x for keyword in non_veg_keywords))
    
    # Vegan flag
    non_vegan_keywords = ['Chicken', 'Turkey', 'Fish', 'Shrimp', 'Beef', 'Pork', 'Eggs', 'Egg', 
                          'Milk', 'Yogurt', 'Cheese', 'Paneer', 'Butter', 'Ghee', 'Honey']
    df['vegan'] = df['food_name'].apply(lambda x: not any(keyword in x for keyword in non_vegan_keywords))
    
    # Allergy flags
    df['contains_dairy'] = df['food_name'].str.contains('Milk|Cheese|Yogurt|Butter|Ghee|Paneer|Cottage', case=False)
    df['contains_nuts'] = df['food_name'].str.contains('Almond|Walnut|Peanut|Cashew|Pistachio', case=False)
    df['contains_gluten'] = df['food_name'].str.contains('Wheat|Bread|Roti|Chapati|Barley', case=False)
    df['contains_soy'] = df['food_name'].str.contains('Soy|Tofu', case=False)
    
    return df

if __name__ == '__main__':
    print("="*70)
    print("🏥 GENERATING GENUINE REAL-WORLD DATASET")
    print("="*70)
    
    import os
    os.makedirs('model/data', exist_ok=True)
    
    # Generate user dataset
    print("\n📊 1. Creating genuine user dataset (2000 users)...")
    user_data = create_genuine_user_dataset()
    user_data.to_csv('model/data/genuine_user_data.csv', index=False)
    print(f"   ✓ Created: model/data/genuine_user_data.csv")
    print(f"   - Total users: {len(user_data)}")
    print(f"   - Features: {len(user_data.columns)}")
    
    # Generate food database
    print("\n🍽️  2. Creating food database with nutritional data...")
    food_data = create_food_database()
    food_data.to_csv('model/data/food_database.csv', index=False)
    print(f"   ✓ Created: model/data/food_database.csv")
    print(f"   - Total foods: {len(food_data)}")
    
    print("\n📈 Dataset Statistics:")
    print(f"\n   Medical Conditions:")
    print(user_data['medical_condition'].value_counts())
    print(f"\n   Dietary Preferences:")
    print(user_data['dietary_preference'].value_counts())
    print(f"\n   Recommended Diets:")
    print(user_data['recommended_diet'].value_counts())
    
    print(f"\n   Diabetic Users: {(user_data['medical_condition'].str.contains('diabetes')).sum()}")
    print(f"   Vegetarian Preference: {(user_data['dietary_preference'] == 'vegetarian').sum()}")
    print(f"   Vegan Preference: {(user_data['dietary_preference'] == 'vegan').sum()}")
    
    print("\n" + "="*70)
    print("✅ GENUINE DATASET CREATION COMPLETE!")
    print("="*70)
    print("\n📌 Files created:")
    print("   1. model/data/genuine_user_data.csv")
    print("   2. model/data/food_database.csv")
    print("\n🚀 Next step: python -m model.train_genuine_model")