# meal_customizer.py - Advanced meal customization system

import pandas as pd
import random

class MealCustomizer:
    def __init__(self):
        self.food_db = None
        self.load_food_database()
    
    def load_food_database(self):
        """Load food database"""
        try:
            self.food_db = pd.read_csv('model/data/food_database.csv')
        except:
            print("⚠ Food database not found. Generate it first.")
            self.food_db = None
    
    def get_meal_plan(self, user_profile, preferences=None):
        """
        Generate personalized meal plan
        
        Args:
            user_profile: dict with user data (diet_type, target_calories, allergies, etc.)
            preferences: dict with user customization preferences
        
        Returns:
            dict with complete meal plan
        """
        if self.food_db is None:
            return None
        
        diet_type = user_profile.get('diet_type', 'balanced_maintenance')
        target_calories = user_profile.get('target_calories', 2000)
        allergies = user_profile.get('allergies', 'none')
        is_vegetarian = user_profile.get('is_vegetarian', False)
        is_diabetic = 'diabetes' in user_profile.get('medical_condition', '')
        
        # Apply preferences if provided
        if preferences:
            excluded_foods = preferences.get('excluded_foods', [])
            favorite_foods = preferences.get('favorite_foods', [])
            meal_frequency = preferences.get('meal_frequency', 5)
        else:
            excluded_foods = []
            favorite_foods = []
            meal_frequency = 5
        
        # Filter foods based on requirements
        available_foods = self.filter_foods(
            is_vegetarian=is_vegetarian,
            is_diabetic=is_diabetic,
            allergies=allergies,
            excluded_foods=excluded_foods
        )
        
        if available_foods.empty:
            return None
        
        # Generate meal plan
        meal_plan = self.generate_plan(
            available_foods,
            target_calories,
            meal_frequency,
            favorite_foods,
            diet_type
        )
        
        return meal_plan
    
    def filter_foods(self, is_vegetarian=False, is_diabetic=False, allergies='none', excluded_foods=[]):
        """Filter foods based on dietary requirements"""
        filtered = self.food_db.copy()
        
        # Vegetarian filter
        if is_vegetarian:
            filtered = filtered[filtered['vegetarian'] == True]
        
        # Diabetic filter
        if is_diabetic:
            filtered = filtered[filtered['diabetic_friendly'] == True]
        
        # Allergy filters
        if allergies != 'none':
            if allergies == 'dairy':
                filtered = filtered[filtered['contains_dairy'] == False]
            elif allergies == 'nuts':
                filtered = filtered[filtered['contains_nuts'] == False]
            elif allergies == 'gluten':
                filtered = filtered[filtered['contains_gluten'] == False]
            elif allergies == 'soy':
                filtered = filtered[filtered['contains_soy'] == False]
        
        # Exclude specific foods
        if excluded_foods:
            for food in excluded_foods:
                filtered = filtered[~filtered['food_name'].str.contains(food, case=False)]
        
        return filtered
    
    def generate_plan(self, available_foods, target_calories, meal_frequency, favorites, diet_type):
        """Generate complete meal plan"""
        calories_per_meal = target_calories / meal_frequency
        
        # Get macro targets based on diet type
        macros = self.get_macro_targets(diet_type)
        
        meals = []
        total_calories = 0
        total_protein = 0
        total_carbs = 0
        total_fats = 0
        
        meal_names = ['Breakfast', 'Mid-Morning Snack', 'Lunch', 'Evening Snack', 'Dinner', 'Night Snack']
        
        for i in range(meal_frequency):
            meal_name = meal_names[i] if i < len(meal_names) else f'Meal {i+1}'
            
            # Select foods for this meal
            meal_foods = self.select_meal_foods(
                available_foods,
                calories_per_meal,
                macros,
                favorites
            )
            
            meal_data = {
                'meal_name': meal_name,
                'foods': meal_foods,
                'total_calories': sum(f['calories'] for f in meal_foods),
                'total_protein': sum(f['protein'] for f in meal_foods),
                'total_carbs': sum(f['carbs'] for f in meal_foods),
                'total_fats': sum(f['fats'] for f in meal_foods)
            }
            
            meals.append(meal_data)
            total_calories += meal_data['total_calories']
            total_protein += meal_data['total_protein']
            total_carbs += meal_data['total_carbs']
            total_fats += meal_data['total_fats']
        
        return {
            'meals': meals,
            'daily_totals': {
                'calories': round(total_calories, 0),
                'protein': round(total_protein, 1),
                'carbs': round(total_carbs, 1),
                'fats': round(total_fats, 1)
            },
            'target_calories': target_calories,
            'compliance': round((total_calories / target_calories) * 100, 1)
        }
    
    def select_meal_foods(self, available_foods, target_calories, macros, favorites):
        """Select foods for a single meal"""
        selected_foods = []
        remaining_calories = target_calories
        
        # Try to include favorites if available
        if favorites:
            for fav in favorites:
                fav_foods = available_foods[available_foods['food_name'].str.contains(fav, case=False)]
                if not fav_foods.empty and remaining_calories > 100:
                    food = fav_foods.sample(1).iloc[0]
                    selected_foods.append({
                        'name': food['food_name'],
                        'calories': food['calories'],
                        'protein': food['protein_g'],
                        'carbs': food['carbs_g'],
                        'fats': food['fats_g']
                    })
                    remaining_calories -= food['calories']
        
        # Fill remaining calories
        attempts = 0
        while remaining_calories > 50 and attempts < 10:
            # Select random food
            food = available_foods.sample(1).iloc[0]
            
            if food['calories'] <= remaining_calories * 1.2:  # Allow 20% overage
                selected_foods.append({
                    'name': food['food_name'],
                    'calories': food['calories'],
                    'protein': food['protein_g'],
                    'carbs': food['carbs_g'],
                    'fats': food['fats_g']
                })
                remaining_calories -= food['calories']
            
            attempts += 1
        
        return selected_foods
    
    def get_macro_targets(self, diet_type):
        """Get macro distribution for diet type"""
        macros = {
            'diabetic_diet': {'protein': 30, 'carbs': 40, 'fats': 30},
            'diabetic_vegetarian': {'protein': 25, 'carbs': 45, 'fats': 30},
            'cardiac_diet': {'protein': 25, 'carbs': 50, 'fats': 25},
            'dash_diet': {'protein': 20, 'carbs': 55, 'fats': 25},
            'vegetarian_weight_loss': {'protein': 30, 'carbs': 40, 'fats': 30},
            'balanced_vegetarian': {'protein': 20, 'carbs': 55, 'fats': 25},
            'balanced_vegan': {'protein': 18, 'carbs': 60, 'fats': 22},
            'ketogenic': {'protein': 25, 'carbs': 5, 'fats': 70},
            'weight_loss_balanced': {'protein': 35, 'carbs': 35, 'fats': 30},
            'muscle_building': {'protein': 35, 'carbs': 45, 'fats': 20},
        }
        return macros.get(diet_type, {'protein': 25, 'carbs': 45, 'fats': 30})
    
    def get_alternative_meals(self, user_profile, current_meal, n_alternatives=3):
        """Get alternative meal options"""
        if self.food_db is None:
            return []
        
        # Filter foods
        available_foods = self.filter_foods(
            is_vegetarian=user_profile.get('is_vegetarian', False),
            is_diabetic='diabetes' in user_profile.get('medical_condition', ''),
            allergies=user_profile.get('allergies', 'none')
        )
        
        # Generate alternatives
        alternatives = []
        target_calories = current_meal.get('total_calories', 400)
        
        for _ in range(n_alternatives):
            alt_foods = []
            remaining = target_calories
            
            while remaining > 50 and len(alt_foods) < 5:
                food = available_foods.sample(1).iloc[0]
                if food['calories'] <= remaining * 1.2:
                    alt_foods.append({
                        'name': food['food_name'],
                        'calories': food['calories'],
                        'protein': food['protein_g'],
                        'carbs': food['carbs_g'],
                        'fats': food['fats_g']
                    })
                    remaining -= food['calories']
            
            if alt_foods:
                alternatives.append({
                    'foods': alt_foods,
                    'total_calories': sum(f['calories'] for f in alt_foods)
                })
        
        return alternatives

# Global customizer instance
_customizer = None

def get_meal_customizer():
    """Get singleton meal customizer"""
    global _customizer
    if _customizer is None:
        _customizer = MealCustomizer()
    return _customizer