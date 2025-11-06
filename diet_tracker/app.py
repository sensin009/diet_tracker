from flask import Flask, render_template, redirect, url_for, request, flash
from flask import Response
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from flask_wtf import CSRFProtect
from functools import wraps
from datetime import datetime, date, timedelta
import pandas as pd
import re
import os
import sys
from flask import jsonify
from sqlalchemy.exc import IntegrityError
from sqlalchemy import func

# Add model directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'model'))

from models import db, User, Intake, Admin
from forms import RegisterForm, LoginForm, IntakeForm
from flask_wtf import FlaskForm
from wtforms import StringField, IntegerField, FloatField, SelectField, SubmitField
from wtforms.validators import InputRequired, Email, NumberRange, Length
from forms_customization import MealPreferenceForm
from flask import session

# Import ML predictor
try:
    from model.predictor import get_predictor
    ml_predictor = get_predictor()
    print("✅ ML Predictor loaded successfully!")
except Exception as e:
    ml_predictor = None
    print(f"⚠️ ML Predictor not available: {e}")

# --------------------------------------------------------------------------------------
# Input validation helpers
# --------------------------------------------------------------------------------------

def _is_number(value):
    try:
        float(value)
        return True
    except Exception:
        return False

def _clamp(value, minimum, maximum):
    if value is None:
        return minimum
    if value < minimum:
        return minimum
    if value > maximum:
        return maximum
    return value

def _validate_user_data(raw):
    """Return a sanitized user_data dict or raise ValueError."""
    if not isinstance(raw, dict):
        raise ValueError("user_data must be a dict")

    required_numeric = ['age', 'weight', 'height']
    for key in required_numeric:
        if key not in raw or not _is_number(raw.get(key)):
            raise ValueError(f"{key} must be a number")

    age = int(float(raw['age']))
    weight = float(raw['weight'])
    height = float(raw['height'])
    if age <= 0 or age > 120:
        raise ValueError("age out of range")
    if weight <= 0 or weight > 400:
        raise ValueError("weight out of range")
    if height <= 0 or height > 300:
        raise ValueError("height out of range")

    gender = (raw.get('gender') or '').strip().lower()
    if gender not in {'male', 'female'}:
        gender = 'male'

    activity_level = (raw.get('activity_level') or '').strip().lower()
    if activity_level not in {'sedentary','light','moderate','active','very_active'}:
        activity_level = 'sedentary'

    goal = (raw.get('goal') or 'maintenance').strip().lower()
    if goal not in {'weight_loss','muscle_gain','maintenance','endurance'}:
        goal = 'maintenance'

    medical_condition = (raw.get('medical_condition') or 'none').strip().lower()
    dietary_preference = (raw.get('dietary_preference') or 'none').strip().lower()
    allergies = (raw.get('allergies') or 'none').strip().lower()
    budget = (raw.get('budget') or 'medium').strip().lower()
    if budget not in {'low','medium','high'}:
        budget = 'medium'
    cooking_time = (raw.get('cooking_time') or 'medium').strip().lower()
    if cooking_time not in {'low','medium','high'}:
        cooking_time = 'medium'
    metabolism_rate = (raw.get('metabolism_rate') or 'normal').strip().lower()
    if metabolism_rate not in {'slow','normal','fast'}:
        metabolism_rate = 'normal'
    sleep_hours = raw.get('sleep_hours', 7.0)
    sleep_hours = float(sleep_hours) if _is_number(sleep_hours) else 7.0
    sleep_hours = _clamp(sleep_hours, 0.0, 24.0)
    stress_level = (raw.get('stress_level') or 'medium').strip().lower()
    if stress_level not in {'low','medium','high'}:
        stress_level = 'medium'

    return {
        'age': age,
        'weight': weight,
        'height': height,
        'gender': gender,
        'activity_level': activity_level,
        'goal': goal,
        'medical_condition': medical_condition,
        'dietary_preference': dietary_preference,
        'allergies': allergies,
        'budget': budget,
        'cooking_time': cooking_time,
        'metabolism_rate': metabolism_rate,
        'sleep_hours': sleep_hours,
        'stress_level': stress_level,
    }

# --------------------------------------------------------------------------------------
# App + extensions
# --------------------------------------------------------------------------------------
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'change_this_secret')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///diet.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
csrf = CSRFProtect(app)

# --------------------------------------------------------------------------------------
# Auth helpers
# --------------------------------------------------------------------------------------
@login_manager.user_loader
def load_user(user_id):
    if not isinstance(user_id, str):
        return None
    try:
        if user_id.startswith('user_'):
            return User.query.get(int(user_id.split('_')[1]))
        elif user_id.startswith('admin_'):
            return Admin.query.get(int(user_id.split('_')[1]))
    except Exception:
        return None
    return None

def admin_required(fn):
    @wraps(fn)
    @login_required
    def wrapper(*args, **kwargs):
        if not isinstance(current_user._get_current_object(), Admin):
            flash('Admins only', 'danger')
            return redirect(url_for('dashboard'))
        return fn(*args, **kwargs)
    return wrapper

with app.app_context():
    db.create_all()

# --------------------------------------------------------------------------------------
# ML Prediction Function (FIXED)
# --------------------------------------------------------------------------------------
def predict_diet_type_ml(user_data):
    """Use ML model to predict optimal diet type"""
    try:
        user_data = _validate_user_data(user_data)
    except Exception:
        # If invalid, fall back to defaults via fallback logic
        return fallback_diet_prediction(user_data or {})

    if ml_predictor is None:
        print("⚠️ ML predictor not available, using fallback")
        return fallback_diet_prediction(user_data)
    
    try:
        # Prepare user info for predictor
        user_info = {
            'age': user_data['age'],
            'weight': user_data['weight'],
            'height': user_data['height'],
            'gender': user_data['gender'],
            'activity_level': user_data['activity_level'],
            'goal': user_data['goal'],
            'medical_condition': user_data.get('medical_condition', 'none'),
            'dietary_preference': user_data.get('dietary_preference', 'none'),
            'allergies': user_data.get('allergies', 'none'),
            'budget': user_data.get('budget', 'medium'),
            'cooking_time': user_data.get('cooking_time', 'medium'),
            'metabolism_rate': user_data.get('metabolism_rate', 'normal'),
            'sleep_hours': user_data.get('sleep_hours', 7.0),
            'stress_level': user_data.get('stress_level', 'medium')
        }
        
        # Get prediction
        result = ml_predictor.predict_diet(user_info)
        
        if result['success']:
            print(f"✅ ML Prediction: {result['diet_type']} (confidence: {result['confidence']:.2%})")
            return {
                'diet_type': result['diet_type'],
                'confidence': result['confidence'],
                'bmi': result['bmi'],
                'bmr': result['bmr'],
                'tdee': result['tdee'],
                'target_calories': result['target_calories'],
                'recommendations': result.get('recommendations', []),
                'alternatives': result.get('alternatives', [])
            }
        else:
            return fallback_diet_prediction(user_data)
            
    except Exception as e:
        print(f"❌ ML Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return fallback_diet_prediction(user_data)

def fallback_diet_prediction(user_data):
    """Fallback rule-based prediction when ML fails"""
    try:
        cleaned = _validate_user_data(user_data)
    except Exception:
        # minimal safe defaults
        cleaned = {
            'age': 30,
            'weight': 70.0,
            'height': 170.0,
            'gender': 'male',
            'activity_level': 'sedentary',
            'goal': 'maintenance',
            'medical_condition': 'none',
            'dietary_preference': 'none',
        }

    age = cleaned['age']
    weight = cleaned['weight']
    height = cleaned['height']
    gender = cleaned['gender']
    activity = cleaned['activity_level']
    goal = cleaned['goal']
    medical = cleaned.get('medical_condition', 'none')
    dietary = cleaned.get('dietary_preference', 'none')
    
    # Calculate BMI
    bmi = weight / ((height / 100) ** 2)
    
    # Calculate BMR
    if gender == 'male':
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    else:
        bmr = 10 * weight + 6.25 * height - 5 * age - 161
    
    # Calculate TDEE
    multipliers = {
        'sedentary': 1.2, 'light': 1.375, 'moderate': 1.55,
        'active': 1.725, 'very_active': 1.9
    }
    tdee = bmr * multipliers.get(activity, 1.2)
    
    # Determine diet type
    if medical == 'diabetes':
        diet_type = 'weight_loss' if bmi > 25 else 'maintenance'
    elif medical == 'heart_disease' or medical == 'hypertension':
        diet_type = 'maintenance'
    elif dietary == 'vegetarian':
        diet_type = 'maintenance'
    elif dietary == 'vegan':
        diet_type = 'maintenance'
    elif dietary == 'keto':
        diet_type = 'weight_loss'
    elif goal == 'weight_loss' or bmi > 27:
        diet_type = 'weight_loss'
    elif goal == 'muscle_gain':
        diet_type = 'muscle_gain'
    elif goal == 'endurance':
        diet_type = 'endurance'
    else:
        diet_type = 'maintenance'
    
    # Calculate target calories
    if goal == 'weight_loss':
        target_calories = tdee - 500
    elif goal == 'muscle_gain':
        target_calories = tdee + 400
    elif goal == 'endurance':
        target_calories = tdee + 300
    else:
        target_calories = tdee
    
    return {
        'diet_type': diet_type,
        'confidence': 0.75,
        'bmi': round(bmi, 1),
        'bmr': round(bmr, 0),
        'tdee': round(tdee, 0),
        'target_calories': round(target_calories, 0),
        'recommendations': [],
        'alternatives': []
    }

# --------------------------------------------------------------------------------------
# Public routes
# --------------------------------------------------------------------------------------
@app.route('/')
def home():
    if current_user.is_authenticated:
        if isinstance(current_user._get_current_object(), Admin):
            return redirect(url_for('admin_dashboard'))
        return redirect(url_for('dashboard'))
    return render_template('landing.html')

@app.route('/register', methods=['GET','POST'])
def register():
    form = RegisterForm()
    if form.validate_on_submit():
        username_input = (form.username.data or '').strip()
        email_input = (form.email.data or '').strip().lower()

        if User.query.filter(func.lower(User.username) == func.lower(username_input)).first():
            flash('Username already exists', 'danger')
            return render_template('register.html', form=form)
        if User.query.filter(func.lower(User.email) == email_input).first():
            flash('Email already registered', 'danger')
            return render_template('register.html', form=form)

        # Prepare user data for ML prediction
        user_data = {
            'age': form.age.data,
            'weight': form.weight.data,
            'height': form.height.data,
            'gender': form.gender.data,
            'activity_level': form.activity_level.data,
            'goal': form.goal.data,
            'medical_condition': form.medical_condition.data if hasattr(form, 'medical_condition') else 'none',
            'allergies': form.allergies.data if hasattr(form, 'allergies') else 'none',
            'dietary_preference': form.dietary_preference.data if hasattr(form, 'dietary_preference') else 'none',
            'budget': form.budget.data if hasattr(form, 'budget') else 'medium',
            'cooking_time': form.cooking_time.data if hasattr(form, 'cooking_time') else 'medium',
            'metabolism_rate': form.metabolism_rate.data if hasattr(form, 'metabolism_rate') else 'normal',
            'sleep_hours': form.sleep_hours.data if hasattr(form, 'sleep_hours') else 7.0,
            'stress_level': form.stress_level.data if hasattr(form, 'stress_level') else 'medium'
        }
        
        # Get ML prediction
        prediction_result = predict_diet_type_ml(user_data)
        
        # Hash password
        hashed_password = generate_password_hash(form.password.data)
        
        # Create user
        user = User(
            username=username_input,
            email=email_input,
            password=hashed_password,
            age=form.age.data,
            weight=form.weight.data,
            height=form.height.data,
            gender=form.gender.data,
            activity_level=form.activity_level.data,
            goal=form.goal.data,
            medical_condition=user_data['medical_condition'],
            allergies=user_data['allergies'],
            dietary_preference=user_data['dietary_preference'],
            budget=user_data['budget'],
            cooking_time=user_data['cooking_time'],
            metabolism_rate=user_data['metabolism_rate'],
            sleep_hours=user_data['sleep_hours'],
            stress_level=user_data['stress_level'],
            tdee=float(prediction_result['tdee']),
            bmi=float(prediction_result['bmi']),
            bmr=float(prediction_result['bmr']),
            diet_type=prediction_result['diet_type'],
            target_calories=float(prediction_result['target_calories'])
        )

        db.session.add(user)
        try:
            db.session.commit()
        except IntegrityError:
            db.session.rollback()
            flash('Email already registered', 'danger')
            return render_template('register.html', form=form)
        
        # Show success message with ML info
        confidence_emoji = "🎯" if prediction_result['confidence'] > 0.8 else "✅"
        flash(f'{confidence_emoji} Registration successful! Your personalized {prediction_result["diet_type"].replace("_", " ").title()} plan is ready! (ML Confidence: {prediction_result["confidence"]:.0%})', 'success')
        
        return redirect(url_for('login'))
    else:
        # If POST but invalid, show field-level errors for easier debugging
        if request.method == 'POST':
            for field, errors in getattr(form, 'errors', {}).items():
                for err in errors:
                    flash(f"{field}: {err}", 'danger')
    return render_template('register.html', form=form)

@app.route('/login', methods=['GET','POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        # Try admin first
        admin = Admin.query.filter_by(username=form.username.data).first()
        if admin and check_password_hash(admin.password, form.password.data):
            login_user(admin)
            return redirect(url_for('admin_dashboard'))
        
        # Try user
        user = User.query.filter_by(username=form.username.data).first()
        if user and check_password_hash(user.password, form.password.data):
            login_user(user)
            return redirect(url_for('dashboard'))

        flash('Invalid credentials', 'danger')
    return render_template('login.html', form=form)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

# --------------------------------------------------------------------------------------
# User routes (ENHANCED WITH ML INFO)
# --------------------------------------------------------------------------------------
@app.route('/dashboard')
@login_required
def dashboard():
    if isinstance(current_user._get_current_object(), Admin):
        return redirect(url_for('admin_dashboard'))

    intakes = Intake.query.filter_by(user_id=current_user.id).order_by(Intake.date).all()
    dates = [i.date.strftime('%Y-%m-%d') for i in intakes]
    calories = [i.calories for i in intakes]
    
    # Calculate stats
    total_intakes = len({i.date for i in intakes})
    avg_calories = sum(calories) / len(calories) if calories else 0
    
    # Last 7 days
    seven_days_ago = date.today() - timedelta(days=7)
    recent_intakes = [i for i in intakes if i.date >= seven_days_ago]
    week_avg = sum([i.calories for i in recent_intakes]) / len(recent_intakes) if recent_intakes else 0
    
    # Streak calculation
    streak = calculate_streak(intakes)
    
    # ML model status
    ml_status = "Active ✅" if ml_predictor is not None else "Fallback Mode ⚠️"
    
    return render_template('dashboard.html',
                           tdee=current_user.tdee,
                           diet_type=current_user.diet_type,
                           target_calories=current_user.target_calories,
                           bmi=current_user.bmi,
                           dates=dates,
                           calories=calories,
                           total_intakes=total_intakes,
                           avg_calories=round(avg_calories, 1),
                           week_avg=round(week_avg, 1),
                           streak=streak,
                           ml_status=ml_status)

@app.route('/diet-plan')
@login_required
def diet_plan():
    """Show personalized meal plan from dataset + ML, honoring customization"""
    if isinstance(current_user._get_current_object(), Admin):
        flash('Admins cannot view diet plans', 'warning')
        return redirect(url_for('admin_dashboard'))

    # Preferences from session or sensible defaults
    prefs = session.get('meal_preferences') or {
        'meal_frequency': 5,
        'excluded_foods': [],
        'favorite_foods': [],
        'budget': getattr(current_user, 'budget', 'medium') or 'medium',
        'include_smoothies': False,
        'include_juices': False
    }

    # Primary: generate plan via helper
    try:
        plan, totals, suggestions = generate_daily_plan(current_user, prefs)
    except Exception:
        # Minimal fallback if dataset fails
        plan = { 'daily_meals': {'Breakfast': [], 'Lunch': [], 'Dinner': [], 'Snack 1': [], 'Snack 2': [], 'Smoothie': []} }
        totals = {'calories': 0, 'protein_g': 0.0, 'carbs_g': 0.0, 'fats_g': 0.0}

    # ML recommendations based on current user data + preferences
    ml_recommendations = []
    try:
        base = current_user.to_dict() if hasattr(current_user, 'to_dict') else {}
        base.update({
            'dietary_preference': getattr(current_user, 'dietary_preference', 'none') or 'none',
            'allergies': getattr(current_user, 'allergies', 'none') or 'none',
            'budget': prefs.get('budget', 'medium'),
        })
        result = predict_diet_type_ml(base)
        ml_recommendations = result.get('recommendations', []) if isinstance(result, dict) else []
    except Exception:
        pass

    # If ML has no recommendations, derive from dataset suggestions
    if not ml_recommendations:
        try:
            # Build a set of items already in the plan to avoid duplicates
            in_plan = set()
            for lst in plan.get('daily_meals', {}).values():
                for it in lst:
                    name = it.get('food_name') if isinstance(it, dict) else None
                    if name:
                        in_plan.add(str(name).lower())
            # Use earlier suggestions if available; otherwise compute fresh
            if 'suggestions' not in locals():
                suggestions, _ = build_food_suggestions(current_user, prefs)
            recs = []
            for s in suggestions:
                name = str(s.get('food_name', '')).strip()
                if not name or name.lower() in in_plan:
                    continue
                cal = s.get('calories'); prot = s.get('protein_g'); carb = s.get('carbs_g'); fat = s.get('fats_g')
                recs.append(f"{name} — {int(cal)} kcal, P {prot:.0f}g, C {carb:.0f}g, F {fat:.0f}g")
                if len(recs) >= 6:
                    break
            ml_recommendations = recs
        except Exception:
            ml_recommendations = []

    return render_template('diet_plan.html', 
                           plan=plan,
                           totals=totals,
                           user_tdee=current_user.tdee,
                           target_calories=current_user.target_calories,
                           diet_type=(current_user.diet_type or 'maintenance'),
                           ml_recommendations=ml_recommendations,
                           allergies=current_user.allergies,
                           medical_condition=current_user.medical_condition)

@app.route('/customize-diet', methods=['GET','POST'])
@login_required
def customize_diet():
    if isinstance(current_user._get_current_object(), Admin):
        flash('Admins cannot customize diet', 'warning')
        return redirect(url_for('admin_dashboard'))

    form = MealPreferenceForm()
    suggestions = []
    sample_plan = None

    if form.validate_on_submit():
        prefs = {
            'meal_frequency': int(form.meal_frequency.data),
            'excluded_foods': [x.strip().lower() for x in (form.excluded_foods.data or '').split(',') if x.strip()],
            'favorite_foods': [x.strip().lower() for x in (form.favorite_foods.data or '').split(',') if x.strip()],
            'budget': form.budget_preference.data,
            'dietary_preference': form.dietary_preference.data,
            'health_conditions': list(form.health_conditions.data or []),
            'include_smoothies': bool(form.include_smoothies.data),
            'include_juices': bool(form.include_juices.data)
        }
        try:
            plan, totals, suggestions = generate_daily_plan(current_user, prefs)
            sample_plan = {'items': [], 'totals': totals}
            # Flatten a simple preview list from generated plan for the page
            preview = []
            for label, items in plan.get('daily_meals', {}).items():
                for it in items:
                    preview.append({'meal': label, **it})
            # Keep only first few for preview
            sample_plan['items'] = preview[:max(4, prefs.get('meal_frequency', 5))]
            flash('Preferences applied. Here are your suggested foods.', 'success')
            # Persist for diet plan
            session['meal_preferences'] = prefs
        except Exception as e:
            print(e)
            flash('Could not build suggestions from CSV data.', 'danger')

    return render_template('customize_diet.html', form=form, suggestions=suggestions, sample_plan=sample_plan)

@app.route('/profile')
@login_required
def profile():
    if isinstance(current_user._get_current_object(), Admin):
        return redirect(url_for('admin_dashboard'))
    
    # BMI category
    if current_user.bmi is not None:
        bmi = current_user.bmi
    else:
        try:
            bmi = (current_user.weight or 0) / (((current_user.height or 0) / 100) ** 2) if current_user.weight and current_user.height else 0
        except Exception:
            bmi = 0
    if bmi < 18.5:
        bmi_category = "Underweight"
        bmi_color = "info"
    elif bmi < 25:
        bmi_category = "Normal"
        bmi_color = "success"
    elif bmi < 30:
        bmi_category = "Overweight"
        bmi_color = "warning"
    else:
        bmi_category = "Obese"
        bmi_color = "danger"
    
    return render_template('profile.html', 
                           user=current_user, 
                           bmi=round(bmi, 1) if bmi is not None else 0,
                           bmi_category=bmi_category,
                           bmi_color=bmi_color,
                           ml_active=ml_predictor is not None)

@app.route('/progress')
@login_required
def progress():
    if isinstance(current_user._get_current_object(), Admin):
        return redirect(url_for('admin_dashboard'))
    
    intakes = Intake.query.filter_by(user_id=current_user.id).order_by(Intake.date).all()
    
    # Last 30 days data
    thirty_days_ago = date.today() - timedelta(days=30)
    recent = [i for i in intakes if i.date >= thirty_days_ago]
    
    dates = [i.date.strftime('%Y-%m-%d') for i in recent]
    calories = [i.calories for i in recent]
    
    # Goals
    tdee = current_user.tdee
    target = current_user.target_calories
    deficit_goal = (tdee or 0) - 500
    surplus_goal = (tdee or 0) + 300
    
    return render_template('progress.html', 
                           dates=dates, 
                           calories=calories,
                           tdee=tdee,
                           target_calories=target,
                           deficit_goal=deficit_goal,
                           surplus_goal=surplus_goal)

@app.route('/add_intake', methods=['GET','POST'])
@login_required
def add_intake():
    if isinstance(current_user._get_current_object(), Admin):
        flash('Admins cannot add intake', 'warning')
        return redirect(url_for('admin_dashboard'))
    form = IntakeForm()
    if form.validate_on_submit():
        intake_date = form.date.data if hasattr(form, 'date') and form.date.data else date.today()
        # Merge multiple entries for the same day into one record
        existing = Intake.query.filter_by(user_id=current_user.id, date=intake_date).first()
        if existing:
            existing.calories = (existing.calories or 0) + form.calories.data
            db.session.commit()
            saved_cal = int(existing.calories)
        else:
            intake = Intake(date=intake_date, calories=form.calories.data, user_id=current_user.id)
            db.session.add(intake)
            db.session.commit()
            saved_cal = int(form.calories.data)
        flash('Intake logged successfully! Keep crushing it! 💪', 'success')
        # Show confirmation on the same page with the stored value
        return render_template('add_intake.html', form=form, saved_calories=saved_cal, saved_date=intake_date.strftime('%Y-%m-%d'))
    return render_template('add_intake.html', form=form)

@app.route('/intake_list')
@login_required
def intake_list():
    if isinstance(current_user._get_current_object(), Admin):
        flash('Admins use admin dashboard', 'info')
        return redirect(url_for('admin_dashboard'))
    intakes = Intake.query.filter_by(user_id=current_user.id).order_by(Intake.date.desc()).all()

    # Today's calories
    today = date.today()
    today_calories = sum(i.calories for i in intakes if i.date == today)

    # Weekly average (last 7 days)
    seven_days_ago = today - timedelta(days=7)
    week = [i.calories for i in intakes if i.date >= seven_days_ago]
    weekly_avg = round(sum(week) / len(week), 1) if week else 0

    return render_template('intake_list.html', intakes=intakes, today_calories=today_calories, weekly_avg=weekly_avg)

@app.route('/delete_intake/<int:intake_id>', methods=['POST'])
@login_required
def delete_intake(intake_id):
    intake = Intake.query.get_or_404(intake_id)
    if intake.user_id != current_user.id and not isinstance(current_user._get_current_object(), Admin):
        flash('Not authorized', 'danger')
        return redirect(url_for('dashboard'))
    db.session.delete(intake)
    db.session.commit()
    flash('Intake deleted', 'success')
    return redirect(url_for('intake_list'))

# --------------------------------------------------------------------------------------
# Admin routes
# --------------------------------------------------------------------------------------
@app.route('/admin/dashboard')
@admin_required
def admin_dashboard():
    stats = {
        'users': User.query.count(),
        'intakes': Intake.query.count(),
        'active_today': Intake.query.filter_by(date=date.today()).count(),
        'ml_status': 'Active ✅' if ml_predictor else 'Inactive ❌'
    }
    return render_template('admin_dashboard.html', stats=stats)

@app.route('/admin/users')
@admin_required
def admin_users():
    page = request.args.get('page', 1, type=int)
    q = request.args.get('q', '', type=str)

    query = User.query
    if q:
        like = f"%{q}%"
        query = query.filter((User.username.ilike(like)) | (User.email.ilike(like)))

    pagination = query.order_by(User.id.desc()).paginate(page=page, per_page=15, error_out=False)
    users = pagination.items
    return render_template('admin_users.html', users=users, pagination=pagination, q=q)

@app.route('/admin/user/<int:user_id>')
@admin_required
def admin_view_user(user_id):
    user = User.query.get_or_404(user_id)
    intakes = Intake.query.filter_by(user_id=user.id).order_by(Intake.date.desc()).all()
    return render_template('admin_view_user.html', user=user, intakes=intakes)

# Lightweight edit form (no password fields)
class EditUserForm(FlaskForm):
    username = StringField('Username', validators=[InputRequired(), Length(min=3, max=150)])
    email = StringField('Email', validators=[InputRequired(), Email()])
    age = IntegerField('Age', validators=[InputRequired(), NumberRange(min=5, max=120)])
    weight = FloatField('Weight (kg)', validators=[InputRequired(), NumberRange(min=20, max=300)])
    height = FloatField('Height (cm)', validators=[InputRequired(), NumberRange(min=100, max=250)])
    gender = SelectField('Gender', choices=[('male', 'Male'), ('female', 'Female')], validators=[InputRequired()])
    activity_level = SelectField('Activity Level', choices=[
        ('sedentary', 'Sedentary'),
        ('light', 'Light'),
        ('moderate', 'Moderate'),
        ('active', 'Active'),
        ('very_active', 'Very Active')
    ], validators=[InputRequired()])
    submit = SubmitField('Save Changes')

@app.route('/admin/user/<int:user_id>/edit', methods=['GET','POST'])
@admin_required
def admin_edit_user(user_id):
    user = User.query.get_or_404(user_id)
    form = EditUserForm(obj=user)
    if form.validate_on_submit():
        # Update fields
        user.username = form.username.data
        user.email = form.email.data
        user.age = form.age.data
        user.weight = form.weight.data
        user.height = form.height.data
        user.gender = form.gender.data
        user.activity_level = form.activity_level.data

        # Recompute BMI/BMR/TDEE and keep existing goal/diet_type/target if present
        try:
            bmi = user.weight / ((user.height / 100) ** 2)
            if user.gender == 'male':
                bmr = 10 * user.weight + 6.25 * user.height - 5 * user.age + 5
            else:
                bmr = 10 * user.weight + 6.25 * user.height - 5 * user.age - 161
            multipliers = {'sedentary':1.2,'light':1.375,'moderate':1.55,'active':1.725,'very_active':1.9}
            tdee = bmr * multipliers.get(user.activity_level, 1.2)
            user.bmi = round(bmi, 1)
            user.bmr = round(bmr, 0)
            user.tdee = round(tdee, 0)
        except Exception:
            pass

        db.session.commit()
        flash('User updated successfully', 'success')
        return redirect(url_for('admin_view_user', user_id=user.id))

    # Extra stats for the sidebar cards in template
    user_intakes_count = Intake.query.filter_by(user_id=user.id).count()
    user_activity_days = len({i.date for i in Intake.query.filter_by(user_id=user.id).all()})
    return render_template('edit_user.html', form=form, user=user, user_intakes_count=user_intakes_count, user_activity_days=user_activity_days)

@app.route('/admin/user/<int:user_id>/delete', methods=['POST'])
@admin_required
def admin_delete_user(user_id):
    user = User.query.get_or_404(user_id)
    # Delete user and cascade deletes intakes (relationship configured)
    db.session.delete(user)
    db.session.commit()
    flash('User deleted successfully', 'success')
    return redirect(url_for('admin_users'))

@app.route('/admin/export-users')
@admin_required
def admin_export_users():
    users = User.query.order_by(User.id.asc()).all()
    # CSV header
    header = [
        'id','username','email','age','weight','height','gender','activity_level','goal',
        'diet_type','tdee','target_calories','bmi','bmr','medical_condition','allergies','dietary_preference'
    ]
    # Rows
    lines = [','.join(header)]
    for u in users:
        row = [
            str(u.id),
            str(u.username or ''),
            str(u.email or ''),
            str(u.age or ''),
            str(u.weight or ''),
            str(u.height or ''),
            str(u.gender or ''),
            str(u.activity_level or ''),
            str(u.goal or ''),
            str(u.diet_type or ''),
            str(int(u.tdee) if u.tdee is not None else ''),
            str(int(u.target_calories) if u.target_calories is not None else ''),
            str(round(u.bmi,1) if u.bmi is not None else ''),
            str(int(u.bmr) if u.bmr is not None else ''),
            str(u.medical_condition or ''),
            str(u.allergies or ''),
            str(u.dietary_preference or '')
        ]
        # Escape commas in fields if needed by wrapping in quotes
        safe_row = []
        for value in row:
            v = value
            if ',' in v or '"' in v:
                v = '"' + v.replace('"', '""') + '"'
            safe_row.append(v)
        lines.append(','.join(safe_row))
    csv_data = '\n'.join(lines)
    return Response(csv_data, mimetype='text/csv', headers={
        'Content-Disposition': 'attachment; filename=users_export.csv'
    })

@app.route('/admin/user/<int:user_id>/export-intakes')
@admin_required
def admin_export_user_intakes(user_id):
    user = User.query.get_or_404(user_id)
    intakes = Intake.query.filter_by(user_id=user.id).order_by(Intake.date.asc()).all()
    header = ['id','user_id','date','calories']
    lines = [','.join(header)]
    for i in intakes:
        row = [str(i.id), str(i.user_id), i.date.isoformat(), str(i.calories)]
        lines.append(','.join(row))
    csv_data = '\n'.join(lines)
    filename = f"user_{user.id}_intakes.csv"
    return Response(csv_data, mimetype='text/csv', headers={'Content-Disposition': f'attachment; filename={filename}'})

@app.route('/admin/intake/<int:intake_id>/delete', methods=['POST'])
@admin_required
def admin_delete_intake(intake_id):
    intake = Intake.query.get_or_404(intake_id)
    db.session.delete(intake)
    db.session.commit()
    flash('Intake deleted', 'success')
    # Try to redirect back to referring user page
    return redirect(request.referrer or url_for('admin_dashboard'))

@app.route('/admin/impersonate/<int:user_id>', methods=['POST'])
@admin_required
def admin_impersonate(user_id):
    user = User.query.get_or_404(user_id)
    login_user(user)  # switch to user session
    flash(f'You are now impersonating {user.username}', 'info')
    return redirect(url_for('dashboard'))

# --------------------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------------------
def calculate_streak(intakes):
    if not intakes:
        return 0
    # Ensure all dates are date objects
    try:
        dates = sorted([i.date if isinstance(i.date, date) else datetime.strptime(str(i.date), '%Y-%m-%d').date() for i in intakes], reverse=True)
    except Exception:
        dates = sorted([i.date for i in intakes if isinstance(i.date, date)], reverse=True)
    streak = 0
    expected_date = date.today()
    
    for intake_date in dates:
        if intake_date == expected_date:
            streak += 1
            expected_date -= timedelta(days=1)
        elif intake_date < expected_date:
            break
    
    return streak

# ---------------------- CSV-driven customization helpers ----------------------
_food_db_cache = None

def load_food_database():
    global _food_db_cache
    if _food_db_cache is None:
        base_dir = os.path.dirname(__file__)
        csv_path = os.path.join(base_dir, 'model', 'data', 'food_database.csv')
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Food database CSV not found at {csv_path}")
        df = pd.read_csv(csv_path)
        if 'food_name' not in df.columns:
            raise ValueError('food_database.csv must contain a food_name column')
        _food_db_cache = df
    return _food_db_cache

def generate_daily_plan(user, prefs):
    # Use existing suggestion builder to get ranked items and a sample plan list
    suggestions, sample_plan = build_food_suggestions(user, prefs)
    items = sample_plan.get('items') or []
    totals = sample_plan.get('totals') or {'calories': 0, 'protein_g': 0.0, 'carbs_g': 0.0, 'fats_g': 0.0}

    daily_meals = {
        'Breakfast': [], 'Lunch': [], 'Dinner': [], 'Snack 1': [], 'Snack 2': [], 'Smoothie': []
    }
    # Place items reliably
    order_labels = ['Breakfast','Lunch','Dinner','Snack 1','Snack 2','Smoothie']
    for idx, it in enumerate(items):
        mt = ''
        if isinstance(it, dict):
            mt = str(it.get('meal_type') or '').strip().lower()
        mapping = {'breakfast':'Breakfast','lunch':'Lunch','dinner':'Dinner','snack':'Snack 1','smoothie':'Smoothie'}
        label = mapping.get(mt)
        if not label:
            label = order_labels[idx] if idx < len(order_labels) else 'Snack 1'
        daily_meals[label].append(it)

    # If empty, relax prefs and retry once
    if all(len(v) == 0 for v in daily_meals.values()):
        relaxed = {
            'meal_frequency': prefs.get('meal_frequency') or 5,
            'excluded_foods': [],
            'favorite_foods': prefs.get('favorite_foods') or [],
            'budget': 'high',
            'include_smoothies': bool(prefs.get('include_smoothies')),
            'include_juices': bool(prefs.get('include_juices')),
        }
        suggestions, sample_plan = build_food_suggestions(user, relaxed)
        items = sample_plan.get('items') or []
        totals = sample_plan.get('totals') or totals
        for k in daily_meals.keys():
            daily_meals[k] = []
        for idx, it in enumerate(items):
            label = order_labels[idx] if idx < len(order_labels) else 'Snack 1'
            daily_meals[label].append(it)

    return { 'daily_meals': daily_meals }, totals, suggestions

@app.route('/api/foods')
@login_required
def api_foods():
    """Autocomplete/search foods from CSV by substring q."""
    q = (request.args.get('q') or '').strip().lower()
    # limit input length and characters
    if len(q) > 50:
        return jsonify({'error': 'query too long'}), 400
    if not re.fullmatch(r'[\w\s\-]*', q):  # letters, numbers, underscore, dash, space
        return jsonify({'error': 'invalid characters in query'}), 400
    df = load_food_database().copy()
    # Ensure numeric types
    for col in ['calories','protein_g','carbs_g','fats_g']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    if q:
        if 'food_name' not in df.columns:
            return jsonify({'results': []})
        mask = df['food_name'].str.lower().str.contains(re.escape(q), na=False)
        df = df[mask]
    results = df.head(15)[['food_name','calories','protein_g','carbs_g','fats_g']].to_dict(orient='records')
    return jsonify({'results': results})

def build_food_suggestions(user, prefs):
    # validate prefs
    prefs = prefs or {}
    meal_frequency = prefs.get('meal_frequency')
    try:
        meal_frequency = int(meal_frequency)
    except Exception:
        meal_frequency = 5
    meal_frequency = _clamp(meal_frequency, 1, 8)

    def _as_str_list(value):
        if not value:
            return []
        if isinstance(value, list):
            return [str(x).strip().lower() for x in value if str(x).strip()]
        return [str(x).strip().lower() for x in str(value).split(',') if str(x).strip()]

    prefs['excluded_foods'] = _as_str_list(prefs.get('excluded_foods'))
    prefs['favorite_foods'] = _as_str_list(prefs.get('favorite_foods'))
    prefs['include_smoothies'] = bool(prefs.get('include_smoothies'))

    df = load_food_database().copy()

    # Exclude based on allergies
    allergy = (user.allergies or 'none').lower()
    if allergy == 'dairy' and 'contains_dairy' in df.columns:
        df = df[~df['contains_dairy']]
    if allergy == 'nuts' and 'contains_nuts' in df.columns:
        df = df[~df['contains_nuts']]
    if allergy == 'gluten' and 'contains_gluten' in df.columns:
        df = df[~df['contains_gluten']]
    if allergy == 'soy' and 'contains_soy' in df.columns:
        df = df[~df['contains_soy']]

    # Respect dietary preference if set (prefs override user)
    pref = (prefs.get('dietary_preference') or user.dietary_preference or 'none').lower()
    if pref == 'vegetarian' and 'vegetarian' in df.columns:
        df = df[df['vegetarian']]
    if pref == 'vegan' and 'vegan' in df.columns:
        df = df[df['vegan']]

    # Apply health condition strict exclusions from prefs
    health = set(prefs.get('health_conditions') or [])
    if 'lactose_intolerance' in health and 'contains_dairy' in df.columns:
        df = df[~df['contains_dairy']]
    if 'nut_allergy' in health and 'contains_nuts' in df.columns:
        df = df[~df['contains_nuts']]
    if 'celiac' in health and 'contains_gluten' in df.columns:
        df = df[~df['contains_gluten']]
    if 'soy_allergy' in health and 'contains_soy' in df.columns:
        df = df[~df['contains_soy']]

    # Excluded foods
    excluded = set(prefs.get('excluded_foods') or [])
    if excluded:
        pattern = '|'.join(map(re.escape, excluded))
        df = df[~df['food_name'].str.lower().str.contains(pattern, na=False)]

    # Favor favorites (we'll bump them later)
    favorites = set(prefs.get('favorite_foods') or [])

    # Optional budget filter if available
    budget_pref = (prefs.get('budget') or 'medium').strip().lower()
    if 'cost_level' in df.columns:
        allowed_costs = {
            'low': {'low'},
            'medium': {'low','medium'},
            'high': {'low','medium','high'}
        }.get(budget_pref, {'low','medium'})
        df = df[df['cost_level'].astype(str).str.lower().isin(allowed_costs)]

    # Strategy based on goal
    goal = (user.goal or 'maintenance').lower()
    target = float(user.target_calories or user.tdee or 2000)

    # Ensure numeric fields
    for col in ['calories','protein_g','carbs_g','fats_g']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    def score_row(row):
        score = 0.0
        # Favor items depending on goal
        if goal == 'muscle_gain':
            score += row.get('protein_g', 0) * 2 - row.get('fats_g', 0) * 0.2
        elif 'weight_loss' in goal:
            score += -row.get('calories', 0) * 0.02 + row.get('protein_g', 0) * 0.8
        elif goal == 'endurance':
            score += row.get('carbs_g', 0) * 1.2 + row.get('protein_g', 0) * 0.2
        else:
            score += row.get('protein_g', 0) * 0.6 + row.get('carbs_g', 0) * 0.4

        # Favor favorites
        if any(fav in str(row['food_name']).lower() for fav in favorites):
            score += 5
        # Health-adjusted scoring
        if 'diabetes' in (prefs.get('health_conditions') or []):
            if pd.notna(row.get('carbs_g')):
                score += -0.02 * float(row.get('carbs_g') or 0)
        if 'hypertension' in (prefs.get('health_conditions') or []) or 'heart_disease' in (prefs.get('health_conditions') or []):
            if 'sodium_mg' in row and pd.notna(row.get('sodium_mg')):
                score += -0.01 * float(row.get('sodium_mg') or 0)
        return score

    df['score'] = df.apply(score_row, axis=1)
    top = df.sort_values('score', ascending=False).head(20)

    suggestions = [
        {
            'food_name': r['food_name'],
            'calories': int(r.get('calories', 0)),
            'protein_g': float(r.get('protein_g', 0)),
            'carbs_g': float(r.get('carbs_g', 0)),
            'fats_g': float(r.get('fats_g', 0))
        }
        for _, r in top.iterrows()
    ]

    # Build sample day plan honoring meal_frequency and smoothies
    slots = ['Breakfast','Lunch','Dinner']
    if meal_frequency >= 4:
        slots.append('Snack 1')
    if meal_frequency >= 5:
        slots.append('Snack 2')
    if meal_frequency >= 6:
        slots.append('Smoothie' if prefs.get('include_smoothies') else 'Snack 3')

    selected_frames = []
    used_names = set()

    if 'meal_type' in top.columns:
        def take_first(mt):
            pool = top[(top['meal_type'].astype(str).str.lower() == mt) & (~top['food_name'].isin(used_names))]
            return pool.head(1)
        mapping = {
            'Breakfast': 'breakfast',
            'Lunch': 'lunch',
            'Dinner': 'dinner',
            'Snack 1': 'snack',
            'Snack 2': 'snack',
            'Snack 3': 'snack',
            'Smoothie': 'smoothie'
        }
        for slot in slots:
            mt = mapping.get(slot, 'snack')
            pick = take_first(mt)
            if pick.empty and mt == 'snack':
                # fallback to any remaining
                pick = top[~top['food_name'].isin(used_names)].head(1)
            if not pick.empty:
                used_names.add(str(pick.iloc[0]['food_name']))
                selected_frames.append(pick)
        selected = pd.concat(selected_frames) if selected_frames else top.head(len(slots))
    else:
        def pick_by_keywords(candidates, keywords, limit):
            mask = candidates['food_name'].str.lower().str.contains('|'.join(map(re.escape, keywords)), na=False)
            return candidates[mask]
        keywords = {
            'Breakfast': ['oat','omelet','egg','pancake','yogurt','chia','oatmeal'],
            'Lunch': ['rice','bowl','salad','wrap','pasta','upma','chilla','sandwich'],
            'Dinner': ['grilled','tikka','curry','bowl','stir-fry','tacos'],
            'Smoothie': ['smoothie'],
            'Snack 1': ['bar','nuts','sprouts','hummus','roasted','fruit'],
            'Snack 2': ['bar','nuts','sprouts','hummus','roasted','fruit'],
            'Snack 3': ['bar','nuts','sprouts','hummus','roasted','fruit']
        }
        picks = []
        pool = top.copy()
        for slot in slots:
            kw = keywords.get(slot, [])
            cand = pick_by_keywords(pool, kw, 1)
            if cand.empty:
                cand = pool.head(1)
            if not cand.empty:
                name = str(cand.iloc[0]['food_name'])
                pool = pool[pool['food_name'] != name]
                picks.append(cand.head(1))
        selected = pd.concat(picks) if picks else top.head(len(slots))
    if selected.empty:
        selected = top.head(5)

    plan_items = [
        {
            'food_name': r['food_name'],
            'calories': int(r['calories']),
            'protein_g': float(r['protein_g']),
            'carbs_g': float(r['carbs_g']),
            'fats_g': float(r['fats_g'])
        }
        for _, r in selected.iterrows()
    ]
    totals = {
        'calories': int(sum(x['calories'] for x in plan_items)),
        'protein_g': round(sum(x['protein_g'] for x in plan_items), 1),
        'carbs_g': round(sum(x['carbs_g'] for x in plan_items), 1),
        'fats_g': round(sum(x['fats_g'] for x in plan_items), 1)
    }
    sample_plan = {
        'items': plan_items,
        'totals': totals,
        'target': int(target)
    }

    return suggestions, sample_plan

# --------------------------------------------------------------------------------------
# Utility
# --------------------------------------------------------------------------------------
@app.route('/create_admin')
def create_admin():
    if Admin.query.filter_by(username='admin').first():
        return 'Admin exists'
    admin = Admin(username='admin', password=generate_password_hash('adminpass'))
    db.session.add(admin)
    db.session.commit()
    return 'Admin created (username: admin, password: adminpass)'

if __name__ == '__main__':
    app.run(debug=True)