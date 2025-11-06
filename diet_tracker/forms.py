# forms.py - Enhanced forms with allergy and medical information

from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, IntegerField, FloatField, SelectField, DateField, SelectMultipleField
from wtforms.validators import InputRequired, Length, EqualTo, Email, NumberRange, Optional

class RegisterForm(FlaskForm):
    """Enhanced registration form with health information"""
    # Basic Info
    username = StringField('Username', validators=[InputRequired(), Length(min=3, max=150)])
    email = StringField('Email', validators=[InputRequired(), Email()])
    password = PasswordField('Password', validators=[InputRequired(), Length(min=6)])
    confirm = PasswordField('Confirm Password', validators=[EqualTo('password', message='Passwords must match')])
    
    # Physical Stats
    age = IntegerField('Age', validators=[InputRequired(), NumberRange(min=5, max=120)])
    weight = FloatField('Weight (kg)', validators=[InputRequired(), NumberRange(min=20, max=300)])
    height = FloatField('Height (cm)', validators=[InputRequired(), NumberRange(min=100, max=250)])
    gender = SelectField('Gender', 
                        choices=[('male', 'Male'), ('female', 'Female')], 
                        validators=[InputRequired()])
    
    # Activity & Goal
    activity_level = SelectField('Activity Level', 
                                choices=[
                                    ('sedentary', 'Sedentary - Little or no exercise'),
                                    ('light', 'Light - Exercise 1-3 days/week'),
                                    ('moderate', 'Moderate - Exercise 3-5 days/week'),
                                    ('active', 'Active - Exercise 6-7 days/week'),
                                    ('very_active', 'Very Active - Physical job or 2x training/day')
                                ], 
                                validators=[InputRequired()])
    
    goal = SelectField('Fitness Goal', 
                      choices=[
                          ('weight_loss', '🔥 Weight Loss - Lose fat and get lean'),
                          ('muscle_gain', '💪 Muscle Gain - Build muscle mass'),
                          ('maintenance', '⚖️ Maintenance - Stay at current weight'),
                          ('endurance', '🏃 Endurance - Train for stamina')
                      ], 
                      validators=[InputRequired()])
    
    # NEW: Medical & Health Information
    medical_condition = SelectField('Medical Condition', 
                                   choices=[
                                       ('none', 'None'),
                                       ('diabetes', 'Diabetes'),
                                       ('hypertension', 'Hypertension / High Blood Pressure'),
                                       ('heart_disease', 'Heart Disease'),
                                   ],
                                   default='none')
    
    allergies = SelectField('Food Allergies', 
                           choices=[
                               ('none', 'No Allergies'),
                               ('dairy', 'Dairy (Milk, Cheese, Yogurt)'),
                               ('nuts', 'Nuts (Peanuts, Almonds, etc.)'),
                               ('gluten', 'Gluten (Wheat, Bread)'),
                               ('shellfish', 'Shellfish'),
                               ('eggs', 'Eggs'),
                               ('soy', 'Soy'),
                               ('fish', 'Fish')
                           ],
                           default='none')
    
    dietary_preference = SelectField('Dietary Preference', 
                                    choices=[
                                        ('none', 'No Preference'),
                                        ('vegetarian', 'Vegetarian'),
                                        ('vegan', 'Vegan'),
                                        ('keto', 'Ketogenic'),
                                        ('paleo', 'Paleo')
                                    ],
                                    default='none')
    
    # NEW: Lifestyle Factors
    budget = SelectField('Food Budget', 
                        choices=[
                            ('low', '$ Low Budget'),
                            ('medium', '$$ Medium Budget'),
                            ('high', '$$$ High Budget')
                        ],
                        default='medium')
    
    cooking_time = SelectField('Available Cooking Time', 
                              choices=[
                                  ('quick', '⚡ Quick (< 15 min)'),
                                  ('medium', '🕐 Medium (15-30 min)'),
                                  ('long', '🕑 Long (> 30 min)')
                              ],
                              default='medium')
    
    metabolism_rate = SelectField('Metabolism Rate', 
                                 choices=[
                                     ('slow', 'Slow - Gain weight easily'),
                                     ('normal', 'Normal - Average'),
                                     ('fast', 'Fast - Hard to gain weight')
                                 ],
                                 default='normal')
    
    sleep_hours = FloatField('Average Sleep (hours/night)', 
                            validators=[Optional(), NumberRange(min=3, max=12)],
                            default=7.0)
    
    stress_level = SelectField('Stress Level', 
                              choices=[
                                  ('low', 'Low Stress'),
                                  ('medium', 'Medium Stress'),
                                  ('high', 'High Stress')
                              ],
                              default='medium')
    
    submit = SubmitField('Create Account')

class LoginForm(FlaskForm):
    """Login form for users and admins"""
    username = StringField('Username', validators=[InputRequired()])
    password = PasswordField('Password', validators=[InputRequired()])
    submit = SubmitField('Login')

class IntakeForm(FlaskForm):
    """Form for logging daily calorie intake"""
    date = DateField('Date (optional)', 
                    format='%Y-%m-%d', 
                    validators=[Optional()], 
                    default=None)
    calories = IntegerField('Calories', 
                           validators=[InputRequired(), NumberRange(min=0, max=10000)])
    submit = SubmitField('Add Intake')