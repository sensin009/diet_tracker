# forms_customization.py - Forms for meal customization

from flask_wtf import FlaskForm
from wtforms import StringField, SelectField, TextAreaField, IntegerField, BooleanField, SubmitField, SelectMultipleField
from wtforms.validators import Optional, NumberRange

class MealPreferenceForm(FlaskForm):
    """Form for customizing meal preferences"""
    
    # Meal frequency
    meal_frequency = SelectField('Meals Per Day',
                                choices=[
                                    ('3', '3 meals - Traditional'),
                                    ('4', '4 meals - Balanced'),
                                    ('5', '5 meals - Recommended'),
                                    ('6', '6 meals - Muscle gain/Diabetic')
                                ],
                                default='5')
    
    # Foods to exclude
    excluded_foods = TextAreaField('Foods to Exclude (comma separated)',
                                  description='e.g., Chicken, Rice, Milk',
                                  validators=[Optional()])
    
    # Favorite foods
    favorite_foods = TextAreaField('Favorite Foods (comma separated)',
                                  description='e.g., Paneer, Oats, Banana',
                                  validators=[Optional()])
    
    # Cooking time preference
    max_cooking_time = SelectField('Maximum Cooking Time',
                                  choices=[
                                      ('quick', '⚡ Under 15 minutes'),
                                      ('medium', '🕐 15-30 minutes'),
                                      ('long', '🕑 30+ minutes - I enjoy cooking')
                                  ],
                                  default='medium')
    
    # Budget preference
    budget_preference = SelectField('Budget Per Day',
                                   choices=[
                                       ('low', '$ Budget-friendly (₹100-200)'),
                                       ('medium', '$$ Moderate (₹200-400)'),
                                       ('high', '$$$ Premium (₹400+)')
                                   ],
                                   default='medium')

    # Dietary type
    dietary_preference = SelectField('Dietary Preference',
                                     choices=[
                                         ('none', 'No restriction'),
                                         ('vegetarian', 'Vegetarian'),
                                         ('vegan', 'Vegan'),
                                         ('non_vegetarian', 'Non-Vegetarian')
                                     ],
                                     default='none')
    
    # Cuisine preference
    cuisine_preference = SelectMultipleField('Preferred Cuisines',
                                            choices=[
                                                ('indian', 'Indian'),
                                                ('western', 'Western'),
                                                ('continental', 'Continental'),
                                                ('asian', 'Asian'),
                                                ('mediterranean', 'Mediterranean')
                                            ])
    
    # Spice level
    spice_level = SelectField('Spice Level',
                             choices=[
                                 ('no_spice', 'No Spice - Bland'),
                                 ('mild', 'Mild'),
                                 ('medium', 'Medium'),
                                 ('hot', 'Hot - I love spicy!')
                             ],
                             default='medium')

    # Health conditions
    health_conditions = SelectMultipleField('Health Conditions',
                                            choices=[
                                                ('diabetes', 'Diabetes'),
                                                ('hypertension', 'Hypertension'),
                                                ('heart_disease', 'Heart Disease'),
                                                ('celiac', 'Celiac / Gluten-sensitive'),
                                                ('lactose_intolerance', 'Lactose Intolerance'),
                                                ('nut_allergy', 'Nut Allergy'),
                                                ('soy_allergy', 'Soy Allergy')
                                            ])
    
    # Meal timing preferences
    breakfast_time = SelectField('Breakfast Time',
                                choices=[
                                    ('early', 'Early (6-7 AM)'),
                                    ('normal', 'Normal (7-9 AM)'),
                                    ('late', 'Late (9-11 AM)'),
                                    ('skip', 'I skip breakfast')
                                ],
                                default='normal')
    
    # Snack preferences
    prefer_snacks = BooleanField('I like snacks between meals', default=True)
    
    # Liquid calories
    include_smoothies = BooleanField('Include smoothies/shakes', default=True)
    include_juices = BooleanField('Include fresh juices', default=True)
    
    # Special requests
    special_notes = TextAreaField('Special Dietary Notes',
                                 description='Any other preferences or requirements',
                                 validators=[Optional()])
    
    submit = SubmitField('Save Preferences')

class QuickMealSwapForm(FlaskForm):
    """Quick form to swap a meal"""
    meal_id = IntegerField('Meal ID', validators=[Optional()])
    reason = SelectField('Why swap this meal?',
                        choices=[
                            ('dont_like', "I don't like these foods"),
                            ('allergic', "I'm allergic to something"),
                            ('not_available', "Ingredients not available"),
                            ('too_expensive', "Too expensive"),
                            ('takes_too_long', "Takes too long to prepare"),
                            ('just_different', "Just want something different")
                        ])
    submit = SubmitField('Get Alternative Meal')    