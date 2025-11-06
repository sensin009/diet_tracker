# models.py - Enhanced database models with health information

from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime

db = SQLAlchemy()

class User(UserMixin, db.Model):
    __tablename__ = 'user'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    
    # Physical Stats
    age = db.Column(db.Integer, nullable=False)
    weight = db.Column(db.Float, nullable=False)
    height = db.Column(db.Float, nullable=False)
    gender = db.Column(db.String(10), nullable=False)
    
    # Activity & Goals
    activity_level = db.Column(db.String(50), nullable=False)
    goal = db.Column(db.String(50))
    
    # NEW: Health Information
    medical_condition = db.Column(db.String(100), default='none')
    allergies = db.Column(db.String(100), default='none')
    dietary_preference = db.Column(db.String(50), default='none')
    
    # NEW: Lifestyle Factors
    budget = db.Column(db.String(20), default='medium')
    cooking_time = db.Column(db.String(20), default='medium')
    metabolism_rate = db.Column(db.String(20), default='normal')
    sleep_hours = db.Column(db.Float, default=7.0)
    stress_level = db.Column(db.String(20), default='medium')
    
    # Calculated Fields
    tdee = db.Column(db.Float)
    bmi = db.Column(db.Float)
    bmr = db.Column(db.Float)
    diet_type = db.Column(db.String(100))
    target_calories = db.Column(db.Float)
    
    # Relationships
    intakes = db.relationship('Intake', backref='user', lazy=True, cascade='all, delete-orphan')
    
    def get_id(self):
        return f"user_{self.id}"
    
    def to_dict(self):
        """Convert user to dictionary for ML prediction"""
        return {
            'age': self.age,
            'weight': self.weight,
            'height': self.height,
            'gender': self.gender,
            'activity_level': self.activity_level,
            'goal': self.goal,
            'medical_condition': self.medical_condition or 'none',
            'allergies': self.allergies or 'none',
            'dietary_preference': self.dietary_preference or 'none',
            'budget': self.budget or 'medium',
            'cooking_time': self.cooking_time or 'medium',
            'metabolism_rate': self.metabolism_rate or 'normal',
            'sleep_hours': self.sleep_hours or 7.0,
            'stress_level': self.stress_level or 'medium'
        }

class Intake(db.Model):
    __tablename__ = 'intake'
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.Date, nullable=False, default=datetime.utcnow)
    calories = db.Column(db.Integer, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

class Admin(UserMixin, db.Model):
    __tablename__ = 'admin'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    
    def get_id(self):
        return f"admin_{self.id}"