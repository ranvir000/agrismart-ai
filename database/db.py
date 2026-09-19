from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime

db = SQLAlchemy()

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    role = db.Column(db.String(20), default='farmer')  # 'farmer' or 'admin'
    city = db.Column(db.String(100), default='New Delhi')
    state = db.Column(db.String(100), default='Delhi')
    phone = db.Column(db.String(20), nullable=True)
    farm_name = db.Column(db.String(150), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    scans = db.relationship('Scan', backref='user', lazy=True, cascade='all, delete-orphan')

    @property
    def is_admin(self):
        return self.role == 'admin'

    @property
    def is_farmer(self):
        return self.role == 'farmer'


class Scan(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    image_filename = db.Column(db.String(255), nullable=False)
    crop_type = db.Column(db.String(100), nullable=True)
    disease_key = db.Column(db.String(100), nullable=False)
    disease_name = db.Column(db.String(200), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    severity = db.Column(db.String(20), nullable=False)  # 'Healthy', 'Mild', 'Moderate', 'Severe'
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    notes = db.Column(db.Text, nullable=True)


class ContactMessage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), nullable=False)
    subject = db.Column(db.String(200), nullable=False)
    message = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    is_read = db.Column(db.Boolean, default=False)
