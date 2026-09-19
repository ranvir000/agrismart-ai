from flask import Blueprint, render_template, redirect, url_for
from flask_login import login_required, current_user
from database.db import Scan
from models.model_loader import get_weather, get_weather_advisory

dashboard_bp = Blueprint('dashboard', __name__)

@dashboard_bp.route('/dashboard')
@login_required
def dashboard():
    if current_user.is_admin:
        return redirect(url_for('admin.admin_panel'))
        
    recent_scans = Scan.query.filter_by(user_id=current_user.id).order_by(Scan.timestamp.desc()).limit(5).all()
    weather_data = get_weather(current_user.city or "New Delhi")
    advisories = get_weather_advisory(weather_data, recent_scans)
    
    total_scans = Scan.query.filter_by(user_id=current_user.id).count()
    healthy_scans = Scan.query.filter_by(user_id=current_user.id, severity='Healthy').count()
    
    return render_template('dashboard.html', 
                           recent_scans=recent_scans,
                           weather=weather_data,
                           advisories=advisories,
                           total_scans=total_scans,
                           healthy_scans=healthy_scans)

@dashboard_bp.route('/weather')
@login_required
def weather():
    weather_data = get_weather(current_user.city or "New Delhi")
    recent_scans = Scan.query.filter_by(user_id=current_user.id).order_by(Scan.timestamp.desc()).limit(3).all()
    advisories = get_weather_advisory(weather_data, recent_scans)
    return render_template('weather.html', weather=weather_data, advisories=advisories)
