from flask import Blueprint, render_template, jsonify, abort
from flask_login import login_required, current_user
from database.db import db, Scan
from sqlalchemy import func
from datetime import datetime, timedelta

analytics_bp = Blueprint('analytics', __name__)

@analytics_bp.route('/analytics')
@login_required
def analytics_page():
    if current_user.is_admin:
        abort(403)
    return render_template('analytics.html')

@analytics_bp.route('/api/analytics-data')
@login_required
def analytics_data():
    if current_user.is_admin:
        return jsonify({"error": "Unauthorized"}), 403
        
    # Disease Distribution (Top 5)
    disease_counts = db.session.query(Scan.disease_name, func.count(Scan.id)).\
        filter_by(user_id=current_user.id).\
        group_by(Scan.disease_name).\
        order_by(func.count(Scan.id).desc()).limit(5).all()
        
    disease_labels = [row[0] for row in disease_counts]
    disease_data = [row[1] for row in disease_counts]
    
    # Severity Distribution
    severity_counts = db.session.query(Scan.severity, func.count(Scan.id)).\
        filter_by(user_id=current_user.id).\
        group_by(Scan.severity).all()
        
    severity_dict = {row[0]: row[1] for row in severity_counts}
    severity_labels = ['Healthy', 'Mild', 'Moderate', 'Severe']
    severity_data = [severity_dict.get(label, 0) for label in severity_labels]
    
    # Scans over last 7 days
    today = datetime.utcnow().date()
    dates = [(today - timedelta(days=i)) for i in range(6, -1, -1)]
    date_labels = [d.strftime('%b %d') for d in dates]
    
    # SQLite friendly date grouping
    daily_scans = []
    for d in dates:
        # Very simple count for each day
        count = Scan.query.filter(
            Scan.user_id == current_user.id,
            func.date(Scan.timestamp) == d
        ).count()
        daily_scans.append(count)
        
    return jsonify({
        "disease_distribution": {
            "labels": disease_labels,
            "data": disease_data
        },
        "severity_distribution": {
            "labels": severity_labels,
            "data": severity_data
        },
        "trend_data": {
            "labels": date_labels,
            "data": daily_scans
        }
    })
