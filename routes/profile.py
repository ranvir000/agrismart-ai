from flask import Blueprint, render_template, request, flash, redirect, url_for
from flask_login import login_required, current_user
from werkzeug.security import check_password_hash, generate_password_hash
from database.db import db, User, Scan

profile_bp = Blueprint('profile', __name__)

@profile_bp.route('/profile', methods=['GET', 'POST'])
@login_required
def profile_page():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        city = request.form.get('city', '').strip()
        state = request.form.get('state', '').strip()
        farm_name = request.form.get('farm_name', '').strip()
        phone = request.form.get('phone', '').strip()
        
        # Password update
        current_pwd = request.form.get('current_password', '')
        new_pwd = request.form.get('new_password', '')
        
        if username and username != current_user.username:
            # usernames are UNIQUE in the database: without this check a rename would
            # crash the page with a 500 the moment another account already uses it.
            taken = User.query.filter(User.username == username, User.id != current_user.id).first()
            if taken:
                flash('That username is already taken. Please choose another.', 'danger')
                return redirect(url_for('profile.profile_page'))
            current_user.username = username
            
        current_user.city = city
        current_user.state = state
        current_user.farm_name = farm_name
        current_user.phone = phone
        
        if current_pwd and new_pwd:
            if check_password_hash(current_user.password_hash, current_pwd):
                current_user.password_hash = generate_password_hash(new_pwd)
                flash('Profile and password updated successfully.', 'success')
            else:
                flash('Incorrect current password.', 'danger')
                return redirect(url_for('profile.profile_page'))
        else:
            flash('Profile updated successfully.', 'success')
            
        db.session.commit()
        return redirect(url_for('profile.profile_page'))
        
    total_scans = Scan.query.filter_by(user_id=current_user.id).count()
    healthy_scans = Scan.query.filter_by(user_id=current_user.id, severity='Healthy').count()
    return render_template(
        'profile.html', total_scans=total_scans, healthy_scans=healthy_scans
    )
