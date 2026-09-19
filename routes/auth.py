from flask import Blueprint, render_template, redirect, url_for, flash, request
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.security import check_password_hash, generate_password_hash
from database.db import db, User

auth_bp = Blueprint('auth', __name__)


@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard.dashboard'))

    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        remember = request.form.get('remember') == 'on'

        if not email or not password:
            flash('Please fill in all fields.', 'danger')
            return render_template('login.html')

        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password_hash, password):
            login_user(user, remember=remember)
            flash(f'Welcome back, {user.username}!', 'success')
            next_page = request.args.get('next')
            if next_page:
                return redirect(next_page)
            if user.is_admin:
                return redirect(url_for('admin.admin_panel'))
            return redirect(url_for('dashboard.dashboard'))
        else:
            flash('Invalid email or password. Please try again.', 'danger')

    return render_template('login.html')


@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard.dashboard'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')
        city = request.form.get('city', 'New Delhi').strip()
        farm_name = request.form.get('farm_name', '').strip()

        # Validation
        if not username or not email or not password:
            flash('All required fields must be filled.', 'danger')
            return render_template('register.html')

        if len(password) < 6:
            flash('Password must be at least 6 characters.', 'danger')
            return render_template('register.html')

        if password != confirm_password:
            flash('Passwords do not match.', 'danger')
            return render_template('register.html')

        if User.query.filter_by(email=email).first():
            flash('An account with this email already exists.', 'danger')
            return render_template('register.html')

        if User.query.filter_by(username=username).first():
            flash('This username is already taken. Please choose another.', 'danger')
            return render_template('register.html')

        user = User(
            username=username,
            email=email,
            password_hash=generate_password_hash(password),
            city=city or 'New Delhi',
            farm_name=farm_name,
            role='farmer'
        )
        db.session.add(user)
        db.session.commit()
        login_user(user)
        flash(f'Welcome to AgriSmart, {username}! Your account has been created.', 'success')
        return redirect(url_for('dashboard.dashboard'))

    return render_template('register.html')


@auth_bp.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out successfully.', 'info')
    return redirect(url_for('main.index'))


@auth_bp.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        # In production: send actual reset email
        # For demo: always show success message (security best practice)
        flash('If an account with that email exists, a password reset link has been sent. Please check your inbox.', 'info')
        return redirect(url_for('auth.login'))
    return render_template('forgot_password.html')
