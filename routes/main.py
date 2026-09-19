from flask import Blueprint, render_template, request, flash, redirect, url_for
from database.db import db, ContactMessage

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    return render_template('index.html')

@main_bp.route('/about')
def about():
    return render_template('about.html')

@main_bp.route('/how-it-works')
def how_it_works():
    return render_template('how_it_works.html')

@main_bp.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        subject = request.form.get('subject', '').strip()
        message_text = request.form.get('message', '').strip()
        
        if not name or not email or not subject or not message_text:
            flash('Please fill in all fields.', 'danger')
            return render_template('contact.html')
            
        msg = ContactMessage(name=name, email=email, subject=subject, message=message_text)
        db.session.add(msg)
        db.session.commit()
        
        flash('Your message has been sent successfully! We will get back to you soon.', 'success')
        return redirect(url_for('main.contact'))
        
    return render_template('contact.html')
