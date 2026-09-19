from flask import Blueprint, render_template, redirect, url_for, flash, abort, request
from flask_login import login_required, current_user
from database.db import db, User, Scan, ContactMessage

admin_bp = Blueprint('admin', __name__)

@admin_bp.before_request
@login_required
def require_admin():
    if not current_user.is_admin:
        abort(403)

@admin_bp.route('/admin')
def admin_panel():
    total_users = User.query.count()
    total_scans = Scan.query.count()
    total_messages = ContactMessage.query.count()
    
    users = User.query.order_by(User.created_at.desc()).all()
    recent_scans = Scan.query.order_by(Scan.timestamp.desc()).limit(10).all()
    messages = ContactMessage.query.order_by(ContactMessage.timestamp.desc()).limit(10).all()
    
    return render_template('admin.html', 
                           total_users=total_users, 
                           total_scans=total_scans, 
                           total_messages=total_messages,
                           users=users,
                           recent_scans=recent_scans,
                           messages=messages)

@admin_bp.route('/admin/toggle-admin/<int:user_id>', methods=['POST'])
def toggle_admin(user_id):
    user = User.query.get_or_404(user_id)
    if user.id == current_user.id:
        flash('You cannot change your own admin status.', 'warning')
    else:
        user.role = 'farmer' if user.is_admin else 'admin'
        db.session.commit()
        flash(f"User {user.username} role updated to {user.role}.", 'success')
    return redirect(url_for('admin.admin_panel'))

@admin_bp.route('/admin/delete-user/<int:user_id>', methods=['POST'])
def delete_user(user_id):
    user = User.query.get_or_404(user_id)
    if user.id == current_user.id:
        flash('You cannot delete yourself.', 'warning')
    else:
        db.session.delete(user)
        db.session.commit()
        flash(f"User {user.username} deleted.", 'success')
    return redirect(url_for('admin.admin_panel'))

@admin_bp.route('/admin/mark-read/<int:msg_id>', methods=['POST'])
def mark_read(msg_id):
    msg = ContactMessage.query.get_or_404(msg_id)
    msg.is_read = True
    db.session.commit()
    return redirect(url_for('admin.admin_panel'))
