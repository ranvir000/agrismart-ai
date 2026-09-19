import io
import csv
from flask import Blueprint, render_template, redirect, url_for, flash, request, Response, abort
from flask_login import login_required, current_user
from database.db import db, Scan

history_bp = Blueprint('history', __name__)

@history_bp.route('/history')
@login_required
def history_page():
    if current_user.is_admin:
        abort(403)
        
    page = request.args.get('page', 1, type=int)
    disease_filter = request.args.get('disease', '')
    
    query = Scan.query.filter_by(user_id=current_user.id)
    if disease_filter:
        query = query.filter(Scan.disease_key.contains(disease_filter))
        
    scans = query.order_by(Scan.timestamp.desc()).paginate(page=page, per_page=10, error_out=False)
    
    return render_template('history.html', scans=scans, disease_filter=disease_filter)


@history_bp.route('/history/delete/<int:scan_id>', methods=['POST'])
@login_required
def delete_scan(scan_id):
    scan = Scan.query.get_or_404(scan_id)
    if scan.user_id != current_user.id:
        abort(403)
        
    db.session.delete(scan)
    db.session.commit()
    flash('Scan record deleted.', 'success')
    return redirect(url_for('history.history_page'))


@history_bp.route('/history/export')
@login_required
def export_csv():
    scans = Scan.query.filter_by(user_id=current_user.id).order_by(Scan.timestamp.desc()).all()
    
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['ID', 'Date', 'Crop', 'Disease', 'Severity', 'Confidence'])
    
    for scan in scans:
        writer.writerow([
            scan.id, 
            scan.timestamp.strftime('%Y-%m-%d %H:%M'),
            scan.crop_type,
            scan.disease_name,
            scan.severity,
            f"{scan.confidence:.2f}"
        ])
        
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=agrismart_history.csv"}
    )
