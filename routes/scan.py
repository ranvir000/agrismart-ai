import os
from werkzeug.utils import secure_filename
from flask import Blueprint, render_template, request, flash, redirect, url_for, current_app, abort
from flask_login import login_required, current_user
from database.db import db, Scan
from models.model_loader import analyze_image, allowed_file, UPLOAD_FOLDER
from models.disease_info import get_disease

scan_bp = Blueprint('scan', __name__)

@scan_bp.route('/scan', methods=['GET', 'POST'])
@login_required
def scan_crop():
    if current_user.is_admin:
        abort(403)  # Admins cannot scan crops

    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part', 'danger')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No selected file', 'danger')
            return redirect(request.url)
            
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # Create a unique filename to avoid overwrites
            import uuid
            unique_filename = f"{uuid.uuid4().hex}_{filename}"
            
            # Ensure upload directory exists
            os.makedirs(UPLOAD_FOLDER, exist_ok=True)
            filepath = os.path.join(UPLOAD_FOLDER, unique_filename)
            file.save(filepath)
            
            # Analyze image
            disease_key, confidence, severity = analyze_image(filepath)
            disease_info = get_disease(disease_key)
            
            # Save to database
            scan = Scan(
                user_id=current_user.id,
                image_filename=unique_filename,
                crop_type=disease_info['crop'],
                disease_key=disease_key,
                disease_name=disease_info['name'],
                confidence=confidence,
                severity=severity
            )
            db.session.add(scan)
            db.session.commit()
            
            return redirect(url_for('scan.scan_results', scan_id=scan.id))
            
        else:
            flash('Allowed image types are -> png, jpg, jpeg, gif, webp', 'danger')
            return redirect(request.url)

    return render_template('scan.html')

@scan_bp.route('/scan/results/<int:scan_id>')
@login_required
def scan_results(scan_id):
    scan = Scan.query.get_or_404(scan_id)
    if scan.user_id != current_user.id:
        abort(403)
        
    disease_info = get_disease(scan.disease_key)
    return render_template('scan_results.html', scan=scan, disease_info=disease_info)

@scan_bp.route('/scan/download-pdf/<int:scan_id>')
@login_required
def download_pdf(scan_id):
    scan = Scan.query.get_or_404(scan_id)
    if scan.user_id != current_user.id:
        abort(403)
        
    from fpdf import FPDF
    import io
    from flask import send_file
    
    disease_info = get_disease(scan.disease_key)
    
    def clean_text(s):
        if not s: return ""
        replacements = {
            '—': '-', '–': '-', '“': '"', '”': '"', "’": "'", "‘": "'",
            '•': '-', '…': '...'
        }
        for k, v in replacements.items():
            s = s.replace(k, v)
        return s.encode('latin-1', 'replace').decode('latin-1')

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("helvetica", style='B', size=16)
    pdf.cell(0, 10, text="AgriSmart AI - Crop Health Report", new_x="LMARGIN", new_y="NEXT", align='C')
    pdf.ln(5)
    
    pdf.set_font("helvetica", size=11)
    pdf.cell(0, 8, text=clean_text(f"Date: {scan.timestamp.strftime('%Y-%m-%d %H:%M')}"), new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, text=clean_text(f"Crop: {scan.crop_type}"), new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, text=clean_text(f"Detected Disease: {scan.disease_name}"), new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, text=clean_text(f"Severity: {scan.severity}"), new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, text=clean_text(f"Confidence: {scan.confidence * 100:.1f}%"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)
    
    pdf.set_font("helvetica", style='B', size=12)
    pdf.cell(0, 8, text="Recommendations & Treatment Plan:", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", size=10)
    for t in disease_info.get('treatment', []):
        pdf.multi_cell(0, 7, text=clean_text(f"- {t}"), new_x="LMARGIN", new_y="NEXT")
        
    pdf_bytes = pdf.output()
    buffer = io.BytesIO(pdf_bytes)
    
    return send_file(
        buffer,
        as_attachment=True,
        download_name=f"Agrismart_Report_{scan.id}.pdf",
        mimetype='application/pdf'
    )
