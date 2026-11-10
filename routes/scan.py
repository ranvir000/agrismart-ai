import base64
import json
import os
from flask import Blueprint, render_template, request, flash, redirect, url_for, current_app, abort, Response, send_from_directory
from flask_login import login_required, current_user
from database.db import db, Scan
from models.model_loader import (
    analyze_image, allowed_file, normalize_upload, ImageFormatError,
    SUPPORTED_FORMATS_LABEL, LEGACY_UPLOAD_FOLDER,
)
from models.disease_info import get_disease

scan_bp = Blueprint('scan', __name__)

# Below this model score the app abstains instead of naming a disease.
# This is not a calibrated probability or a reliable unknown-crop detector.
MIN_CONFIDENCE = float(os.environ.get('MIN_CONFIDENCE', '0.60'))

def _prediction_candidates(prediction):
    """Return the CNN's top candidates, clearly labelled as unverified suggestions."""
    candidates = []
    seen = set()
    if not prediction or not prediction.get('ok'):
        return candidates
    options = [prediction] + list(prediction.get('alternatives') or [])
    for option in options:
        key = option.get('key') or option.get('disease_key')
        if not key or key in seen:
            continue
        info = get_disease(key) or {}
        candidates.append({
            'name': option.get('name') or info.get('name') or key,
            'crop': info.get('crop', ''),
            'confidence': float(option.get('confidence') or 0),
        })
        seen.add(key)
        if len(candidates) == 3:
            break
    return candidates


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
            # Any supported format is decoded and re-saved as a clean RGB JPEG, so
            # the preview, the stored photo and the model always get a usable image.
            try:
                unique_filename, filepath, image_bytes = normalize_upload(file)
            except ImageFormatError as exc:
                flash(str(exc), 'danger')
                return redirect(request.url)
            image_mimetype = 'image/jpeg'
            
            # This viva build deliberately runs only the trained CNN; no external
            # vision API receives the uploaded image.
            result = None
            try:
                result = analyze_image(filepath)
            except Exception:
                current_app.logger.exception('CNN crop analysis failed.')
            finally:
                # Images are stored in the database after a successful scan. Do not
                # leave the normalized upload in a public static directory.
                try:
                    os.remove(filepath)
                except OSError:
                    pass

            if not result or not result.get('ok'):
                flash('We could not analyse that photo. Please try a clearer, closer shot of the affected leaf.', 'warning')
                return redirect(url_for('scan.scan_crop'))

            preview_b64 = base64.b64encode(image_bytes).decode('ascii')
            disease_key = result['disease_key']
            confidence = result['confidence']
            severity = result['severity']
            disease_info = get_disease(disease_key)

            # Honest rejection: a weak best guess - or a photo of a plant outside the
            # library - means "I do not know this one", not a made-up disease name.
            if confidence < MIN_CONFIDENCE:
                return render_template(
                    'scan_unknown.html',
                    preview=f"data:{image_mimetype};base64,{preview_b64}",
                    candidates=_prediction_candidates(result),
                    threshold=MIN_CONFIDENCE,
                )
            
            # Save to database
            scan = Scan(
                user_id=current_user.id,
                image_filename=unique_filename,
                crop_type=disease_info['crop'],
                disease_key=disease_key,
                disease_name=disease_info['name'],
                confidence=confidence,
                severity=severity,
                image_data=image_bytes,
                image_mimetype=image_mimetype,
                alternatives=json.dumps(result.get('alternatives', [])),
                analysis_source=result.get('source', 'on-device'),
            )
            db.session.add(scan)
            db.session.commit()
            
            return redirect(url_for('scan.scan_results', scan_id=scan.id))
            
        else:
            flash(f'Allowed image types are -> {SUPPORTED_FORMATS_LABEL}', 'danger')
            return redirect(request.url)

    return render_template('scan.html')

@scan_bp.route('/scan/image/<int:scan_id>')
@login_required
def scan_image(scan_id):
    scan = Scan.query.get_or_404(scan_id)
    if scan.user_id != current_user.id and not current_user.is_admin:
        abort(403)

    if scan.image_data:
        return Response(scan.image_data, mimetype=scan.image_mimetype or 'image/jpeg')

    # Scans saved before images were kept in the database
    return send_from_directory(LEGACY_UPLOAD_FOLDER, scan.image_filename)


@scan_bp.route('/scan/results/<int:scan_id>')
@login_required
def scan_results(scan_id):
    scan = Scan.query.get_or_404(scan_id)
    if scan.user_id != current_user.id:
        abort(403)
        
    disease_info = get_disease(scan.disease_key)
    try:
        alternatives = json.loads(scan.alternatives) if scan.alternatives else []
    except ValueError:
        alternatives = []
    return render_template('scan_results.html', scan=scan, disease_info=disease_info,
                           alternatives=alternatives)

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
    pdf.cell(0, 8, text=clean_text(f"Model-predicted class: {scan.disease_name}"), new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, text=clean_text(f"Class-level severity default (not image-measured): {scan.severity}"), new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, text=clean_text(f"Model score (not a calibrated field probability): {scan.confidence * 100:.1f}%"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)
    
    pdf.set_font("helvetica", style='B', size=12)
    pdf.cell(0, 8, text="General Reference Guidance:", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", size=10)
    pdf.multi_cell(0, 7, text=clean_text(
        "Educational class-level information only. Confirm with a local agricultural expert "
        "and follow product labels; this is not an individualized prescription."
    ), new_x="LMARGIN", new_y="NEXT")
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
