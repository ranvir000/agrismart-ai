import base64
import json
import os
from werkzeug.utils import secure_filename
import mimetypes
from flask import Blueprint, render_template, request, flash, redirect, url_for, current_app, abort, Response, send_from_directory
from flask_login import login_required, current_user
from database.db import db, Scan
from models.model_loader import (
    analyze_image, allowed_file, normalize_upload, ImageFormatError,
    SUPPORTED_FORMATS_LABEL, LEGACY_UPLOAD_FOLDER,
)
from models.vision_analyzer import analyze_with_vision, is_available, last_error
from models.disease_info import get_disease

scan_bp = Blueprint('scan', __name__)

# Below this probability the app answers "outside my scope" instead of naming a
# disease it is not confident about. Override per deployment with MIN_CONFIDENCE.
MIN_CONFIDENCE = float(os.environ.get('MIN_CONFIDENCE', '0.60'))

# Display guidance only: >=0.85 was 100% accurate on the held-out in-scope test
# split, but this is not a calibrated probability or a field/OOD guarantee.
CNN_TRUST_CONFIDENCE = float(os.environ.get('CNN_TRUST_CONFIDENCE', '0.85'))


def _engine_summary(prediction):
    """One engine's own verdict, shaped for the side-by-side comparison view."""
    if not prediction:
        return {'status': 'unavailable'}
    if prediction.get('out_of_scope'):
        return {
            'status': 'outside',
            'crop': prediction.get('crop') or 'Unknown',
            'note': prediction.get('note') or '',
        }
    if not prediction.get('ok'):
        return {'status': 'unavailable'}
    info = get_disease(prediction['disease_key']) or {}
    return {
        'status': 'named',
        'key': prediction['disease_key'],
        'name': info.get('name') or prediction['disease_key'],
        'crop': info.get('crop') or '',
        'confidence': round(float(prediction.get('confidence') or 0), 4),
        'note': prediction.get('note') or '',
    }


def build_comparison(cnn, cloud):
    """Both engines' answers for the same photo, kept apart so they can be compared."""
    local = _engine_summary(cnn)
    vision = _engine_summary(cloud)
    if local['status'] == 'named' and vision['status'] == 'named':
        verdict = 'agree' if local['key'] == vision['key'] else 'differ'
    else:
        verdict = 'partial'
    return {'local': local, 'vision': vision, 'verdict': verdict}


def _engine_crop(prediction):
    """The crop implied by an engine's own answer ('' when it named nothing)."""
    if not prediction or not prediction.get('ok') or prediction.get('out_of_scope'):
        return ''
    info = get_disease(prediction.get('disease_key')) or {}
    return (info.get('crop') or '').strip()


def _unknown_candidates(cnn, cloud):
    """Return a small, de-duplicated list of guesses clearly labelled as unreliable."""
    candidates = []
    seen = set()
    for prediction in (cnn, cloud):
        if not prediction or not prediction.get('ok'):
            continue
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
                return candidates
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
            
            # The CNN is the local classifier. Gemini is an optional cross-check; any
            # explicit out-of-scope verdict from it takes precedence over a closed-set
            # CNN guess, which cannot independently prove that an input is in scope.
            cnn = None
            cloud = None
            try:
                cnn = analyze_image(filepath)
            except Exception:
                current_app.logger.exception('On-device crop analysis failed.')
            try:
                cloud = analyze_with_vision(filepath)
            except Exception:
                current_app.logger.exception('Vision cross-check failed.')
            finally:
                # Images are stored in the database after a successful scan. Do not
                # leave the normalized upload in a public static directory.
                try:
                    os.remove(filepath)
                except OSError:
                    pass

            cnn_ok = bool(cnn and cnn.get('ok'))
            cloud_ok = bool(cloud and cloud.get('ok'))

            # Keep both raw answers, whatever the app decides to show as its answer.
            comparison = build_comparison(cnn, cloud)
            scope_refusal = cloud if (cloud and cloud.get('out_of_scope')) else None
            cnn_crop = _engine_crop(cnn)
            cloud_crop = _engine_crop(cloud)
            crop_conflict = bool(cnn_crop and cloud_crop and cnn_crop.lower() != cloud_crop.lower())

            preview_b64 = base64.b64encode(image_bytes).decode('ascii')
            if scope_refusal or crop_conflict:
                if crop_conflict:
                    reason = (
                        f'The two engines disagreed about the plant itself: the on-device model '
                        f'answered for {cnn_crop} while the vision model answered for {cloud_crop}. '
                        'No disease name was saved because the crop could be wrong.'
                    )
                else:
                    reason = (
                        f"The AI cross-check did not match this plant to the {scope_refusal.get('crop_count', 14)} "
                        'crops this app covers, so no disease name was saved.'
                    )
                    if scope_refusal.get('note'):
                        reason += f" It described the image as: {scope_refusal['note']}"
                return render_template(
                    'scan_unknown.html',
                    preview=f'data:{image_mimetype};base64,{preview_b64}',
                    candidates=_unknown_candidates(cnn, cloud),
                    threshold=MIN_CONFIDENCE,
                    source='Gemini vision' if scope_refusal else 'model comparison',
                    reason=reason,
                    comparison=comparison,
                )

            # Keep the trained local model as the headline answer whenever it returns a
            # supported-class prediction; the optional vision result remains a comparison.
            if cnn_ok:
                result = cnn
            elif cloud_ok:
                result = cloud
            else:
                result = cnn
                if is_available() and not scope_refusal:
                    flash('The cloud analysis could not be reached just now, so this result comes from the on-device model only.', 'warning')
            if not result or not result.get('ok'):
                flash('We could not analyse that photo. Please try a clearer, closer shot of the affected leaf.', 'warning')
                return redirect(url_for('scan.scan_crop'))

            # A disagreement between the two engines must never be hidden behind one
            # confident-looking name: the other opinion is shown as an alternative.
            other = None
            if result is cloud and cnn_ok and cnn['disease_key'] != result['disease_key']:
                other = cnn
            elif result is cnn and cloud_ok and cloud['disease_key'] != result['disease_key']:
                other = cloud
            if other:
                other_info = get_disease(other['disease_key'])
                result = dict(result)
                result['alternatives'] = [{
                    'key': other['disease_key'],
                    'name': other_info['name'],
                    'confidence': other['confidence'],
                }] + list(result.get('alternatives') or [])

            disease_key = result['disease_key']
            confidence = result['confidence']
            severity = result['severity']
            disease_info = get_disease(disease_key)

            # Honest rejection: a weak best guess - or a photo of a plant outside the
            # library - means "I do not know this one", not a made-up disease name.
            if confidence < MIN_CONFIDENCE:
                candidates = [{
                    'name': disease_info['name'],
                    'crop': disease_info['crop'],
                    'confidence': confidence,
                }]
                for alt in result.get('alternatives', []):
                    alt_info = {}
                    if alt.get('key'):
                        try:
                            alt_info = get_disease(alt['key']) or {}
                        except Exception:
                            alt_info = {}
                    candidates.append({
                        'name': alt.get('name') or alt_info.get('name', '-'),
                        'crop': alt_info.get('crop', ''),
                        'confidence': alt.get('confidence', 0),
                    })

                return render_template(
                    'scan_unknown.html',
                    preview=f"data:{image_mimetype};base64,{preview_b64}",
                    candidates=candidates[:3],
                    threshold=MIN_CONFIDENCE,
                    source=result.get('source'),
                    reason=None,
                    comparison=comparison,
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
                engine_outputs=json.dumps(comparison)
            )
            db.session.add(scan)
            db.session.commit()
            
            return redirect(url_for('scan.scan_results', scan_id=scan.id))
            
        else:
            flash(f'Allowed image types are -> {SUPPORTED_FORMATS_LABEL}', 'danger')
            return redirect(request.url)

    return render_template('scan.html', vision_available=is_available())

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
    comparison = None
    if scan.engine_outputs:
        try:
            comparison = json.loads(scan.engine_outputs)
        except ValueError:
            comparison = None
    return render_template('scan_results.html', scan=scan, disease_info=disease_info,
                           alternatives=alternatives, comparison=comparison,
                           trust_threshold=CNN_TRUST_CONFIDENCE)

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
