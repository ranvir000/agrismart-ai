from flask import Blueprint, render_template, abort
from models.disease_info import DISEASES

diseases_bp = Blueprint('diseases', __name__)

@diseases_bp.route('/diseases')
def diseases_list():
    # Group diseases by crop for the UI
    crops = {}
    for key, info in DISEASES.items():
        crop = info['crop']
        if crop not in crops:
            crops[crop] = []
        crops[crop].append({'key': key, **info})
        
    return render_template('diseases.html', crops=crops)

@diseases_bp.route('/diseases/<string:disease_key>')
def disease_detail(disease_key):
    if disease_key not in DISEASES:
        abort(404)
        
    disease = DISEASES[disease_key]
    return render_template('disease_detail.html', key=disease_key, disease=disease)
