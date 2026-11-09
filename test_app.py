import os
import unittest
from io import BytesIO
from app import create_app
from database.db import db, User, Scan
from werkzeug.security import generate_password_hash

class AgriSmartTests(unittest.TestCase):
    def setUp(self):
        self.app = create_app({
            'TESTING': True,
            'SECRET_KEY': 'test-secret-key-123',
            'SQLALCHEMY_DATABASE_URI': 'sqlite:///:memory:',
            'WTF_CSRF_ENABLED': False
        })
        self.client = self.app.test_client()
        self.app_context = self.app.app_context()
        self.app_context.push()
        
        db.create_all()
        
        # Seed a farmer
        farmer = User(
            username='test_farmer', 
            email='farmer@test.com', 
            password_hash=generate_password_hash('password'),
            role='farmer'
        )
        # Seed an admin
        admin = User(
            username='test_admin', 
            email='admin@test.com', 
            password_hash=generate_password_hash('password'),
            role='admin'
        )
        db.session.add_all([farmer, admin])
        db.session.commit()

    def tearDown(self):
        db.session.remove()
        db.drop_all()
        self.app_context.pop()
        
    def login(self, email, password):
        return self.client.post('/login', data=dict(
            email=email,
            password=password
        ), follow_redirects=True)

    def logout(self):
        return self.client.get('/logout', follow_redirects=True)

    def test_auth_pages_accessible(self):
        rv = self.client.get('/login')
        self.assertEqual(rv.status_code, 200)
        rv = self.client.get('/register')
        self.assertEqual(rv.status_code, 200)

    def test_farmer_can_access_dashboard(self):
        self.login('farmer@test.com', 'password')
        rv = self.client.get('/dashboard')
        self.assertEqual(rv.status_code, 200)
        
    def test_admin_blocked_from_scan(self):
        self.login('admin@test.com', 'password')
        rv = self.client.get('/scan')
        self.assertEqual(rv.status_code, 403)  # Forbidden
        
    def test_farmer_can_access_scan(self):
        self.login('farmer@test.com', 'password')
        rv = self.client.get('/scan')
        self.assertEqual(rv.status_code, 200)

    def test_admin_can_access_admin_panel(self):
        self.login('admin@test.com', 'password')
        rv = self.client.get('/admin')
        self.assertEqual(rv.status_code, 200)
        
    def test_farmer_blocked_from_admin_panel(self):
        self.login('farmer@test.com', 'password')
        rv = self.client.get('/admin')
        self.assertEqual(rv.status_code, 403)

    def test_analytics_api_returns_json(self):
        self.login('farmer@test.com', 'password')
        rv = self.client.get('/api/analytics-data')
        self.assertEqual(rv.status_code, 200)
        self.assertTrue(rv.is_json)

    def test_diseases_pages(self):
        rv = self.client.get('/diseases')
        self.assertEqual(rv.status_code, 200)
        rv = self.client.get('/diseases/tomato_early_blight')
        self.assertEqual(rv.status_code, 200)

    def test_weather_page(self):
        self.login('farmer@test.com', 'password')
        rv = self.client.get('/weather')
        self.assertEqual(rv.status_code, 200)

    def _seed_scan(self, user_id):
        """A stored scan, independent of whatever the classifier returns for a test image."""
        scan = Scan(
            user_id=user_id,
            image_filename='seeded_leaf.jpg',
            crop_type='Tomato',
            disease_key='tomato_early_blight',
            disease_name='Early Blight',
            confidence=0.91,
            severity='Moderate',
            image_data=b'\xff\xd8\xff\xd9',
            image_mimetype='image/jpeg',
            alternatives='[]',
            analysis_source='on-device',
        )
        db.session.add(scan)
        db.session.commit()
        return scan

    def test_scan_upload_is_answered(self):
        """Any upload must get a page back - a result or an honest refusal, never an error."""
        self.login('farmer@test.com', 'password')
        from PIL import Image
        img_io = BytesIO()
        Image.new('RGB', (100, 100), color='green').save(img_io, 'JPEG')
        img_io.seek(0)

        rv = self.client.post('/scan', data={
            'file': (img_io, 'test_leaf.jpg')
        }, content_type='multipart/form-data', follow_redirects=True)
        self.assertEqual(rv.status_code, 200)

    def test_pdf_and_csv_export(self):
        self.login('farmer@test.com', 'password')
        farmer = User.query.filter_by(email='farmer@test.com').first()
        scan = self._seed_scan(farmer.id)

        pdf_rv = self.client.get(f'/scan/download-pdf/{scan.id}')
        self.assertEqual(pdf_rv.status_code, 200)
        self.assertEqual(pdf_rv.mimetype, 'application/pdf')
        self.assertTrue(len(pdf_rv.data) > 500)  # real PDF bytes

        csv_rv = self.client.get('/history/export')
        self.assertEqual(csv_rv.status_code, 200)
        self.assertIn(b'Crop', csv_rv.data)

    def test_database_url_uses_psycopg_v3(self):
        from app import normalize_database_url
        self.assertEqual(normalize_database_url('postgres://u:p@host/db'),
                         'postgresql+psycopg://u:p@host/db')
        self.assertEqual(normalize_database_url('postgresql://u:p@host/db'),
                         'postgresql+psycopg://u:p@host/db')

    def test_production_requires_secret_key(self):
        from unittest.mock import patch
        with patch.dict(os.environ, {'FLASK_ENV': 'production', 'SECRET_KEY': '', 'RENDER': ''}):
            with self.assertRaisesRegex(RuntimeError, 'SECRET_KEY'):
                create_app()

    def test_admin_is_not_bootstrapped_with_a_default_password(self):
        from unittest.mock import patch
        from app import seed_admin
        with patch.dict(os.environ, {'ADMIN_PASSWORD': '', 'ADMIN_EMAIL': 'admin@agrismart.com'}):
            seed_admin()
        self.assertIsNone(User.query.filter_by(email='admin@agrismart.com').first())

    def test_legacy_default_admin_requires_secret_or_is_rotated(self):
        from unittest.mock import patch
        from app import seed_admin
        from werkzeug.security import check_password_hash
        legacy = User(username='admin', email='admin@agrismart.com',
                      password_hash=generate_password_hash('admin123'), role='admin')
        db.session.add(legacy)
        db.session.commit()
        with patch.dict(os.environ, {'ADMIN_PASSWORD': '', 'ADMIN_EMAIL': 'admin@agrismart.com'}):
            with self.assertRaisesRegex(RuntimeError, 'public default password'):
                seed_admin()
        with patch.dict(os.environ, {'ADMIN_PASSWORD': 'new-private-admin-password',
                                     'ADMIN_EMAIL': 'admin@agrismart.com'}):
            seed_admin()
        self.assertTrue(check_password_hash(legacy.password_hash, 'new-private-admin-password'))
        self.assertFalse(check_password_hash(legacy.password_hash, 'admin123'))

    def test_admin_confirmation_does_not_interpolate_username_in_javascript(self):
        attacker = User(username="bad');alert(1);//", email='xss@test.com',
                        password_hash=generate_password_hash('password'), role='farmer')
        db.session.add(attacker)
        db.session.commit()
        self.login('admin@test.com', 'password')
        response = self.client.get('/admin')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"confirm('Delete this user and all their data?')", response.data)
        self.assertNotIn(b"confirm('Delete user bad", response.data)

    def test_legacy_static_upload_path_is_not_public(self):
        response = self.client.get('/static/uploads/old-user-photo.jpg')
        self.assertEqual(response.status_code, 404)

    def test_gemini_out_of_scope_refusal_overrules_confident_cnn(self):
        from unittest.mock import patch
        from PIL import Image
        self.login('farmer@test.com', 'password')
        image = BytesIO()
        Image.new('RGB', (100, 100), color='green').save(image, 'JPEG')
        image.seek(0)
        cnn = {'ok': True, 'disease_key': 'tomato_early_blight', 'confidence': 0.96,
               'severity': 'Moderate', 'alternatives': [], 'source': 'on-device'}
        refusal = {'ok': False, 'out_of_scope': True, 'crop': 'Rice', 'crop_count': 14,
                   'note': 'a rice leaf'}
        with patch('routes.scan.analyze_image', return_value=cnn), \
                patch('routes.scan.analyze_with_vision', return_value=refusal):
            response = self.client.post('/scan', data={'file': (image, 'rice.jpg')},
                                        content_type='multipart/form-data')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Outside My Scope', response.data)
        self.assertIn(b'no disease name was saved', response.data)
        self.assertEqual(Scan.query.count(), 0)

    def test_low_confidence_scan_renders_refusal_without_crashing(self):
        from unittest.mock import patch
        from PIL import Image
        self.login('farmer@test.com', 'password')
        image = BytesIO()
        Image.new('RGB', (100, 100), color='green').save(image, 'JPEG')
        image.seek(0)
        cnn = {'ok': True, 'disease_key': 'tomato_early_blight', 'confidence': 0.2,
               'severity': 'Moderate', 'alternatives': [], 'source': 'on-device'}
        with patch('routes.scan.analyze_image', return_value=cnn), \
                patch('routes.scan.analyze_with_vision', return_value=None):
            response = self.client.post('/scan', data={'file': (image, 'leaf.jpg')},
                                        content_type='multipart/form-data')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Outside My Scope', response.data)
        self.assertIn(b'60%', response.data)

    def test_successful_scan_deletes_temporary_upload(self):
        from unittest.mock import patch
        from PIL import Image
        from models.model_loader import normalize_upload as real_normalize_upload
        import routes.scan as scan_routes
        self.login('farmer@test.com', 'password')
        image = BytesIO()
        Image.new('RGB', (100, 100), color='green').save(image, 'JPEG')
        image.seek(0)
        saved_paths = []

        def save_and_capture(file_storage):
            result = real_normalize_upload(file_storage)
            saved_paths.append(result[1])
            return result

        cnn = {'ok': True, 'disease_key': 'tomato_early_blight', 'confidence': 0.91,
               'severity': 'Moderate', 'alternatives': [], 'source': 'on-device'}
        with patch.object(scan_routes, 'normalize_upload', side_effect=save_and_capture), \
                patch.object(scan_routes, 'analyze_image', return_value=cnn), \
                patch.object(scan_routes, 'analyze_with_vision', return_value=None):
            response = self.client.post('/scan', data={'file': (image, 'leaf.jpg')},
                                        content_type='multipart/form-data', follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(saved_paths), 1)
        self.assertFalse(os.path.exists(saved_paths[0]))
        self.assertEqual(Scan.query.count(), 1)

    def test_vision_network_timeout_degrades_gracefully(self):
        import requests
        from unittest.mock import patch
        import models.vision_analyzer as vision
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'test-key'}), \
                patch.object(vision, '_post', side_effect=requests.Timeout('timed out')):
            result = vision.analyze_with_vision('/unused/mock-image.jpg')
        self.assertIsNone(result)
        self.assertIn('Network error', vision.last_error())

    def test_weather_api_aggregates_40_intervals_into_five_days(self):
        from datetime import datetime, timedelta, timezone
        from types import SimpleNamespace
        from unittest.mock import patch
        import models.model_loader as model_loader
        start = datetime(2026, 11, 9, tzinfo=timezone.utc)
        periods = []
        for index in range(40):
            stamp = start + timedelta(hours=3 * index)
            periods.append({'dt': int(stamp.timestamp()),
                            'main': {'temp': 22, 'temp_max': 28, 'temp_min': 18, 'humidity': 60},
                            'weather': [{'description': 'clear sky', 'icon': '01d'}],
                            'wind': {'speed': 2.0}})
        payload = {'city': {'name': 'Delhi', 'country': 'IN', 'timezone': 0}, 'list': periods}
        response = SimpleNamespace(status_code=200, json=lambda: payload)
        with patch.dict(os.environ, {'WEATHER_API_KEY': 'test-key'}), \
                patch.object(model_loader.requests, 'get', return_value=response) as get:
            weather = model_loader.get_weather('Delhi')
        self.assertEqual(weather['source'], 'live')
        self.assertEqual(len(weather['forecasts']), 5)
        self.assertIn('https://', get.call_args.args[0])
        self.assertIn('cnt=40', get.call_args.args[0])

if __name__ == '__main__':
    unittest.main()
