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

if __name__ == '__main__':
    unittest.main()
