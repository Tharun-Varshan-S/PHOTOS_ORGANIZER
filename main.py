import sys
import os
import json
import sqlite3
import hashlib
import bcrypt
import time
from datetime import datetime, timedelta
from pathlib import Path
import shutil
import threading
from typing import Dict, List, Optional, Tuple

# GUI imports
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QGridLayout, QLabel, QLineEdit, 
                            QPushButton, QListWidget, QTextEdit, QTabWidget,
                            QScrollArea, QFrame, QCalendarWidget, QSlider,
                            QComboBox, QCheckBox, QProgressBar, QMessageBox,
                            QInputDialog, QListWidgetItem, QSplitter,
                            QTreeWidget, QTreeWidgetItem, QDialog, QDialogButtonBox)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QSize, QDate
from PyQt5.QtGui import QPixmap, QFont, QPalette, QColor, QIcon, QMovie

# Image processing imports
from PIL import Image, ImageTk, ExifTags
import cv2
import numpy as np
import face_recognition

class DatabaseManager:
    """Manages SQLite database operations for the photo management app"""
    
    def __init__(self, db_path: str = "photo_manager.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Users table for authentication
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                failed_attempts INTEGER DEFAULT 0,
                last_failed_attempt TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Photos table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS photos (
                id INTEGER PRIMARY KEY,
                filepath TEXT UNIQUE NOT NULL,
                filename TEXT NOT NULL,
                thumbnail_path TEXT,
                date_taken TIMESTAMP,
                date_added TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                file_size INTEGER,
                width INTEGER,
                height INTEGER,
                hash_value TEXT UNIQUE
            )
        ''')
        
        # Faces table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS faces (
                id INTEGER PRIMARY KEY,
                photo_id INTEGER,
                person_name TEXT,
                face_encoding BLOB,
                face_location TEXT,
                confidence REAL,
                FOREIGN KEY (photo_id) REFERENCES photos (id)
            )
        ''')
        
        # Memories table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                date_created TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                tags TEXT,
                description TEXT
            )
        ''')
        
        # Memory photos junction table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS memory_photos (
                memory_id INTEGER,
                photo_id INTEGER,
                PRIMARY KEY (memory_id, photo_id),
                FOREIGN KEY (memory_id) REFERENCES memories (id),
                FOREIGN KEY (photo_id) REFERENCES photos (id)
            )
        ''')
        
        # Tags table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tags (
                id INTEGER PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                color TEXT DEFAULT '#3498db'
            )
        ''')
        
        # Photo tags junction table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS photo_tags (
                photo_id INTEGER,
                tag_id INTEGER,
                PRIMARY KEY (photo_id, tag_id),
                FOREIGN KEY (photo_id) REFERENCES photos (id),
                FOREIGN KEY (tag_id) REFERENCES tags (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def create_user(self, username: str, password: str) -> bool:
        """Create a new user with hashed password"""
        try:
            password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                (username, password_hash)
            )
            conn.commit()
            conn.close()
            return True
        except sqlite3.IntegrityError:
            return False
    
    def authenticate_user(self, username: str, password: str) -> Tuple[bool, bool]:
        """Authenticate user and handle failed attempts"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if user exists and get current failed attempts
        cursor.execute(
            "SELECT password_hash, failed_attempts, last_failed_attempt FROM users WHERE username = ?",
            (username,)
        )
        result = cursor.fetchone()
        
        if not result:
            conn.close()
            return False, False
        
        password_hash, failed_attempts, last_failed_attempt = result
        
        # Check if user is locked out (3 failed attempts within 60 seconds)
        if failed_attempts >= 3:
            if last_failed_attempt:
                last_attempt_time = datetime.fromisoformat(last_failed_attempt)
                if datetime.now() - last_attempt_time < timedelta(seconds=60):
                    conn.close()
                    return False, True  # Locked out
            
            # Reset failed attempts after timeout
            cursor.execute(
                "UPDATE users SET failed_attempts = 0 WHERE username = ?",
                (username,)
            )
            conn.commit()
        
        # Verify password
        if bcrypt.checkpw(password.encode('utf-8'), password_hash):
            # Reset failed attempts on successful login
            cursor.execute(
                "UPDATE users SET failed_attempts = 0 WHERE username = ?",
                (username,)
            )
            conn.commit()
            conn.close()
            return True, False
        else:
            # Increment failed attempts
            cursor.execute(
                "UPDATE users SET failed_attempts = failed_attempts + 1, last_failed_attempt = ? WHERE username = ?",
                (datetime.now().isoformat(), username)
            )
            conn.commit()
            conn.close()
            return False, False
    
    def add_photo(self, filepath: str, **kwargs) -> int:
        """Add a photo to the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO photos (filepath, filename, thumbnail_path, date_taken, 
                              file_size, width, height, hash_value)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            filepath,
            kwargs.get('filename', os.path.basename(filepath)),
            kwargs.get('thumbnail_path'),
            kwargs.get('date_taken'),
            kwargs.get('file_size'),
            kwargs.get('width'),
            kwargs.get('height'),
            kwargs.get('hash_value')
        ))
        
        photo_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return photo_id
    
    def get_photos_by_date(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Get photos within a date range"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM photos 
            WHERE date_taken BETWEEN ? AND ?
            ORDER BY date_taken
        ''', (start_date.isoformat(), end_date.isoformat()))
        
        photos = []
        for row in cursor.fetchall():
            photos.append({
                'id': row[0],
                'filepath': row[1],
                'filename': row[2],
                'thumbnail_path': row[3],
                'date_taken': row[4],
                'date_added': row[5],
                'file_size': row[6],
                'width': row[7],
                'height': row[8],
                'hash_value': row[9]
            })
        
        conn.close()
        return photos
    
    def add_face(self, photo_id: int, face_encoding: np.ndarray, 
                 face_location: Tuple, confidence: float, person_name: str = None) -> int:
        """Add a face detection result to the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO faces (photo_id, person_name, face_encoding, face_location, confidence)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            photo_id,
            person_name,
            face_encoding.tobytes(),
            json.dumps(face_location),
            confidence
        ))
        
        face_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return face_id
    
    def get_all_faces(self) -> List[Dict]:
        """Get all faces from the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT f.*, p.filepath FROM faces f
            JOIN photos p ON f.photo_id = p.id
        ''')
        
        faces = []
        for row in cursor.fetchall():
            faces.append({
                'id': row[0],
                'photo_id': row[1],
                'person_name': row[2],
                'face_encoding': np.frombuffer(row[3], dtype=np.float64),
                'face_location': json.loads(row[4]),
                'confidence': row[5],
                'filepath': row[6]
            })
        
        conn.close()
        return faces
    
    def update_face_name(self, face_id: int, person_name: str):
        """Update the person name for a face"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "UPDATE faces SET person_name = ? WHERE id = ?",
            (person_name, face_id)
        )
        
        conn.commit()
        conn.close()


class PhotoProcessor:
    """Handles photo processing, face detection, and thumbnail generation"""
    
    def __init__(self, photos_dir: str = "photos", thumbnails_dir: str = "thumbnails"):
        self.photos_dir = Path(photos_dir)
        self.thumbnails_dir = Path(thumbnails_dir)
        self.thumbnails_dir.mkdir(exist_ok=True)
        
        # Supported image formats
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'}
    
    def generate_thumbnail(self, image_path: str, size: Tuple[int, int] = (200, 200)) -> str:
        """Generate thumbnail for an image"""
        try:
            image_path = Path(image_path)
            thumbnail_path = self.thumbnails_dir / f"thumb_{image_path.stem}.jpg"
            
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Create thumbnail
                img.thumbnail(size, Image.Resampling.LANCZOS)
                img.save(thumbnail_path, "JPEG", quality=85)
            
            return str(thumbnail_path)
        except Exception as e:
            print(f"Error generating thumbnail for {image_path}: {e}")
            return None
    
    def extract_exif_date(self, image_path: str) -> Optional[datetime]:
        """Extract date taken from EXIF data"""
        try:
            with Image.open(image_path) as img:
                exif = img._getexif()
                if exif:
                    for tag, value in exif.items():
                        tag_name = ExifTags.TAGS.get(tag, tag)
                        if tag_name == 'DateTime':
                            return datetime.strptime(value, "%Y:%m:%d %H:%M:%S")
        except Exception as e:
            print(f"Error extracting EXIF date from {image_path}: {e}")
        
        # Fall back to file modification time
        try:
            return datetime.fromtimestamp(os.path.getmtime(image_path))
        except Exception:
            return datetime.now()
    
    def calculate_file_hash(self, filepath: str) -> str:
        """Calculate MD5 hash of file for duplicate detection"""
        hash_md5 = hashlib.md5()
        try:
            with open(filepath, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            print(f"Error calculating hash for {filepath}: {e}")
            return ""
    
    def detect_faces(self, image_path: str) -> List[Dict]:
        """Detect faces in an image using face_recognition library"""
        try:
            # Load image
            image = face_recognition.load_image_file(image_path)
            
            # Find face locations
            face_locations = face_recognition.face_locations(image)
            
            # Find face encodings
            face_encodings = face_recognition.face_encodings(image, face_locations)
            
            faces = []
            for i, (face_location, face_encoding) in enumerate(zip(face_locations, face_encodings)):
                faces.append({
                    'location': face_location,
                    'encoding': face_encoding,
                    'confidence': 1.0  # face_recognition doesn't provide confidence directly
                })
            
            return faces
        except Exception as e:
            print(f"Error detecting faces in {image_path}: {e}")
            return []
    
    def find_duplicate_photos(self, photos: List[Dict]) -> List[List[Dict]]:
        """Find duplicate photos based on hash values"""
        hash_groups = {}
        
        for photo in photos:
            hash_value = photo.get('hash_value')
            if hash_value:
                if hash_value not in hash_groups:
                    hash_groups[hash_value] = []
                hash_groups[hash_value].append(photo)
        
        # Return groups with more than one photo (duplicates)
        duplicates = [group for group in hash_groups.values() if len(group) > 1]
        return duplicates
    
    def scan_photos_directory(self) -> List[str]:
        """Scan the photos directory for image files"""
        photo_files = []
        
        if self.photos_dir.exists():
            for file_path in self.photos_dir.rglob("*"):
                if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                    photo_files.append(str(file_path))
        
        return photo_files


class FaceClusterer:
    """Handles face clustering and person identification"""
    
    def __init__(self, tolerance: float = 0.6):
        self.tolerance = tolerance
    
    def cluster_faces(self, faces: List[Dict]) -> List[List[Dict]]:
        """Cluster faces by similarity"""
        if not faces:
            return []
        
        # Extract encodings
        encodings = [face['face_encoding'] for face in faces]
        
        # Simple clustering based on face distance
        clusters = []
        used_indices = set()
        
        for i, face in enumerate(faces):
            if i in used_indices:
                continue
            
            cluster = [face]
            used_indices.add(i)
            
            # Find similar faces
            for j, other_face in enumerate(faces):
                if j in used_indices:
                    continue
                
                distance = face_recognition.face_distance([face['face_encoding']], 
                                                        other_face['face_encoding'])[0]
                
                if distance < self.tolerance:
                    cluster.append(other_face)
                    used_indices.add(j)
            
            clusters.append(cluster)
        
        return clusters
    
    def identify_person(self, face_encoding: np.ndarray, known_faces: List[Dict]) -> Optional[str]:
        """Identify a person based on face encoding"""
        if not known_faces:
            return None
        
        known_encodings = [face['face_encoding'] for face in known_faces 
                          if face.get('person_name')]
        known_names = [face['person_name'] for face in known_faces 
                      if face.get('person_name')]
        
        if not known_encodings:
            return None
        
        # Find matches
        matches = face_recognition.compare_faces(known_encodings, face_encoding, 
                                               tolerance=self.tolerance)
        
        # Find best match
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        
        if matches[best_match_index]:
            return known_names[best_match_index]
        
        return None


class PhotoScannerThread(QThread):
    """Background thread for scanning and processing photos"""
    
    progress_updated = pyqtSignal(int, str)
    photo_processed = pyqtSignal(dict)
    face_detected = pyqtSignal(int, dict)
    scanning_complete = pyqtSignal()
    
    def __init__(self, db_manager: DatabaseManager, photo_processor: PhotoProcessor):
        super().__init__()
        self.db_manager = db_manager
        self.photo_processor = photo_processor
        self.running = True
    
    def run(self):
        """Main scanning loop"""
        try:
            # Scan for photo files
            photo_files = self.photo_processor.scan_photos_directory()
            total_files = len(photo_files)
            
            for i, photo_path in enumerate(photo_files):
                if not self.running:
                    break
                
                self.progress_updated.emit(
                    int((i / total_files) * 100), 
                    f"Processing {os.path.basename(photo_path)}"
                )
                
                # Process photo
                self.process_photo(photo_path)
            
            self.scanning_complete.emit()
        except Exception as e:
            print(f"Error in photo scanning thread: {e}")
    
    def process_photo(self, photo_path: str):
        """Process a single photo"""
        try:
            # Calculate hash for duplicate detection
            file_hash = self.photo_processor.calculate_file_hash(photo_path)
            
            # Extract date from EXIF
            date_taken = self.photo_processor.extract_exif_date(photo_path)
            
            # Get file info
            file_stat = os.stat(photo_path)
            
            # Generate thumbnail
            thumbnail_path = self.photo_processor.generate_thumbnail(photo_path)
            
            # Get image dimensions
            with Image.open(photo_path) as img:
                width, height = img.size
            
            # Add photo to database
            photo_id = self.db_manager.add_photo(
                filepath=photo_path,
                filename=os.path.basename(photo_path),
                thumbnail_path=thumbnail_path,
                date_taken=date_taken.isoformat(),
                file_size=file_stat.st_size,
                width=width,
                height=height,
                hash_value=file_hash
            )
            
            # Emit photo processed signal
            photo_data = {
                'id': photo_id,
                'filepath': photo_path,
                'filename': os.path.basename(photo_path),
                'thumbnail_path': thumbnail_path,
                'date_taken': date_taken.isoformat(),
                'width': width,
                'height': height
            }
            self.photo_processed.emit(photo_data)
            
            # Detect faces
            faces = self.photo_processor.detect_faces(photo_path)
            for face in faces:
                face_id = self.db_manager.add_face(
                    photo_id=photo_id,
                    face_encoding=face['encoding'],
                    face_location=face['location'],
                    confidence=face['confidence']
                )
                
                # Emit face detected signal
                face_data = {
                    'id': face_id,
                    'photo_id': photo_id,
                    'location': face['location'],
                    'encoding': face['encoding'],
                    'confidence': face['confidence']
                }
                self.face_detected.emit(photo_id, face_data)
        
        except Exception as e:
            print(f"Error processing photo {photo_path}: {e}")
    
    def stop(self):
        """Stop the scanning thread"""
        self.running = False


class LoginDialog(QDialog):
    """Login dialog for user authentication"""
    
    def __init__(self, db_manager: DatabaseManager, parent=None):
        super().__init__(parent)
        self.db_manager = db_manager
        self.authenticated = False
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the login dialog UI"""
        self.setWindowTitle("Photo Manager - Login")
        self.setFixedSize(400, 300)
        
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("AI Photo Manager")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 24px; font-weight: bold; margin: 20px;")
        layout.addWidget(title)
        
        # Username field
        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("Username")
        self.username_input.setStyleSheet("padding: 10px; font-size: 14px;")
        layout.addWidget(self.username_input)
        
        # Password field
        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("Password")
        self.password_input.setEchoMode(QLineEdit.Password)
        self.password_input.setStyleSheet("padding: 10px; font-size: 14px;")
        layout.addWidget(self.password_input)
        
        # Login button
        self.login_button = QPushButton("Login")
        self.login_button.setStyleSheet("padding: 10px; font-size: 14px; font-weight: bold;")
        self.login_button.clicked.connect(self.login)
        layout.addWidget(self.login_button)
        
        # Create account button
        self.create_account_button = QPushButton("Create Account")
        self.create_account_button.setStyleSheet("padding: 10px; font-size: 14px;")
        self.create_account_button.clicked.connect(self.create_account)
        layout.addWidget(self.create_account_button)
        
        # Status label
        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: red; margin: 10px;")
        layout.addWidget(self.status_label)
        
        self.setLayout(layout)
        
        # Connect Enter key to login
        self.password_input.returnPressed.connect(self.login)
        self.username_input.returnPressed.connect(self.login)
    
    def login(self):
        """Handle login attempt"""
        username = self.username_input.text().strip()
        password = self.password_input.text()
        
        if not username or not password:
            self.status_label.setText("Please enter both username and password")
            return
        
        # Authenticate user
        success, locked_out = self.db_manager.authenticate_user(username, password)
        
        if success:
            self.authenticated = True
            self.accept()
        elif locked_out:
            self.status_label.setText("Account locked. Please wait 60 seconds and try again.")
            self.login_button.setEnabled(False)
            QTimer.singleShot(60000, lambda: self.login_button.setEnabled(True))
        else:
            self.status_label.setText("Invalid username or password")
            self.password_input.clear()
    
    def create_account(self):
        """Handle account creation"""
        username = self.username_input.text().strip()
        password = self.password_input.text()
        
        if not username or not password:
            self.status_label.setText("Please enter both username and password")
            return
        
        if len(password) < 6:
            self.status_label.setText("Password must be at least 6 characters")
            return
        
        # Create user account
        if self.db_manager.create_user(username, password):
            self.status_label.setText("Account created successfully! Please login.")
            self.status_label.setStyleSheet("color: green; margin: 10px;")
            self.password_input.clear()
        else:
            self.status_label.setText("Username already exists")


class PhotoViewer(QWidget):
    """Widget for displaying photos in a grid or list view"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.photos = []
        self.current_index = 0
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the photo viewer UI"""
        layout = QVBoxLayout()
        
        # Controls
        controls_layout = QHBoxLayout()
        
        self.view_mode_combo = QComboBox()
        self.view_mode_combo.addItems(["Grid View", "List View"])
        self.view_mode_combo.currentTextChanged.connect(self.change_view_mode)
        controls_layout.addWidget(self.view_mode_combo)
        
        self.slideshow_button = QPushButton("Start Slideshow")
        self.slideshow_button.clicked.connect(self.toggle_slideshow)
        controls_layout.addWidget(self.slideshow_button)
        
        controls_layout.addStretch()
        layout.addLayout(controls_layout)
        
        # Photo display area
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.photo_container = QWidget()
        self.photo_layout = QGridLayout(self.photo_container)
        self.scroll_area.setWidget(self.photo_container)
        layout.addWidget(self.scroll_area)
        
        # Slideshow timer
        self.slideshow_timer = QTimer()
        self.slideshow_timer.timeout.connect(self.next_slide)
        self.slideshow_active = False
        
        self.setLayout(layout)
    
    def load_photos(self, photos: List[Dict]):
        """Load photos into the viewer"""
        self.photos = photos
        self.current_index = 0
        self.update_display()
    
    def update_display(self):
        """Update the photo display"""
        # Clear existing photos
        for i in reversed(range(self.photo_layout.count())):
            child = self.photo_layout.itemAt(i).widget()
            if child:
                child.deleteLater()
        
        # Add photos based on view mode
        if self.view_mode_combo.currentText() == "Grid View":
            self.display_grid_view()
        else:
            self.display_list_view()
    
    def display_grid_view(self):
        """Display photos in grid view"""
        columns = 4
        for i, photo in enumerate(self.photos):
            row = i // columns
            col = i % columns
            
            photo_widget = self.create_photo_widget(photo)
            self.photo_layout.addWidget(photo_widget, row, col)
    
    def display_list_view(self):
        """Display photos in list view"""
        for i, photo in enumerate(self.photos):
            photo_widget = self.create_photo_widget(photo, list_view=True)
            self.photo_layout.addWidget(photo_widget, i, 0)
    
    def create_photo_widget(self, photo: Dict, list_view: bool = False) -> QWidget:
        """Create a widget for displaying a single photo"""
        widget = QFrame()
        widget.setFrameStyle(QFrame.Box)
        widget.setStyleSheet("""
            QFrame {
                border: 2px solid #3498db;
                border-radius: 5px;
                background-color: #2c3e50;
                margin: 5px;
            }
            QFrame:hover {
                border-color: #e74c3c;
            }
        """)
        
        if list_view:
            layout = QHBoxLayout()
            widget.setFixedHeight(150)
        else:
            layout = QVBoxLayout()
            widget.setFixedSize(200, 250)
        
        # Photo thumbnail
        thumbnail_label = QLabel()
        thumbnail_label.setAlignment(Qt.AlignCenter)
        
        if photo.get('thumbnail_path') and os.path.exists(photo['thumbnail_path']):
            pixmap = QPixmap(photo['thumbnail_path'])
            if list_view:
                thumbnail_label.setPixmap(pixmap.scaled(120, 120, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            else:
                thumbnail_label.setPixmap(pixmap.scaled(180, 180, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            thumbnail_label.setText("No Thumbnail")
            thumbnail_label.setStyleSheet("color: #bdc3c7;")
        
        layout.addWidget(thumbnail_label)
        
        # Photo info
        info_widget = QWidget()
        info_layout = QVBoxLayout(info_widget)
        
        filename_label = QLabel(photo.get('filename', 'Unknown'))
        filename_label.setWordWrap(True)
        filename_label.setStyleSheet("color: white; font-weight: bold;")
        info_layout.addWidget(filename_label)
        
        if photo.get('date_taken'):
            date_label = QLabel(photo['date_taken'][:10])  # Show only date part
            date_label.setStyleSheet("color: #bdc3c7; font-size: 12px;")
            info_layout.addWidget(date_label)
        
        layout.addWidget(info_widget)
        widget.setLayout(layout)
        
        # Add click handler
        widget.mousePressEvent = lambda event, p=photo: self.open_photo(p)
        
        return widget
    
    def open_photo(self, photo: Dict):
        """Open a photo in full size"""
        # This would open a full-size photo viewer
        # For now, just print the photo info
        print(f"Opening photo: {photo['filename']}")
    
    def change_view_mode(self, mode: str):
        """Change between grid and list view"""
        self.update_display()
    
    def toggle_slideshow(self):
        """Toggle slideshow mode"""
        if self.slideshow_active:
            self.stop_slideshow()
        else:
            self.start_slideshow()
    
    def start_slideshow(self):
        """Start slideshow"""
        if not self.photos:
            return
        
        self.slideshow_active = True
        self.slideshow_button.setText("Stop Slideshow")
        self.slideshow_timer.start(3000)  # 3 seconds per slide
        self.show_current_slide()
    
    def stop_slideshow(self):
        """Stop slideshow"""
        self.slideshow_active = False
        self.slideshow_button.setText("Start Slideshow")
        self.slideshow_timer.stop()
        self.update_display()
    
    def next_slide(self):
        """Show next slide in slideshow"""
        if self.photos:
            self.current_index = (self.current_index + 1) % len(self.photos)
            self.show_current_slide()
    
    def show_current_slide(self):
        """Show current slide in slideshow mode"""
        if not self.photos:
            return
        
        # Clear layout
        for i in reversed(range(self.photo_layout.count())):
            child = self.photo_layout.itemAt(i).widget()
            if child:
                child.deleteLater()
        
        # Show current photo
        photo = self.photos[self.current_index]
        
        slide_widget = QWidget()
        slide_layout = QVBoxLayout(slide_widget)
        
        # Photo
        photo_label = QLabel()
        photo_label.setAlignment(Qt.AlignCenter)
        
        if photo.get('thumbnail_path') and os.path.exists(photo['thumbnail_path']):
            pixmap = QPixmap(photo['thumbnail_path'])
            photo_label.setPixmap(pixmap.scaled(600, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            photo_label.setText("No Image")
            photo_label.setStyleSheet("color: #bdc3c7; font-size: 24px;")
        
        slide_layout.addWidget(photo_label)
        
        # Photo info
        info_label = QLabel(f"{photo.get('filename', 'Unknown')} - {photo.get('date_taken', '')[:10]}")
        info_label.setAlignment(Qt.AlignCenter)
        info_label.setStyleSheet("color: white; font-size: 18px; margin: 20px;")
        slide_layout.addWidget(info_label)
        
        # Progress indicator
        progress_label = QLabel(f"{self.current_index + 1} / {len(self.photos)}")
        progress_label.setAlignment(Qt.AlignCenter)
        progress_label.setStyleSheet("color: #bdc3c7; font-size: 14px;")
        slide_layout.addWidget(progress_label)
        
        self.photo_layout.addWidget(slide_widget, 0, 0)


class MemoryManager(QWidget):
    """Widget for managing photo memories by date"""
    
    def __init__(self, db_manager: DatabaseManager, parent=None):
        super().__init__(parent)
        self.db_manager = db_manager
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the memory manager UI"""
        layout = QHBoxLayout()
        
        # Left panel - Calendar
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        left_layout.addWidget(QLabel("Select Date:"))
        
        self.calendar = QCalendarWidget()
        self.calendar.setStyleSheet("""
            QCalendarWidget {
                background-color: #2c3e50;
                color: white;
            }
            QCalendarWidget QWidget {
                alternate-background-color: #34495e;
            }
            QCalendarWidget QAbstractItemView:enabled {
                background-color: #2c3e50;
                selection-background-color: #3498db;
            }
        """)
        self.calendar.selectionChanged.connect(self.date_selected)
        left_layout.addWidget(self.calendar)
        
        # Memory tags
        left_layout.addWidget(QLabel("Memory Tags:"))
        self.tags_input = QLineEdit()
        self.tags_input.setPlaceholderText("Enter tags separated by commas")
        left_layout.addWidget(self.tags_input)
        
        self.add_tag_button = QPushButton("Add Tag")
        self.add_tag_button.clicked.connect(self.add_tag)
        left_layout.addWidget(self.add_tag_button)
        
        layout.addWidget(left_panel)
        
        # Right panel - Photos for selected date
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        self.date_label = QLabel("Select a date to view photos")
        self.date_label.setStyleSheet("font-size: 18px; font-weight: bold; color: white;")
        right_layout.addWidget(self.date_label)
        
        # Photo viewer for selected date
        self.photo_viewer = PhotoViewer()
        right_layout.addWidget(self.photo_viewer)
        
        layout.addWidget(right_panel)
        
        self.setLayout(layout)
    
    def date_selected(self):
        """Handle date selection from calendar"""
        selected_date = self.calendar.selectedDate().toPyDate()
        self.date_label.setText(f"Photos from {selected_date.strftime('%B %d, %Y')}")
        
        # Load photos for selected date
        start_date = datetime.combine(selected_date, datetime.min.time())
        end_date = datetime.combine(selected_date, datetime.max.time())
        
        photos = self.db_manager.get_photos_by_date(start_date, end_date)
        self.photo_viewer.load_photos(photos)
    
    def add_tag(self):
        """Add tag to current memory"""
        tags = self.tags_input.text().strip()
        if tags:
            # Implementation for adding tags would go here
            self.tags_input.clear()
            print(f"Added tags: {tags}")


class FaceManager(QWidget):
    """Widget for managing face recognition and person identification"""
    
    def __init__(self, db_manager: DatabaseManager, parent=None):
        super().__init__(parent)
        self.db_manager = db_manager
        self.face_clusterer = FaceClusterer()
        self.setup_ui()
        self.load_faces()
    
    def setup_ui(self):
        """Setup the face manager UI"""
        layout = QVBoxLayout()
        
        # Controls
        controls_layout = QHBoxLayout()
        
        self.cluster_button = QPushButton("Cluster Faces")
        self.cluster_button.clicked.connect(self.cluster_faces)
        controls_layout.addWidget(self.cluster_button)
        
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.load_faces)
        controls_layout.addWidget(self.refresh_button)
        
        controls_layout.addStretch()
        layout.addLayout(controls_layout)
        
        # Face groups
        self.face_groups_widget = QWidget()
        self.face_groups_layout = QVBoxLayout(self.face_groups_widget)
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.face_groups_widget)
        layout.addWidget(scroll_area)
        
        self.setLayout(layout)
    
    def load_faces(self):
        """Load all faces from database"""
        self.faces = self.db_manager.get_all_faces()
        self.display_faces()
    
    def display_faces(self):
        """Display faces in groups"""
        # Clear existing face groups
        for i in reversed(range(self.face_groups_layout.count())):
            child = self.face_groups_layout.itemAt(i).widget()
            if child:
                child.deleteLater()
        
        # Group faces by person name
        person_groups = {}
        unknown_faces = []
        
        for face in self.faces:
            person_name = face.get('person_name')
            if person_name:
                if person_name not in person_groups:
                    person_groups[person_name] = []
                person_groups[person_name].append(face)
            else:
                unknown_faces.append(face)
        
        # Display known person groups
        for person_name, faces in person_groups.items():
            self.create_person_group(person_name, faces)
        
        # Display unknown faces
        if unknown_faces:
            self.create_person_group("Unknown", unknown_faces)
    
    def create_person_group(self, person_name: str, faces: List[Dict]):
        """Create a widget for a person's face group"""
        group_widget = QFrame()
        group_widget.setFrameStyle(QFrame.Box)
        group_widget.setStyleSheet("""
            QFrame {
                border: 2px solid #3498db;
                border-radius: 5px;
                background-color: #34495e;
                margin: 5px;
                padding: 10px;
            }
        """)
        
        layout = QVBoxLayout(group_widget)
        
        # Person name header
        header_layout = QHBoxLayout()
        
        name_label = QLabel(person_name)
        name_label.setStyleSheet("color: white; font-size: 16px; font-weight: bold;")
        header_layout.addWidget(name_label)
        
        if person_name == "Unknown":
            assign_name_button = QPushButton("Assign Name")
            assign_name_button.clicked.connect(lambda: self.assign_person_name(faces))
            header_layout.addWidget(assign_name_button)
        
        count_label = QLabel(f"({len(faces)} photos)")
        count_label.setStyleSheet("color: #bdc3c7; font-size: 14px;")
        header_layout.addWidget(count_label)
        
        header_layout.addStretch()
        layout.addLayout(header_layout)
        
        # Face thumbnails
        faces_layout = QHBoxLayout()
        
        for i, face in enumerate(faces[:5]):  # Show first 5 faces
            face_widget = self.create_face_thumbnail(face)
            faces_layout.addWidget(face_widget)
        
        if len(faces) > 5:
            more_label = QLabel(f"+{len(faces) - 5} more")
            more_label.setStyleSheet("color: #bdc3c7; font-size: 12px;")
            faces_layout.addWidget(more_label)
        
        faces_layout.addStretch()
        layout.addLayout(faces_layout)
        
        self.face_groups_layout.addWidget(group_widget)
    
    def create_face_thumbnail(self, face: Dict) -> QWidget:
        """Create a thumbnail widget for a face"""
        widget = QFrame()
        widget.setFrameStyle(QFrame.Box)
        widget.setFixedSize(80, 80)
        widget.setStyleSheet("""
            QFrame {
                border: 1px solid #7f8c8d;
                border-radius: 3px;
                background-color: #2c3e50;
            }
        """)
        
        layout = QVBoxLayout(widget)
        
        # Extract face from original photo
        try:
            image = cv2.imread(face['filepath'])
            if image is not None:
                top, right, bottom, left = face['face_location']
                face_image = image[top:bottom, left:right]
                
                # Convert to RGB and create thumbnail
                face_image_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
                face_pil = Image.fromarray(face_image_rgb)
                face_pil.thumbnail((70, 70))
                
                # Convert to QPixmap and display
                face_pil.save("temp_face.jpg")
                pixmap = QPixmap("temp_face.jpg")
                
                face_label = QLabel()
                face_label.setAlignment(Qt.AlignCenter)
                face_label.setPixmap(pixmap)
                layout.addWidget(face_label)
                
                # Clean up temp file
                os.remove("temp_face.jpg")
            else:
                face_label = QLabel("No Face")
                face_label.setAlignment(Qt.AlignCenter)
                face_label.setStyleSheet("color: #bdc3c7; font-size: 10px;")
                layout.addWidget(face_label)
        except Exception as e:
            face_label = QLabel("Error")
            face_label.setAlignment(Qt.AlignCenter)
            face_label.setStyleSheet("color: #e74c3c; font-size: 10px;")
            layout.addWidget(face_label)
        
        return widget
    
    def assign_person_name(self, faces: List[Dict]):
        """Assign a name to a group of faces"""
        name, ok = QInputDialog.getText(self, "Assign Name", "Enter person's name:")
        
        if ok and name.strip():
            # Update all faces in the group
            for face in faces:
                self.db_manager.update_face_name(face['id'], name.strip())
            
            # Refresh display
            self.load_faces()
    
    def cluster_faces(self):
        """Cluster unknown faces by similarity"""
        unknown_faces = [face for face in self.faces if not face.get('person_name')]
        
        if not unknown_faces:
            QMessageBox.information(self, "Info", "No unknown faces to cluster.")
            return
        
        # Cluster faces
        clusters = self.face_clusterer.cluster_faces(unknown_faces)
        
        # Display clustering results
        msg = f"Found {len(clusters)} face clusters:\n"
        for i, cluster in enumerate(clusters):
            msg += f"Cluster {i+1}: {len(cluster)} faces\n"
        
        QMessageBox.information(self, "Clustering Results", msg)


class SearchWidget(QWidget):
    """Widget for searching photos by various criteria"""
    
    def __init__(self, db_manager: DatabaseManager, parent=None):
        super().__init__(parent)
        self.db_manager = db_manager
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the search widget UI"""
        layout = QVBoxLayout()
        
        # Search controls
        search_layout = QHBoxLayout()
        
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search by person name, date, or tags...")
        self.search_input.returnPressed.connect(self.search)
        search_layout.addWidget(self.search_input)
        
        self.search_button = QPushButton("Search")
        self.search_button.clicked.connect(self.search)
        search_layout.addWidget(self.search_button)
        
        # Search filters
        filters_layout = QHBoxLayout()
        
        self.search_type_combo = QComboBox()
        self.search_type_combo.addItems(["All", "Person", "Date", "Tags"])
        filters_layout.addWidget(self.search_type_combo)
        
        self.date_from = QLineEdit()
        self.date_from.setPlaceholderText("From date (YYYY-MM-DD)")
        filters_layout.addWidget(self.date_from)
        
        self.date_to = QLineEdit()
        self.date_to.setPlaceholderText("To date (YYYY-MM-DD)")
        filters_layout.addWidget(self.date_to)
        
        filters_layout.addStretch()
        
        layout.addLayout(search_layout)
        layout.addLayout(filters_layout)
        
        # Results
        self.results_label = QLabel("Enter search terms to find photos")
        self.results_label.setStyleSheet("color: #bdc3c7; font-size: 14px; margin: 10px;")
        layout.addWidget(self.results_label)
        
        # Photo viewer for results
        self.photo_viewer = PhotoViewer()
        layout.addWidget(self.photo_viewer)
        
        self.setLayout(layout)
    
    def search(self):
        """Perform search based on current criteria"""
        search_term = self.search_input.text().strip()
        search_type = self.search_type_combo.currentText()
        
        if not search_term and search_type == "All":
            self.results_label.setText("Enter search terms to find photos")
            return
        
        # This would implement the actual search logic
        # For now, just show a placeholder
        self.results_label.setText(f"Searching for: {search_term} (Type: {search_type})")
        
        # Load sample results (would be replaced with actual search results)
        results = []
        self.photo_viewer.load_photos(results)


class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.db_manager = DatabaseManager()
        self.photo_processor = PhotoProcessor()
        self.scanner_thread = None
        self.setup_ui()
        self.apply_dark_theme()
    
    def setup_ui(self):
        """Setup the main window UI"""
        self.setWindowTitle("AI Photo Manager")
        self.setGeometry(100, 100, 1200, 800)
        
        # Central widget with tabs
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout(central_widget)
        
        # Status bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.statusBar().addPermanentWidget(self.progress_bar)
        
        self.status_label = QLabel("Ready")
        self.statusBar().addWidget(self.status_label)
        
        # Tab widget
        self.tab_widget = QTabWidget()
        
        # Memories tab
        self.memory_manager = MemoryManager(self.db_manager)
        self.tab_widget.addTab(self.memory_manager, "Memories")
        
        # Faces tab
        self.face_manager = FaceManager(self.db_manager)
        self.tab_widget.addTab(self.face_manager, "Faces")
        
        # Search tab
        self.search_widget = SearchWidget(self.db_manager)
        self.tab_widget.addTab(self.search_widget, "Search")
        
        layout.addWidget(self.tab_widget)
        
        # Menu bar
        self.create_menu_bar()
    
    def create_menu_bar(self):
        """Create the menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('File')
        
        scan_action = file_menu.addAction('Scan Photos')
        scan_action.triggered.connect(self.scan_photos)
        
        exit_action = file_menu.addAction('Exit')
        exit_action.triggered.connect(self.close)
        
        # Tools menu
        tools_menu = menubar.addMenu('Tools')
        
        duplicates_action = tools_menu.addAction('Find Duplicates')
        duplicates_action.triggered.connect(self.find_duplicates)
        
        # Help menu
        help_menu = menubar.addMenu('Help')
        
        about_action = help_menu.addAction('About')
        about_action.triggered.connect(self.show_about)
    
    def apply_dark_theme(self):
        """Apply dark theme to the application"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2c3e50;
                color: white;
            }
            QTabWidget::pane {
                border: 1px solid #34495e;
                background-color: #2c3e50;
            }
            QTabWidget::tab-bar {
                left: 5px;
            }
            QTabBar::tab {
                background-color: #34495e;
                color: white;
                padding: 8px 16px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #3498db;
            }
            QTabBar::tab:hover {
                background-color: #4a6741;
            }
            QWidget {
                background-color: #2c3e50;
                color: white;
            }
            QLineEdit {
                background-color: #34495e;
                border: 2px solid #3498db;
                border-radius: 5px;
                padding: 5px;
                color: white;
            }
            QPushButton {
                background-color: #3498db;
                border: none;
                border-radius: 5px;
                padding: 8px 16px;
                color: white;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #21618c;
            }
            QComboBox {
                background-color: #34495e;
                border: 2px solid #3498db;
                border-radius: 5px;
                padding: 5px;
                color: white;
            }
            QListWidget {
                background-color: #34495e;
                border: 2px solid #3498db;
                border-radius: 5px;
                color: white;
            }
            QScrollArea {
                background-color: #2c3e50;
                border: none;
            }
            QProgressBar {
                border: 2px solid #3498db;
                border-radius: 5px;
                text-align: center;
                color: white;
            }
            QProgressBar::chunk {
                background-color: #3498db;
                border-radius: 3px;
            }
        """)
    
    def scan_photos(self):
        """Start scanning photos in the background"""
        if self.scanner_thread and self.scanner_thread.isRunning():
            QMessageBox.information(self, "Info", "Photo scanning is already in progress.")
            return
        
        # Create photos directory if it doesn't exist
        os.makedirs("photos", exist_ok=True)
        
        # Check if photos directory has any images
        photo_files = self.photo_processor.scan_photos_directory()
        if not photo_files:
            QMessageBox.information(self, "Info", "No photos found in the 'photos' directory.")
            return
        
        # Start scanning thread
        self.scanner_thread = PhotoScannerThread(self.db_manager, self.photo_processor)
        self.scanner_thread.progress_updated.connect(self.update_progress)
        self.scanner_thread.photo_processed.connect(self.photo_processed)
        self.scanner_thread.face_detected.connect(self.face_detected)
        self.scanner_thread.scanning_complete.connect(self.scanning_complete)
        
        self.scanner_thread.start()
        
        # Show progress bar
        self.progress_bar.setVisible(True)
        self.status_label.setText("Scanning photos...")
    
    def update_progress(self, value: int, message: str):
        """Update progress bar and status"""
        self.progress_bar.setValue(value)
        self.status_label.setText(message)
    
    def photo_processed(self, photo_data: Dict):
        """Handle photo processed signal"""
        # Update UI as needed
        pass
    
    def face_detected(self, photo_id: int, face_data: Dict):
        """Handle face detected signal"""
        # Update face manager
        pass
    
    def scanning_complete(self):
        """Handle scanning complete signal"""
        self.progress_bar.setVisible(False)
        self.status_label.setText("Scanning complete")
        
        # Refresh face manager
        self.face_manager.load_faces()
        
        QMessageBox.information(self, "Complete", "Photo scanning completed successfully!")
    
    def find_duplicates(self):
        """Find and display duplicate photos"""
        # This would implement duplicate detection
        QMessageBox.information(self, "Duplicates", "Duplicate detection feature coming soon!")
    
    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(self, "About", 
                         "AI Photo Manager v1.0\n\n"
                         "A comprehensive photo management application with:\n"
                         "• Face recognition and grouping\n"
                         "• Memory organization by date\n"
                         "• Secure access with password protection\n"
                         "• Slideshow and search capabilities\n"
                         "• Offline operation with local storage")


class PhotoManagerApp(QApplication):
    """Main application class"""
    
    def __init__(self, sys_argv):
        super().__init__(sys_argv)
        self.db_manager = DatabaseManager()
        self.main_window = None
    
    def run(self):
        """Run the application"""
        # Show login dialog
        login_dialog = LoginDialog(self.db_manager)
        
        if login_dialog.exec_() == QDialog.Accepted:
            # Show main window
            self.main_window = MainWindow()
            self.main_window.show()
            return self.exec_()
        else:
            return 0


def main():
    """Main entry point"""
    app = PhotoManagerApp(sys.argv)
    sys.exit(app.run())

if __name__ == "__main__":
    main()