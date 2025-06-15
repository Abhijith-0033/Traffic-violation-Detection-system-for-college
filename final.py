import os
import sys
import cv2
import numpy as np
import sqlite3
from datetime import datetime, timezone, timedelta
from ultralytics import YOLO
from tqdm import tqdm
import warnings
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from insightface.app import FaceAnalysis
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QLabel, QVBoxLayout, 
    QHBoxLayout, QGridLayout, QLineEdit, QFileDialog, QTabWidget, 
    QTableWidget, QTableWidgetItem, QMessageBox, QProgressBar, QFrame,
    QSplashScreen, QScrollArea, QComboBox, QHeaderView, QDialog
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize
from PyQt5.QtGui import QPixmap, QFont, QIcon, QImage, QPalette, QColor, QBrush
import time

warnings.filterwarnings('ignore')

class LoginDialog(QDialog):
    """Login dialog for teacher authentication"""
    login_successful = pyqtSignal(str)
    
    def __init__(self, db_path):
        super().__init__()
        self.db_path = db_path
        self.initUI()
    
    def initUI(self):
        self.setWindowTitle("Teacher Login")
        self.setFixedSize(500, 500)
        
        layout = QVBoxLayout()
        
        # Logo or icon
        self.logo_label = QLabel()
        self.logo_label.setAlignment(Qt.AlignCenter)
        
        # Create a simple text-based logo
        font = QFont("Arial", 20, QFont.Bold)
        self.logo_label.setFont(font)
        self.logo_label.setText("Traffic Violation\nDetection System")
        self.logo_label.setStyleSheet("color: #2c3e50; margin-bottom: 20px;")
        
        layout.addWidget(self.logo_label)
        
        # Username field
        self.email_input = QLineEdit()
        self.email_input.setPlaceholderText("Email")
        self.email_input.setStyleSheet("padding: 10px; margin: 5px; border: 1px solid #ccc; border-radius: 5px;")
        layout.addWidget(self.email_input)
        
        # Password field
        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("Password")
        self.password_input.setEchoMode(QLineEdit.Password)
        self.password_input.setStyleSheet("padding: 10px; margin: 5px; border: 1px solid #ccc; border-radius: 5px;")
        layout.addWidget(self.password_input)
        
        # Login button
        self.login_button = QPushButton("Login")
        self.login_button.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                padding: 10px;
                margin: 10px 5px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        self.login_button.clicked.connect(self.authenticate)
        layout.addWidget(self.login_button)
        
        # Status message
        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: red;")
        layout.addWidget(self.status_label)
        
        # Add some vertical space
        layout.addStretch()
        
        self.setLayout(layout)
    
    def authenticate(self):
        email = self.email_input.text()
        password = self.password_input.text()
        
        # Connect to database and check if teacher exists
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if the email exists in tutors table
            cursor.execute("SELECT email FROM tutors WHERE email = ?", (email,))
            result = cursor.fetchone()
            
            conn.close()
            
            if result and password == "12345678":
                self.status_label.setText("")
                self.login_successful.emit(email)
                self.close()
            else:
                self.status_label.setText("Invalid email or password")
        except Exception as e:
            self.status_label.setText(f"Login error: {str(e)}")


class ProcessingThread(QThread):
    """Thread for processing video without freezing UI"""
    progress_update = pyqtSignal(int)
    frame_update = pyqtSignal(QPixmap)
    status_update = pyqtSignal(str)
    processing_finished = pyqtSignal()
    
    def __init__(self, video_path, db_path):
        super().__init__()
        self.video_path = video_path
        self.db_path = db_path
        self.running = True
        
    def run(self):
        try:
            self.status_update.emit("Loading models...")
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Load YOLO Models
            vehicle_model = YOLO(r"D:\mini project\myproject\yolov8n.pt")
            helmet_model = YOLO(r"D:\mini project\myproject\final.pt")
            
            # Initialize InsightFace
            app = FaceAnalysis(name="buffalo_l")
            app.prepare(ctx_id=0, det_size=(640, 640))
            
            # Video Source
            cap = cv2.VideoCapture(self.video_path)
            
            if not cap.isOpened():
                self.status_update.emit("Error: Cannot open video file!")
                return
                
            # Process video frames
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Process only a subset of frames to make it faster
            for frame_number in range(0, total_frames, 75):  # Every 75th frame
                if not self.running:
                    break
                
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = cap.read()
                
                if not ret:
                    self.status_update.emit(f"Error: Could not read frame {frame_number}")
                    continue
                
                # Create a copy for visualization
                visualization_frame = frame.copy()
                
                # Step 1: Detect motorcycles & persons
                vehicle_results = vehicle_model(frame, conf=0.35, imgsz=640)
                vehicle_preds = vehicle_results[0].boxes.data.cpu().numpy()
                
                motorcycles, persons = [], []
                
                for box in vehicle_preds:
                    x1, y1, x2, y2, conf, cls = map(int, box[:6])
                    class_id = int(cls)
                    if class_id == 3:  # Motorcycle
                        motorcycles.append((x1, y1, x2, y2))
                        # Draw motorcycle boxes in red
                        visualization_frame = self.draw_boxes(visualization_frame, [(x1, y1, x2, y2)], "Motorcycle", (0, 0, 255))
                    elif class_id == 0:  # Person
                        persons.append((x1, y1, x2, y2))
                        # Draw person boxes in green
                        visualization_frame = self.draw_boxes(visualization_frame, [(x1, y1, x2, y2)], "Person", (0, 255, 0))
                
                # Step 2: Associate riders with motorcycles
                motorcycle_riders = {m: [] for m in motorcycles}  # Dictionary to track riders per motorcycle
                
                for px1, py1, px2, py2 in persons:
                    for mx1, my1, mx2, my2 in motorcycles:
                        if px1 < mx2 and px2 > mx1 and py1 < my2 and py2 > my1:  # Overlap check
                            motorcycle_riders[(mx1, my1, mx2, my2)].append((px1, py1, px2, py2))
                
                # Get current time and date
                now = datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=5, minutes=30)))
                time_str = now.strftime("%H-%M-%S")
                date_str = now.strftime("%d-%m-%Y")
                
                
                # Step 3: Run helmet detection on valid riders
                helmet_results = helmet_model(frame, conf=0.25, imgsz=640)
                helmet_preds = helmet_results[0].boxes.data.cpu().numpy()
                
                for box in helmet_preds:
                    x1, y1, x2, y2, conf, cls = map(int, box[:6])
                    class_name = helmet_model.names[int(cls)]
                    
                    if class_name == "without helmet":
                        # Mark helmet violation in magenta
                        visualization_frame = self.draw_boxes(visualization_frame, [(x1, y1, x2, y2)], "VIOLATION: No Helmet", (255, 0, 255))
                        
                        # Crop the image of the person without helmet
                        violator_image = self.crop_image(frame, x1, y1, x2, y2)
                        
                        for motorcycle, riders in motorcycle_riders.items():
                            for rx1, ry1, rx2, ry2 in riders:
                                if x1 < rx2 and x2 > rx1 and y1 < ry2 and y2 > ry1:  # Ensure it's a rider
                                    self.status_update.emit(f"Helmet violation detected on Frame {frame_number}!")
                                    
                                    # Convert to blobs
                                    image_blob = self.convert_image_to_blob(frame)
                                    face_blob = self.convert_image_to_blob(violator_image)  # Store cropped image in the face column
                                    
                                    cursor.execute("INSERT INTO violations (violation_type, time, date, image, face, student_name) VALUES (?, ?, ?, ?, ?, ?)",
                                               ("Without Helmet", time_str, date_str, image_blob, face_blob, "Unknown"))
                                    conn.commit()
                
                # Add timestamp to the frame
                cv2.putText(visualization_frame, f"Frame: {frame_number} | Date: {date_str} | Time: {time_str}", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Convert to QPixmap and emit signal to update UI
                rgb_frame = cv2.cvtColor(visualization_frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_frame.shape
                qt_image = QImage(rgb_frame.data, w, h, ch * w, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qt_image)
                self.frame_update.emit(pixmap)
                
                # Update progress
                progress = int((frame_number / total_frames) * 100)
                self.progress_update.emit(progress)
                
                # Sleep briefly to allow UI updates
                time.sleep(0.05)
            
            # Done processing, now match faces and send emails
            self.status_update.emit("Processing face recognition and preparing emails...")
            self.match_faces_and_send_emails(conn)
            
            cap.release()
            conn.close()
            self.status_update.emit("Video processing completed.")
            self.processing_finished.emit()
            
        except Exception as e:
            self.status_update.emit(f"Error: {str(e)}")
            self.processing_finished.emit()
    
    def stop(self):
        self.running = False
    
    def convert_image_to_blob(self, image):
        """Convert OpenCV image to BLOB format for storing in SQLite."""
        _, buffer = cv2.imencode(".jpg", image)
        return buffer.tobytes()
    
    def crop_image(self, frame, x1, y1, x2, y2, padding=10):
        """Crop image with padding and ensure coordinates are within frame bounds."""
        height, width = frame.shape[:2]
        
        # Add padding but ensure within frame bounds
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(width, x2 + padding)
        y2 = min(height, y2 + padding)
        
        return frame[y1:y2, x1:x2]
    
    def draw_boxes(self, frame, boxes, label, color):
        """Draw bounding boxes with labels on the frame."""
        for box in boxes:
            if len(box) == 4:  # Simple box coordinates
                x1, y1, x2, y2 = box
            else:  # Full prediction with confidence and class
                x1, y1, x2, y2 = box[:4]
            
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label background
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(frame, (x1, y1 - text_size[1] - 10), (x1 + text_size[0] + 10, y1), color, -1)
            
            # Draw label text
            cv2.putText(frame, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame
    
    def cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors."""
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    def match_faces_and_send_emails(self, conn):
        """Match faces in violations with student database and send emails."""
        cursor = conn.cursor()
        
        # Initialize InsightFace
        app = FaceAnalysis(name="buffalo_l")
        app.prepare(ctx_id=0, det_size=(640, 640))
        
        # Fetch all student records
        cursor.execute("SELECT roll_no, name, photo FROM students")
        students = cursor.fetchall()
        
        # Load student images and encode faces
        student_encodings = {}
        for roll_no, name, photo_blob in students:
            if photo_blob:
                self.status_update.emit(f"Processing student: {name}")
                nparr = np.frombuffer(photo_blob, np.uint8)
                student_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                faces = app.get(student_image)
                if faces:
                    student_encodings[roll_no] = (name, faces[0].embedding)
        
        # Fetch all violation records
        cursor.execute("SELECT id, face, image, violation_type, time FROM violations WHERE face IS NOT NULL")
        violations = cursor.fetchall()
        
        # Dictionary to store violations per student
        student_violations = {}
        
        for violation_id, face_blob, frame_blob, violation_type, time in violations:
            if face_blob and frame_blob:
                # Convert BLOBs to images
                face_image = cv2.imdecode(np.frombuffer(face_blob, np.uint8), cv2.IMREAD_COLOR)
                frame_image = cv2.imdecode(np.frombuffer(frame_blob, np.uint8), cv2.IMREAD_COLOR)
                
                faces = app.get(face_image)
                if faces:
                    violation_encoding = faces[0].embedding
                    
                    for roll_no, (student_name, student_encoding) in student_encodings.items():
                        similarity = self.cosine_similarity(student_encoding, violation_encoding)
                        if similarity > 0.55:
                            # Update the violations table with matched student name
                            cursor.execute("UPDATE violations SET student_name = ? WHERE id = ?", (student_name, violation_id))
                            conn.commit()
                            
                            if roll_no not in student_violations:
                                student_violations[roll_no] = {"name": student_name, "violations": [], "frame": frame_image}
                            student_violations[roll_no]["violations"].append((violation_type, time, violation_id))  # Include violation ID
                            break  # Stop checking once a match is found
        
        # Send emails with grouped violations
        for roll_no, data in student_violations.items():
            cursor.execute("SELECT tutors.email FROM tutors JOIN students ON students.tutor_id = tutors.tutor_id WHERE students.roll_no = ?", (roll_no,))
            tutor_result = cursor.fetchone()
            if tutor_result:
                tutor_email = tutor_result[0]
                self.status_update.emit(f"Sending email to {tutor_email} for student {data['name']}")
                image_path = f"violation_{roll_no}.jpg"
                cv2.imwrite(image_path, data["frame"])
                self.send_email(tutor_email, data["name"], roll_no, data["violations"], image_path)
                os.remove(image_path)
    
    def send_email(self, to_email, student_name, roll_no, violations, image_path):
        """Send email notification about violations."""
        sender_email = "s6aidsminiproject@gmail.com"
        sender_password = "lfis audy vult qrhu"
        smtp_server = "smtp.gmail.com"
        smtp_port = 587
        
        msg = MIMEMultipart()
        msg["From"] = sender_email
        msg["To"] = to_email
        msg["Subject"] = f"Violation Alert: {student_name} ({roll_no})"
        
        body = f"""
        Dear Tutor,
        
        Multiple violations have been detected for your student:
        - Student Name: {student_name}
        - Roll No: {roll_no}
        
        Violations:
        """
        
        for violation in violations:
            body += f"- {violation[0]} (Time: {violation[1]})\n"
        
        body += "\nPlease take necessary action.\n\nRegards,\nViolation Detection System"
        
        msg.attach(MIMEText(body, "plain"))
        
        try:
            with open(image_path, "rb") as img_file:
                img = MIMEImage(img_file.read(), name=os.path.basename(image_path))
            msg.attach(img)
            
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, to_email, msg.as_string())
            server.quit()
            
            # Log the email in the database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            violation_ids = ",".join([str(v[2]) for v in violations])  # Assuming violation IDs are stored in the tuple
            cursor.execute("""
                INSERT INTO email_logs (student_name, roll_no, tutor_email, status, violation_ids)
                VALUES (?, ?, ?, ?, ?)
            """, (student_name, roll_no, to_email, "Sent", violation_ids))
            conn.commit()
            conn.close()
            
            self.status_update.emit(f"Email sent to {to_email}")
        except Exception as e:
            # Log the failed email attempt
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            violation_ids = ",".join([str(v[2]) for v in violations])
            cursor.execute("""
                INSERT INTO email_logs (student_name, roll_no, tutor_email, status, violation_ids)
                VALUES (?, ?, ?, ?, ?)
            """, (student_name, roll_no, to_email, "Failed", violation_ids))
            conn.commit()
            conn.close()
            
            self.status_update.emit(f"Failed to send email: {e}")


class MainWindow(QMainWindow):
    """Main application window"""
    def __init__(self):
        super().__init__()
        # Setup database path
        self.db_path = os.path.join(os.getcwd(), r"D:\mini project\myproject\violations_dbb.db")
        self.ensure_database_exists()
        
        # Setup UI
        self.initUI()
        
        # Show login dialog
        self.show_login()
    
    def ensure_database_exists(self):
        """Ensure the database exists and has required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables if they don't exist
        cursor.execute('''CREATE TABLE IF NOT EXISTS violations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        violation_type TEXT NOT NULL,
                        name TEXT NOT NULL DEFAULT 'Unknown',
                        time TEXT NOT NULL,
                        date TEXT NOT NULL,
                        image BLOB NOT NULL,
                        face BLOB,
                        student_name TEXT DEFAULT 'Unknown',
                        confidence REAL
                    )''')
        
        # Create tutors table if it doesn't exist
        cursor.execute('''CREATE TABLE IF NOT EXISTS tutors (
                        tutor_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        email TEXT NOT NULL UNIQUE,
                        department TEXT NOT NULL
                    )''')
        
        # Create students table if it doesn't exist
        cursor.execute('''CREATE TABLE IF NOT EXISTS students (
                        roll_no TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        tutor_id INTEGER,
                        department TEXT NOT NULL,
                        photo BLOB,
                        FOREIGN KEY (tutor_id) REFERENCES tutors(tutor_id)
                    )''')
        
        # Create email_logs table if it doesn't exist
        cursor.execute('''CREATE TABLE IF NOT EXISTS email_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        student_name TEXT NOT NULL,
                        roll_no TEXT NOT NULL,
                        tutor_email TEXT NOT NULL,
                        send_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        status TEXT NOT NULL,
                        violation_ids TEXT NOT NULL
                    )''')
        
        conn.commit()
        conn.close()
    
    def initUI(self):
        self.setWindowTitle("Traffic Violation Detection System")
        self.setGeometry(100, 100, 1200, 800)
        
        # Create main tab widget
        self.tabs = QTabWidget()
        
        # Create the various tabs
        self.create_dashboard_tab()
        self.create_violations_tab()
        self.create_students_tab()
        self.create_detection_tab()
        self.create_email_logs_tab()
        
        # Set the main tab widget as the central widget
        self.setCentralWidget(self.tabs)
    
    def show_login(self):
        """Show the login dialog"""
        self.login_dialog = LoginDialog(self.db_path)
        self.login_dialog.login_successful.connect(self.on_login_successful)
        self.login_dialog.show()
    
    def on_login_successful(self, email):
        """Called when login is successful"""
        self.teacher_email = email
        # Get teacher name
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM tutors WHERE email = ?", (email,))
        result = cursor.fetchone()
        conn.close()
        
        if result:
            self.teacher_name = result[0]
            self.setWindowTitle(f"Traffic Violation Detection System - Welcome, {self.teacher_name}")
        
        # Refresh data in tabs
        self.refresh_all_data()
        
        # Show main window
        self.show()
    
    def create_dashboard_tab(self):
        """Create the dashboard tab"""
        dashboard_tab = QWidget()
        layout = QVBoxLayout()
        
        # Welcome banner
        welcome_frame = QFrame()
        welcome_frame.setFrameShape(QFrame.StyledPanel)
        welcome_frame.setStyleSheet("background-color: #3498db; padding: 20px; border-radius: 10px;")
        welcome_layout = QVBoxLayout()
        
        welcome_label = QLabel("Traffic Violation Detection System")
        welcome_label.setStyleSheet("color: white; font-size: 24px; font-weight: bold;")
        welcome_layout.addWidget(welcome_label)
        
        desc_label = QLabel("Monitor traffic violations, identify students, and send automated notifications.")
        desc_label.setStyleSheet("color: white; font-size: 16px;")
        welcome_layout.addWidget(desc_label)
        
        welcome_frame.setLayout(welcome_layout)
        layout.addWidget(welcome_frame)
        
        # Stats section
        stats_layout = QHBoxLayout()
        
        # Violations stats
        violations_frame = self.create_stat_frame("Total Violations", "0", "rgb(231, 76, 60)")
        stats_layout.addWidget(violations_frame)
        
        # Students stats
        students_frame = self.create_stat_frame("Total Students", "0", "rgb(46, 204, 113)")
        stats_layout.addWidget(students_frame)
        
        # Email stats
        emails_frame = self.create_stat_frame("Emails Sent", "0", "rgb(52, 152, 219)")
        stats_layout.addWidget(emails_frame)
        
        layout.addLayout(stats_layout)
        
        # Quick actions section
        actions_frame = QFrame()
        actions_frame.setFrameShape(QFrame.StyledPanel)
        actions_frame.setStyleSheet("background-color: #f9f9f9; padding: 15px; border-radius: 10px;")
        
        actions_layout = QVBoxLayout()
        
        actions_title = QLabel("Quick Actions")
        actions_title.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 10px;")
        actions_layout.addWidget(actions_title)
        
        # Action buttons
        buttons_layout = QHBoxLayout()
        
        detection_btn = QPushButton("Start Detection")
        detection_btn.setStyleSheet(self.get_button_style("rgb(52, 152, 219)"))
        detection_btn.clicked.connect(lambda: self.tabs.setCurrentIndex(3))  # Switch to detection tab
        buttons_layout.addWidget(detection_btn)
        
        view_violations_btn = QPushButton("View Violations")
        view_violations_btn.setStyleSheet(self.get_button_style("rgb(155, 89, 182)"))
        view_violations_btn.clicked.connect(lambda: self.tabs.setCurrentIndex(1))  # Switch to violations tab
        buttons_layout.addWidget(view_violations_btn)
        
        view_students_btn = QPushButton("View Students")
        view_students_btn.setStyleSheet(self.get_button_style("rgb(46, 204, 113)"))
        view_students_btn.clicked.connect(lambda: self.tabs.setCurrentIndex(2))  # Switch to students tab
        buttons_layout.addWidget(view_students_btn)
        
        actions_layout.addLayout(buttons_layout)
        actions_frame.setLayout(actions_layout)
        
        layout.addWidget(actions_frame)
        
        # Recent violations table
        recent_frame = QFrame()
        recent_frame.setFrameShape(QFrame.StyledPanel)
        recent_frame.setStyleSheet("background-color: #f9f9f9; padding: 15px; border-radius: 10px;")
        
        recent_layout = QVBoxLayout()
        
        recent_title = QLabel("Recent Violations")
        recent_title.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 10px;")
        recent_layout.addWidget(recent_title)
        
        self.recent_violations_table = QTableWidget(0, 5)
        self.recent_violations_table.setHorizontalHeaderLabels(["ID", "Violation Type", "Student", "Date", "Time"])
        self.recent_violations_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.recent_violations_table.setStyleSheet("background: white;")
        
        recent_layout.addWidget(self.recent_violations_table)
        recent_frame.setLayout(recent_layout)
        
        layout.addWidget(recent_frame)
        
        # Set the layout
        dashboard_tab.setLayout(layout)
        
        # Add tab to main tab widget
        self.tabs.addTab(dashboard_tab, "Dashboard")
        
        # Store references for updating stats
        self.violations_count_label = violations_frame.findChild(QLabel, "count_label")
        self.students_count_label = students_frame.findChild(QLabel, "count_label")
        self.emails_count_label = emails_frame.findChild(QLabel, "count_label")
    
    def create_stat_frame(self, title, count, color):
        """Create a statistics frame for the dashboard"""
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        frame.setStyleSheet(f"background-color: {color}; padding: 20px; border-radius: 10px;")
        
        layout = QVBoxLayout()
        
        title_label = QLabel(title)
        title_label.setStyleSheet("color: white; font-size: 16px;")
        layout.addWidget(title_label)
        
        count_label = QLabel(count)
        count_label.setObjectName("count_label")  # Set object name for later reference
        count_label.setStyleSheet("color: white; font-size: 28px; font-weight: bold;")
        layout.addWidget(count_label)
        
        frame.setLayout(layout)
        return frame
    
    def get_button_style(self, color):
        """Get the style string for a button"""
        rgba_color = color.replace('rgb', 'rgba').replace(')', ', 0.8)')
        return f"""
            QPushButton {{
                background-color: {color};
                color: white;
                padding: 12px;
                border-radius: 5px;
                font-weight: bold;
                font-size: 14px;
            }}
            QPushButton:hover {{
                background-color: {rgba_color};
            }}
        """
    def create_violations_tab(self):
        """Create the violations tab"""
        violations_tab = QWidget()
        layout = QVBoxLayout()
        
        # Header
        title_label = QLabel("Violations Database")
        title_label.setStyleSheet("font-size: 20px; font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(title_label)
        
        # Filter controls
        filter_frame = QFrame()
        filter_frame.setFrameShape(QFrame.StyledPanel)
        filter_frame.setStyleSheet("background-color: #f5f5f5; padding: 10px; border-radius: 5px;")
        
        filter_layout = QHBoxLayout()
        
        # Violation type filter
        filter_layout.addWidget(QLabel("Filter by type:"))
        self.violation_type_combo = QComboBox()
        self.violation_type_combo.addItems(["All", "Triple Riding", "Without Helmet"])
        self.violation_type_combo.currentTextChanged.connect(self.refresh_violations_table)
        filter_layout.addWidget(self.violation_type_combo)
        
        # Date filter
        filter_layout.addWidget(QLabel("Filter by date:"))
        self.date_filter = QLineEdit()
        self.date_filter.setPlaceholderText("DD-MM-YYYY")
        self.date_filter.textChanged.connect(self.refresh_violations_table)
        filter_layout.addWidget(self.date_filter)
        
        # Student name filter
        filter_layout.addWidget(QLabel("Filter by student:"))
        self.student_filter = QLineEdit()
        self.student_filter.setPlaceholderText("Student name")
        self.student_filter.textChanged.connect(self.refresh_violations_table)
        filter_layout.addWidget(self.student_filter)
        
        # Refresh button
        refresh_btn = QPushButton("Refresh")
        refresh_btn.setStyleSheet(self.get_button_style("rgb(52, 152, 219)"))
        refresh_btn.clicked.connect(self.refresh_violations_table)
        filter_layout.addWidget(refresh_btn)
        
        filter_frame.setLayout(filter_layout)
        layout.addWidget(filter_frame)
        
        # Violations table
        self.violations_table = QTableWidget(0, 7)
        self.violations_table.setHorizontalHeaderLabels(["ID", "Violation Type", "Student Name", "Date", "Time", "Image", "Actions"])
        self.violations_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.violations_table.setSelectionBehavior(QTableWidget.SelectRows)
        layout.addWidget(self.violations_table)
        
        violations_tab.setLayout(layout)
        self.tabs.addTab(violations_tab, "Violations")
    
    def display_student_image(self):
        """Display the selected student's image in a larger box."""
        selected_row = self.students_table.currentRow()
        if selected_row >= 0:
            # Get the photo blob from the selected row
            photo_item = self.students_table.item(selected_row, 4)
            if photo_item and photo_item.data(Qt.UserRole):  # Check if photo exists
                photo_blob = photo_item.data(Qt.UserRole)
                
                # Convert blob to image
                nparr = np.frombuffer(photo_blob, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Resize image to fit the display area while maintaining aspect ratio
                max_width = self.student_image_label.width()
                max_height = self.student_image_label.height()
                h, w, _ = image_rgb.shape
                aspect_ratio = w / h
                
                if w > max_width or h > max_height:
                    if w > max_width:
                        w = max_width
                        h = int(w / aspect_ratio)
                    if h > max_height:
                        h = max_height
                        w = int(h * aspect_ratio)
                
                image_rgb = cv2.resize(image_rgb, (w, h))
                
                # Convert to QPixmap and display
                q_img = QImage(image_rgb.data, w, h, image_rgb.strides[0], QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_img)
                self.student_image_label.setPixmap(pixmap)
            else:
                self.student_image_label.clear()  # Clear the label if no image is available
        else:
            self.student_image_label.clear()  # Clear the label if no row is selected
    
    def create_students_tab(self):
        """Create the students tab with a larger image display."""
        students_tab = QWidget()
        layout = QVBoxLayout()
        
        # Header
        title_label = QLabel("Students Database")
        title_label.setStyleSheet("font-size: 20px; font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(title_label)
        
        # Filter controls
        filter_frame = QFrame()
        filter_frame.setFrameShape(QFrame.StyledPanel)
        filter_frame.setStyleSheet("background-color: #f5f5f5; padding: 10px; border-radius: 5px;")
        
        filter_layout = QHBoxLayout()
        
        # Department filter
        filter_layout.addWidget(QLabel("Filter by department:"))
        self.dept_filter = QLineEdit()
        self.dept_filter.setPlaceholderText("Department")
        self.dept_filter.textChanged.connect(self.refresh_students_table)
        filter_layout.addWidget(self.dept_filter)
        
        # Name filter
        filter_layout.addWidget(QLabel("Filter by name:"))
        self.name_filter = QLineEdit()
        self.name_filter.setPlaceholderText("Student name")
        self.name_filter.textChanged.connect(self.refresh_students_table)
        filter_layout.addWidget(self.name_filter)
        
        # Roll number filter
        filter_layout.addWidget(QLabel("Filter by roll no:"))
        self.roll_filter = QLineEdit()
        self.roll_filter.setPlaceholderText("Roll number")
        self.roll_filter.textChanged.connect(self.refresh_students_table)
        filter_layout.addWidget(self.roll_filter)
        
        # Refresh button
        refresh_btn = QPushButton("Refresh")
        refresh_btn.setStyleSheet(self.get_button_style("rgb(52, 152, 219)"))
        refresh_btn.clicked.connect(self.refresh_students_table)
        filter_layout.addWidget(refresh_btn)
        
        filter_frame.setLayout(filter_layout)
        layout.addWidget(filter_frame)
        
        # Students table
        self.students_table = QTableWidget(0, 5)
        self.students_table.setHorizontalHeaderLabels(["Roll No", "Name", "Department", "Tutor", "Photo"])
        self.students_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.students_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.students_table.setSelectionMode(QTableWidget.SingleSelection)
        self.students_table.itemSelectionChanged.connect(self.display_student_image)  # Connect selection change
        layout.addWidget(self.students_table)
        
        # Add a QLabel to display the larger image
        self.student_image_label = QLabel()
        self.student_image_label.setAlignment(Qt.AlignCenter)
        self.student_image_label.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ccc; border-radius: 5px;")
        self.student_image_label.setMinimumSize(300, 300)  # Set minimum size for the image display
        layout.addWidget(self.student_image_label)
        
        students_tab.setLayout(layout)
        self.tabs.addTab(students_tab, "Students")

        def display_student_image(self):
            """Display the selected student's image in a larger box."""
            selected_row = self.students_table.currentRow()
            if selected_row >= 0:
                # Get the photo blob from the selected row
                photo_item = self.students_table.item(selected_row, 4)
                if photo_item and photo_item.data(Qt.UserRole):  # Check if photo exists
                    photo_blob = photo_item.data(Qt.UserRole)
                    
                    # Convert blob to image
                    nparr = np.frombuffer(photo_blob, np.uint8)
                    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # Resize image to fit the display area while maintaining aspect ratio
                    max_width = self.student_image_label.width()
                    max_height = self.student_image_label.height()
                    h, w, _ = image_rgb.shape
                    aspect_ratio = w / h
                    
                    if w > max_width or h > max_height:
                        if w > max_width:
                            w = max_width
                            h = int(w / aspect_ratio)
                        if h > max_height:
                            h = max_height
                            w = int(h * aspect_ratio)
                    
                    image_rgb = cv2.resize(image_rgb, (w, h))
                    
                    # Convert to QPixmap and display
                    q_img = QImage(image_rgb.data, w, h, image_rgb.strides[0], QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(q_img)
                    self.student_image_label.setPixmap(pixmap)
                else:
                    self.student_image_label.clear()  # Clear the label if no image is available
            else:
                self.student_image_label.clear()  # Clear the label if no row is selected
    
    def create_detection_tab(self):
        """Create the detection tab for video processing"""
        detection_tab = QWidget()
        layout = QVBoxLayout()
        
        # Header
        title_label = QLabel("Traffic Violation Detection")
        title_label.setStyleSheet("font-size: 20px; font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(title_label)
        
        # Video selection controls
        controls_frame = QFrame()
        controls_frame.setFrameShape(QFrame.StyledPanel)
        controls_frame.setStyleSheet("background-color: #f5f5f5; padding: 15px; border-radius: 5px;")
        
        controls_layout = QHBoxLayout()
        
        self.video_path_input = QLineEdit()
        self.video_path_input.setPlaceholderText("Select video file...")
        self.video_path_input.setReadOnly(True)
        controls_layout.addWidget(self.video_path_input, 3)
        
        browse_btn = QPushButton("Browse")
        browse_btn.setStyleSheet(self.get_button_style("rgb(52, 152, 219)"))
        browse_btn.clicked.connect(self.browse_video)
        controls_layout.addWidget(browse_btn, 1)
        
        self.process_btn = QPushButton("Start Processing")
        self.process_btn.setStyleSheet(self.get_button_style("rgb(46, 204, 113)"))
        self.process_btn.clicked.connect(self.start_video_processing)
        self.process_btn.setEnabled(False)
        controls_layout.addWidget(self.process_btn, 1)
        
        controls_frame.setLayout(controls_layout)
        layout.addWidget(controls_frame)
        
        # Video display area
        self.video_frame = QLabel("No video loaded")
        self.video_frame.setAlignment(Qt.AlignCenter)
        self.video_frame.setMinimumSize(640, 480)
        self.video_frame.setStyleSheet("background-color: #444; color: white; border-radius: 5px;")
        layout.addWidget(self.video_frame)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #bbb;
                border-radius: 5px;
                text-align: center;
                height: 20px;
            }
            QProgressBar::chunk {
                background-color: #3498db;
                border-radius: 5px;
            }
        """)
        layout.addWidget(self.progress_bar)
        
        # Status label
        self.status_label = QLabel("Ready to process video")
        self.status_label.setStyleSheet("font-style: italic; color: #555;")
        layout.addWidget(self.status_label)
        
        detection_tab.setLayout(layout)
        self.tabs.addTab(detection_tab, "Detection")
    
    def create_email_logs_tab(self):
        """Create the email logs tab"""
        email_tab = QWidget()
        layout = QVBoxLayout()
        
        # Header
        title_label = QLabel("Email Notification Logs")
        title_label.setStyleSheet("font-size: 20px; font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(title_label)
        
        # Email logs table
        self.email_logs_table = QTableWidget(0, 6)
        self.email_logs_table.setHorizontalHeaderLabels(["ID", "Student Name", "Roll No", "Tutor Email", "Time", "Status"])
        self.email_logs_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.email_logs_table)
        
        # Refresh button
        refresh_btn = QPushButton("Refresh Logs")
        refresh_btn.setStyleSheet(self.get_button_style("rgb(52, 152, 219)"))
        refresh_btn.clicked.connect(self.refresh_email_logs)
        layout.addWidget(refresh_btn)
        
        email_tab.setLayout(layout)
        self.tabs.addTab(email_tab, "Email Logs")
    
    def browse_video(self):
        """Open file dialog to select a video"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video", "", "Video Files (*.mp4 *.avi *.mov *.mkv)"
        )
        
        if file_path:
            self.video_path_input.setText(file_path)
            self.process_btn.setEnabled(True)
            self.status_label.setText(f"Video selected: {os.path.basename(file_path)}")
            
            # Show a preview frame
            cap = cv2.VideoCapture(file_path)
            ret, frame = cap.read()
            if ret:
                # Resize the frame to fit the display area
                frame = cv2.resize(frame, (640, 480))
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_frame.shape
                qt_image = QImage(rgb_frame.data, w, h, ch * w, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qt_image)
                self.video_frame.setPixmap(pixmap)
            cap.release()
    
    def start_video_processing(self):
        """Start processing the selected video"""
        video_path = self.video_path_input.text()
        
        if not video_path:
            QMessageBox.warning(self, "Error", "Please select a video file first.")
            return
        
        # Disable the process button during processing
        self.process_btn.setEnabled(False)
        self.process_btn.setText("Processing...")
        self.progress_bar.setValue(0)
        self.status_label.setText("Initializing processing...")
        
        # Create and start the processing thread
        self.processing_thread = ProcessingThread(video_path, self.db_path)
        self.processing_thread.progress_update.connect(self.update_progress)
        self.processing_thread.frame_update.connect(self.update_frame)
        self.processing_thread.status_update.connect(self.update_status)
        self.processing_thread.processing_finished.connect(self.processing_complete)
        self.processing_thread.start()
    
    def update_progress(self, value):
        """Update the progress bar"""
        self.progress_bar.setValue(value)
    
    def update_frame(self, pixmap):
        """Update the video frame display"""
        self.video_frame.setPixmap(pixmap)
    
    def update_status(self, message):
        """Update the status label"""
        self.status_label.setText(message)
    
    def processing_complete(self):
        """Called when video processing is complete"""
        self.process_btn.setEnabled(True)
        self.process_btn.setText("Start Processing")
        self.progress_bar.setValue(100)
        self.status_label.setText("Processing complete!")
        
        # Refresh all data
        self.refresh_all_data()
        
        # Show completion message
        QMessageBox.information(self, "Processing Complete", 
                                "Video processing has been completed. Violations have been recorded and notifications sent.")
    
    def refresh_all_data(self):
        """Refresh all data displays"""
        self.refresh_dashboard_stats()
        self.refresh_violations_table()
        self.refresh_students_table()
        self.refresh_email_logs()
    
    def refresh_dashboard_stats(self):
        """Refresh statistics on the dashboard"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get violation count
            cursor.execute("SELECT COUNT(*) FROM violations")
            violations_count = cursor.fetchone()[0]
            self.violations_count_label.setText(str(violations_count))
            
            # Get students count
            cursor.execute("SELECT COUNT(*) FROM students")
            students_count = cursor.fetchone()[0]
            self.students_count_label.setText(str(students_count))
            
            # Get email count
            cursor.execute("SELECT COUNT(*) FROM email_logs")
            emails_count = cursor.fetchone()[0]
            self.emails_count_label.setText(str(emails_count))
            
            # Refresh recent violations
            cursor.execute("""
                SELECT id, violation_type, student_name, date, time 
                FROM violations 
                ORDER BY id DESC LIMIT 5
            """)
            recent_violations = cursor.fetchall()
            
            self.recent_violations_table.setRowCount(0)
            for row_idx, violation in enumerate(recent_violations):
                self.recent_violations_table.insertRow(row_idx)
                for col_idx, value in enumerate(violation):
                    self.recent_violations_table.setItem(row_idx, col_idx, QTableWidgetItem(str(value)))
            
            conn.close()
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to refresh dashboard: {str(e)}")
    
    def refresh_violations_table(self):
        """Refresh the violations table with filters applied"""
        try:
            # Get filter values
            violation_type = self.violation_type_combo.currentText()
            date_filter = self.date_filter.text()
            student_filter = self.student_filter.text()
            
            # Build query
            query = "SELECT id, violation_type, student_name, date, time, image FROM violations WHERE 1=1"
            params = []
            
            if violation_type != "All":
                query += " AND violation_type = ?"
                params.append(violation_type)
            
            if date_filter:
                query += " AND date LIKE ?"
                params.append(f"%{date_filter}%")
            
            if student_filter:
                query += " AND student_name LIKE ?"
                params.append(f"%{student_filter}%")
            
            query += " ORDER BY id DESC"
            
            # Execute query
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(query, params)
            violations = cursor.fetchall()
            
            # Update table
            self.violations_table.setRowCount(0)
            for row_idx, violation in enumerate(violations):
                self.violations_table.insertRow(row_idx)
                
                # Add data columns
                for col_idx in range(5):  # ID, Type, Student, Date, Time
                    self.violations_table.setItem(row_idx, col_idx, QTableWidgetItem(str(violation[col_idx])))
                
                # Add image thumbnail
                if violation[5]:  # Image blob
                    img_data = np.frombuffer(violation[5], np.uint8)
                    img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
                    img = cv2.resize(img, (80, 60))
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    h, w, ch = img_rgb.shape
                    q_img = QImage(img_rgb.data, w, h, ch * w, QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(q_img)
                    
                    label = QLabel()
                    label.setPixmap(pixmap)
                    label.setAlignment(Qt.AlignCenter)
                    self.violations_table.setCellWidget(row_idx, 5, label)
                
                # Add view button
                view_btn = QPushButton("View")
                view_btn.setStyleSheet(self.get_button_style("rgb(52, 152, 219)"))
                view_btn.clicked.connect(lambda checked, vid=violation[0]: self.view_violation(vid))
                self.violations_table.setCellWidget(row_idx, 6, view_btn)
            
            conn.close()
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to refresh violations: {str(e)}")
    
    def refresh_students_table(self):
        """Refresh the students table with filters applied."""
        try:
            # Get filter values
            dept_filter = self.dept_filter.text()
            name_filter = self.name_filter.text()
            roll_filter = self.roll_filter.text()
            
            # Build query
            query = """
                SELECT s.roll_no, s.name, s.department, t.name, s.photo 
                FROM students s
                LEFT JOIN tutors t ON s.tutor_id = t.tutor_id
                WHERE 1=1
            """
            params = []
            
            if dept_filter:
                query += " AND s.department LIKE ?"
                params.append(f"%{dept_filter}%")
            
            if name_filter:
                query += " AND s.name LIKE ?"
                params.append(f"%{name_filter}%")
            
            if roll_filter:
                query += " AND s.roll_no LIKE ?"
                params.append(f"%{roll_filter}%")
            
            query += " ORDER BY s.roll_no"
            
            # Execute query
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(query, params)
            students = cursor.fetchall()
            
            # Update table
            self.students_table.setRowCount(0)
            for row_idx, student in enumerate(students):
                self.students_table.insertRow(row_idx)
                
                # Add data columns
                for col_idx in range(4):  # Roll No, Name, Department, Tutor
                    self.students_table.setItem(row_idx, col_idx, QTableWidgetItem(str(student[col_idx] or "")))
                
                # Add photo thumbnail if available
                if student[4]:  # Photo blob
                    img_data = np.frombuffer(student[4], np.uint8)
                    img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
                    img = cv2.resize(img, (60, 60))
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    h, w, ch = img_rgb.shape
                    q_img = QImage(img_rgb.data, w, h, ch * w, QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(q_img)
                    
                    label = QLabel()
                    label.setPixmap(pixmap)
                    label.setAlignment(Qt.AlignCenter)
                    self.students_table.setCellWidget(row_idx, 4, label)
                    
                    # Store the photo blob in the item's UserRole
                    photo_item = QTableWidgetItem()
                    photo_item.setData(Qt.UserRole, student[4])
                    self.students_table.setItem(row_idx, 4, photo_item)
                else:
                    self.students_table.setItem(row_idx, 4, QTableWidgetItem("No Photo"))
            
            conn.close()
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to refresh students: {str(e)}")
    
    def refresh_email_logs(self):
        """Refresh the email logs table"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, student_name, roll_no, tutor_email, send_time, status 
                FROM email_logs 
                ORDER BY send_time DESC
            """)
            logs = cursor.fetchall()
            
            self.email_logs_table.setRowCount(0)
            for row_idx, log in enumerate(logs):
                self.email_logs_table.insertRow(row_idx)
                for col_idx, value in enumerate(log):
                    self.email_logs_table.setItem(row_idx, col_idx, QTableWidgetItem(str(value)))
            
            conn.close()
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to refresh email logs: {str(e)}")
    
    def view_violation(self, violation_id):
        """View details of a specific violation"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, violation_type, student_name, date, time, image 
                FROM violations 
                WHERE id = ?
            """, (violation_id,))
            violation = cursor.fetchone()
            
            if not violation:
                QMessageBox.warning(self, "Error", f"Violation ID {violation_id} not found.")
                return
            
            # Create a details dialog
            details_dialog = QDialog(self)
            details_dialog.setWindowTitle(f"Violation #{violation_id} Details")
            details_dialog.setMinimumSize(800, 600)
            
            layout = QVBoxLayout()
            
            # Information section
            info_frame = QFrame()
            info_frame.setFrameShape(QFrame.StyledPanel)
            info_frame.setStyleSheet("background-color: #f5f5f5; padding: 15px; border-radius: 5px;")
            
            info_layout = QGridLayout()
            
            info_layout.addWidget(QLabel("Violation ID:"), 0, 0)
            info_layout.addWidget(QLabel("Violation ID:"), 0, 0)
            info_layout.addWidget(QLabel(str(violation[0])), 0, 1)

            info_layout.addWidget(QLabel("Violation Type:"), 1, 0)
            info_layout.addWidget(QLabel(violation[1]), 1, 1)  # Removed extra ')'

            info_layout.addWidget(QLabel("Student:"), 2, 0)
            info_layout.addWidget(QLabel(violation[2]), 2, 1)  # Removed extra ')'

            info_layout.addWidget(QLabel("Date:"), 3, 0)
            info_layout.addWidget(QLabel(violation[3]), 3, 1)  # Removed extra ')'

            info_layout.addWidget(QLabel("Time:"), 4, 0)
            info_layout.addWidget(QLabel(violation[4]), 4, 1)  # Removed extra ')'

            
            info_frame.setLayout(info_layout)
            layout.addWidget(info_frame)
            
            # Image section
            if violation[5]:  # Image blob
                img_data = np.frombuffer(violation[5], np.uint8)
                img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w, ch = img_rgb.shape
                
                # Calculate aspect ratio for resizing
                max_height = 400
                max_width = 700
                
                if h > max_height or w > max_width:
                    aspect = w / h
                    if h > max_height:
                        h = max_height
                        w = int(aspect * h)
                    if w > max_width:
                        w = max_width
                        h = int(w / aspect)
                
                img_rgb = cv2.resize(img_rgb, (w, h))
                q_img = QImage(img_rgb.data, w, h, ch * w, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_img)
                
                image_label = QLabel()
                image_label.setPixmap(pixmap)
                image_label.setAlignment(Qt.AlignCenter)
                layout.addWidget(image_label)
            
            # Close button
            close_btn = QPushButton("Close")
            close_btn.setStyleSheet(self.get_button_style("rgb(52, 152, 219)"))
            close_btn.clicked.connect(details_dialog.close)
            layout.addWidget(close_btn)
            
            details_dialog.setLayout(layout)
            details_dialog.exec_()
            
            conn.close()
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to view violation: {str(e)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Show splash screen
    splash_pixmap = QPixmap(300, 200)
    splash_pixmap.fill(QColor(52, 152, 219))
    
    splash = QSplashScreen(splash_pixmap)
    splash.showMessage("Starting Traffic Violation Detection System...", 
                      Qt.AlignCenter | Qt.AlignBottom, Qt.white)
    splash.show()
    
    # Process events to display splash screen
    app.processEvents()
    
    # Create main window after a short delay
    time.sleep(2)
    window = MainWindow()
    
    # Close splash once main window is ready
    splash.finish(window)
    
    sys.exit(app.exec_())