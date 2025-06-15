# Traffic-violation-Detection-system-for-college
This project is a Traffic Violation Detection System that uses computer vision to detect helmet violations and other traffic violations from video feeds. The system can identify violators, match them with a student database, and send email notifications to their tutors.
Features
Helmet violation detection using YOLO models

Face recognition for violator identification

Database storage of violations

Email notification system

Graphical user interface with PyQt5

Dashboard with statistics and recent violations

Student and violation management

Requirements
The system requires the following Python packages:

text
ultralytics==8.0.0
opencv-python==4.5.5.64
numpy==1.21.5
sqlite3==2.6.0
PyQt5==5.15.7
tqdm==4.64.0
insightface==0.7.3
Installation
Clone this repository

Install the required packages:

bash
pip install -r requirements.txt
Download the YOLO models:

yolov8n.pt (for vehicle detection)

helmet detection model (custom trained)

Ensure you have a SQLite database file at the specified path (D:\mini project\myproject\violations_dbb.db)

Usage
Run the application:

bash
python traffic_violation_system.py
Login with teacher credentials (default password is "12345678")

Main functions:

Detection Tab: Process video files to detect violations

Violations Tab: View and manage detected violations

Students Tab: View and manage student database

Email Logs: View sent email notifications

Configuration
The system requires the following configuration:

Path to YOLO models (modify in ProcessingThread class)

Database path (modify in MainWindow class)

Email credentials (modify in send_email method)

File Structure
text
traffic_violation_system/
├── traffic_violation_system.py  # Main application file
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── models/                     # YOLO model files
    ├── yolov8n.pt
    └── helmet_model.pt
Notes
The system uses pre-trained YOLO models for object detection

Face recognition requires properly formatted student photos in the database

Email functionality requires valid SMTP credentials

License
This project is licensed under the MIT License.




**Important Note: Database Setup**
Before running this Traffic Violation Detection System, you must populate the database with initial data. The system requires the following tables to be properly initialized:

Required Database Tables
tutors table - Stores teacher/tutor information

students table - Stores student information with tutor associations

violations table - Will store detected violations (auto-populated)

email_logs table - Will store email notifications (auto-populated)

How to Add Initial Data
1. Adding Tutors/Teachers
You need to add at least one tutor record to enable login:

sql
INSERT INTO tutors (name, email, department) 
VALUES ('Admin Teacher', 'teacher@school.edu', 'Computer Science');
2. Adding Students
Add student records with their photos (as BLOBs) and associate them with tutors:

sql
INSERT INTO students (roll_no, name, tutor_id, department, photo)
VALUES 
('CS001', 'John Doe', 1, 'Computer Science', [photo_blob]),
('CS002', 'Jane Smith', 1, 'Computer Science', [photo_blob]);
3. Important Notes
The system comes with a default login password "12345678"

Student photos should be properly cropped face images for best recognition results

You can use SQLite browser tools to import photos as BLOBs

The tutor email in the database must match the login email

Verification
After adding data, verify your tables contain records by running:

sql
SELECT * FROM tutors;
SELECT * FROM students;
The system will automatically create and manage the violations and email_logs tables during operation.
