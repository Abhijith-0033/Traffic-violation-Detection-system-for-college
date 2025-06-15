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
