# IC1001 Smart Classroom Surveillance System

A Flask-based CCTV monitoring system with motion detection, face authentication, MySQL logging, and CSV data import.

---

## Requirements

- Python 3.11 or 3.12
- MySQL or MariaDB
- Two phone cameras (iPhone via SimpleIPCam, Android via IP Webcam app)

---

## 1. Install Python

Download from https://www.python.org/downloads/

> ⚠️ During installation, check **"Add Python to PATH"** before clicking Install.

Verify:
```bash
python --version
pip --version
```

---

## 2. Install Python Packages

```bash
pip install flask opencv-contrib-python numpy python-dotenv mysql-connector-python
```

> ⚠️ Must use `opencv-contrib-python`, NOT `opencv-python`. The contrib version includes the face recognizer (`cv2.face`) required by this project. If you already have `opencv-python` installed, remove it first:
> ```bash
> pip uninstall opencv-python
> pip install opencv-contrib-python
> ```

### Package summary

| Package | Purpose |
|---|---|
| `flask` | Web framework |
| `opencv-contrib-python` | Camera streams, motion detection, face recognition |
| `numpy` | Required by OpenCV |
| `python-dotenv` | Loads `.env` config file |
| `mysql-connector-python` | MySQL / MariaDB database connection |

---

## 3. Set Up MySQL

Create the database (tables are auto-created on first run):

```sql
CREATE DATABASE cctv_db;
```

---

## 4. Configure `.env`

Fill in your details in the `.env` file:

```env
# Flask
SECRET_KEY=change_this_to_something_random

# MySQL / MariaDB
DB_HOST=localhost
DB_PORT=3306
DB_USER=root
DB_PASSWORD=your_password_here
DB_NAME=cctv_db

# Camera stream URLs
PHONE_1_URL=http://192.168.68.105:8080/stream.mjpg
PHONE_2_URL=http://172.19.245.134:8080/videofeed

# Admin backup login ID
BACKUP_ADMIN_ID=ADMIN2026

# Motion detection sensitivity (in pixels²)
MOG2_MIN_AREA=1500
```

---

## 5. Project Structure

```
your-project/
├── app.py
├── .env
├── templates/
│   ├── index.html
│   ├── login.html
│   ├── logs.html
│   └── upload.html
├── static/          ← auto-created, stores alert snapshots
└── admin_profile/
    └── admin.jpg    ← your face photo for face login (add manually)
```

> The `static/` and `admin_profile/` folders are created automatically on first run. You only need to add `admin.jpg` manually — it should be a clear, well-lit photo of your face.

---

## 6. Run the App

```bash
cd your-project
python app.py
```

Then open your browser at:
```
http://localhost:5000
```

---

## 7. Camera Setup

### iPhone — SimpleIPCam
1. Download **SimpleIPCam** from the App Store
2. Start the stream
3. Copy the stream URL into `.env` as `PHONE_1_URL`

### Android — IP Webcam
1. Download **IP Webcam** from the Play Store
2. Scroll to the bottom and tap **Start Server**
3. Copy the URL shown on screen into `.env` as `PHONE_2_URL`

> Both phones must be on the **same Wi-Fi network** as the computer running the app.

---

## 8. CSV Upload

Go to `http://localhost:5000/upload` to import data into the database.

### Rooms CSV
```
room_id,camera
IC1001,192.168.1.10:8080
IC1002,192.168.1.11:8080
```

### Classrooms / Schedule CSV
```
room_id,offer_id,time,day_of_week
IC1001,CS301,09:00-10:30,Monday
IC1001,CS301,09:00-10:30,Wednesday
IC1002,IT401,13:00-14:30,Tuesday
```

> Column names are case-insensitive. Duplicate entries are updated automatically (upsert).

---

## 9. Login Methods

**Face Recognition** — scans your webcam and matches against `admin_profile/admin.jpg`.
- Make sure Teams, Zoom, or any other app using your webcam is fully closed before using this.

**Backup Admin ID** — enter the `BACKUP_ADMIN_ID` value from your `.env` file.

---

## 10. Database Tables

Auto-created on first run:

| Table | Columns |
|---|---|
| `rooms` | `id`, `room_id`, `camera` |
| `classrooms` | `id`, `room_id`, `offer_id`, `time`, `day_of_week` |
| `logs` | `id`, `timestamp`, `message`, `snapshot` |

---

## Troubleshooting

**`pip` not recognized after installing Python**
Close and reopen your terminal, or run:
```bash
python -m pip install flask opencv-contrib-python numpy python-dotenv mysql-connector-python
```

**`cv2.face` not found / AttributeError**
You have the wrong OpenCV package. Fix:
```bash
pip uninstall opencv-python opencv-contrib-python
pip install opencv-contrib-python
```

**Webcam blocked during face login**
Close Microsoft Teams, Zoom, or any video call app running in the background (check the system tray).

**Can't connect to camera stream**
- Make sure both phones are on the same Wi-Fi as your PC
- Double-check the IP addresses in `.env` match what the apps show
- Try opening the stream URL directly in your browser first