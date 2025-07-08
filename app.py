import os, sys, json, bcrypt, time, hashlib
import face_recognition
import cv2
import numpy as np
from datetime import datetime
from PIL import Image
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt

PHOTO_FOLDER = "photos"
THUMB_FOLDER = "thumbnails"
FACE_DATA = "faces.json"
CONFIG_FILE = "config.json"
MEMORY_INDEX = "memory_index.json"
def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt())

def setup_password():
    if not os.path.exists(CONFIG_FILE):
        pw = input("Set a password: ")
        with open(CONFIG_FILE, 'w') as f:
            json.dump({'password': hash_password(pw).decode()}, f)

def check_password():
    with open(CONFIG_FILE) as f:
        saved = json.load(f)['password'].encode()
    tries = 0
    while tries < 3:
        pw = input("Enter password: ")
        if bcrypt.checkpw(pw.encode(), saved): return True
        print("Wrong password!")
        tries += 1
        if tries == 3:
            print("Wait 1 minute.")
            time.sleep(60)
    return False

def scan_photos():
    memories = {}
    for file in os.listdir(PHOTO_FOLDER):
        if file.lower().endswith(('.jpg', '.png')):
            path = os.path.join(PHOTO_FOLDER, file)
            img = Image.open(path)
            try:
                date = img._getexif()[36867].split(" ")[0].replace(":", "-")
            except:
                date = "unknown"
            if date not in memories:
                memories[date] = []
            memories[date].append(file)
    with open(MEMORY_INDEX, 'w') as f:
        json.dump(memories, f, indent=2)

def cluster_faces():
    faces = {}
    for file in os.listdir(PHOTO_FOLDER):
        path = os.path.join(PHOTO_FOLDER, file)
        image = face_recognition.load_image_file(path)
        encs = face_recognition.face_encodings(image)
        for enc in encs:
            key = hashlib.sha1(enc).hexdigest()
            if key not in faces: faces[key] = []
            faces[key].append(file)
    with open(FACE_DATA, 'w') as f:
        json.dump(faces, f, indent=2)

def name_faces():
    with open(FACE_DATA) as f:
        data = json.load(f)
    name_map = {}
    for i, key in enumerate(data.keys()):
        print(f"Face #{i+1} appears in: {data[key][:3]}...")
        name = input("Enter name for this face: ")
        name_map[name] = data[key]
    with open("face_names.json", 'w') as f:
        json.dump(name_map, f, indent=2)

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Photo Memories App")
        self.setGeometry(100, 100, 400, 300)
        self.setStyleSheet("background-color: #1e1e1e; color: white;")

        layout = QVBoxLayout()

        self.label = QLabel("Welcome to Photo Memories!")
        layout.addWidget(self.label)

        btn1 = QPushButton("📷 Scan Photos by Date")
        btn1.clicked.connect(scan_photos)
        layout.addWidget(btn1)

        btn2 = QPushButton("🙂 Group Faces")
        btn2.clicked.connect(cluster_faces)
        layout.addWidget(btn2)

        btn3 = QPushButton("📝 Name Faces")
        btn3.clicked.connect(name_faces)
        layout.addWidget(btn3)

        btn4 = QPushButton("📅 View Memories Calendar")
        btn4.clicked.connect(self.show_calendar)
        layout.addWidget(btn4)

        btn5 = QPushButton("🎞️ Start Slideshow")
        btn5.clicked.connect(self.slideshow)
        layout.addWidget(btn5)

        btn6 = QPushButton("🔍 Search Memories")
        btn6.clicked.connect(self.search_memories)
        layout.addWidget(btn6)

        self.setLayout(layout)

    def show_calendar(self):
        with open(MEMORY_INDEX) as f:
            data = json.load(f)
        dates = list(data.keys())
        plt.barh(dates, [len(v) for v in data.values()])
        plt.title("Photos by Date")
        plt.show()

    def slideshow(self):
        files = os.listdir(PHOTO_FOLDER)
        for file in files:
            if file.lower().endswith(('.jpg', '.png')):
                img = cv2.imread(os.path.join(PHOTO_FOLDER, file))
                cv2.imshow("Slideshow", img)
                cv2.waitKey(1000)
        cv2.destroyAllWindows()

    def search_memories(self):
        text, ok = QInputDialog.getText(self, 'Search', 'Enter name or date:')
        if ok:
            results = []
            try:
                with open("face_names.json") as f:
                    names = json.load(f)
                    if text in names:
                        results = names[text]
            except:
                pass
            try:
                with open(MEMORY_INDEX) as f:
                    dates = json.load(f)
                    if text in dates:
                        results = dates[text]
            except:
                pass

            if results:
                msg = QMessageBox()
                msg.setText("Found these photos:\n" + "\n".join(results))
                msg.exec_()
            else:
                QMessageBox.information(self, "Not Found", "No photos found.")


if __name__ == "__main__":
    setup_password()
    if not check_password():
        sys.exit()

    app = QApplication(sys.argv)
    win = App()
    win.show()
    sys.exit(app.exec_())

