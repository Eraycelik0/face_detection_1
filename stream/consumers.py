import base64
import cv2
import json
import numpy as np
from channels.generic.websocket import WebsocketConsumer
from geopy.geocoders import Nominatim
import random
import os

class VideoStreamConsumer(WebsocketConsumer):
    def connect(self):
        self.accept()

    def receive(self, text_data):
        data_url = text_data.split(',')[1]
        image_data = base64.b64decode(data_url)
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) > 0:
            x, y, w, h = faces[0]
            face_image = frame[y:y+h, x:x+w]
            file_name = f"found_face_{random.randint(1000,9999)}.jpg"
            file_path = os.path.join('media/found_faces', file_name)
            cv2.imwrite(file_path, face_image)

            geolocator = Nominatim(user_agent="geoapiExercises")
            location = geolocator.geocode("Your location data")
            location_info = f"{location.latitude}, {location.longitude}" if location else "Unknown"

            self.send(json.dumps({
                'found': True,
                'image_url': file_path,
                'location': location_info,
            }))
