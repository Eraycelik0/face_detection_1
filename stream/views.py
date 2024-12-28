from django.shortcuts import render
from django.http import JsonResponse, StreamingHttpResponse
import os
import random
import face_recognition
import cv2
import os
import cv2
import numpy as np
from django.shortcuts import render
from django.http import JsonResponse, StreamingHttpResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

MODEL_PATH = "best_model.keras"
model = load_model(MODEL_PATH)


IMAGE_SIZE = (128, 128)

def preprocess_image_1(image):
    """Görüntüyü model için ön işleme tabi tutar."""
    image = cv2.resize(image, IMAGE_SIZE)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    return image

def gen_frames_1(target_image_path=None):
    """Kamera görüntülerini oluşturur ve yüz algılama yapar."""
    cap = cv2.VideoCapture(0)
    target_encoding = None

    if target_image_path:
        target_image = cv2.imread(target_image_path)
        target_image = preprocess_image_1(target_image)
        target_encoding = model.predict(target_image)[0]

    while True:
        success, frame = cap.read()
        if not success:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized_frame = preprocess_image_1(rgb_frame)
        frame_encoding = model.predict(resized_frame)[0]

        if target_encoding is not None:
            similarity = np.linalg.norm(frame_encoding - target_encoding)
            if similarity < 0.5:  # Benzerlik eşik değeri
                cv2.putText(frame, "Match Found", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.rectangle(frame, (10, 10), (300, 300), (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()

def video_feed_1(request):
    """Kamera görüntüsünü canlı olarak akışa verir."""
    return StreamingHttpResponse(gen_frames(), content_type='multipart/x-mixed-replace; boundary=frame')

def upload_face_image_1(request):
    """Yüklenen hedef yüzü işler."""
    if request.method == 'POST' and request.FILES.get('image'):
        image = request.FILES['image']
        image_path = f'media/target_face/target_{np.random.randint(1000, 9999)}.jpg'
        os.makedirs(os.path.dirname(image_path), exist_ok=True)

        with open(image_path, 'wb') as f:
            for chunk in image.chunks():
                f.write(chunk)

        return JsonResponse({'status': 'success', 'image_path': image_path})

    return JsonResponse({'status': 'error'})



def gen_frames(target_face_encoding=None):
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        if target_face_encoding is not None:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces([target_face_encoding], face_encoding)
                if True in matches:
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()


def video_feed(request):
    return StreamingHttpResponse(gen_frames(), content_type='multipart/x-mixed-replace; boundary=frame')


def index(request):
    return render(request, 'stream/index.html')


def upload_face_image(request):
    if request.method == 'POST' and request.FILES.get('image'):
        image = request.FILES['image']
        image_path = f'media/target/target_{random.randint(1000, 9999)}.jpg'
        os.makedirs(os.path.dirname(image_path), exist_ok=True)

        with open(image_path, 'wb') as f:
            for chunk in image.chunks():
                f.write(chunk)

        target_image = face_recognition.load_image_file(image_path)
        target_face_encoding = face_recognition.face_encodings(target_image)[0]

        cap = cv2.VideoCapture(0)
        found = False
        image_url = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces([target_face_encoding], face_encoding)
                if True in matches:
                    found = True
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    found_image_path = f'media/found_face/found_face_{random.randint(1000, 9999)}.jpg'
                    os.makedirs(os.path.dirname(found_image_path), exist_ok=True)
                    cv2.imwrite(found_image_path, frame)
                    image_url = f'/{found_image_path}'
                    break
            if found:
                break

        cap.release()
        return JsonResponse({'status': 'found' if found else 'not found', 'image_url': image_url})

    return JsonResponse({'status': 'error'})
