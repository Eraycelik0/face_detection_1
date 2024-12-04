from django.shortcuts import render
from django.http import JsonResponse, StreamingHttpResponse
import os
import random
import face_recognition
import cv2


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
