"""
Este módulo contiene un ejemplo de detección de rostros en un video utilizando OpenCV.
"""
import cv2

# Cargar el clasificador de cascada para detección de rostros
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Capturar el video
video_capture = cv2.VideoCapture("mesa.mp4")

while True:
    ret, frame = video_capture.read()

    if not ret:
        break

    # Redimensionar la imagen
    height, width = frame.shape[:2]
    new_height = int(height * 0.6)
    new_width = int(width * 0.6)
    frame = cv2.resize(frame, (new_width, new_height))

    # Convertir a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar rostros en la imagen
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    if len(faces):
        print("Rostros encontrados: ", len(faces))
        for (x, y, w, h) in faces:
            # Dibujar un rectángulo alrededor de cada rostro detectado
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 3)

    # Mostrar el frame con los rostros detectados
    cv2.imshow("Frame", frame)

    # Salir del bucle al presionar 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar el video capture y cerrar las ventanas
video_capture.release()
cv2.destroyAllWindows()