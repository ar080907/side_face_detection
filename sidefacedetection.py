import cv2
import time

# Load cascade
face_cascade = cv2.CascadeClassifier(r"cascade2.xml")#add the full directory in which the cascade is present 
# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

prev_time = time.time()
fps = 0

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(300,300))

    for i, (x, y, w, h) in enumerate(faces, start=1):
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 255), 1)
        cv2.putText(frame, f"side face #{i}", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1)
    # FPS display
    now = time.time()
    fps = 0.9 * fps + 0.1 * (1 / (now - prev_time))
    prev_time = now
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 255, 50), 1)

    # Show video
    cv2.imshow("Face and Eye Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
