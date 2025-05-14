from flask import Flask, request, jsonify
import cv2
import numpy as np
import mediapipe as mp
import base64

app = Flask(__name__)

# Initialize mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

# EAR / MAR threshold
EYE_AR_THRESH = 0.22
MOUTH_AR_THRESH = 0.6
HEAD_TILT_THRESH = 35.0  # Degrees threshold for head tilt

# Eye and mouth landmark indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [263, 387, 385, 362, 380, 373]
MOUTH = [13, 14, 78, 308]

# Head pose estimation landmarks
# Mid of forehead
FOREHEAD = 10
# Bottom of chin
CHIN = 152
# Left and right sides of face
LEFT_FACE = 234
RIGHT_FACE = 454

def eye_aspect_ratio(landmarks, eye_indices):
    # Extract points as (x,y) coordinates
    points = np.array([landmarks[i] for i in eye_indices])
    
    # Calculate vertical distances (top of the fraction)
    top = np.linalg.norm(points[1] - points[5]) + np.linalg.norm(points[2] - points[4])
    
    # Calculate horizontal distance (bottom of the fraction)
    bottom = 2 * np.linalg.norm(points[0] - points[3])
    
    # Avoid division by zero
    if bottom == 0:
        return 0
    
    return top / bottom

def mouth_aspect_ratio(landmarks):
    # Extract points as (x,y) coordinates
    points = np.array([landmarks[i] for i in MOUTH])
    
    # Calculate vertical distance
    vertical = np.linalg.norm(points[0] - points[1])
    
    # Calculate horizontal distance
    horizontal = np.linalg.norm(points[2] - points[3])
    
    # Avoid division by zero
    if horizontal == 0:
        return 0
    
    return vertical / horizontal

def detect_head_tilt(landmarks):
    """
    Detect if the head is tilted/bent over indicating sleeping posture
    Uses the angle between the vertical line and the line connecting forehead to chin
    """
    # Extract key points for head pose
    forehead = landmarks[FOREHEAD]
    chin = landmarks[CHIN]
    
    # Calculate angle between vertical line and face orientation
    # A vertical line would have dx=0, so we're measuring deviation from vertical
    dy = chin[1] - forehead[1]
    dx = chin[0] - forehead[0]
    
    # Calculate angle in degrees
    # When head is upright, angle will be close to 0
    # When head is tilted sideways or bent over, angle will increase
    if dy == 0:  # Avoid division by zero
        angle = 90.0
    else:
        angle = abs(np.degrees(np.arctan(dx/dy)))
    
    # Also check if chin is higher than forehead (completely bent over)
    if forehead[1] > chin[1]:  # In image coordinates, y increases downward
        return True, angle
    
    # Otherwise check if the head is significantly tilted
    return angle > HEAD_TILT_THRESH, angle

@app.route('/detect', methods=['POST'])
def detect():
    try:
        # Get the image data from the request
        data = request.json
        img_data = base64.b64decode(data['image'])
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'error': 'Invalid image data'})
            
        h, w, _ = img.shape

        # Process the image with MediaPipe
        results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if not results.multi_face_landmarks:
            return jsonify({'status': 'No face detected'})

        # Extract landmarks
        face_landmarks = results.multi_face_landmarks[0]  # We're only processing one face
        landmarks = {}
        
        # Store all landmarks in a dictionary for easy access
        for idx, landmark in enumerate(face_landmarks.landmark):
            landmarks[idx] = np.array([landmark.x * w, landmark.y * h])
        
        # Calculate eye aspect ratios
        left_ear = eye_aspect_ratio(landmarks, LEFT_EYE)
        right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE)
        ear = (left_ear + right_ear) / 2.0
        
        # Calculate mouth aspect ratio
        mar = mouth_aspect_ratio(landmarks)
        
        # Detect head tilt/bent over posture
        is_slept, head_angle = detect_head_tilt(landmarks)

        # Determine eye and mouth status
        eye_status = "Closed" if ear < EYE_AR_THRESH else "Open"
        mouth_status = "Yawning" if mar > MOUTH_AR_THRESH else "Normal"
        sleep_status = "Slept" if is_slept else "Awake"
        
        # Determine if drowsy based on current frame (including sleep state)
        drowsy = (eye_status == "Closed" or mouth_status == "Yawning" or sleep_status == "Slept")

        # Return the results
        return jsonify({
            'eye_status': eye_status,
            'mouth_status': mouth_status,
            'sleep_status': sleep_status,
            'ear_value': float(ear),
            'mar_value': float(mar),
            'head_angle': float(head_angle),
            'drowsy': drowsy
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
