import cv2
import numpy as np
import mediapipe as mp

# TF error
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}


# to detect all landmarks 
mp_pose = mp.solutions.pose

mp_drawing = mp.solutions.drawing_utils

# Calculating angle
def angle(a,b,c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians =  np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    ang = np.abs(radians*180.0/np.pi)

    if ang>180.0:
        ang = 360 - ang
        return ang

# input 
cap = cv2.VideoCapture('./KneeBendVideo.mp4')

import cv2
import numpy as np

# function to handle fluctuations
def handle_fluctuations(keypoints, dummy_frame_interval):
    # Initialize a list to store filtered keypoints
    filtered_keypoints = []
    
    # Iterate through keypoints and check for fluctuations caused by dummy frames
    for i in range(len(keypoints)):
        if i % dummy_frame_interval != 0:
            filtered_keypoints.append(keypoints[i])
            
    return filtered_keypoints

# function to count knee bent frames
def count_knee_bent(keypoints, knee_joint_index):
    count = 0
    for keypoint in keypoints:
        knee_joint = keypoint[knee_joint_index]
        # check if knee joint is bent
        if knee_joint[2] > 0.5:  # confidence threshold
            count += 1
    return count

# Read in the video
video = cv2.VideoCapture("path/to/video.mp4")

# Extract the frames
frames = []
while True:
    ret, frame = video.read()
    if not ret:
        break
    frames.append(frame)

# Initialize MediaPipe graph
graph = mp.Graph()

# Add a keypoint detection model to the graph
keypoint_detection = mp.solutions.pose.PoseDetectionGraph()
keypoint_detection.load('path/to/mediapipe_model.pbtxt', 'path/to/mediapipe_model.pb')

# Run the graph on the frames
keypoints = []
for frame in frames:
    keypoint_detection.process(frame)
    keypoints.append(keypoint_detection.outputs['keypoints'])

dummy_frame_interval = 3

# Filter the keypoints
filtered_keypoints = handle_fluctuations(keypoints, dummy_frame_interval)


# Characteristics
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
size = (width,height)
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Variables
relax_counter = 0
bent_counter = 0
counter = 0
stage = None
feedback = None
images_array = []
bent_time = 0
relax_time = 0

# Instance
with mp_pose.Pose(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # BGR --> RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable= False

        # make detection
        results = pose.process(image)

        # RGB --> BGR
        image.flags.writeable = True
        image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # landmarks
        try:
            landmarks = results.pose_landmarks.landmark

            # coordinates of knee, hip, ankle, shoulder
            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]

            # Calculating angle
            angle = angle(hip,knee,ankle)

            # Directions
            a0 = int(ankle[0] * width)
            a1 = int(ankle[1] * height)

            k0 = int(knee[0] * width)
            k1 = int(knee[1] * height)

            h0 = int(hip[0] * width)
            h1 = int(hip[1] * height)

            cv2.line(image, (h0,h1), (k0,k1), (255,255,0), 2)
            cv2.line(image, (k0, k1), (a0, a1), (255, 255, 0), 2)
            cv2.circle(image, (h0, h1), 5, (0, 0, 0), cv2.FILLED)
            cv2.circle(image, (k0, k1), 5, (0, 0, 0), cv2.FILLED)
            cv2.circle(image, (a0, a1), 5, (0, 0, 0), cv2.FILLED)       
            
            # displaying angle
            cv2.putText(image, str(round(angle,4)), tuple(np.multiply(shoulder, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_PLAIN, 0.5, (255,255,255), 2, cv2.LINEAA)

            relax_time = (1/fps) * relax_counter
            bent_time = (1/fps) * bent_counter

            # Logic
            if angle > 140:
                relax_counter += 1
                bent_counter = 0
                stage = "Relaxed"
                feedback = ""

            if angle < 140:
                relax_counter = 0
                bent_counter += 1
                stage = "Bent"
                feedback = ""

            # rep
            if bent_time == 8:
                counter += 1
                feedback = 'Rep completed'
                
            elif bent_time < 8 and stage == 'Bent':
                feedback = 'Keep Your Knee Bent'
            
            else:
                feedback = ""

        except:
            pass

        # Status Box
        cv2.rectangle(image,(0,0), (int(width), 60), (255,255,0), -1)

        # rep data
        cv2.putText(image, 'REPS', (10,15), 
                    cv2.FONT_HERSHEY_PLAIN, 0.5, (0,0,0), 1, cv2.LINE_AA)
        
        cv2.putText(image, str(counter), 
                    (10,50), 
                    cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 2, cv2.LINE_AA)
        
        # Stage data
        cv2.putText(image, 'STAGE', (105,15), 
                    cv2.FONT_HERSHEY_PLAIN, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, stage, 
                    (105,50), 
                    cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 2, cv2.LINE_AA)
        
        # Feedback
        cv2.putText(image, 'FEEDBACK', (315,15), 
                    cv2.FONT_HERSHEY_PLAIN, 0.5, (0,0,0), 1, cv2.LINE_AA)
        
        cv2.putText(image, feedback, 
                    (315,50), 
                    cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 2, cv2.LINE_AA)
        
        # Bent Time
        cv2.putText(image, 'BENT TIME', (725,15), 
                    cv2.FONT_HERSHEY_PLAIN, 0.5, (0,0,0), 1, cv2.LINE_AA)
        
        cv2.putText(image, str(round(bent_time,2)), 
                    (725,50), 
                    cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 2, cv2.LINE_AA)  

        images_array.append(image) 
        
        cv2.imshow('Knee Bend Excercise', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break


        cap.release()
        cv2.destroyAllWindows()

    # Output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('Output.mp4', fourcc , fps, size)
    for i in range(len(images_array)):
        out.write(images_array[i])
    out.release()




            