import cv2
import numpy as np
import mediapipe as mp

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
video = cv2.VideoCapture("./KneeBendVideo.mp4")

# Extract the frames
frames = []
while True:
    ret, frame = video.read()
    if not ret:
        break
    frames.append(frame)

# Initialize MediaPipe graph
graph = mp.CalculatorGraph()

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

# Get the index of the knee joint
knee_joint_index = 7

# Count the number of bent knee frames
knee_bent_count = count_knee_bent(filtered_keypoints, knee_joint_index)

print("Number of bent knee frames:", knee_bent_count)
