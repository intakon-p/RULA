import math as m
import cv2
import numpy as np
import time
import mediapipe as mp
import matplotlib.pyplot as plt
from MainAllcopyWeight import *
import time

# Setup Pose function for video.
# pose_video = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.7, model_complexity=1)
pose_video = mp_pose.Holistic(static_image_mode=False, min_detection_confidence=0.7, model_complexity=1)

# Initialize the VideoCapture object to read from the webcam.
camera_video = cv2.VideoCapture(0)
# camera_video.set(3,1280)
# camera_video.set(4,960)

# Initialize a resizable window.
# cv2.namedWindow('Pose Classification', cv2.WINDOW_NORMAL)

# Calculate distance
def findDistance(x1, y1, x2, y2):
    dist = m.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dist


# Calculate angle.
def findAngle(x1, y1, x2, y2):
    theta = m.acos((y2 - y1) * (-y1) / (m.sqrt(
        (x2 - x1) ** 2 + (y2 - y1) ** 2) * y1))
    degree = int(180 / m.pi) * theta
    return degree


"""
Function to send alert. Use this function to send alert when bad posture detected.
Feel free to get creative and customize as per your convenience.
"""


def sendWarning(x):
    pass


# =============================CONSTANTS and INITIALIZATIONS=====================================#
# Initilize frame counters.
good_frames = 0
bad_frames = 0

# Font type.
font = cv2.FONT_HERSHEY_SIMPLEX

# Colors.
blue = (255, 127, 0)
red = (50, 50, 255)
green = (127, 255, 0)
dark_blue = (127, 20, 0)

yellow = (0, 255, 255)
pink = (255, 0, 255)
white = (255,255,255)
# Initialize mediapipe pose class.
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
# ===============================================================================================#

# Iterate until the webcam is accessed successfully.
while camera_video.isOpened():

    # Read a frame.
    ok, frame = camera_video.read()
    
    # Check if frame is not read properly.
    if not ok:
        
        # Continue to the next iteration to read the next frame and ignore the empty camera frame.
        continue
    
    # Flip the frame horizontally for natural (selfie-view) visualization.
    frame = cv2.flip(frame, 1)
    
    # Get the width and height of the frame
    frame_height, frame_width, _ =  frame.shape
    
    # Resize the frame while keeping the aspect ratio.
    frame = cv2.resize(frame, (int(frame_width * (640 / frame_height)), 640))
    
    # print(frame.shape)
    t1 = time.time()
    # Perform Pose landmark detection.
    frame, landmarks = detectPose(frame, pose_video, display=False)
    
    ###

### Neck  and Torso

    # Get fps.
    fps = camera_video.get(cv2.CAP_PROP_FPS)
    # Get height and width.
    frame_height, frame_width = frame.shape[:2]

    # Convert the BGR frame to RGB.
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame.
    keypoints = pose.process(frame)

    # Convert the frame back to BGR.
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Use lm and lmPose as representative of the following methods.
    lm = keypoints.pose_landmarks
    lmPose = mp_pose.PoseLandmark
    
    if lm:
    # Acquire the landmark coordinates.
    # Once aligned properly, left or right should not be a concern.
    # Left shoulder.
     l_shldr_x = int(lm.landmark[lmPose.LEFT_SHOULDER].x * frame_width)
     l_shldr_y = int(lm.landmark[lmPose.LEFT_SHOULDER].y * frame_height)
    # Right shoulder
     r_shldr_x = int(lm.landmark[lmPose.RIGHT_SHOULDER].x * frame_width)
     r_shldr_y = int(lm.landmark[lmPose.RIGHT_SHOULDER].y * frame_height)
    # Left ear.
     l_ear_x = int(lm.landmark[lmPose.LEFT_EAR].x * frame_width)
     l_ear_y = int(lm.landmark[lmPose.LEFT_EAR].y * frame_height)
    # Left hip.
     l_hip_x = int(lm.landmark[lmPose.LEFT_HIP].x * frame_width)
     l_hip_y = int(lm.landmark[lmPose.LEFT_HIP].y * frame_height)

    # # Acquire the landmark coordinates.
    # # Once aligned properly, left or right should not be a concern.      
    # # Left shoulder.
    # l_shldr_x = int(lm.landmark[lmPose.LEFT_SHOULDER].x * frame_width)
    # l_shldr_y = int(lm.landmark[lmPose.LEFT_SHOULDER].y * frame_height)
    # # Right shoulder
    # r_shldr_x = int(lm.landmark[lmPose.RIGHT_SHOULDER].x * frame_width)
    # r_shldr_y = int(lm.landmark[lmPose.RIGHT_SHOULDER].y * frame_height)
    # # Left ear.
    # l_ear_x = int(lm.landmark[lmPose.LEFT_EAR].x * frame_width)
    # l_ear_y = int(lm.landmark[lmPose.LEFT_EAR].y * frame_height)
    # # Left hip.
    # l_hip_x = int(lm.landmark[lmPose.LEFT_HIP].x * frame_width)
    # l_hip_y = int(lm.landmark[lmPose.LEFT_HIP].y * frame_height)

    # Calculate distance between left shoulder and right shoulder points.
    ###offset = findDistance(l_shldr_x, l_shldr_y, r_shldr_x, r_shldr_y)

    # Assist to align the camera to point at the side view of the person.
    # Offset threshold 30 is based on results obtained from analysis over 100 samples.
    
    
    # if offset < 100:
    #     cv2.putText(frame, str(int(offset)) + ' Aligned', (frame_width - 150, 30), font, 0.9, green, 2)
    # else:
    #     cv2.putText(frame, str(int(offset)) + ' Not Aligned', (frame_width - 150, 30), font, 0.9, red, 2)

    # Calculate angles.
    neck_inclination = findAngle(l_shldr_x, l_shldr_y, l_ear_x, l_ear_y)
    torso_inclination = findAngle(l_hip_x, l_hip_y, l_shldr_x, l_shldr_y)

    # Text string for display.
    angle_text_string = 'Neck : ' + str(int(neck_inclination)) + '  Torso : ' + str(int(torso_inclination))


###
    
    # Check if the landmarks are detected.
    if landmarks:
        
        # Perform the Pose Classification.
        frame, _ = classifyPose(landmarks, frame, display=False)

    t2 = time.time() - t1
    cv2.putText(frame, "{:.0f} ms".format(
            t2*1000), (10, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (203, 52, 247), 1)
    
    cv2.putText(frame, angle_text_string, (10, frame_height - 50), font, 0.9, white, 2)

    
    
    cv2.putText(frame, str(int(neck_inclination)), (l_shldr_x + 10, l_shldr_y), font, 0.9, dark_blue, 2)
    
    if 0 < neck_inclination <= 10 :
         Neck_score = 1 
    elif    10 < neck_inclination <= 20 :  
         Neck_score = 2
    elif    20 < neck_inclination :
        Neck_score = 3
    elif  neck_inclination < 0 :
        Neck_score = 4

    cv2.putText(frame, "Neck_Score : " + str(Neck_score), (500, frame_height - 50), font, 0.9, white, 2)


    cv2.putText(frame, str(int(torso_inclination)), (l_hip_x + 10, l_hip_y), font, 0.9, white, 2)
    if 0 < torso_inclination <= 10 :
         Trunk_score = 1 
    elif    10 < torso_inclination <= 20 :  
         Trunk_score = 2
    elif    20 < torso_inclination <= 60 :
        Trunk_score = 3
    elif  60 <torso_inclination  :
        Trunk_score = 4
    cv2.putText(frame, "Trunk_Score : " + str(Trunk_score), (500, frame_height - 10), font, 0.9, white, 2)
    
    
    #cv2.putText(frame, str(int(neck_inclination)), (700, frame_height - 50), font, 0.9, white, 2)

    # Display the frame.
    cv2.imshow('Pose Classification', frame)
    
    # Wait until a key is pressed.
    # Retreive the ASCII code of the key pressed
    k = cv2.waitKey(1) & 0xFF
    # Check if 'ESC' is pressed.
    if k == 27 and landmarks:
    # Break the loop.
     break
    
# Release the VideoCapture object and close the windows.
camera_video.release()
cv2.destroyAllWindows()

