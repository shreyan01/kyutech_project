#!/usr/bin/env python
import mediapipe as mp # Import mediapipe
import cv2 # Import opencv
import pyrealsense2 as rs
import datetime as dt
import time
import csv
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline 
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score # Accuracy metrics 
from std_msgs.msg import String
#from calibrate import DLT, get_projection_matrix
import pickle 
import rospy
import moveit_commander
import sys
from math import pi
import moveit_msgs.msg
import geometry_msgs.msg
from geometry_msgs.msg import TwistStamped
from subprocess import *
import serial
from time import sleep
import pyfirmata
from pyfirmata import Arduino ,SERVO, util
import urx
import subprocess
import joblib

gripper_frame_conditions = []
gripper_closed = False
gripper_intent = []

publish_deltar = False
publish_deltat = False
establish_normal = False
initial_normals = [] #this will be populated with N vectors then averaged to make default normal
default_normal = [] #this will contain the default normal once established
tracking_point = []
tracking_thetas = [] 
sequence = []
sentence = []
predictions = []
threshold = 0.1

# Buffer to store last few readings
buffer_size = 5  # Adjust the buffer size as needed
intention_buffer = []





class intention_predict:
    
    def __init__(self):
        rospy.init_node('intention_publisher', anonymous = True)
        self.pub = rospy.Publisher('intention_publisher_commands', String, queue_size = 1)
        self.current_intention = None
        self.intention_start_time = None
        self.consistent_duration = 2  # Duration in seconds to hold the same intention before sending
        self.probability_threshold = 0.05
        self.publish = False
        self.publish_solder = False # solder mode
        self.publish_data = True  # Flag to control data publishing
        self.class_buffer = []
        self.prob_buffer = []
        self.buffer_time = 5  # seconds
        self.last_detected_class = None  # Initialize here
        self.consecutive_count = 0
        # Define your conditions
        self.condition_for_program1 = True  # Change this condition as needed
        self.condition_for_program2 = False  # Change this condition as needed

        # Start both programs initially
        #self.program1_process = subprocess.Popen(["python", "RobotControl_V5.py"])
        #self.program2_process = subprocess.Popen(["python", "keypoints_publisher_sync.py"])
        
        
      
        self.detection_start_time = None
        self.publish_timestamp = None
        self.recent_probabilities = []
        # self.probability_threshold = 0.5
    

    #New def
    def prob_viz(self,res, actions, input_frame, colors):
        output_frame = input_frame.copy()
        for num, prob in enumerate(res):
            cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
            cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
        return output_frame
    
    def save_intention_to_csv(self, intention, probability):
        # with open('published_intentions.csv', 'a', newline='') as csvfile:
        #     csvwriter = csv.writer(csvfile)
        #     csvwriter.writerow([intention, probability])

        file_name = f'published_intentions.csv'
        file_path = os.path.join('/home/non/catkin_ws/src/robot_control/scripts/', file_name)
        with open(file_path, mode='a', newline='') as f:
            csvwriter = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csvwriter.writerow([intention, probability])


    
    def update_buffers(self, detected_class, probability):
        current_time = time.time()
        self.class_buffer.append((detected_class, current_time))
        self.prob_buffer.append((probability, current_time))
        self.class_buffer = [(cls, t) for cls, t in self.class_buffer if current_time - t <= self.buffer_time]
        self.prob_buffer = [(prob, t) for prob, t in self.prob_buffer if current_time - t <= self.buffer_time]

    def check_consistency(self):
        if not self.class_buffer:
            return None, None
        classes = [cls for cls, _ in self.class_buffer]
        most_frequent_class = max(set(classes), key=classes.count)
        probs = [prob for cls, prob in zip(classes, self.prob_buffer) if cls == most_frequent_class]
        avg_prob = sum(probs) / len(probs) if probs else 0
        return most_frequent_class, avg_prob
    

    def IntentionPrediction(self):
        
        current_intention = None
        intention_start_time = None
        consistent_duration = 5  # Duration in seconds to hold the same intention before sending

        mp_drawing = mp.solutions.drawing_utils # Drawing helpers
        mp_holistic = mp.solutions.holistic # Mediapipe Solutions
        df = pd.read_csv('/home/non/catkin_ws/src/robot_control/scripts/coords_ROMAN.csv')
        X = df.drop('class', axis=1) # features
        y = df['class'] # target value
        model = pickle.load(open('/home/non/catkin_ws/src/robot_control/scripts/Data_trained1.pkl','rb'))
        #model = joblib.load('Data_trained1.joblib')
        font = cv2.FONT_HERSHEY_COMPLEX
        org = (20,100)
        fontScale = .5
        thickness = 1 
        color = (0,150,255)
        realsense_ctx = rs.context()
        connected_devices = [] # List of serial numbers for present cameras
        for i in range(len(realsense_ctx.devices)):
                detected_camera =  realsense_ctx.devices[i].get_info(rs.camera_info.serial_number)
                print(f"{detected_camera}")
                connected_devices.append(detected_camera)
                
        device = connected_devices[0] # In this example we are only using one camera    
        pipeline = rs.pipeline()    
        config = rs.config()    
        background_removed_color = 153 # Grey

        config.enable_device(device)
        stream_res_x = 640
        stream_res_y = 480
        stream_fps = 30
        config.enable_stream(rs.stream.depth, stream_res_x, stream_res_y, rs.format.z16, stream_fps)
        config.enable_stream(rs.stream.color, stream_res_x, stream_res_y, rs.format.bgr8, stream_fps)
        profile = pipeline.start(config)
        align_to = rs.stream.color
        align = rs.align(align_to)
        # ====== Get depth Scale ======
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        # ====== Set clipping distance ======
        clipping_distance_in_meters = 2
        clipping_distance = clipping_distance_in_meters / depth_scale
        print(f"\tConfiguration Successful for SN {device}")

        # Initiate holistic model
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            
            while True:
                start_time = dt.datetime.today().timestamp() # Necessary for FPS calculations

                        # Get and align frames
                frames = pipeline.wait_for_frames()
                aligned_frames = align.process(frames)
                aligned_depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()

                if not aligned_depth_frame or not color_frame:
                    continue

                    # Process images
                depth_image = np.asanyarray(aligned_depth_frame.get_data())
                depth_image_flipped = cv2.flip(depth_image,1)
                color_image = np.asanyarray(color_frame.get_data())

                depth_image_3d = np.dstack((depth_image,depth_image,depth_image))
                background_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), background_removed_color, color_image)

                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                image = cv2.flip(background_removed,1)
                color_image = cv2.flip(color_image,1)
                color_images_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                    
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
                image.flags.writeable = False                  # Image is no longer writeable
                
                # Recolor Feed
                    
                
                # Make Detections
                results = holistic.process(image)
                # print(results.face_landmarks)
                
                # face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks
                
                # Recolor image back to BGR for rendering
                image.flags.writeable = True   
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                # 1. Draw face landmarks
                mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, 
                                        mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                        mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                                        )
                
                # 2. Right hand
                mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                        )

                # 3. Left Hand
                mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                        )

                # 4. Pose Detections
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                        )
            
                try:
                    # Extract Pose landmarks
                    pose = results.pose_landmarks.landmark
                    pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
                    
                    # Extract Face landmarks
                    face = results.face_landmarks.landmark
                    face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())

                    # Extract Right hand landmarks
                    # right_hand = results.right_hand_landmarks.landmark
                    # right_hand_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in right_hand]).flatten())
                    
                    # Extract Left hand landmarks
                    left_hand = results.left_hand_landmarks.landmark
                    left_hand_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in left_hand]).flatten())

                    # Concate rows
                    # row = pose_row+face_row+right_hand_row+left_hand_row
                    row = pose_row+face_row+left_hand_row

                    X = pd.DataFrame([row])
                    body_language_class = model.predict(X)[0]
                    body_language_prob = model.predict_proba(X)[0]
                    #print(body_language_class, body_language_prob)
                    max_body_language_prob = round(body_language_prob[np.argmax(body_language_prob)])

                     # Visualization logic
                    # if results.pose_landmarks:
                    #     coords = tuple(np.multiply(
                    #             np.array(
                    #                 (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x, 
                    #                 results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y)),
                    #             [640,480]).astype(int))
                            
                    #     cv2.rectangle(image, 
                    #                     (coords[0], coords[1]+5), 
                    #                     (coords[0]+len(body_language_class)*20, coords[1]-30), 
                    #                     (245, 117, 16), -1)
                    #     cv2.putText(image, body_language_class, coords, 
                    #                     cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                    # # Get status box
                    # cv2.rectangle(image, (0,0), (250, 60), (245, 117, 16), -1)
                    
                    # # Display Class
                    # cv2.putText(image, 'CLASS', (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    # cv2.putText(image, body_language_class.split(' ')[0], (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    # # Display Probability
                    # cv2.putText(image, 'PROB', (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    # cv2.putText(image, str(max_body_language_prob), (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)


                    # Visualization logic
                   # Grab ear coords
                    coords = tuple(np.multiply(
                                    np.array(
                                        (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x, 
                                        results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y))
                                , [640,480]).astype(int))
                    

                    cv2.rectangle(image, 
                                (coords[0], coords[1]+5), 
                                (coords[0]+len(body_language_class)*20, coords[1]-30), 
                                (245, 117, 16), -1)
                    cv2.putText(image, body_language_class, coords, 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    # Get status box
                    cv2.rectangle(image, (0,0), (250, 60), (245, 117, 16), -1)
                    
                    # Display Class
                    cv2.putText(image, 'CLASS'
                                , (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, body_language_class.split(' ')[0]
                                , (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    # Display Probability
                    cv2.putText(image, 'PROB'
                                , (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)],2))
                                , (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
                    

                    colors = [(245,117,16), (117,245,16),(117,245,16),(117,245,16),(117,245,16),(117,245,16),(117,245,16)]

                    publish = True
                    if self.publish:

                        self.publish_solder = False
                        #     #  cv2.putText(image, 'Publishing', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255))
                        #  # Process "on" and "off" intentions with high confidence
                        # # Handle "on" and "off" intentions with high confidence
                        # if max_body_language_prob >= self.probability_threshold:
                        #     if body_language_class == "Manual":
                        #         if self.current_intention == "Manual" and time.time() - self.intention_start_time >= self.consistent_duration:
                        #             self.publish_data = False
                        #             print("System turned off, stopping data publishing.")
                        #             self.current_intention = None
                        #     elif body_language_class == "Auto":
                        #         if self.current_intention == "Auto" and time.time() - self.intention_start_time >= self.consistent_duration:
                        #             self.publish_data = True
                        #             print("System turned on, resuming data publishing.")
                        #             self.current_intention = None

                        # # Publish other intentions with high confidence when allowed
                        # if self.publish_data and body_language_class not in ["on", "off"] and max_body_language_prob >= self.probability_threshold:
                        #     if self.current_intention is None or self.current_intention != body_language_class:
                        #         self.current_intention = body_language_class
                        #         self.intention_start_time = time.time()
                        #     elif time.time() - self.intention_start_time >= self.consistent_duration:
                        #         intent_to_publish = str(self.current_intention)
                        #         print(f"Publishing consistent intention: {intent_to_publish} with probability {max_body_language_prob}")
                        #         self.pub.publish(intent_to_publish)
                        #          # Visualization for publishing status
                        #         cv2.putText(image, f"Publishing: {intent_to_publish}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                        #            # Save the published intention and its probability to CSV
                        #         self.save_intention_to_csv(intent_to_publish,  str(round(body_language_prob[np.argmax(body_language_prob)],2)))

                        #         self.current_intention = None  # Reset for next intention
                        # else:
                        #     # Display waiting status if not publishing yet
                        #     cv2.putText(image, "Waiting for consistent intention...", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
                          # Check if the detected class is the same as the last detected class
                        if body_language_class == self.last_detected_class:
                            self.consecutive_count += 1
                        else:
                            self.last_detected_class = body_language_class
                            self.consecutive_count = 1

                        if self.consecutive_count == 20:
                            if body_language_class == "Auto":
                                self.program1_process.poll()  # Check if program1 is still running
                                if self.program1_process.returncode is not None:  # If program1 has terminated
                                        self.program1_process = subprocess.Popen(["python", "RobotControl_V5.py"])  # Restart program1
                                if self.program2_process.poll() is None:  # If program2 is still running, terminate it
                                        self.program2_process.terminate()
                                self.pub.publish(body_language_class)  # Publish the class
                            
                            elif body_language_class == "Manual":

                                self.program2_process.poll()  # Check if program2 is still running
                                if self.program2_process.returncode is not None:  # If program2 has terminated
                                    self.program2_process = subprocess.Popen(["python", "keypointpubllisher"])  # Restart program2
                                if self.program1_process.poll() is None:  # If program1 is still running, terminate it
                                    self.program1_process.terminate()
                                self.pub.publish(body_language_class)  # Publish the class
                            
                            else :
                                self.pub.publish(body_language_class)  # Publish the class
                            self.publish_timestamp = time.time()  # Update the timestamp of the last published class
                            self.save_intention_to_csv(body_language_class, max(body_language_prob))  # Save to CSV (optional)

                            # Print the published class name to the terminal
                            print(f"Published Class: {body_language_class}")

                            self.consecutive_count = 0  # Reset the consecutive count after publishing
                        
                        if self.publish_timestamp and (time.time() - self.publish_timestamp) <= 1:
                            cv2.putText(image, f"Published: {self.last_detected_class}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                        else:
                            cv2.putText(image, "Waiting...", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    
                    if self.publish_solder:

                        self.publish = False
                        #cv2.putText(image, "Solder...", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                        if body_language_class == self.last_detected_class:
                            self.consecutive_count += 1
                        else:
                                self.last_detected_class = body_language_class
                                self.consecutive_count = 1

                        if self.consecutive_count == 20:
                                if body_language_class == "Up" :
                                    cv2.putText(image, "Solder...", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                                    self.pub.publish(body_language_class)  # Publish the class

                                    self.publish_timestamp = time.time()  # Update the timestamp of the last published class
                                    self.save_intention_to_csv(body_language_class, max(body_language_prob))  # Save to CSV (optional)

                                    # Print the published class name to the terminal
                                    print(f"Published Class: {body_language_class}")
                                
                                    self.consecutive_count = 0  # Reset the consecutive count after publishing
                            
                                elif body_language_class == "Down" :
                                    cv2.putText(image, "Solder...", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                                    self.pub.publish(body_language_class)  # Publish the class

                                    self.publish_timestamp = time.time()  # Update the timestamp of the last published class
                                    self.save_intention_to_csv(body_language_class, max(body_language_prob))  # Save to CSV (optional)

                                    # Print the published class name to the terminal
                                    print(f"Published Class: {body_language_class}")
                                
                                    self.consecutive_count = 0  # Reset the consecutive count after publishing
                            
                                elif body_language_class == "Left" :
                                    cv2.putText(image, "Solder...", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                                    self.pub.publish(body_language_class)  # Publish the class

                                    self.publish_timestamp = time.time()  # Update the timestamp of the last published class
                                    self.save_intention_to_csv(body_language_class, max(body_language_prob))  # Save to CSV (optional)

                                    # Print the published class name to the terminal
                                    print(f"Published Class: {body_language_class}")
                                
                                    self.consecutive_count = 0  # Reset the consecutive count after publishing
                                
                                elif body_language_class == "Right" :
                                    cv2.putText(image, "Solder...", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                                    self.pub.publish(body_language_class)  # Publish the class

                                    self.publish_timestamp = time.time()  # Update the timestamp of the last published class
                                    self.save_intention_to_csv(body_language_class, max(body_language_prob))  # Save to CSV (optional)

                                    # Print the published class name to the terminal
                                    print(f"Published Class: {body_language_class}")
                                
                                    self.consecutive_count = 0  # Reset the consecutive count after publishing

                                elif body_language_class == "Forward" :
                                    cv2.putText(image, "Solder...", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                                    self.pub.publish(body_language_class)  # Publish the class

                                    self.publish_timestamp = time.time()  # Update the timestamp of the last published class
                                    self.save_intention_to_csv(body_language_class, max(body_language_prob))  # Save to CSV (optional)

                                    # Print the published class name to the terminal
                                    print(f"Published Class: {body_language_class}")
                                
                                    self.consecutive_count = 0  # Reset the consecutive count after publishing

                                elif body_language_class == "Backward" :
                                    cv2.putText(image, "Solder...", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                                    self.pub.publish(body_language_class)  # Publish the class

                                    self.publish_timestamp = time.time()  # Update the timestamp of the last published class
                                    self.save_intention_to_csv(body_language_class, max(body_language_prob))  # Save to CSV (optional)

                                    # Print the published class name to the terminal
                                    print(f"Published Class: {body_language_class}")
                                
                                    self.consecutive_count = 0  # Reset the consecutive count after publishing

                                elif body_language_class == "Feeding" :
                                    cv2.putText(image, "Solder...", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                                    self.pub.publish(body_language_class)  # Publish the class

                                    self.publish_timestamp = time.time()  # Update the timestamp of the last published class
                                    self.save_intention_to_csv(body_language_class, max(body_language_prob))  # Save to CSV (optional)

                                    # Print the published class name to the terminal
                                    print(f"Published Class: {body_language_class}")
                                
                                    self.consecutive_count = 0  # Reset the consecutive count after publishing

                                elif body_language_class == "Stop_Feeding" :
                                    cv2.putText(image, "Solder...", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                                    self.pub.publish(body_language_class)  # Publish the class

                                    self.publish_timestamp = time.time()  # Update the timestamp of the last published class
                                    self.save_intention_to_csv(body_language_class, max(body_language_prob))  # Save to CSV (optional)

                                    # Print the published class name to the terminal
                                    print(f"Published Class: {body_language_class}")
                                
                                    self.consecutive_count = 0  # Reset the consecutive count after publishing


                        if self.publish_timestamp and (time.time() - self.publish_timestamp) <= 1: 
                                
                                cv2.putText(image, f"Published: {self.last_detected_class}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                        else:
                                cv2.putText(image, "Waiting...", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                except Exception as e:
                    print(f"Error processing frame: {e}")
                    pass
            
                
                cv2.imshow('Raw Webcam Feed', image)

                k = cv2.waitKey(1)
                if k == 27:
                    quit()
                        
                if k == ord('p'):
                        
                    self.publish = not self.publish

                if k == ord('o'):
                        
                    self.publish_solder = not self.publish_solder
                
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    

    Icontroller = intention_predict()

    try:
        Icontroller.IntentionPrediction()
    except rospy.ROSInterruptException:
        pass