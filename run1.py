import argparse
import cv2
import numpy as np
import PIL
from norfair import Tracker, Video
from norfair.camera_motion import MotionEstimator
from norfair.distances import mean_euclidean
from PIL import Image

from inference import Converter, HSVClassifier, InertiaClassifier, YoloV5
from inference.filters import filters
from run_utils import (
    get_ball_detections,
    get_main_ball,
    get_player_detections,
    get_ball_player_detections,
    update_motion_estimator,
)
from soccer import Match, Player, Team
from soccer.draw import AbsolutePath
from soccer.pass_event import Pass
from utils.track_math import *

parser = argparse.ArgumentParser()
parser.add_argument(
    "--video",
    default="videos/soccer_possession.mp4",
    type=str,
    help="Path to the input video",
)
parser.add_argument(
    "--model", default="models/ball.pt", type=str, help="Path to the model"
)
parser.add_argument(
    "--passes",
    action="store_true",
    help="Enable pass detection",
)
parser.add_argument(
    "--possession",
    action="store_true",
    help="Enable possession counter",
)
args = parser.parse_args()

video = Video(input_path=args.video)
fps = video.video_capture.get(cv2.CAP_PROP_FPS)

# Object Detectors
# player_detector = YoloV5(model_path ="yolov5s_class3_400_25k.pt")
# ball_detector = YoloV5(model_path=args.model)

object_detector = YoloV5(model_path=args.model)
# HSV Classifier
hsv_classifier = HSVClassifier(filters=filters)
# Add inertia to classifier
classifier = InertiaClassifier(classifier=hsv_classifier, inertia=50)

# Teams and Match
chelsea = Team(
    name="Chelsea",
    abbreviation="RED",
    color=(0, 0, 255),
    board_color=(244, 86, 64),
    text_color=(255, 255, 255),
)
man_city = Team(
    name="Man City", 
    abbreviation="WHITE", 
    color=(250, 250, 250))
teams = [chelsea, man_city]
match = Match(home=chelsea, away=man_city, fps=fps)
match.team_possession = man_city

# Tracking
player_tracker = Tracker(
    distance_function=mean_euclidean,
    distance_threshold=250,
    initialization_delay=3,
    hit_counter_max=90,
)

ball_tracker = Tracker(
    distance_function=mean_euclidean,
    distance_threshold=250,
    initialization_delay=3,
    hit_counter_max=2000,
)
motion_estimator = MotionEstimator()
coord_transformations = None

# Paths
path = AbsolutePath()

# Get Counter img
possession_background = match.get_possession_background()
passes_background = match.get_passes_background()

ret, frame = video.video_capture.read()


############################################# CROP #############################################
smooth_factor = 0.1
last_known_position = None
last_known_velocity = [0, 0]
missed_detections = 0
max_missed_detections = 35
crop_width = 1280
crop_height = 720

fx = 1280  # focal length in x-direction
fy = 1250  # focal length in y-direction
cx = 530 # result.shape[1] // 2  # principal point x-coordinate
cy = 540 # result.shape[0] // 2  # principal point y-coordinate

k1 = 0.1  # radial distortion coefficient
k2 = 0.01  # radial distortion coefficient
p1 = 0.001  # tangential distortion coefficient
p2 = -0.002  # tangential distortion coefficient
k3 = 0.00001  # radial distortion coefficient

fps = 25
fourcc = cv2.VideoWriter_fourcc(*'XVID')

vid_writer = cv2.VideoWriter("out.avi", fourcc, fps, (crop_width, crop_height))
vid_h, vid_w, _ = frame.shape

while ret:
    
    players_detections, ball_detections, ball_value = get_ball_player_detections(object_detector, frame)
    detections = ball_detections + players_detections
    coord_transformations = update_motion_estimator(
        motion_estimator=motion_estimator,
        detections=detections,
        frame=frame,
    )

    player_track_objects = player_tracker.update(
        detections=players_detections, coord_transformations=coord_transformations
    )

    ball_track_objects = ball_tracker.update(
        detections=ball_detections, coord_transformations=coord_transformations
    )

    player_detections = Converter.TrackedObjects_to_Detections(player_track_objects)
    ball_detections = Converter.TrackedObjects_to_Detections(ball_track_objects)

    player_detections = classifier.predict_from_detections(
        detections=player_detections,
        img=frame,
    )
    # Match update
    ball = get_main_ball(ball_detections)
    players = Player.from_detections(detections=players_detections, teams=teams)
   
    if len(ball_value) > 0:
        try:
            
            left, top, right, bottom, confidence = ball_value.xmin, ball_value.ymin, ball_value.xmax, ball_value.ymax, ball_value.confidence
            left, top, right, bottom = int(left), int(top), int(right), int(bottom)
            new_x, new_y = (left + right) // 2, (top + bottom) // 2
        except:
            
            high_ball = ball_value.sort_values(by='confidence', ascending=False).iloc[0]
            left, top, right, bottom, confidence = high_ball.xmin, high_ball.ymin, high_ball.xmax, high_ball.ymax, high_ball.confidence
            left, top, right, bottom = int(left), int(top), int(right), int(bottom)
            new_x, new_y = (left + right) // 2, (top + bottom) // 2
            

        
        missed_detections = 0  # Reset missed detections counter
        if last_known_position:
            last_known_velocity = [new_x - last_known_position[0], new_y - last_known_position[1]]
        last_known_position = [new_x, new_y]
    else:
        # Count missed detections
        missed_detections += 1
        if missed_detections <= max_missed_detections and last_known_position:
            # Predict the next position based on last known velocity
            new_x, new_y = [last_known_position[0] + last_known_velocity[0], last_known_position[1] + last_known_velocity[1]]
            last_known_position = [new_x, new_y]
        else:
            # Too many missed detections, might stop prediction or handle differently
            continue  # Skip this frame or implement alternative logic

    x_pos, y_pos = last_known_position
    x_pos = int(x_pos + (new_x - x_pos) * smooth_factor)
    y_pos = int(y_pos + (new_y - y_pos) * smooth_factor)
    last_known_position = [x_pos, y_pos]

    # Ensure cropped frame stays within bounds
    start_x = max(0, min(x_pos - crop_width // 2, vid_w - crop_width))
    start_y = max(0, min(y_pos - crop_height // 2, vid_h - crop_height))

    # Crop and write the frame
    cropped_frame = frame[start_y:start_y+crop_height, start_x:start_x+crop_width]
    

    # Optionally show the frame (for debugging)
    cv2.imshow('Cropped Frame', cropped_frame)
    vid_writer.write(cropped_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


    print("########################< END >########################")

    ret, frame = video.video_capture.read()
    match.update(players, ball)
