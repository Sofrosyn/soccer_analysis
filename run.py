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


########################################### CROP #############################################

ball_bbox = [2300, 338, 2323, 361]
res_width = 1280
res_height = 720
wd, ht = res_width, res_height
cropCoords = [2300,338, 2300 + res_width, 338 + res_height]
[box_left, box_top, box_right, box_bottom] = cropCoords    
lastCoords = [box_left, box_top, box_right, box_bottom]
lastBoxCoords = lastCoords
box_width = box_right-box_left
box_height = box_bottom-box_top
fps = 20 

# vid_writer = cv2.VideoWriter(f"{id}_out.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (res_width, res_height))
# ### ==== other resolution 
# res_width_2 = 320
# res_height_2 = 180
# wd_2, ht_2 = res_width_2, res_height_2
# vid_writer_2 = cv2.VideoWriter(f"{id}_low.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (res_width_2, res_height_2))

acc_num = 27
temp_x_vel = 0    
temp_y_vel = 0
l_val = -800
h_val = 800
pre_center_x = 0
pre_center_y = 0
ht_rate = 0
frame_num = 0
acc_flg = 0
last_ball_bbox = [0, 0, 0, 0]

fx = 1280  # focal length in x-direction
fy = 1250  # focal length in y-direction
cx = 530 # result.shape[1] // 2  # principal point x-coordinate
cy = 540 # result.shape[0] // 2  # principal point y-coordinate

k1 = 0.1  # radial distortion coefficient
k2 = 0.01  # radial distortion coefficient
p1 = 0.001  # tangential distortion coefficient
p2 = -0.002  # tangential distortion coefficient
k3 = 0.00001  # radial distortion coefficient


vid_h, vid_w, _ = frame.shape

while ret:

    # Get Detections
    # players_detections = get_player_detections(player_detector, frame)
    # ball_detections = get_ball_detections(ball_detector, frame)
    players_detections, ball_detections, ball_value = get_ball_player_detections(object_detector, frame)

    detections = ball_detections + players_detections

    # Update trackers
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
    

    # # Draw
    # frame = PIL.Image.fromarray(frame)  

    # if args.possession:

    #     frame = Player.draw_players(
    #         players=players, frame=frame, confidence=False, id=True
    #     )

    #     frame = path.draw(
    #         img=frame,
    #         detection=ball.detection,
    #         coord_transformations=coord_transformations,
    #         color=match.team_possession.color,
    #     )

    #     frame = match.draw_possession_counter(
    #         frame, counter_background=possession_background, debug=False
    #     )

    #     if ball:
    #         frame = ball.draw(frame)

    # if args.passes:
    #     pass_list = match.passes

    #     frame = Pass.draw_pass_list(
    #         img=frame, passes=pass_list, coord_transformations=coord_transformations
    #     )

    #     frame = match.draw_passes_counter(
    #         frame, counter_background=passes_background, debug=False
    #     )

    # frame = np.array(frame)
    # # Write video

    # video.write(frame)
    if len(ball_value) > 0:
        try:
            left, top, right, bottom, confidence = ball_value.xmin, ball_value.ymin, ball_value.xmax, ball_value.ymax, ball_value.confidence
            left, top, right, bottom = int(left), int(top), int(right), int(bottom)
        except:

            high_ball = ball_value.sort_values(by='confidence', ascending=False).iloc[0]
            left, top, right, bottom, confidence = high_ball.xmin, high_ball.ymin, high_ball.xmax, high_ball.ymax, high_ball.confidence
            left, top, right, bottom = int(left), int(top), int(right), int(bottom)
            

        temp_bbox = [left, top, right, bottom]
        [pre_center_x, pre_center_y] = boxCenter(ball_bbox)
        [cur_center_x, cur_center_y] = boxCenter(temp_bbox)
        ball_ht = bottom - top
        point1 = (pre_center_x, pre_center_y)    
        point2 = (cur_center_x, cur_center_y)
        dis_val = euclidean_distance(point1, point2)
        ball_bbox = temp_bbox
        limit_ht = math.ceil(vid_h * 0.85 + 0.5) 
        ht_rate = ((limit_ht  - ((ball_ht / 20) - 1) * 54) - ht) / acc_num
        
  
  

    
    newCoords = adjustBoxSize(ball_bbox, box_width, box_height) 
    newCoords = adjustBoundaries(newCoords,[vid_w, vid_h]) 
    
    [box_left, box_top, box_right, box_bottom] = newCoords

    [cur_center_x, cur_center_y] = boxCenter(newCoords)
    [pre_center_x, pre_center_y] = boxCenter(lastBoxCoords)
    
    point1 = (pre_center_x, pre_center_y)    
    point2 = (cur_center_x, cur_center_y)

    cur_camera_dis = math.ceil(euclidean_distance(point1, point2) + 0.5)

    if acc_flg == 0:
        if  cur_camera_dis >= l_val and cur_camera_dis < h_val:
            res_points, temp_x_vel, temp_y_vel = interpolate_points_uniform_acceleration(point1, point2, temp_x_vel, temp_y_vel, acc_num)
            acc_flg = 1
        elif cur_camera_dis < l_val:
            res_points, temp_x_vel, temp_y_vel = interpolate_points(point1, point2, acc_num)

        elif cur_camera_dis > h_val:
            res_points, temp_x_vel, temp_y_vel = interpolate_points_uniform_deceleration(point1, point2, temp_x_vel, temp_y_vel, acc_num)
            # temp_x_vel, temp_y_vel = 0, 0
    else:
        res_points, temp_x_vel, temp_y_vel = interpolate_points_uniform_deceleration(point1, point2, temp_x_vel, temp_y_vel, acc_num)
        # temp_x_vel, temp_y_vel = 0, 0
        acc_flg = 0
    

    lastBoxCoords = newCoords  


    ln = len(res_points)
    print(res_points[ln - 1], "--ball---")

    frame_num += 1
    
    for res_point in res_points:
        try:
            ret, frame = video.video_capture.read()
        except: 
            break

        
        
        center_x, center_y = res_point
        angle = calculate_angle(center_x, center_y, vid_w, vid_h)
        undistorted_img = screen_processing(frame, center_x, center_y, wd, ht, angle, vid_w, vid_h, res_width, res_height, fx, fy, cx, cy, k1, k2, p1, p2, k3)
        
        opencv_image_rgb = cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(opencv_image_rgb)

        if args.possession:

            

            pil_image = match.draw_possession_counter(
                pil_image, counter_background=possession_background, debug=False
            )

            


        pil_image_array = np.array(pil_image)

    
        final_frame = cv2.cvtColor(pil_image_array, cv2.COLOR_RGB2BGR)
        cv2.namedWindow("result", 0)
        cv2.imshow("result", final_frame)

        # undistorted_img_2 = screen_processing(im0, center_x, center_y, wd_2, ht_2, angle, vid_w, vid_h, res_width_2, res_height_2, fx, fy, cx, cy, k1, k2, p1, p2, k3)
        # vid_writer.write(undistorted_img)
        # vid_writer_2.write(undistorted_img_2)
        cv2.waitKey(1)  # Check for key press every 1ms
    #         _, buffer = cv2.imencode('.jpg', undistorted_img)
    #         decoded_img = buffer.tobytes()
    #         yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + decoded_img + b'\r\n')
    print("########################< END >########################")
    ret, frame = video.video_capture.read()
    match.update(players, ball)
