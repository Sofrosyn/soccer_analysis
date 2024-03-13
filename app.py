import cv2
import numpy as np
import PIL
from norfair import Tracker, Video
from norfair.camera_motion import MotionEstimator
from norfair.distances import mean_euclidean
from PIL import Image
from collections import deque
from flask import Flask, Response, request
import json
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
from utils.stream_output import *
from utils.playermap import *

app = Flask(__name__)


def generate_frames(video_path, model_path, enable_pass_detection, enable_possession_counter, id, team1, team2, crop_basis):

    output_dir = os.path.normpath(os.path.join(BASE_DIR, "..", "rtmp_out"))
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Folder '{output_dir}' created successfully.")
        except OSError as e:
            print(f"Error creating folder '{output_dir}': {str(e)}")

    
    video = Video(input_path=video_path)
    fps = video.video_capture.get(cv2.CAP_PROP_FPS)

    object_detector = YoloV5(model_path=model_path)
    hsv_classifier = HSVClassifier(filters=filters)
    classifier = InertiaClassifier(classifier=hsv_classifier, inertia=50)

    team_one = Team(
        name=team1,
        abbreviation=team1[:3],
        color=(0, 0, 255),
        board_color=(244, 86, 64),
        text_color=(255, 255, 255),
    )
    team_two = Team(
        name=team2, 
        abbreviation=team2[:3], 
        color=(250, 250, 250))
    teams = [team_one, team_two]
    match = Match(home=team_one, away=team_two, fps=fps)
    match.team_possession = team_one

    player_tracker = Tracker(
        distance_function=mean_euclidean,
        distance_threshold=250,
        initialization_delay=3,
        hit_counter_max=90,
    )

    ball_tracker = Tracker(
        distance_function=mean_euclidean,
        distance_threshold=20,
        initialization_delay=3,
        hit_counter_max=2000,
    )
    motion_estimator = MotionEstimator()
    coord_transformations = None

    possession_background = match.get_possession_background()
    passes_background = match.get_passes_background()

    ret, frame = video.video_capture.read()

    ##################################################### initialize variables #####################################################
    MAX_LENGTH = 3
    last_ball_positions = deque(maxlen=MAX_LENGTH)
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
    fps = 24

    acc_num = 27
    
    l_val = 150
    h_val = 800

    if crop_basis > 0:
        l_val *= 0.2
    
    pre_center_x = 0
    pre_center_y = 0
    ht_rate = 0
    acc_flg = 0

    temp_x_vel = 0    
    temp_y_vel = 0
    

    fx = 1280  # focal length in x-direction
    fy = 1250  # focal length in y-direction
    cx = 530 # result.shape[1] // 2  # principal point x-coordinate
    cy = 540 # result.shape[0] // 2  # principal point y-coordinate

    k1 = 0.1  # radial distortion coefficient
    k2 = 0.01  # radial distortion coefficient
    p1 = 0.001  # tangential distortion coefficient
    p2 = -0.002  # tangential distortion coefficient
    k3 = 0.00001  # radial distortion coefficient
    
    center_x = 0
    center_y = 0
    os.makedirs("videos", exist_ok=True)
    vid_writer = cv2.VideoWriter(f"{output_dir}/{id}_hd_out.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (res_width, res_height))
    vid_writer_sd = cv2.VideoWriter(f"{output_dir}/{id}_sd_out.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (320, 240))
    vid_h, vid_w, _ = frame.shape
    full_writer = cv2.VideoWriter(f"{output_dir}/{id}_full.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (vid_w, vid_h))
    
    players_list = []

    if crop_basis == 0:
        full_writer.write(frame)
        
        for _ in range(25):
            tile_img = 255 * np.ones((256,144,3), np.uint8)
            players_list.append(tile_img)


    ####################################  Auxiliary Variable ####################################
    frame_num = 0
    stream_num = 1
    
    map_count = 0
    
    while ret:
        players_detections, ball_detections, players_value, ball_value = get_ball_player_detections(object_detector, frame)

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
        players_ln = len(players)
        print(match.home.passes, "******** HOME PASS *******")
        print(match.away.passes, "******** AWAY PASS *******")
        print(match.away.possession, "******** AWAY POSS *******")
        print(match.home.possession, "******** HOME POSS *******")
        ############################################################### CROP PLAYERS MAP ###############################################################
        if crop_basis == 0:
        
            for player in players:
                try:
                    player_id = player.detection.data["id"]
                    if np.sum(np.all(players_list[player_id - 1] == [255, 255, 255], axis=2)) != 36864:
                        continue
                    pxmin, pymin, pxmax, pymax = player.detection.points[0][0], player.detection.points[0][1], player.detection.points[1][0], player.detection.points[1][1]
                    tmp_img = frame[pymin:pymax, pxmin:pxmax]
                    tmp_img = cv2.resize(tmp_img, (144, 256))
                    players_list[player_id - 1] = tmp_img
                    
                except:
                    continue

            

                
                

        ############################################################### INERTIA ###############################################################
        
        if crop_basis == 0 and len(ball_value) > 0:
        # if crop_basis == 0 and ball.detection is not None:
            try:
                left, top, right, bottom, confidence = ball_value.xmin, ball_value.ymin, ball_value.xmax, ball_value.ymax, ball_value.confidence
                left, top, right, bottom = int(left), int(top), int(right), int(bottom)
            except:
                high_ball = ball_value.sort_values(by='confidence', ascending=False).iloc[0]
                left, top, right, bottom, confidence = high_ball.xmin, high_ball.ymin, high_ball.xmax, high_ball.ymax, high_ball.confidence
                left, top, right, bottom = int(left), int(top), int(right), int(bottom)

            # left, top, right, bottom = ball.detection.points[0][0], ball.detection.points[0][1], ball.detection.points[1][0], ball.detection.points[1][1]
                
            ball_bbox = [left, top, right, bottom]
            [cur_center_x, cur_center_y] = boxCenter(ball_bbox)
            ball_ht = bottom - top
            
            limit_ht = math.ceil(vid_h * 0.7 + 0.5) 
            ht_rate = ((limit_ht  - ((ball_ht / 20) - 1) * acc_num * 4) - ht) / acc_num
            last_ball_positions.append([cur_center_x, cur_center_y])

        elif crop_basis > 0 and players_ln > 0:
            
            last_ball_ln = len(last_ball_positions)
            if len(last_ball_positions) == 0:
                pxmin, pymin, pxmax, pymax = 2300, 338, 2323, 361
            else:
                pxmin, pymin, pxmax, pymax = last_ball_positions[last_ball_ln - 1][0] - 30, last_ball_positions[last_ball_ln - 1][1] - 50, last_ball_positions[last_ball_ln - 1][0] + 30, last_ball_positions[last_ball_ln - 1][1] + 50
            for player in players:
                try:
                    if player.detection.data["id"] == crop_basis:
                        pxmin, pymin, pxmax, pymax = player.detection.points[0][0], player.detection.points[0][1], player.detection.points[1][0], player.detection.points[1][1]
                except:
                    print("Player missed")
            left, top, right, bottom = pxmin, pymin, pxmax, pymax
            ball_bbox = [left, top, right, bottom]
            [cur_center_x, cur_center_y] = boxCenter(ball_bbox)
            ball_ht = bottom - top
            
            limit_ht = math.ceil(vid_h * 0.7 + 0.5) 
            ht_rate = ((limit_ht  - ((ball_ht / 20) - 1) * acc_num * 4) - ht) / acc_num
            last_ball_positions.append([cur_center_x, cur_center_y])
            
        else:
            

            try:
                predict_position = predict_next_position(last_ball_positions)
                
                
                ball_bbox = [predict_position[0] - 15, predict_position[1] - 15, predict_position[0] + 15, predict_position[1] + 15]
                [cur_center_x, cur_center_y] = boxCenter(ball_bbox)
                ball_ht = bottom - top
                
                limit_ht = math.ceil(vid_h * 0.7 + 0.5) 
                ht_rate = ((limit_ht  - ((ball_ht / 20) - 1) * acc_num * 4) - ht) / acc_num
                last_ball_positions.append([cur_center_x, cur_center_y])
            except:
                
                num = len(last_ball_positions)
        
        newCoords = adjustBoxSize(ball_bbox, box_width, box_height)
        newCoords = adjustBoundaries(newCoords,[vid_w, vid_h])
        
        [box_left, box_top, box_right, box_bottom] = newCoords
        [cur_center_x, cur_center_y] = boxCenter(newCoords)
        [pre_center_x, pre_center_y] = boxCenter(lastBoxCoords)
        
        point1 = (pre_center_x, pre_center_y)
        point2 = (cur_center_x, cur_center_y)

        cur_camera_dis = abs(math.ceil(euclidean_distance(point1, point2) + 0.5))

        if acc_flg == 0:

            if  cur_camera_dis >= l_val and cur_camera_dis < h_val:
                
                res_points, temp_x_vel, temp_y_vel = interpolate_points(point1, point2, acc_num, vid_w, vid_h)
                acc_num = 24
                print("Nor")
            elif cur_camera_dis < l_val:
                res_points, temp_x_vel, temp_y_vel = interpolate_points_uniform_acceleration(point1, point2, temp_x_vel, temp_y_vel, acc_num, vid_w, vid_h)
                acc_flg = 1
                

                print("Acc")

            elif cur_camera_dis > h_val:
                res_points, temp_x_vel, temp_y_vel = interpolate_points_uniform_deceleration(point1, point2, temp_x_vel, temp_y_vel, acc_num, vid_w, vid_h)
                acc_num = 15
                acc_flg = 0
                print("DEC")

        else:
            res_points, temp_x_vel, temp_y_vel= interpolate_points_uniform_deceleration(point1, point2, temp_x_vel, temp_y_vel, acc_num, vid_w, vid_h)
            acc_flg = 0
            acc_num = 27
        #     print("DEC")
        
        lastBoxCoords = newCoords  
        

        
        for res_point in res_points:

            
            if frame_num > 200:
                vid_writer.release()
                convert_mp4_to_hls(f"{output_dir}/{id}_hd_out.mp4", f"{id}{stream_num}")
                vid_writer = cv2.VideoWriter(f"{output_dir}/{id}_hd_out.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (res_width, res_height))
                stream_num += 1
                frame_num = 0
            try:
                ret, frame = video.video_capture.read()
                if crop_basis == 0:
                    full_writer.write(frame)
    
                center_x, center_y = res_point
                # ht += ht_rate
                # wd = (res_width * ht) / res_height
                angle = calculate_angle(center_x, center_y, vid_w, vid_h)
                undistorted_img = screen_processing(frame, center_x, center_y, wd, ht, angle, vid_w, vid_h, res_width, res_height, fx, fy, cx, cy, k1, k2, p1, p2, k3)
            except:
                vid_writer.release()
                convert_mp4_to_hls(f"{output_dir}/{id}_hd_out.mp4", f"{id}{stream_num}")
                if crop_basis == 0:
                    player_map_img = create_player_map(players_list)
                    cv2.imwrite(f"{output_dir}/{id}.png", player_map_img)

                    analysis_file_path = f"{output_dir}/{id}.json"
                    analaysis_data = {
                        "data": [
                            {
                                "possession": {
                                    "team1": match.home.possession,
                                    "team2": match.away.possession
                                }
                            },
                            {
                                "passes": {
                                    "team1": match.home.passes,
                                    "team2": match.away.passes
                                }
                            }
                        ]
                    }
                    try:
                        with open(analysis_file_path, 'w') as json_file:
                            json.dump(analaysis_data, json_file, indent=4)
                        print(f"Data has been written to '{analysis_file_path}' successfully.")
                    except Exception as e:
                        print(f"An error occurred while writing to the file: {e}")


                break
            opencv_image_rgb = cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(opencv_image_rgb)
            
            if enable_possession_counter:
                
                pil_image = match.draw_possession_counter(
                    pil_image, counter_background=possession_background, debug=False
                )
                
            if enable_pass_detection:
                
                pass_list = match.passes
                pil_image = Pass.draw_pass_list(
                    img=pil_image, passes=pass_list, coord_transformations=coord_transformations
                )
                pil_image = match.draw_passes_counter(
                    pil_image, counter_background=passes_background, debug=False
                )

            pil_image_array = np.array(pil_image)
            final_frame = cv2.cvtColor(pil_image_array, cv2.COLOR_RGB2BGR)
            
            _, buffer = cv2.imencode('.jpg', final_frame)
            decoded_img = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + decoded_img + b'\r\n')


            vid_writer.write(final_frame)
            frame_num += 1


            try:
                match.update(players, ball)
            except:
                continue
            cv2.waitKey(1)  # Check for key press every 1ms
        
        try:
            ret, frame = video.video_capture.read()
        except:
            # vid_writer.release()
            convert_mp4_to_hls(f"{output_dir}/{id}_ht_out.mp4", f"{id}{stream_num}")
            if crop_basis == 0:
                player_map_img = create_player_map(players_list)
                cv2.imwrite(f"{output_dir}/{id}.png", player_map_img)

                analysis_file_path = f"{output_dir}/{id}.json"
                analaysis_data = {
                    "data": [
                        {
                            "possession": {
                                "team1": match.home.possession,
                                "team2": match.away.possession
                            }
                        },
                        {
                            "passes": {
                                "team1": match.home.passes,
                                "team2": match.away.passes
                            }
                        }
                    ]
                }
                try:
                    with open(analysis_file_path, 'w') as json_file:
                        json.dump(analaysis_data, json_file, indent=4)
                    print(f"Data has been written to '{analysis_file_path}' successfully.")
                except Exception as e:
                    print(f"An error occurred while writing to the file: {e}")


        if crop_basis == 0:
            full_writer.write(frame)
        
    print("########################< END >########################")


@app.route('/video')
def video():
    video_path = request.args.get('source', default="videos/soccer_possession.mp4", type=str)
    model_path = request.args.get('weights', default="models/yolov5s_class3_300_25k.pt", type=str)
    enable_pass_detection = request.args.get('passes', type=lambda x: x.lower() in ['true', '1', 'yes'])
    enable_possession_counter = request.args.get('possession', type=lambda x: x.lower() in ['true', '1', 'yes'])

    
    id = request.args.get('id')
    team1 = request.args.get('team1')
    team2 = request.args.get('team2')
    crop_basis = request.args.get('crop_basis', type=int)

    
    return Response(generate_frames(video_path, model_path, enable_pass_detection, enable_possession_counter, id, team1, team2, crop_basis), mimetype='multipart/x-mixed-replace; boundary=frame')
    

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
