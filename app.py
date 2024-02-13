import argparse
import csv
import os
import platform
import sys
from pathlib import Path
import math
import torch
import numpy as np
from flask import Flask, Response, request




app = Flask(__name__)

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    strip_optimizer,
    xyxy2xywh,

)
from utils.torch_utils import select_device, smart_inference_mode
from utils.track_math import *

def detect_object(path, im, im0s, dt, model, save_dir, augment, conf_thres, iou_thres, classes, agnostic_nms, max_det, visualize):

    with dt[0]:
        im = torch.from_numpy(im).to(model.device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        if model.xml and im.shape[0] > 1:
            ims = torch.chunk(im, im.shape[0], 0)

    # Inference
    with dt[1]:
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        if model.xml and im.shape[0] > 1:
            pred = None
            for image in ims:
                if pred is None:
                    pred = model(image, augment=augment, visualize=visualize).unsqueeze(0)
                else:
                    pred = torch.cat((pred, model(image, augment=augment, visualize=visualize).unsqueeze(0)), dim=0)
            pred = [pred, None]
        else:
            pred = model(im, augment=augment, visualize=visualize)
    # NMS
    with dt[2]:
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

    
    det = pred[0]
    im0 = im0s.copy()


    try: 
        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
    except: 
        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0[0].shape).round()


    ball_rows = det[det[:, 5] == 0]
    player_rows = det[det[:, 5] == 1]

    return ball_rows, player_rows





@smart_inference_mode()
def generate_frames(
    weights=ROOT / "yolov5s.pt",  # model path or triton URL
    source=ROOT / "data/images",  # file/dir/URL/glob/screen/0(webcam)
    id="123",
    team1="A", 
    team2="B",
    data=ROOT / "data/coco128.yaml",  # dataset.yaml path
    imgsz=(640, 640),  # inference size (height, width)
    conf_thres=0.1,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    max_det=1000,  # maximum detections per image
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img=False,  # show results
    save_txt=False,  # save results to *.txt
    save_csv=False,  # save results in CSV format
    save_conf=False,  # save confidences in --save-txt labels
    save_crop=False,  # save cropped prediction boxes
    nosave=False,  # do not save images/videos
    classes=None,  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False,  # class-agnostic NMS
    augment=False,  # augmented inference
    visualize=False,  # visualize features
    update=False,  # update all models
    project=ROOT / "runs/detect",  # save results to project/name
    name="exp",  # save results to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    line_thickness=3,  # bounding box thickness (pixels)
    hide_labels=False,  # hide labels
    hide_conf=False,  # hide confidences
    half=False,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    vid_stride=1,  # video frame-rate stride
):
    
    fx = 1280  # focal length in x-direction
    fy = 1250  # focal length in y-direction
    cx = 530 # result.shape[1] // 2  # principal point x-coordinate
    cy = 540 # result.shape[0] // 2  # principal point y-coordinate

    k1 = 0.1  # radial distortion coefficient
    k2 = 0.01  # radial distortion coefficient
    p1 = 0.001  # tangential distortion coefficient
    p2 = -0.002  # tangential distortion coefficient
    k3 = 0.00001  # radial distortion coefficient
    

    source = str(source)
    save_img = not nosave and not source.endswith(".txt")  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)
    screenshot = source.lower().startswith("screen")
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        # view_img = check_imshow(warn=True)
        
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
        
    elif screenshot:
        
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
        
    else:
        
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    _, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    _, _, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))

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
    
    vid_writer = cv2.VideoWriter(f"{id}_out.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (res_width, res_height))

    ### ==== other resolution 
    res_width_2 = 320
    res_height_2 = 180
    wd_2, ht_2 = res_width_2, res_height_2
    
    vid_writer_2 = cv2.VideoWriter(f"{id}_low.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (res_width_2, res_height_2))

    acc_num = 27
    vid_h = 0
    vid_w = 0  
    temp_x_vel = 0    
    temp_y_vel = 0
    iterator = iter(dataset)
    
    l_val = -800
    h_val = 800
    
    pre_center_x = 0
    pre_center_y = 0
    ht_rate = 0

    frame_num = 0
    acc_flg = 0

    last_ball_bbox = [0, 0, 0, 0]
    
    temp_player_points = [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
                          (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)
                          ]

    while True:

        try: 
            path, im, im0s, _, _ = next(iterator)
        except:
            break
        ball_rows, player_rows = detect_object(path, im, im0s, dt, model, save_dir, augment, conf_thres, iou_thres, classes, agnostic_nms, max_det, visualize) 
       
        

        
        try:
            vid_h, vid_w, _ = im0s.shape
        except: 
            vid_h, vid_w, _ = im0s[0].shape


        if ball_rows.numel() > 0:
            
            _, max_idx = ball_rows[:, 4].max(0)
            bbox = ball_rows[max_idx][None][0]
            left, top, right, bottom, confidence, class_id = bbox.tolist()
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
            
            temp_player_points = [(int(p[0]) + int(p[2]) // 2, int(p[1]) + int(p[3]) // 2) for p in player_rows[:, :4]]
            
        else:
            if last_ball_bbox == ball_bbox:

                cur_player_points = [(int(p[0]) + int(p[2]) // 2, int(p[1]) + int(p[3]) // 2) for p in player_rows[:, :4]]
                result_x = sum(point[0] for point in cur_player_points)
                result_y = sum(point[1] for point in cur_player_points)

                min_len = min(len(temp_player_points), len(cur_player_points))
                for i in range(min_len):
                    direction_x = cur_player_points[i][0] - temp_player_points[i][0]
                    direction_y = cur_player_points[i][1] - temp_player_points[i][1]
                    result_x += direction_x
                    result_y += direction_y

                
                result_x /= min_len
                result_y /= min_len
                result_x = math.ceil(result_x + 0.5)
                result_y = math.ceil(result_y + 0.5)

                
                temp_bbox = [result_x - 15, result_y - 15, result_x + 15, result_y + 15]
                [pre_center_x, pre_center_y] = boxCenter(ball_bbox)
                [cur_center_x, cur_center_y] = boxCenter(temp_bbox)
                
                point1 = (pre_center_x, pre_center_y)    
                point2 = (cur_center_x, cur_center_y)

                dis_val = euclidean_distance(point1, point2)

                if dis_val < 400:
                    ball_bbox = temp_bbox
                    print(result_x, result_y, "--Predicted Ball")


                temp_player_points = cur_player_points
                ht_rate = 0

            else:
                ball_bbox = last_ball_bbox

                ball_ht = last_ball_bbox[3] - last_ball_bbox[1]
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
                path, im, im0s, cap, s = next(iterator)
            except: 
                break

            ball_rows, player_rows = detect_object(path, im, im0s, dt, model, save_dir, augment, conf_thres, iou_thres, classes, agnostic_nms, max_det, visualize)

            if ball_rows.numel() > 0:
            
                _, max_idx = ball_rows[:, 4].max(0)
                bbox = ball_rows[max_idx][None][0]
                left, top, right, bottom, confidence, class_id = bbox.tolist()
                left, top, right, bottom = int(left), int(top), int(right), int(bottom)
                last_ball_bbox = [left, top, right, bottom]
            temp_player_points = [(int(p[0]) + int(p[2]) // 2, int(p[1]) + int(p[3]) // 2) for p in player_rows[:, :4]]
            ht = ht_rate + ht 
            wd = (res_width * ht) / res_height

            ht_2 = ht_rate + ht_2
            wd_2 = (res_width_2 * ht_2) / res_height_2

            try:
                _, _, _ = im0s.shape    
                im0 = im0s.copy()
            except: 
                _, _, _ = im0s[0].shape    
                im0 = im0s[0].copy()
            
            center_x, center_y = res_point

            angle = calculate_angle(center_x, center_y, vid_w, vid_h)

            undistorted_img = screen_processing(im0, center_x, center_y, wd, ht, angle, vid_w, vid_h, res_width, res_height, fx, fy, cx, cy, k1, k2, p1, p2, k3)
            undistorted_img_2 = screen_processing(im0, center_x, center_y, wd_2, ht_2, angle, vid_w, vid_h, res_width_2, res_height_2, fx, fy, cx, cy, k1, k2, p1, p2, k3)

            vid_writer.write(undistorted_img)
            vid_writer_2.write(undistorted_img_2)
            cv2.waitKey(1)  # Check for key press every 1ms
            
            _, buffer = cv2.imencode('.jpg', undistorted_img)
            decoded_img = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + decoded_img + b'\r\n')
              
    print("########################< END >########################")


@app.route('/video')
def video():
    weights = request.args.get('weights')
    source = request.args.get('source')
    id = request.args.get('id')
    team1 = request.args.get('team1')
    team2 = request.args.get('team2')

    return Response(generate_frames(weights, source, id, team1, team2), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)