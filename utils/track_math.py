
import numpy as np
import math
import cv2

def calculate_centroid(detections):
    if detections.numel() > 0:
        centers = detections[:, :4].reshape(-1, 4)  # Get the bounding boxes
        centers = (centers[:, :2] + centers[:, 2:]) / 2  # Calculate the center points
        centroid = centers.mean(axis=0).tolist()  # Calculate the average center point
        return [int(centroid[0]), int(centroid[1])]
    return None

def interpolate_points_uniform_acceleration(point1, point2, ini_x_vel, ini_y_vel, n, wd, ht):
    x1, y1 = point1
    x2, y2 = point2
    x1, y1 = max(0, min(x1, wd)), max(0, min(y1, ht))
    x2, y2 = max(0, min(x2, wd)), max(0, min(y2, ht))

    ini_x_vel = 0
    ini_y_vel = 0
    
    interpolated_points = []
    x_deceleration = 2 * (x2 - x1 - ini_x_vel * n ) / n**2
    y_deceleration = 2 * (y2 - y1 - ini_y_vel * n)/ n**2
    for i in range(1, n + 1):
        tm = i  # Assuming each unit of tm is 1 second
        x = x1  + ini_x_vel * tm + 0.5 * x_deceleration * tm**2
        y = y1  + ini_y_vel * tm + 0.5 * y_deceleration * tm**2
    
        interpolated_points.append((x, y))


    x_vel = ini_x_vel + x_deceleration * tm
    y_vel = ini_y_vel + y_deceleration * tm

    return interpolated_points, x_vel, y_vel

def interpolate_points_uniform_deceleration(point1, point2, initial_x_velocity, initial_y_velocity, num_points, width, height):
    x1, y1 = point1
    x2, y2 = point2
    x1, y1 = max(0, min(point1[0], width)), max(0, min(point1[1], height))
    x2, y2 = max(0, min(point2[0], width)), max(0, min(point2[1], height))

    interpolated_points = []
    
    x_deceleration = 2 * (initial_x_velocity * num_points - x2 + x1) / num_points**2
    y_deceleration = 2 * (initial_y_velocity * num_points - y2 + y1) / num_points**2
    
    for t in range(1, num_points):
        x = x1 + initial_x_velocity * t - 0.5 * x_deceleration * t**2
        y = y1 + initial_y_velocity * t - 0.5 * y_deceleration * t**2
        interpolated_points.append((x, y))
    
    final_x_velocity = initial_x_velocity - x_deceleration * num_points
    final_y_velocity = initial_y_velocity - y_deceleration * num_points

    return interpolated_points, final_x_velocity, final_y_velocity



def interpolate_points(point1, point2, n, wd, ht):
     
    x1, y1 = point1
    x2, y2 = point2
    x1, y1 = max(0, min(x1, wd)), max(0, min(y1, ht))
    x2, y2 = max(0, min(x2, wd)), max(0, min(y2, ht))
    

    
    interpolated_points = []
    
    for i in range(1, n + 1):
        ratio = i / n
        x = x1 + ratio * (x2 - x1)
        y = y1 + ratio * (y2 - y1)
        interpolated_points.append((x, y))

    x_vel = (x2 - x1) / n
    y_vel = (y2 - y1) / n

    return interpolated_points, x_vel, y_vel

def interpolate_log_points(point1, point2, n):
     
    x1, y1 = point1
    x2, y2 = point2
        
    interpolated_points = []
    
    x_start = x1
    x_stop = x2
    x_intervals = np.logspace(np.log10(x_start), np.log10(x_stop), num=n)

    y_start = y1
    y_stop = y2
    y_intervals = np.logspace(np.log10(y_start), np.log10(y_stop), num=n)


    for i in range(len(x_intervals)):
        x = x_intervals[i]
        y = y_intervals[i]
        interpolated_points.append((x, y))

    return interpolated_points




def create_polygon_vertices(center_x, center_y, width, height, angle): # Generate polygon vertices based on center point and angle
    # Calculate the rectangle vertices
    rectangle_vertices = np.array([
        [-width / 2, -height / 2],
        [width / 2, -height / 2],
        [width / 2, height / 2],
        [-width / 2, height / 2],
    ], dtype=np.float32)

    # Get the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D((0, 0), angle, 1)

    # Apply rotation to the rectangle vertices
    rotated_rectangle = cv2.transform(np.array([rectangle_vertices]), rotation_matrix)[0]

    # Translate the rotated rectangle to the specified center
    translated_rectangle = rotated_rectangle + np.array([center_x, center_y])
    
    return translated_rectangle


def boxCenter(coords):
    [left, top, right, bottom] = coords
    return [(left+right)/2,(top+bottom)/2]

def closestBox(boxes, coords):
    distance = []
    center = boxCenter(coords)
    for box in boxes:
        boxCent = boxCenter(box)
        distance.append(math.dist(boxCent,center))
    return boxes[distance.index(min(distance))]

def adjustBoxSize(coords, box_width, box_height):
    [centerX, centerY] = boxCenter(coords)
    return [centerX-box_width/2, centerY-box_height/2, centerX+box_width/2, centerY+box_height/2]


def adjustPolyBoundaries(coords, screen):
    left = int(np.min(coords[:, 0]))
    right = int(np.max(coords[:, 0]))
    top = int(np.min(coords[:, 1]))
    bottom = int(np.max(coords[:, 1]))

    [width, height]=screen
        
    boundary_left = 0
    boundary_right = width  # Adjust according to your image width
    boundary_top = 0
    boundary_bottom = height  # Adjust according to your image height
    
    adjustment_x = min(0, left - boundary_left)
    adjustment_y = min(0, top - boundary_top)
    
    
    coords[:, 0] -= adjustment_x
    coords[:, 1] -= adjustment_y

    adjustment_x = min(0, boundary_right - right)
    adjustment_y = min(0, boundary_bottom - bottom)
    
    
    coords[:, 0] += adjustment_x
    coords[:, 1] += adjustment_y

    return coords




def adjustBoundaries(coords, screen):
    [left, top, right, bottom] = coords
    [width, height]=screen
    if left<0:
        right=right-left
        left=0
    if top<0:
        bottom=bottom-top
        top=0
    if right>width:
        left=left-(right-width)
        right=width
    if bottom>height:
        top=top-(bottom-height)
        bottom=height
    return [round(left), round(top), round(right), round(bottom)]

def euclidean_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance




def calculate_angle(center_x, center_y, vid_w, vid_h):
    if center_x < vid_w / 2 and center_x > vid_w / 2 + 374:
        angle = 0
    else:
        ########################################## Calculate Angle ##########################################
        oppo = abs(vid_w / 2 + 187 - center_x)       # Set opposite side of tangent.
        adja = abs(vid_h / 2 - center_y) + 5000 # + cal_lean(x, 5120) # Set adjacent of tangent.
        # oppo = abs(vid_w / 2 + 240 - center_x)       # Set opposite side of tangent.
        # adja = abs(vid_h / 2 - center_y) + 6130 # + cal_lean(x, 5120) # Set adjacent of tangent.

        theta_rad = math.atan2(oppo, adja)          # Calculate tangent value    
        angle = math.degrees(theta_rad)             # Convert radian to angle.

        if (center_x > vid_w // 2):                  # Convert Signals of Angle.
            angle = -angle
    return angle



def screen_processing(im0, center_x, center_y, wd, ht, angle, vid_w, vid_h, res_width, res_height, fx, fy, cx, cy, k1, k2, p1, p2, k3):


    temp_vertices = create_polygon_vertices(center_x, center_y, wd, ht, angle)
    polygon_vertices = adjustPolyBoundaries(temp_vertices, [vid_w, vid_h])
    perspective_matrix = cv2.getPerspectiveTransform(np.float32(polygon_vertices[:4]), np.float32([[0, 0], [res_width, 0], [res_width, res_height], [0, res_height]]))
    result = cv2.warpPerspective(im0, perspective_matrix, (res_width, res_height))
    camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    distortion_coeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float32)
    camera_matrix = camera_matrix.reshape((3, 3))
    distortion_coeffs = distortion_coeffs[:4]
    undistorted_img = cv2.fisheye.undistortImage(result, camera_matrix, distortion_coeffs, Knew=camera_matrix)


    return undistorted_img




def predict_next_position(last_positions):
    # Assuming last_positions is a numpy array of shape (n, 2) where n is the number of previous positions
    # Each row represents a (x, y) coordinate of the ball
    
    if len(last_positions) < 2:
        return None  # Not enough data to predict
    

    
    direction_vector = [0, 0]
    predicted_position = [0, 0]
    # Extract the last known position and the position before that
    last_known_position = last_positions[-1]
    previous_position = last_positions[-2]
    
    # Calculate the direction vector from the previous position to the last known position
    direction_vector[0] = last_known_position[0] - previous_position[0]
    direction_vector[1] = last_known_position[1] - previous_position[1]
    
    # Assuming constant velocity, extrapolate the next position by adding the direction vector to the last known position
    predicted_position[0] = last_known_position[0] + direction_vector[0]
    predicted_position[1] = last_known_position[1] + direction_vector[1] 
    
    return predicted_position