
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

def interpolate_points_uniform_acceleration(point1, point2, ini_x_vel, ini_y_vel, n):
    x1, y1 = point1
    x2, y2 = point2
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

def interpolate_points_uniform_deceleration(point1, point2, ini_x_vel, ini_y_vel, n):
    x1, y1 = point1
    x2, y2 = point2
    
    interpolated_points = []
    x_deceleration = 2 * (ini_x_vel * n - x2 + x1) / n**2
    y_deceleration = 2 * (ini_y_vel * n - y2 + y1)/ n**2
    for i in range(1, n + 1):
        tm = i  # Assuming each unit of tm is 1 second
        x = x1  + ini_x_vel * tm - 0.5 * x_deceleration * tm**2
        y = y1  + ini_y_vel * tm - 0.5 * y_deceleration * tm**2
    
        interpolated_points.append((x, y))


    x_vel = ini_x_vel - x_deceleration * tm
    y_vel = ini_y_vel - y_deceleration * tm

    return interpolated_points, x_vel, y_vel

def interpolate_points(point1, point2, n):
    
    x1, y1 = point1
    x2, y2 = point2

    
    interpolated_points = []
    
    for i in range(1, n + 1):
        ratio = i / n
        x = x1 + ratio * (x2 - x1)
        y = y1 + ratio * (y2 - y1)
        interpolated_points.append((x, y))

    x_vel = (x2 - x1) / n
    y_vel = (y2 - y1) / n

    return interpolated_points, x_vel, y_vel

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
