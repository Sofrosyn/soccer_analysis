import cv2
import numpy as np
import os


def create_player_map(image_list):

    # Define the size of the image map
    rows = 5
    cols = 5

    # Initialize the dimensions of each cell
    cell_width = 144  # Adjust as per your image size
    cell_height = 256  # Adjust as per your image size

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    font_thickness = 3
    font_color = (0, 0, 255)

    # Create a blank canvas for the image map
    image_map = np.zeros((rows * cell_height, cols * cell_width, 3), dtype=np.uint8)

    # Iterate through each image and paste it onto the image map
    for i in range(rows):
        for j in range(cols):
            # Read the image
            image = image_list[i * cols + j]
            if i * cols + j + 1 < 10:

                cv2.putText(image, str(i * cols + j + 1), (56, 214), font, font_scale, font_color, font_thickness, cv2.LINE_AA)
            else:
                cv2.putText(image, str(i * cols + j + 1), (28, 214), font, font_scale, font_color, font_thickness, cv2.LINE_AA)
            # Resize the image to fit the cell dimensions
            

            # Determine the coordinates to paste the image
            y_start = i * cell_height
            y_end = (i + 1) * cell_height
            x_start = j * cell_width
            x_end = (j + 1) * cell_width

            # Paste the image onto the image map
            image_map[y_start:y_end, x_start:x_end] = image

    # Display or save the image map
    return image_map