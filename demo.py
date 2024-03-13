from PIL import ImageFont, ImageDraw, Image
import numpy as np
import cv2

# Load your image
image_path = 'test.jpg'
image = cv2.imread(image_path)

# Convert the OpenCV image (BGR format) to RGB format
image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# Create a drawing context for the image
draw = ImageDraw.Draw(image_pil)

# Specify the font: the font file should support Chinese characters
font_path = 'SimSun.TTF'
font_size = 20
font = ImageFont.truetype(font_path, font_size)

# Define the position where you want to start drawing the text
position = (500, 500)  # Change as per your requirement

# Define the text and color
text = "你好, 世界!"  # 'Hello, World!' in Chinese
color = 'rgb(2, 25, 255)'  # White color

# Draw the text onto the image
draw.text(position, text, fill=color, font=font)

# Convert the PIL image back to an OpenCV image
result_image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

# Display the result image with OpenCV
cv2.imshow("Image with Chinese Text", result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
