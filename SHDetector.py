import cv2
import numpy as np
import cvui
from processImage import process as processImage

class SDHException(Exception):
    def __init__(self, exception):
        self.exception = exception

def detect_safety(frame):
    img = processImage(frame)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

#################### Main driver program #######################

# Main UI Frame and Source Image variable
window_name = 'Safety Helmet Detector'

ui_width = 800
ui_height = 600

toolbar_top_height = 100

# Image Containers
frame = np.zeros((ui_height, ui_width, 3), np.uint8)
camera_frame = np.array([])
detected_frame = np.array([])

image_padding = 10

# Button messages
load_action_message = 'Waiting for camera...'
load_action_message_color = 0x000000

cvui.init(window_name)

# Initialize camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SDHException('Could not open camera')

# main program loop (window property check as while condition allows the close button to end the program)
while cv2.getWindowProperty(window_name, 0) >= 0:
    
    ret, camera_frame = cap.read()
    if not ret:
        load_action_message = 'Could not read frame from camera'
        load_action_message_color = 0xFF0000
        continue

    src_height, src_width = camera_frame.shape[:2]

    new_height = src_height + toolbar_top_height + (image_padding * 2)
    new_width = src_width + (image_padding * 2)

    frame = np.zeros((new_height, new_width, 3), np.uint8)
    frame[:] = (173, 216, 230)  # Light blue color

    # Process the current frame for helmet detection
    detected_frame = detect_safety(camera_frame)
    
    # Display the detected frame with bounding boxes
    cvui.image(frame, image_padding, toolbar_top_height + image_padding, detected_frame)

    # Adding text beside the button to display path or error message
    cvui.text(frame, 126, 18, load_action_message, 0.6 , load_action_message_color)  # Font scale increased to 0.6

    # Show the output on screen
    cvui.imshow(window_name, frame)

    # Exit using ESC button
    if cv2.waitKey(20) == 27:
        break

# Release camera and destroy windows
cap.release()
cv2.destroyAllWindows()
