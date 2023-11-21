import cv2
import numpy as np
import mediapipe as mp
import random
import sounddevice as sd
import sys
import time

from mediapipe import solutions as mp_solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green

def draw_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        # Draw the hand landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])
        mp_solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            mp_solutions.hands.HAND_CONNECTIONS,
            mp_solutions.drawing_styles.get_default_hand_landmarks_style(),
            mp_solutions.drawing_styles.get_default_hand_connections_style())

        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        # Draw handedness (left or right hand) on the image.
        cv2.putText(annotated_image, f"{handedness[0].category_name}",
                    (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    return annotated_image

contrast = 255
brightness = 90

cv2.imshow("img", np.zeros((320, 240)))
cv2.createTrackbar('frequency', "img", 44000, 44000, lambda x:  None)
# Setup camera
cap = cv2.VideoCapture()
cap.open(-1, apiPreference=cv2.CAP_V4L2)
# Necessary for my old webcam
cap.set(cv2.CAP_PROP_FPS, 12.5)
# These are inferred, but left commented just in case
# cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# Setup Mediapipe's hand detector
BaseOptions = mp.tasks.BaseOptions
# For landmark indices
HandLandmark = mp_solutions.hands.HandLandmark
# For detection
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

glob = None
hands_last_known = [None, None]

sd.default.device = "pipewire"
samplerate = sd.query_devices(None, 'output')['default_samplerate']
x = np.arange(0, 2, 100/samplerate) # sample every 100Hz
sin_table = np.sin(np.pi * x)

class WaveManager:
    def triangle(self, x, amplitude, frequency):
        return 2 * amplitude / np.pi * np.arcsin(np.sin(2 * np.pi * x * frequency))

    def sin(self, x, amplitude, frequency):
        return amplitude * np.sin(2 * np.pi * frequency * x)

    def saw(self, x, amplitude, frequency):
        period = 1 / frequency
        return amplitude * 2*(x / period - np.floor(0.5 + x / period))

    def square(self, x, amplitude, frequency):
        return amplitude*np.sign(np.pi * np.sin(x * 2 * np.pi * frequency))

    def noise(self, x, amplitude, frequency):
        return (2*np.random.random(x.shape)-1)*amplitude

    def __init__(self):
        self.start_idx = 0
        self.frequency = 440
        self.amplitude = 0.0
        self.function = self.sin
        self.phase = 0


samplerate = sd.query_devices(None, 'output')['default_samplerate']
wave = WaveManager()
waves = [wave.triangle, wave.sin, wave.saw, wave.square, wave.noise]

def sd_callback(outdata, frames, audio_time, status):
        if status:
            print(status, file=sys.stderr)
        # t = (wave.start_idx + np.arange(frames)) / samplerate
        # t_n0 = np.append([wave.last_t], t[:-1])
        # t = (t_n0 + wave.phase) % (2 * np.pi)
        # t = t.reshape(-1, 1)
        for i in range(frames):
            wave.phase +=  wave.frequency  / samplerate * len(sin_table) 
            outdata[i] = wave.amplitude * sin_table[int(wave.phase) % len(sin_table)]
        # wave.frequency = 440 + np.sin(2 * np.pi * (time.time()+i)*0.1 ) * 220
        wave.start_idx += frames
        # wave.last_t = wave.amplitude * sin_table[int(wave.phase) % len(sin_table)]

sd_out = sd.OutputStream(channels=2, callback=sd_callback)

def hand_callback(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global glob,frequency,amplitude
    # glob = draw_landmarks_on_image(np.zeros(output_image.numpy_view().shape), result)
    img = cv2.cvtColor(output_image.numpy_view(),cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img,(3,3),0)
    _,img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    glob = draw_landmarks_on_image(img, result)
    
    # if len(result.handedness) == 2:
    #     print(result.handedness[0])
    #     hand_indices = result.handedness[0][0].index,result.handedness[1][0].index
    #     freq = 2*((0.5 - result.hand_landmarks[hand_indices[0]][HandLandmark.WRIST].x))
    #     wave.frequency = 440-440*(freq)
    #     print(freq)
    #     wave.amplitude = np.clip((1 - result.hand_landmarks[hand_indices[0]][HandLandmark.WRIST].y),0,1)
    #     # print(result.hand_landmarks[hand_indices[1]][HandLandmark.WRIST].y )
    #     print(wave.amplitude)
    # if result.handedness != []:
    #     freq = 2*((0.5 - result.hand_landmarks[0][HandLandmark.WRIST].x))
    #     wave.frequency = 440-440*(freq)
    #     wave.amplitude = np.clip((1 - result.hand_landmarks[0][HandLandmark.WRIST].y),0,1)

with HandLandmarker.create_from_options(HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
            running_mode=VisionRunningMode.LIVE_STREAM,
            num_hands=2,
            min_tracking_confidence=0.1,
            min_hand_detection_confidence=0.2,
            result_callback=hand_callback)) as landmarker,\
            sd.OutputStream(channels=2, callback=sd_callback,latency=0.05):
    while cap.isOpened():
        # contrast = cv2.getTrackbarPos('contrast', 'img')
        # wave.frequency = cv2.getTrackbarPos('frequency', 'img')/100
        success, img = cap.read()
        # img.flags.writeable = False
        # print(f"cap is {success}")
        if success:
            # img = cv2.GaussianBlur(img,(7,7),0)
            # img = cv2.fastNlMeansDenoisingColored(img,None,10,10,3,10)
            # bright_contrast = (contrast / 255 * img + brightness)
            # img = np.clip(bright_contrast.astype(np.uint8), 0, 255)
            img = cv2.resize(img, dsize=(640,480))
            # img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            # img = cv2.GaussianBlur(img,(5,5),0)
            # _,img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
            # print(dir(mp_img))
            landmarker.detect_async(mp_img, int(
                cap.get(cv2.CAP_PROP_POS_MSEC)))
            # cv2.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

            if glob is not None:
                cv2.imshow("img", glob)
            
            # if not None in hands_last_known:
            #     print(hands_last_known)
        else:
            print("Capture failed!")
            break
        if cv2.waitKey(30) & 0xFF == ord("q"):
            print("Quiting...")
            break

cap.release()
cv2.destroyAllWindows()
