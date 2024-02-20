#!/usr/bin/python3
import sys

from PyQt6.QtCore import Qt, QSize, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QKeyEvent
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout
from PyQt6.QtWidgets import QWidget

import cv2
import numpy as np

import mediapipe as mp
from mediapipe import solutions as mp_solutions

import sounddevice as sd


WINDOW_SIZE = (800, 600)

# Setup Mediapipe's hand detector
BaseOptions = mp.tasks.BaseOptions

# For landmark indices
HandLandmark = mp_solutions.hands.HandLandmark

# For detection
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode


def hand_callback(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    if len(result.handedness) == 2:
        detected_hands = result.handedness[0][0], result.handedness[1][0]
        if detected_hands[0].category_name == detected_hands[1].category_name:
            return
        hand_indices = detected_hands[0].index, detected_hands[1].index
        freq = 2 * \
            ((0.5 -
             result.hand_landmarks[hand_indices[0]][HandLandmark.WRIST].x))
        wave.frequency = 440-440*(freq)
        wave.amplitude = 0.2 * \
            np.clip(
                (1 - result.hand_landmarks[hand_indices[1]][HandLandmark.WRIST].y), 0, 1)


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
sin_table = np.sin(np.pi * np.arange(0, 2, 1 / samplerate))


def sd_callback(outdata, frames, audio_time, status):
    if status:
        print(status, file=sys.stderr)
    for i in range(frames):
        wave.phase += wave.frequency / samplerate * len(sin_table)
        outdata[i] = wave.amplitude * \
            sin_table[int(wave.phase) % len(sin_table)]
    wave.start_idx += frames


class CamThread(QThread):
    img_signal = pyqtSignal(QImage)
    ThreadActive = None

    def run(self):
        self.ThreadActive = True
        cap = cv2.VideoCapture()
        cap.open(0)
        old_hw = None
        if "--old_hw" in sys.argv:
            old_hw = True
            cap.set(cv2.CAP_PROP_FPS, 12.5)

        fps = cap.get(cv2.CAP_PROP_FPS)

        with HandLandmarker.create_from_options(HandLandmarkerOptions(
                base_options=BaseOptions(
                    model_asset_path='hand_landmarker.task'),
                running_mode=VisionRunningMode.LIVE_STREAM,
                num_hands=2,
                # min_tracking_confidence=0.01,
                min_hand_detection_confidence=0.85,
                result_callback=hand_callback)) as landmarker, \
                sd.OutputStream(channels=2, callback=sd_callback):
            frame_count = 0
            while self.ThreadActive:

                success, frame = cap.read()
                # Optimization if we don't edit capture
                frame.flags.writeable = False
                if success:
                    frame_count += 1
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.flip(frame, 1)
                    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                                      data=frame)
                    landmarker.detect_async(mp_img, int(1000*frame_count/fps))
                    qt_fmt = QImage(
                        frame, frame.shape[1], frame.shape[0], QImage.Format.Format_RGB888)
                    qt_fmt = qt_fmt.scaled(*WINDOW_SIZE)
                    self.img_signal.emit(qt_fmt)

                else:
                    print("Capture failed!")
                    cap.release()
                    self.stop()
                    break

    def stop(self):
        self.ThreadActive = False
        self.terminate()


wave = WaveManager()


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("TheremAI")
        self.camera_thread = CamThread()
        self.camera_thread.start()

        camera_feed = QLabel()
        self.camera_thread.img_signal.connect(
            lambda img: camera_feed.setPixmap(QPixmap.fromImage(img)))

        layout = QVBoxLayout()
        layout.addWidget(camera_feed)

        container = QWidget()
        container.setLayout(layout)

        self.setCentralWidget(container)
        self.setMinimumSize(QSize(*WINDOW_SIZE))

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            print("Exiting!")
            self.camera_thread.stop()
            QApplication.quit()


def main():
    print("Starting app, quit with escape key.")
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
