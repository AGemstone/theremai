# theremai
A virtual theremin that uses hand detection to control sound

## Install
Set up a Python environment, install **Mediapipe**, **Python Sounddevice** **OpenCV**, and **Numpy**. Lastly run **main.py**

## How to play
Since the detection method is based on palm detection, for now the point of reference is the base of the hand
* Right hand controls the frequency, the note that plays depends on the distance between the rightmost edge of the screen to the center.
* Left hand controls amplitude, the loudness is determined by how far the hand is located above the lower edge of the screen
