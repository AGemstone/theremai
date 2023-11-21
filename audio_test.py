import random
import sounddevice as sd
import numpy as np
import sys
import time
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import queue

sd.default.device = "pipewire"
samplerate = sd.query_devices(None, 'output')['default_samplerate']
trivial_table_size = int(samplerate)
x = np.arange(0, 2, 1 / trivial_table_size) 
sin_table = np.sin(2 * np.pi * x)
saw_table = 2 * (x / 2 - np.floor(0.5 + x / 2))
tri_table = 2 * abs(2 * (x / 2 - np.floor(0.5 + x / 2))) - 1

#TODO : Check if using rms 
def amp_rms(table):
        return np.sqrt(np.sum(table**2)/len(x))
def amp_peak(table):
        return abs(np.max(table))

#sys.exit(-1)

class WaveManager:
    def triangle(self,buffer_size):
        for i in range(buffer_size):
            self.phase += wave.frequency  / samplerate * len(sin_table) 
            self.phase_buffer[i] = self.amplitude * (tri_table[int(self.phase) % trivial_table_size])
        return self.phase_buffer

    def sin(self,buffer_size):
        for i in range(buffer_size):
            self.phase += wave.frequency  / samplerate * trivial_table_size 
            self.phase_buffer[i] = self.amplitude * (sin_table[int(self.phase) % trivial_table_size])
        return self.phase_buffer

    def saw(self,buffer_size):
        for i in range(buffer_size):
            self.phase += wave.frequency  / samplerate * trivial_table_size 
            self.phase_buffer[i] = self.amplitude * (saw_table[int(self.phase) % trivial_table_size])
        return self.phase_buffer

    def square(self,buffer_size):
        for i in range(buffer_size):
            self.phase += wave.frequency * trivial_table_size / samplerate 
            self.phase_buffer[i] = self.amplitude * np.sign(sin_table[int(self.phase) % trivial_table_size])
        return self.phase_buffer

    def noise(self,buffer_size):
        return (2 * np.random.random((buffer_size,1)) - 1) * amplitude
        

    def __init__(self):
        self.start_idx = 0
        self.frequency = 440
        self.function = self.sin
        self.amplitude = 0.2
        self.phase = 0
        self.phase_buffer = None

def callback(outdata, frames, audio_time, status):
        if status:
            print(status, file=sys.stderr)
        if wave.phase_buffer is None:
            wave.phase_buffer = np.zeros((frames,2))
        #print(trivial_table_size)
        outdata[:] = wave.function(frames)
        # wave.frequency = 440 + np.sin(2 * np.pi * (time.time()+i)*0.57735101135711020 ) * 220
        wave.start_idx += frames
        block_queue.put(outdata[::downsample, mapping])



def plot_callback(frame):
    global plotdata
    while True:
        try:
            data = block_queue.get_nowait()
        except queue.Empty:
            break
        shift = len(data)
        plotdata = np.roll(plotdata, -shift, axis=0)
        plotdata[-shift:, :] = data
    for column, line in enumerate(lines):
        line.set_ydata(plotdata[:, column])
    return lines

mapping = [0,1]  # Channel numbers start with 1
block_queue = queue.Queue()
downsample = 10
wave = WaveManager()

# plot data for debugging signal
length = int(200 * 44100 / (1000 * downsample))
plotdata = np.zeros((length, 2))
fig, ax = plt.subplots()
ax.axis((0, len(plotdata), -1, 1))
lines = ax.plot(plotdata)
fig.tight_layout(pad=0)
ani = FuncAnimation(fig, plot_callback, 60, blit=True)
        
try:
    with sd.OutputStream(channels=2, callback=callback) as sd_stream:
        #print(sd.default.blocksize)
        plt.show()
        # input() 

except KeyboardInterrupt:
    sys.exit(0)
except Exception as e:
    print(e)
    sys.exit(-1)


# freq = 440;          // frequency we want to generate (Hz)
# delta_phase = freq / Fs * LUT_SIZE;
# phase = 0.0f;

# // generate buffer of output
# for (int i = 0; i < BUFF_SIZE; ++i)
# {
#     phase_i = (int)phase;        // get integer part of our phase
#     outdata[i] = sin_table[phase_i];          // get sample value from LUT
#     phase += delta_phi;              // increment phase
#     phase = phase % len(sin_table)
# }
