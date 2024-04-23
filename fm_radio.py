import multiprocessing
import asyncio
import queue
import time
import faulthandler
from rtlsdr import RtlSdr
import numpy as  np
import scipy.signal as signal
import sounddevice as sd
from pynput import keyboard


# Plays FM radio through the computer audio output.

faulthandler.enable()


if __name__ == '__main__':
    sample_queue = multiprocessing.Queue()  # samples added to the queue for processing
    audio_queue = multiprocessing.Queue()  # audio added to the queue to be played
    exitFlag = multiprocessing.Event()  # set flag to exit by pressing ESC


# Handles sampling, processing, and playing from an SDR
# Outputs audio samples to Radio.output_queue for potential use by other programs
# Optional contructor arguments:
#   # sdr sampling rate, must be a multiple of 256
#   # audio sampling rate
#   # sdr listening frequency
#   # sdr tuning offset
#   # buffer time
class Radio():    
    def __init__(self, f_sps=1.0*256*256*16, f_audiosps=48000, f_c=94.9e6, f_offset=250e3, buffer_time=1.0):
        # set up constants
        self.f_sps = f_sps  # sdr sampling frequency
        self.f_audiosps = f_audiosps  # audio sampling frequency (for output)
        self.f_c = f_c  # listening frequency
        self.f_offset = int(f_offset)  # default offset by 250 Khz to avoid sdr center spike
        self.dt = 1.0 / self.f_sps  # time step size between samples
        self.nyquist = self.f_sps / 2.0  # nyquist frequency based on sample rate
        self.buffer_time = buffer_time  # length in seconds of samples in each buffer
        self.N = round(self.f_sps*self.buffer_time)  # number of samples to collect per buffer

        # initialize multiprocessing processes
        self.sample_process = SampleProcess(self.f_sps, self.f_c, self.f_offset, self.N, sample_queue, exitFlag)
        self.extraction_process = ExtractionProcess(self.f_sps, self.f_offset, self.f_audiosps, sample_queue, audio_queue, exitFlag)
        self.exit_listener = keyboard.Listener(on_press=self.on_press)

        # initalize output audio stream
        self.stream = sd.OutputStream(samplerate=self.f_audiosps, blocksize=int(self.N / (self.f_sps / self.f_audiosps)), channels=1)

        self.output_queue = multiprocessing.Queue(25)  # for other programs to use audio samples. max size is 25 to avoid memory overuse if output is not being used.
    
    def run(self):
        print('\nInitialized. Starting streaming. Press <ESC> to exit.\n')
        self.stream.start()
        self.exit_listener.start()
        self.sample_process.start()
        self.extraction_process.start()

        # audio playing in the main process
        while not exitFlag.is_set():
            audio = audio_queue.get(block=True)
            audio = audio.astype(np.float32)

            # play audio and send to output queue if it's not full
            try:
                self.output_queue.put(audio, block=False)
            except queue.Full:
                pass

            self.stream.write(3 * audio)

    def cleanup(self):
        time.sleep(self.buffer_time)  # wait to allow processes to finish
        self.extraction_process.terminate()
        self.sample_process.terminate()
        self.stream.stop()
        self.extraction_process.close()
        self.sample_process.close()
        self.stream.close()
        del self.output_queue  # must be deleted for clean exit

    def on_press(self, key):
        if key == keyboard.Key.esc:
            exitFlag.set()
            self.cleanup()


# Process to sample radio using the sdr
class SampleProcess(multiprocessing.Process):
    def __init__(self, f_sps, f_c, f_offset, buffer_length, sample_queue, exit_flag):
        multiprocessing.Process.__init__(self)
        # initialize SDR
        self.sdr = None
        self.f_sps = f_sps
        self.f_c = f_c
        self.f_offset = f_offset
        self.buffer_length = buffer_length
        self.sample_queue = sample_queue
        self.exit_flag = exit_flag

    def run(self):
        asyncio.run(self.stream_samples(self.buffer_length, self.f_sps, self.f_c, self.f_offset))


    async def stream_samples(self, N, f_sps, f_c, f_offset):
        if self.sdr is None:
            assert RtlSdr is not None
            self.sdr = RtlSdr()
        self.sdr.sample_rate = f_sps
        self.sdr.center_freq = f_c + f_offset
        self.sdr.gain = -1.0  # increase for receiving weaker signals. valid gains (dB): -1.0, 1.5, 4.0, 6.5, 9.0, 11.5, 14.0, 16.5, 19.0, 21.5, 24.0, 29.0, 34.0, 42.0
        samples = np.array([], dtype=np.complex64)
        async for sample_set in self.sdr.stream():  # streams 131072 samples at a time
            samples = np.concatenate((samples, sample_set))
            if len(samples) >= N:
                self.sample_queue.put(samples)
                samples = np.array([], dtype=np.complex64)
            
            if self.exit_flag.is_set():
                return True


# Process to extract audio from the samples
class ExtractionProcess(multiprocessing.Process):
    def __init__(self, f_sps, f_offset, f_audiosps, sample_queue, audio_queue, exit_flag):
        multiprocessing.Process.__init__(self)
        self.f_sps = f_sps
        self.f_offset = f_offset
        self.f_audiosps = f_audiosps
        self.sample_queue = sample_queue
        self.audio_queue = audio_queue
        self.exit_flag = exit_flag

    def run(self):
        while not self.exit_flag.is_set():
            samples = self.sample_queue.get(block=True)
            filteredsignal = self.filter_samples(samples, self.f_sps, self.f_offset)
            audio = self.process_signal(filteredsignal, self.f_sps, self.f_audiosps)
            self.audio_queue.put(audio)

    # returns filtered signal
    def filter_samples(self, samples, f_sps, f_offset):
        f_bw = 100e3  # 100 KHz FM bandwidth
        N = len(samples)

        # shift samples back to center frequency using complex exponential with period f_offset/f_sps
        shift = np.exp(1.0j * 2.0 * np.pi * f_offset / f_sps * np.arange(N))
        shifted_samples = samples * shift

        # filter samples to include only the FM bandwidth
        k = 201  # number of filter taps - increase this to improve filter quality
        firtaps = signal.firwin(k, cutoff=f_bw, fs=f_sps, window='hamming')
        filteredsignal = np.convolve(firtaps, shifted_samples, mode='same')

        return filteredsignal
    
    # returns audio processed from filteredsignal
    def process_signal(self, filteredsignal, f_sps, f_audiosps):
        theta = np.arctan2(filteredsignal.imag, filteredsignal.real)

        # squelch low power signal to remove noise
        abssignal = np.abs(filteredsignal)
        meanabssignal = np.mean(abssignal)
        theta = np.where(abssignal < meanabssignal / 3, 0, theta)

        # calculate derivative of phase (instantaneous frequency)
        # and unwrap phase-wrapping effects
        derivtheta = np.diff(np.unwrap(theta) / (2 * np.pi))

        # downsample by taking average of surrounding values
        dsf = round(f_sps/f_audiosps)  # downsampling factor
        # pad derivtheta with NaN so size is divisible by dsf, then split into rows of size dsf and take the mean of each row
        derivtheta_padded = np.pad(derivtheta.astype(float), (0, dsf - derivtheta.size % dsf), mode='constant', constant_values=np.NaN)
        dsdtheta = np.nanmean(derivtheta_padded.reshape(-1, dsf), axis=1)

        return dsdtheta


if __name__ == '__main__':
    r = Radio(buffer_time=2.5)
    r.run()
