import multiprocessing
import asyncio
from rtlsdr import RtlSdr
import numpy as  np
import scipy.signal as signal
import sounddevice as sd
from pynput import keyboard

exitFlag = multiprocessing.Event()  # set flag to exit by pressing ESC
sample_queue = multiprocessing.Queue()  # samples added to the queue for processing
audio_queue = multiprocessing.Queue()  # audio added to the queue to be played


# Process to sample radio using the sdr
class SampleProcess(multiprocessing.Process):
    def __init__(self, sdr, f_sps, f_c, f_offset, buffer_length):
        multiprocessing.Process.__init__(self)
        self.sdr = sdr
        self.f_sps = f_sps
        self.f_c = f_c
        self.f_offset = f_offset
        self.buffer_length = buffer_length

    def run(self):
        asyncio.run(stream_samples(self.sdr, self.buffer_length, self.f_sps, self.f_c, self.f_offset))


# Process to extract audio from the samples
class ExtractionProcess(multiprocessing.Process):
    def __init__(self, f_sps, f_offset, f_audiosps):
        multiprocessing.Process.__init__(self)
        self.f_sps = f_sps
        self.f_offset = f_offset
        self.f_audiosps = f_audiosps
    def run(self):
        while not exitFlag.is_set():
            samples = sample_queue.get(block=True)
            filteredsignal = filter_samples(samples, self.f_sps, self.f_offset)
            audio = process_signal(filteredsignal, self.f_sps, self.f_audiosps)
            audio_queue.put(audio)


def main():
    print('\nInitializing.\n')

    # set up constants
    f_sps = 2.0*256*256*16  # sdr sampling frequency
    f_audiosps =48000  # audio sampling frequency (for output)
    f_c = 94.9e6  # listening frequency
    f_offset = int(250e3)  # offset by 250 Khz to avoid sdr center spike
    dt = 1.0 / f_sps  # time step size between samples
    nyquist = f_sps / 2.0  # nyquist frequency based on sample rate
    buffer_time = 1.0  # length in seconds of samples in each buffer
    N = round(f_sps*buffer_time)  # number of samples to collect per buffer

    # initialize SDR
    try: 
        sdr.close()
    except NameError:
        pass
    sdr = RtlSdr()

    # initialize multiprocessing processes
    sample_process = SampleProcess(sdr, f_sps, f_c, f_offset, N)
    extraction_process = ExtractionProcess(f_sps, f_offset, f_audiosps)
    exit_listener = keyboard.Listener(on_press=on_press)

    # initalize output audio stream
    stream = sd.OutputStream(samplerate=f_audiosps , blocksize=int(N / (f_sps / f_audiosps)), channels=1)
    stream.start()

    print('\nInitialized. Starting streaming. Press <ESC> to exit.\n')
    exit_listener.start()
    sample_process.start()
    extraction_process.start()

    # audio playing in the main process
    while not exitFlag.is_set():
        audio = audio_queue.get(block=True)

        audio = audio.astype(np.float32)
        stream.write(3 * audio)


async def stream_samples(sdr, N, f_sps, f_c, f_offset):
    sdr.sample_rate = f_sps
    sdr.center_freq = f_c + f_offset
    sdr.gain = -1.0  # increase for receiving weaker signals. valid gains (dB): -1.0, 1.5, 4.0, 6.5, 9.0, 11.5, 14.0, 16.5, 19.0, 21.5, 24.0, 29.0, 34.0, 42.0
    samples_streamed = 0
    sample_length = 131072  # length of each unit of samples streamed from sdr using sdr.stream()
    samples = np.array([], dtype=np.complex64)
    async for sample_set in sdr.stream():
        samples_streamed += 1
        samples = np.concatenate((samples, sample_set))
        if samples_streamed * sample_length >= N:  # if N samples have been taken, send to the player
            sample_queue.put(samples)
            samples_streamed = 0  # reset samples for next batch
            samples = np.array([], dtype=np.complex64)
        
        if exitFlag.is_set():
            break


# returns filtered signal
def filter_samples(samples, f_sps, f_offset):
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
def process_signal(filteredsignal, f_sps, f_audiosps):
    N = len(filteredsignal)

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


def on_press(key):
    if key == keyboard.Key.esc:
        exitFlag.set()
        return True


if __name__ == '__main__':
    main()
