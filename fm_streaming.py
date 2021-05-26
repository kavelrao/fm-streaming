from rtlsdr import RtlSdr
import numpy as  np
import scipy.signal as signal
import sounddevice as sd
import multiprocessing

exitFlag = multiprocessing.Event()  # set flag to exit; not currently implemented
sample_queue = multiprocessing.Queue()  # samples added to the queue for processing
audio_queue = multiprocessing.Queue()  # audio added to the queue to be played


# Process to sample radio using the sdr
class SampleProcess(multiprocessing.Process):
    def __init__(self, sdr, f_sps, f_c, f_offset, num_samples):
        multiprocessing.Process.__init__(self)
        self.sdr = sdr
        self.f_sps = f_sps
        self.f_c = f_c
        self.f_offset = f_offset
        self.num_samples = num_samples
    def run(self):
        while not exitFlag.is_set():
            print('starting sampling...')
            sample = get_samples(self.sdr, self.num_samples, self.f_sps, self.f_c, self.f_offset)
            sample_queue.put(sample)
            print('... sampling finished')


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
            print('starting processing...')
            filteredsignal = filter_samples(samples, self.f_sps, self.f_offset)
            audio = process_signal(filteredsignal, self.f_sps, self.f_audiosps)
            audio_queue.put(audio)
            print('... processing finished')

def main():
    # set up constants
    sdr = RtlSdr()
    f_sps = 1.5*256*256*16  # sdr sampling frequency
    f_audiosps =44100  # audio sampling frequency (for output)
    f_c = 98.1e6  # KUOW Seattle
    f_offset = int(250e3)  # offset by 250 Khz to avoid sdr center spike
    dt = 1.0 / f_sps  # time step size between samples
    nyquist = f_sps / 2.0  # nyquist frequency based on sample rate
    buffer_time = 3.5  # length in seconds of samples in each buffer
    N = round(f_sps*buffer_time)  # number of samples to collect per buffer

    # initialize multiprocessing processes
    sample_process = SampleProcess(sdr, f_sps, f_c, f_offset, N)
    extraction_process = ExtractionProcess(f_sps, f_offset, f_audiosps)

    sample_process.start()
    extraction_process.start()

    # audio playing in the main process
    while not exitFlag.is_set():
        audio = audio_queue.get(block=True)
        sd.play(audio)


# returns SDR samples
def get_samples(sdr, N, f_sps, f_c, f_offset):
    sdr.sample_rate = f_sps 
    sdr.center_freq = f_c + f_offset
    sdr.gain = 42.0
    samples = sdr.read_samples(N)

    return samples   


# returns filtered signal
def filter_samples(samples, f_sps, f_offset):
    f_bw = 100e3  # 100 KHz FM bandwidth
    N = len(samples)

    # shift samples back to center frequency using complex exponential with period -f_offset/f_sps
    shift = np.exp(-1.0j * 2.0 * np.pi * -1 * f_offset / f_sps * np.arange(N))
    shifted_samples = samples * shift

    # filter samples to include only the FM bandwidth
    k = 101  # number of filter taps
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
    for i in range(N):
        if abssignal[i] < meanabssignal / 3.0:
            filteredsignal[i] = 0.0
            theta[i] = 0.0

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
    main()