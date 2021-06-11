from fm_radio import Radio
import multiprocessing
import scipy.io.wavfile
import torch
from glob import glob
from pynput import keyboard


# Plays FM radio audio and outputs captions.
# Captions are generated from Silero STT models at https://github.com/snakers4/silero-models

exitFlag = multiprocessing.Event()  # set flag to exit by pressing ESC

device = torch.device('cpu')
model, decoder, utils = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                       model='silero_stt',
                                       language='en',
                                       device=device)
(read_batch, split_into_batches,
 read_audio, prepare_model_input) = utils  # see function signature for details


# Process to take in audio samples and print captions
class CaptionProcess(multiprocessing.Process):
    def __init__(self, input_queue):
        multiprocessing.Process.__init__(self)
        self.input_queue = input_queue
        self.exit_listener = keyboard.Listener(on_press=self.on_press)
    
    def run(self):
        while not exitFlag.is_set():
            audio = self.input_queue.get(block=True)
            scipy.io.wavfile.write('audio.wav', 48000, audio)
            test_files = glob('audio.wav')
            batches = split_into_batches(test_files, batch_size=10)
            input = prepare_model_input(read_batch(batches[0]),
                            device=device)
            output = model(input)
            for example in output:
                print(decoder(example.cpu()))
            print()
    
    def on_press(self, key):
        if key == keyboard.Key.esc:
            exitFlag.set()
            return True


if __name__ == '__main__':
    r = Radio(buffer_time=5.0, f_sps=1.0*256*256*16)
    c = CaptionProcess(r.output_queue)
    c.start()
    r.run()
    c.terminate()
    c.join()
