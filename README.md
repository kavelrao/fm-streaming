# FM Streaming

This is the code from a project for CSE 490 W at the University of Washington, taught by Professor Joshua Smith in Spring of 2021. The class topic is wireless communication, with a software focus. Through the class, we had access to Software Defined Radio (SDR) devices, which can be used to capture signals in frequency ranges from 55 MHz to 2.3 GHz. In class, we explored FM demodulation using the SDR by capturing a short 2.5 second sample, demodulating it, and playing the audio clip through the computer's speakers. This project is a significant extension of that concept.

The goal of this project is to create a program that streams an FM signal from a software defined radio (SDR), demodulates the signal, plays back the audio, and outputs captions all in real time. The primary goal of the project is the streaming FM radio player, and the captioning is an additional objective.

If you have an SDR that is compatible with the python `rtlsdr` library, you can run `fm_radio.py` for a streaming radio and `fm_captions.py` for the same radio + captions printed in stdout.

More details on the project can be found in my extensive writeup [here](https://kavelrao.dev/assets/files/fm_radio_report.pdf).
