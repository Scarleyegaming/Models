import sounddevice as sd
from scipy.io.wavfile import write

def record_and_save(filename, stop_recording_callback, samplerate=44100):
    print("Recording...")

    # Record audio until the stop button is pressed
    audio_data = []
    while not stop_recording_callback():
        chunk = sd.rec(samplerate, channels=2, dtype='int16')
        audio_data.extend(chunk)
        sd.wait()

    print("Recording complete.")

    # Save audio data as a WAV file
    write(filename, samplerate, audio_data)

    print(f"Audio saved as {filename}")