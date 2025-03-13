import pyaudio
import wave
import openai
from pynput import keyboard

def record_audio(output_file, sample_rate=44100, chunk_size=1024):
    """Record audio with more robust error handling and device selection"""
    audio = pyaudio.PyAudio()
    frames = []
    recording = False
    stream = None

    # Find the default input device
    try:
        default_input_device_info = audio.get_default_input_device_info()
        device_index = default_input_device_info['index']
        max_channels = default_input_device_info['maxInputChannels']
        channels = min(1, max_channels)  # Use mono recording
        
        print(f"Using audio device: {default_input_device_info['name']}")
        
        stream = audio.open(
            format=pyaudio.paInt16,
            channels=channels,
            rate=sample_rate,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=chunk_size
        )
    except Exception as e:
        print(f"Error setting up audio device: {str(e)}")
        print("\nAvailable audio devices:")
        for i in range(audio.get_device_count()):
            try:
                device_info = audio.get_device_info_by_index(i)
                print(f"Device {i}: {device_info['name']} (Max channels: {device_info['maxInputChannels']})")
            except:
                continue
        audio.terminate()
        return False

    def on_press(key):
        nonlocal recording, frames
        if key == keyboard.Key.space:
            if not recording:
                frames = []
                recording = True
                print("\nRecording... (Press space again to stop)")
            else:
                recording = False
                return False

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    try:
        while True:
            if recording:
                try:
                    data = stream.read(chunk_size, exception_on_overflow=False)
                    frames.append(data)
                except Exception as e:
                    print(f"Error during recording: {str(e)}")
                    recording = False
                    break
            if not listener.running:
                break

        if frames:  # Only save if we actually recorded something
            wf = wave.open(output_file, 'wb')
            wf.setnchannels(channels)
            wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
            wf.setframerate(sample_rate)
            wf.writeframes(b''.join(frames))
            wf.close()
            print("Recording saved!")

    except Exception as e:
        print(f"Error during recording process: {str(e)}")
        return False

    finally:
        if stream:
            stream.stop_stream()
            stream.close()
        audio.terminate()

    return bool(frames)

def transcribe_audio(file_path, api_key):
    """Transcribes the given audio file using OpenAI Whisper API."""
    client = openai.OpenAI(api_key=api_key)

    print("Transcribing audio...")

    with open(file_path, "rb") as audio_file:
        response = client.audio.transcriptions.create(
            model="whisper-1", file=audio_file, language="en"
        )

    transcription = response.text
    print("Transcription complete.")
    print("Transcription:", transcription)
    return transcription