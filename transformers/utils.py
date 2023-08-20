
from rhvoice_wrapper import TTS
import subprocess

tts = TTS(threads=1)
sets = {'absolute_rate': 0.3, 'absolute_pitch': 0.5, 'absolute_volume': 0.5, 'voice': ['SLT']}
tts.set_params(**sets)

def generator_audio(text, voice='SLT', format_='wav', buff=4096, sets=None):
    data = tts.get(text, voice, sets)
    subprocess.check_output(['aplay', '-q'], input=data)
