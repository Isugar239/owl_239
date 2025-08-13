import os
import threading
from yt_dlp import YoutubeDL
from playsound import playsound
import sys

def download_audio(track_name):
    ydl_opts = {
        'format': 'bestaudio/best',
        'noplaylist': True,
        'quiet': True,
        'outtmpl': f'{track_name}.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'default_search': 'ytsearch1',  
    }

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(track_name, download=True)
        filename = ydl.prepare_filename(info)
        filename = os.path.splitext(filename)[0] + '.mp3'
        return filename

def play_audio(file_path):
    print("playing")
    playsound(file_path)

if __name__ == "__main__":
    track = input("input name of track: ")
    audio_file = download_audio(track)

    sound_thread = threading.Thread(target=play_audio, args=(audio_file,), daemon=True)
    sound_thread.start()

    try:
        while sound_thread.is_alive():
            sound_thread.join(timeout=1)
    except KeyboardInterrupt:
        sys.exit(0)
