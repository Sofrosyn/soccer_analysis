import subprocess
import time
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
output_dir = os.path.normpath(os.path.join(BASE_DIR, "..", "rtmp_out"))

def update_playlist(playlist_path, ts_file, sequence_num):
    """
    Adds a new .ts file to the HLS playlist (m3u8 file).
    """
    with open(playlist_path, 'a') as playlist:
        if sequence_num == 0:  # Write header if first segment
            playlist.write('#EXTM3U\n#EXT-X-VERSION:3\n#EXT-X-TARGETDURATION:10\n#EXT-X-MEDIA-SEQUENCE:0\n')
        playlist.write(f'#EXTINF:10,\n{ts_file}\n')

        

def convert_mp4_to_hls(mp4_file, stream_ts_name):
    print("=========stream_name", stream_ts_name)
    stream_name, index = stream_ts_name.split("_")
    sequence_num = int(index) - 1
    playlist_path = f"{output_dir}/{stream_name}.m3u8"

    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Folder '{output_dir}' created successfully.")
        except OSError as e:
            print(f"Error creating folder '{output_dir}': {str(e)}")

    ts_file = f'{stream_name}{sequence_num}.ts'  # The new .ts filename
    ts_path = os.path.join(output_dir, ts_file)

    # Command to convert MP4 to TS
    command = [
        'ffmpeg', 
        '-i', mp4_file, 
        '-c', 'copy', 
        '-f', 'mpegts', 
        ts_path
    ]

    # Wait for the mp4 file to be updated
    print(f"Waiting for updates to {mp4_file}...")
    print(f"Converting {mp4_file} to {ts_file}...")
    subprocess.run(command)  # Removed stdout and stderr redirection for debugging
    
    print(f"Updated playlist with {ts_file}")
    # Update the m3u8 playlist
    update_playlist(playlist_path, ts_file, sequence_num)
    
    print(f"Updated playlist with {ts_file}")

    