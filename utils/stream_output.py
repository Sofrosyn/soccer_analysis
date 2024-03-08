import subprocess
import os
def convert_mp4_to_hls(mp4_file_path, stream_name):

    output_dir = "./rtmp_out"
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Folder '{output_dir}' created successfully.")
        except OSError as e:
            print(f"Error creating folder '{output_dir}': {str(e)}")
    
    hls_playlist_path = f"{output_dir}/{stream_name}.m3u8"
    
    command = [
        'ffmpeg',
        '-i', mp4_file_path,  # Input file
        '-codec:v', 'libx264',  # Video codec: H.264
        '-codec:a', 'aac',  # Audio codec: AAC
        '-start_number', '0',  # Start the numbering of HLS segments from 0
        '-hls_time', '10',  # Set the segment length (in seconds)
        '-hls_list_size', '0',  # Keep all segments in the playlist (no limit)
        '-f', 'hls',  # Format: HLS
        hls_playlist_path  # Output playlist (m3u8 file)
    ]
    
    try:
        subprocess.run(command, check=True)
        print(f"Conversion completed: '{mp4_file_path}' to HLS format in '{output_dir}'")
    except subprocess.CalledProcessError as e:
        print(f"Error during conversion: {e}")

