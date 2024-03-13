import subprocess
import os

def segment_stream(rtmp_url, output_dir, segment_time=12):
    """
    Segments an RTMP stream and saves it as HLS (HTTP Live Streaming) format.
    
    Parameters:
    - rtmp_url: URL of the RTMP stream.
    - output_dir: Directory to save the HLS files.
    - segment_time: Duration (in seconds) of each segment.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Command to segment the stream and create HLS playlist and segments
    command = [
        'ffmpeg',
        '-i', rtmp_url,  # Input from the RTMP stream
        '-c', 'copy',  # Use the same codecs for both audio and video
        '-f', 'hls',  # Output format HLS
        '-hls_time', str(segment_time),  # Segment duration
        '-hls_playlist_type', 'event',  # Playlist type
        os.path.join(output_dir, 'stream.m3u8')  # Output HLS playlist file
    ]
    
    # Execute the command
    subprocess.run(command)

if __name__ == "__main__":
    rtmp_url = "1_clip.mp4"
    output_dir = "./videos"
    segment_stream(rtmp_url, output_dir)
