# Soccer Video Analytics with YOLOv5

This project leverages YOLOv5 for soccer video analytics, including possession computation and passes counter. 

## Installation
<details>
<summary>Installation</summary>
This project uses pip for managing dependencies. Follow these steps to set up your environment:

Clone repo and install requirements.txt in a Python>=3.8.0 environment, including PyTorch>=1.8.

```bash
git clone git@github.com:ChengGong0602/soccer_analysis.git
cd soccer_analysis
pip install -r requirements.txt
```
</details>

<details>
<summary>Training</summary>

#### Training YOLOv5 on customized dataset

**Note:** Multi-GPU training times are faster. Use the largest `--batch-size` possible, or `--batch-size -1` for YOLOv5 AutoBatch.

#### Training Command

```bash
python train.py --data data/data.yaml --epochs 300 --weights 'yolov5s.pt' --batch-size 32
```

</details>

<details>
<summary>Inference</summary>

####  Inference with detect.py
detect.py runs inference on a variety of sources, downloading models automatically from the latest YOLOv5 release and saving results to runs/detect.
```bash
python detect.py --weights yolov5s.pt --source 0                               # webcam
                                               img.jpg                         # image
                                               vid.mp4                         # video
                                               screen                          # screenshot
                                               path/                           # directory
                                               list.txt                        # list of images
                                               list.streams                    # list of streams
                                               'path/*.jpg'                    # glob
                                               'https://youtu.be/LNwODJXcvt4'  # YouTube
                                               'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
```
</details>

## How to run
### Run on Flask
```bash
python app.py
```
By using API endpoint, it can process live streaming and convert panorama video into normal tracking video.
Example:
```bash
http://127.0.0.1:8000/video?source=rtmp://stream.wow-sports.com/wow/wow?auth_key=1706097000-0-0-13d1bb6ec74d390e6eea42941bb64bfb&weights=yolov5_class3_300.pt&team1=AA&team2=BB&id=current
```
| Argument | Description | Default value |
| ----------- | ----------- | ----------- |
| URL | basic endpoint| http://127.0.0.1:8000/video|
| source | local video or live streaming url | 1.mp4 |
| weights | path of trained model | models/yolov5s_class3_300_25k.pt |
| team1 | left team | AA |
| team2 | right team | BB |
| id | index | current|


### Run on local


To run one of the applications (possession computation and passes counter) you need to use flags in the console.

These flags are defined in the following table:

| Argument | Description | Default value |
| ----------- | ----------- | ----------- |
| application | Set it to `possession` to run the possession counter or `passes` if you like to run the passes counter | None, but mandatory |
| path-to-the-model | Path to the soccer ball model weights (`pt` format) | `/models/ball.pt` |
| path-to-the-video | Path to the input video | `/videos/soccer_possession.mp4` |

The following command shows you how to run this project.

```
python run.py --<application> --model <path-to-the-model> --video <path-to-the-video>
```

>__Warning__: You have to run this command on the root of the project folder.

Here is an example on how to run the command:
    
```bash
python run.py --possession --model models/ball.pt --video videos/soccer_possession.mp4
```

An mp4 video will be generated after the execution. The name is the same as the input video with the suffix `_out` added.
