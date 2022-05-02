# Real Time Face Recognition
 
## Project Introduction

### This project uses three face detection models (Python) to track people's face in a real-time manner:

S3FD Face Detector: default detector of face-alignment to detect face points/landmarks, which is slow.

BlazeFace: improved from S3FD Face Detector with a faster detection speed.

Naive Face Detector: not required in the handout but I included this since this is quite efficient. It uses the cascade classifier from OpenCV which generates boxes instead of points around faces with a much faster speed than the previous two models.

### Each of the three models can detect face in two ways:

Direct Camera Detection: users turn on their cameras and the system captures the faces that appear in the camera in real time.

Detection in Pre-recorded Videos: users provide pre-recorded videos and the system captures the faces in the video and render the predicitons in real time with the predicitons rendered on the screen.


## How to Run the Project

In the root directory, use 'python3 face_detection.py' (or 'python face_detection.py' if your python version is 3 by default) to run the project with the default settings.

And the optional parameters are as follows:

--face_detector_type: choose which of the three models (S3FD, BlazeFace, Naive) to use, available options are 'sfd', 'blazeface' and 'naive', default value is 'blazeface'

--pre_recorded_video_for_detection: choose whether we want to detect in the pre-recorded video or using the camera directly. Do not provide this parameter if you want to use the camera directly, and provide the location of the video if you want to detect in a video (try '--pre_recorded_video_for_detection pre_recorded_video_for_test.mp4' to see how this works as an exmaple!). Default value is None.

--casc_path: casc file for the naive model, I have set this up and you do not need to provide this usually.

--max_num_frames: maximum number of frames processed before we stop the program, can be adjusted based on how long you want to use the detector.

It is recommended to use the default settings first to get yourself familiar with the app!

(You can also press 'q' on the keyboard to quit the program manually, but this can be slow for S3FD sometimes, sorry about the inconvenience)


## Performance Analysis:

Model #1 - S3FD:

Model Load Time: 0.64 seconds

fps when detecting directly using the camera: 0.13
fps when detecting a pre-recorded video: 0.21

Model #2 - BlazeFace:

Model Load Time: 0.49 seconds

fps when detecting directly using the camera: 1.26
fps when detecting a pre-recorded video: 1.28

Model #3 - Naive Model:

Model Load Time: 0.022 seconds

fps when detecting directly using the camera: 19.38
fps when detecting a pre-recorded video: 15.83

PS: I planned to use GPU to do another comparison to see how well GPU can helpn improving the process speed. The sad fact is that all GPU-available platforms I have access to are kind of "virtual" and fail to support these visualization functions (like camera usage) in this project. Would be willing to do this GPU comparison if I could be granted a temporary access to a valid GPU-equipped device. I included code that switchs the device to cuda when the GPU is available, can look at the code for details, thank you.

