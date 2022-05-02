import cv2
import sys
import time
import numpy as np
import face_alignment
import matplotlib.pyplot as plt
import argparse
import torch

# face detector assigning points and landmarks on the faces. More accurate but slower than the naive approach
class landmarkFaceDetector():
	def __init__(self, preRecordedVideo=None, faceDetectorType='blazeface', maxNumFrames=20):
		# check if we want to do the face recognition on a pre-recorded video, or using our camera directly.
		self.preRecordedVideo = preRecordedVideo

		# initialize the face detector (S3FD or BlazeFace) and the current frame in process
		self.fig = plt.figure()
		self.faceDetectorType = faceDetectorType

		# used to log the average fps
		self.frames_per_second = []

		# we use a GPU to speed up the inference if it is available
		self.device = 'cpu'
		if torch.cuda.is_available():
			print("cuda is available!")
			self.device = 'cuda'

		# log the time it takes to load the model fa
		t_start = time.time()
		self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device=self.device, face_detector=self.faceDetectorType)
		print(f'{self.faceDetectorType}: time to load fa: {time.time() - t_start} seconds')

		# max number of frames to detect before we stop
		self.maxNumFrames = maxNumFrames

	def performFaceDetection(self):

		# handle the case when the face detection is performed using the camera directly
		if not self.preRecordedVideo:
			# use the camera to capture the frames
			self.videoCapture = cv2.VideoCapture(0)
		else:
			# use the give pre-recorded video to capture the frames
			self.videoCapture = cv2.VideoCapture(self.preRecordedVideo)

		for i in range(self.maxNumFrames):
			# capture frame-by-frame
			ret, frame = self.videoCapture.read()

			# stop if we fail to capture a frame
			if not ret:
				break

			# get the prediction for the point on the face and log the time it used to generate the prediction for this frame
			t_start = time.time()
			preds = self.fa.get_landmarks_from_image(frame)
			self.frames_per_second.append(1.0 / (time.time() - t_start))

			# plot the points on the face captured in the frame
			plt.imshow(frame)
			if preds:
				for detection in preds:
					plt.scatter(detection[:,0], detection[:,1], 2)
			plt.axis('off')

			# convert matplotlib plots to OpenCV images
			self.fig.canvas.draw()
			graph_image = np.array(self.fig.canvas.get_renderer()._renderer)
			im = cv2.cvtColor(graph_image,cv2.COLOR_RGB2BGR)

			# show the frame with the points on the face
			cv2.imshow('Video', im)
			plt.clf()
			
			# stop when the video is over
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break

		# stop the capturing
		print("Average fps is: ", round(sum(self.frames_per_second) / len(self.frames_per_second), 2))
		self.videoCapture.release()
		cv2.destroyAllWindows()

# naive face detector using boxes only to circle the faces. Less accurate but quite fast
class naiveFaceDetector():
	def __init__(self, preRecordedVideo=None, cascPath="haarcascade_frontalface_default.xml", maxNumFrames=100):
		# check if we want to do the face recognition on a pre-recorded video, or using our camera directly.
		self.preRecordedVideo = preRecordedVideo

		# initialize the cascade classifier using the provided cascade file path
		self.cascPath = cascPath
		# log the time it takes to load the model faceCascade
		t_start = time.time()
		self.faceCascade = cv2.CascadeClassifier(self.cascPath)
		print(f'Naive detector: time to load: {time.time() - t_start} seconds')

		# used to log the average fps
		self.frames_per_second = []

		# max number of frames to detect before we stop
		self.maxNumFrames = maxNumFrames

	def performFaceDetection(self):
		# handle the case when the face detection is performed using the camera directly
		if not self.preRecordedVideo:
			# use the camera to capture the frames
			self.videoCapture = cv2.VideoCapture(0)
		else:
			# use the give pre-recorded video to capture the frames
			self.videoCapture = cv2.VideoCapture(self.preRecordedVideo)

		for i in range(self.maxNumFrames):

			# capture frame-by-frame
			ret, frame = self.videoCapture.read()

			# stop if we fail to capture a frame
			if not ret:
				break

			# convert the frame to the form suitable for OpenCV
			frameConverted = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

			# Generate the face detections and log the time it takes
			t_start = time.time()
			faces = self.faceCascade.detectMultiScale(
			    frameConverted,
			    scaleFactor=1.1,
			    minNeighbors=5,
			    minSize=(30, 30)
			)
			self.frames_per_second.append(1.0 / (time.time() - t_start))

			# draw the face recognition rectangles on the frame
			for (x, y, w, h) in faces:
				cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

			# show the frame with the detections.
			cv2.imshow('Video', frame)

			# stop when the video is over or we stop it from keyboard
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break

		# stop the capturing
		print("Average fps is: ", round(sum(self.frames_per_second) / len(self.frames_per_second), 2))
		self.videoCapture.release()
		cv2.destroyAllWindows()


if __name__ == "__main__":
	# parse in the arguments
	parser = argparse.ArgumentParser()
	parser.add_argument("--face_detector_type", default="blazeface", help="the type of face detector we choose to use, default to Blaze")
	parser.add_argument("--pre_recorded_video_for_detection", default=None, help="the type of face detector we choose to use, default to Blaze")
	parser.add_argument("--casc_path", default="haarcascade_frontalface_default.xml", help="cascade file path")
	parser.add_argument("--max_num_frames", type=int, default=20, help="max number of frames to detect before we quit")
	args = parser.parse_args()

	# decide if we want to use naive face detector or face detectors using points/landmarks
	if args.face_detector_type == "naive":
		faceDetector = naiveFaceDetector(preRecordedVideo=args.pre_recorded_video_for_detection, cascPath=args.casc_path)
		faceDetector.performFaceDetection()
	else:
		faceDetector = landmarkFaceDetector(preRecordedVideo=args.pre_recorded_video_for_detection, 
											faceDetectorType=args.face_detector_type,
											maxNumFrames=args.max_num_frames)
		faceDetector.performFaceDetection()