import cv2
import sys

import time
import numpy as np
import face_alignment
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt

fig = plt.figure()

naiveFaceDetection = False
blazeFaceDetection = True

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cpu', face_detector='blazeface')

while True:

	ret, frame = video_capture.read()

	if not ret:
		break

	frameConverted = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	if naiveFaceDetection:

		faces = faceCascade.detectMultiScale(
		    frameConverted,
		    scaleFactor=1.1,
		    minNeighbors=5,
		    minSize=(30, 30)
		)

		for (x, y, w, h) in faces:
			cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

		cv2.imshow('Video', frame)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	elif blazeFaceDetection:

		preds = fa.get_landmarks_from_image(frame)

		plt.imshow(frame)

		if preds:
			for detection in preds:
				plt.scatter(detection[:,0], detection[:,1], 2)
		plt.axis('off')

		fig.canvas.draw()

		graph_image = np.array(fig.canvas.get_renderer()._renderer)
		im = cv2.cvtColor(graph_image,cv2.COLOR_RGB2BGR)

		cv2.imshow('Video', im)
		plt.clf()

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break



video_capture.release()
cv2.destroyAllWindows()