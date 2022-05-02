# Real Time Face Recognition
 
## Project Introduction

### This project uses three face detection models (Python) to track people's face in a real-time manner:

S3FD Face Detector: default detector of face-alignment to detect face points/landmarks, which is slow.

BlazeFace: improved from S3FD Face Detector with a faster detection speed.

Naive Face Detector: not required in the handout but I included this since this is quite efficient. It uses the cascade classifier from OpenCV which generates boxes instead of points around faces with a much faster speed than the previous two models.

### Each of the three models can detect face in two ways:

Direct Camera Detection: users turn on their cameras and the system captures the faces that appear in the camera in real time.

Detection in Pre-recorded Videos: users provide pre-recorded videos and the system captures the faces in the video and render the predicitons in real time.

### `Number of briefcases to choose from:`

To choose how many briefcases a contestant will be choosing from (this will be 3 by default according to the requirement).

And this is a parameter to toggle that I think will be useful although not required in the handout.

### `User chance to switch:`

After the host eliminates the briefcase(s), what percentage of the contestants will choose to switch? This can be 30%, 20%, 50%, etc. So I provide the API to choose this percentage to allow various settings of the simulation.

### `Whether the host knows the answer:`

Whether the host knows the answer will affect how the host eliminates the briefcase(s) since the host could reveal the winning briefcase by accident if the host does NOT know which one of the briefcases contains the prize. So I provide the API to set whether the host knows the answer and allows various settings of the simulation.

## start simulation!

After setting all the parameters, we click the "start simulation!" button to run the simulation.

The number of prizes given during the simulation will be shown to give a quick impression of whether or not we are giving too many prizes.

Then users could choose to download a CSV file containing detailed information about the simulation:

### `Detailed information of the simulation in the generated CSV file:`

Number of briefcases to choose from: how many briefcases a contestant will be choosing from (this will be 3 by default according to the requirement)

The host knows the answer: whether the host knows the answer of the winning briefcase

The user switches the choice (for each round): whether the contestant chooses to switch the choice in this round

The case chose (for each round): the # of the briefcase chosen by the contestant

Case contains the prize (for each round): the # of the briefcase containing the prize. 

Win/Fail (for each round): whether this round is a win/fail

Win by accident (for each round): whether the contestant win by accident (host reveals the prize).