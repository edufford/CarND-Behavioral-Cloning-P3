""" drive.py """

import argparse
import base64
from datetime import datetime
import os
import shutil

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO

from keras.models import load_model
import h5py
from keras import __version__ as keras_version

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None


class SimplePIController:
    def __init__(self, Kp, Ki):
        self.Kp = Kp
        self.Ki = Ki
        self.set_point = 0.
        self.error = 0.
        self.integral = 0.

    def set_desired(self, desired):
        self.set_point = desired

    def update(self, measurement):
        # proportional error
        self.error = self.set_point - measurement

        # integral error
        self.integral += self.error

        ### Add min/max guards on integral term to prevent excessive windup during slipping
        self.integral = min(self.integral, 200)
        self.integral = max(self.integral, -200)
        
        print(self.integral)

        return self.Kp * self.error + self.Ki * self.integral


controller = SimplePIController(0.1, 0.002)
set_speed = 9
controller.set_desired(set_speed)


@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        # The current steering angle of the car
        steering_angle = data["steering_angle"]
        # The current throttle of the car
        throttle = data["throttle"]
        # The current speed of the car
        speed = data["speed"]
        # The current image from the center camera of the car
        imgString = data["image"]
        image = Image.open(BytesIO(base64.b64decode(imgString)))
        image_array = np.asarray(image)
        steering_angle = float(model.predict(image_array[None, :, :, :], batch_size=1))

        throttle = controller.update(float(speed))

        print(steering_angle, throttle)
        send_control(steering_angle, throttle)

        # save frame
        if args.image_folder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            image.save('{}.jpg'.format(image_filename))
    else:
        # NOTE: DON'T EDIT THIS.
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder. This is where the images from the run will be saved.'
    )
    args = parser.parse_args()

    # check that model Keras version is same as local Keras version
    f = h5py.File(args.model, mode='r')
    model_version = f.attrs.get('keras_version')
    keras_version = str(keras_version).encode('utf8')

    if model_version != keras_version:
        print('You are using Keras version ', keras_version,
              ', but the model was built using ', model_version)

    model = load_model(args.model)

    if args.image_folder != '':
        print("Creating image folder at {}".format(args.image_folder))
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)


""" load_data.py """

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

"""
Tuning parameters
"""
STEER_ADJUST = 0.3 # steering adjustment for left/right camera images
STEER_RECOVER = 0.9 # steering adjustment for recovery from lane edges
PLOT_HISTOGRAM = 1 # 1 = plot histogram of steering angles in training data set

"""
Helper functions
"""
def get_drive_data(base_path):
	"""
	Loads CSV driving log file to gather arrays of file paths to the driving camera
	images (center, left, right) and the corresponding steering angles.
	"""
	df = pd.read_csv(base_path + 'driving_log.csv', header=None)
	
	filenames = df[0].values
	for i in range(len(filenames)):
		filenames[i] = base_path + 'IMG/' + filenames[i].split('/')[-1]
	img_centercam = filenames

	filenames = df[1].values
	for i in range(len(filenames)):
		filenames[i] = base_path + 'IMG/' + filenames[i].split('/')[-1]
	img_leftcam = filenames

	filenames = df[2].values
	for i in range(len(filenames)):
		filenames[i] = base_path + 'IMG/' + filenames[i].split('/')[-1]
	img_rightcam = filenames

	steer_angle = df[3].values

	return (img_centercam, img_leftcam, img_rightcam, steer_angle)


def get_straight_data_idx(steer_angle, pct_discard=0.7, steer_straight_thresh=0.04):
	"""
	Finds indices of straight driving where steering angle is < steer_straight_thresh and
	returns a percentage of them to be used to discard a portion of straight driving data.
	"""
	idx_straight = np.where(abs(steer_angle) < steer_straight_thresh)
	idx_straight = idx_straight[0]
	idx_straight = np.random.choice(idx_straight, int(pct_discard*len(idx_straight)))
	return idx_straight


"""
Gather base training data and split to make a validation set.
Apply a tuned steering adjusment to left/right camera images.
"""
print('Processing center driving data')
img_centercam, img_leftcam, img_rightcam, steer_angle = get_drive_data('./data/center_drive/')
idx_straight = get_straight_data_idx(steer_angle)
img_centercam_trim = np.delete(img_centercam, idx_straight)
steer_angle_trim = np.delete(steer_angle, idx_straight)
X_train_centercam, X_valid, y_train_centercam, y_valid = train_test_split(img_centercam_trim,
																		  steer_angle_trim,
																		  test_size=0.2,
																		  random_state=321)
X_train_leftcam = img_leftcam
y_train_leftcam = steer_angle + STEER_ADJUST
X_train_rightcam = img_rightcam
y_train_rightcam = steer_angle - STEER_ADJUST

"""
Use backward driving data (driving counter-clockwise) as additional data.
"""
print('Processing backward driving data')
img_centercam, img_leftcam, img_rightcam, steer_angle = get_drive_data('./data/backward_drive/')
idx_straight = get_straight_data_idx(steer_angle)
X_train_backward = np.delete(img_centercam, idx_straight)
y_train_backward = np.delete(steer_angle, idx_straight)

"""
Apply a tuned recovery steering adjustment to the center camera image of driving on
the left/right lane edges.
"""
print('Processing left recovery data')
img_centercam, img_leftcam, img_rightcam, steer_angle = get_drive_data('./data/recover_leftside/')
X_train_leftrecover = img_centercam
y_train_leftrecover = steer_angle + STEER_RECOVER

print('Processing right recovery data')
img_centercam, img_leftcam, img_rightcam, steer_angle = get_drive_data('./data/recover_rightside/')
X_train_rightrecover = img_centercam
y_train_rightrecover = steer_angle - STEER_RECOVER

"""
Use extra data of normal driving to help generalize the model.
"""
print('Processing extra driving data')
img_centercam, img_leftcam, img_rightcam, steer_angle = get_drive_data('./data/extra_normaldrive/')
idx_straight = get_straight_data_idx(steer_angle)
X_train_extranormal = np.delete(img_centercam, idx_straight)
y_train_extranormal = np.delete(steer_angle, idx_straight)

"""
Combine all training data sets.
"""
print('Combining data')
X_train = np.concatenate((X_train_centercam,
						  X_train_backward,
						  X_train_leftcam,
						  X_train_rightcam,
						  X_train_leftrecover,
						  X_train_rightrecover,
						  X_train_extranormal),
						  axis=0)
y_train = np.concatenate((y_train_centercam,
						  y_train_backward,
						  y_train_leftcam,
						  y_train_rightcam,
						  y_train_leftrecover,
						  y_train_rightrecover,
						  y_train_extranormal),
						  axis=0)

print('Training set size: X={}, y={}'.format(X_train.shape, y_train.shape))

"""
Save training and validation data sets for model processing.
"""
np.save("X_train.npy", X_train)
np.save("X_valid.npy", X_valid)
np.save("y_train.npy", y_train)
np.save("y_valid.npy", y_valid)
print('Data set index files saved')


"""
Plot histogram of steering angles in combined training data set.
"""
if PLOT_HISTOGRAM == 1:
	hist, bin_edges = np.histogram(y_train, bins=100)
	plt.bar(bin_edges[:-1], hist, width = 0.02)
	plt.xlim(-1.0, 1.0)
	plt.title('Steering Angle Histogram')
	plt.ylabel('Frequency of steering angle')
	plt.xlabel('Steering angle')
	plt.show()


""" model.py """

import cv2
import numpy as np
from sklearn.utils import shuffle
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Dropout, Lambda, Cropping2D, Convolution2D

"""
Load training and validation data sets generated by 'load_data.py'
"""
X_train = np.load('X_train.npy')
X_valid = np.load('X_valid.npy')
y_train = np.load('y_train.npy')
y_valid = np.load('y_valid.npy')

"""
Helper function
"""
def data_generator(X_samples, y_samples, batch_size=32):
	"""
	Generator for loading batches of image data into memory
	"""
	num_samples = len(X_samples)
	while 1:
		X_samples, y_samples = shuffle(X_samples, y_samples)

		for offset in range(0, num_samples, batch_size):
			X_batch = X_samples[offset:offset+batch_size]
			y_batch = y_samples[offset:offset+batch_size]
			images, steering_angles = [], []

			for X_sample, y_sample in zip(X_batch, y_batch):
				centercam_image = cv2.imread(X_sample) # BGR image
				centercam_image = cv2.cvtColor(centercam_image, cv2.COLOR_BGR2RGB)
				steering_angle = float(y_sample)

				# Add base image/steering angle to batch
				images.append(centercam_image)
				steering_angles.append(steering_angle)

				# Augment data by horizontal flipping
				images.append(np.fliplr(centercam_image))
				steering_angles.append(-1.0*steering_angle)

			X_out = np.array(images)
			y_out = np.array(steering_angles)
			yield shuffle(X_out, y_out)


"""
Set number of epochs and data generators for training and validation
"""
EPOCHS = 10

train_generator = data_generator(X_train, y_train, batch_size=32)
validation_generator = data_generator(X_valid, y_valid, batch_size=32)

"""
Keras model architecture similar to Nvidia paper 'End to End Learning for Self-Driving Cars'
"""
model = Sequential()
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5 ))
model.add(Convolution2D(nb_filter=24, nb_row=5, nb_col=5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(nb_filter=36, nb_row=5, nb_col=5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(nb_filter=48, nb_row=5, nb_col=5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(nb_filter=64, nb_row=3, nb_col=3, activation="relu"))
model.add(Convolution2D(nb_filter=64, nb_row=3, nb_col=3, activation="relu"))
model.add(Flatten())
model.add(Dense(100, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(50, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation="relu"))
model.add(Dense(1))

"""
Compile model using mean-squared error and ADAM optimizer with default learning rate parameters.
Run model fit training using generators and save model and loss history results.
"""
model.compile(loss='mse', optimizer='adam')
trained_history = model.fit_generator(train_generator, 
									samples_per_epoch=(2*len(X_train)),
									validation_data=validation_generator,
									nb_val_samples=(2*len(X_valid)),
									nb_epoch=EPOCHS)

model.summary()
model.save('model.h5')
np.save("trained_history.npy", trained_history.history)
print('Model and history saved.')


""" plot_training_history.py """

import numpy as np
import matplotlib.pyplot as plt

"""
Load training history generated from 'model.py'
"""
history = np.load("trained_history.npy").item()

"""
Plot loss vs epochs
"""
plt.plot(history['loss'], '-')
plt.plot(history['val_loss'], '--')
plt.title('Training history')
plt.ylabel('Mean-Squared Error loss')
plt.xlabel('Epoch (#)')
plt.legend(['Training set', 'Validation set'], loc='upper right')
plt.show()


""" video.py """

from moviepy.editor import ImageSequenceClip
import argparse


def main():
    parser = argparse.ArgumentParser(description='Create driving video.')
    parser.add_argument(
        'image_folder',
        type=str,
        default='',
        help='Path to image folder. The video will be created from these images.'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=60,
        help='FPS (Frames per second) setting for the video.')
    args = parser.parse_args()

    video_file = args.image_folder + '.mp4'
    print("Creating video {}, FPS={}".format(video_file, args.fps))
    clip = ImageSequenceClip(args.image_folder, fps=args.fps)
    clip.write_videofile(video_file)


if __name__ == '__main__':
    main()
