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