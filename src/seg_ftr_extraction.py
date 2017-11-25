#! /usr/bin/python3


import numpy as np
from scipy import signal
import librosa



def get_segment_features_from_file(filename, sample_rate=16000):
	y,_ = librosa.core.load(filename, sample_rate)
	yt,_ = y_trim, idx = librosa.effects.trim(y, top_db=5)
	# print("Len: Original", len(y), "Trimmed", len(yt))
	return get_segment_features(yt, sample_rate)


def get_segment_features(y, sample_rate, frame_width=0.025, frame_interval=0.010, frames_per_segment=25):
	
	frame_width_s = int(sample_rate * frame_width)
	frame_interval_s = int(sample_rate * frame_interval)

	# print("frame_width_s:", frame_width_s)
	# print("frame_interval_s:", frame_interval_s)

	signal_length = len(y)

	frame_features = []
	for i in np.arange(0, signal_length - frame_width_s , frame_interval_s):
		frame = y[i:i+frame_width_s]
		
		feature_mfcc = get_feature_mfcc(frame, sample_rate)
		feature_pitch_period = get_feature_pitch_period(frame, sample_rate)
		feature_hnr = get_feature_hnr(frame, sample_rate)

		feature = np.concatenate([feature_mfcc, feature_pitch_period, feature_hnr])
		frame_features.append(feature)

	# print("Num of frames:", len(frame_features))

	segment_features = []
	for i in range(frames_per_segment//2, len(frame_features)-frames_per_segment//2):
		# print("*", end="", flush=True)
		w = frames_per_segment//2
		feature = np.concatenate(frame_features[i-w:i+w])
		segment_features.append(feature)

	# print("Num of segments", len(segment_features))

	return segment_features


def get_feature_mfcc(frame, sample_rate):
	mfcc = librosa.feature.mfcc(frame, sample_rate)
	return np.ravel(mfcc)


def get_feature_pitch_period(frame, sample_rate):
	return []


def get_feature_hnr(frame, sample_rate):
	return []


def trim_signal(y):
	pass