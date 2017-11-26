#! /usr/bin/python3


import numpy as np
from scipy import signal
import librosa



def get_segment_features_from_file(filename, sample_rate=16000):
	y,_ = librosa.core.load(filename, sample_rate)
	return get_segment_features(y, sample_rate)


def get_segment_features(y, sample_rate, frame_length=0.025, hop_length=0.010, frames_per_segment=25):

	frame_length_s = int(sample_rate * frame_length)
	hop_length_s = int(sample_rate * hop_length)
	# print("frame_length_s:", frame_length_s)
	# print("hop_length_s:", hop_length_s)

	yt,_ = librosa.effects.trim(y, top_db=5)
	# print("Len: Original", len(y), "Trimmed", len(yt))

	# Pad with zeros, to match behaviour of mfcc with yt while getting frames from yp
	yp = np.concatenate([np.zeros(frame_length_s//2), yt, np.zeros(frame_length_s//2)]) 	

	frames = librosa.util.frame(yp,frame_length_s, hop_length_s)
	frames = frames.T


	mfcc = librosa.feature.mfcc(y=yt, sr=sample_rate, n_fft=frame_length_s, hop_length=hop_length_s)	
	mfcc_d = librosa.feature.delta(mfcc)
	mfcc_dd = librosa.feature.delta(mfcc, order=2)

	frame_features_mfcc = mfcc.T
	frame_features_mfcc_d = mfcc_d.T
	frame_features_mfcc_dd = mfcc_dd.T

	threshold = 0.15

	frame_features_pitch = np.empty((0,2))
	for frame in frames:

		if np.max(frame) < threshold:
			frame_features_pitch = np.vstack((frame_features_pitch, [np.nan, np.nan]))
			continue

		feature_pitch_period = get_feature_pitch_period(frame, sample_rate)
		feature_hnr = get_feature_hnr(frame, sample_rate, feature_pitch_period[0])

		feature_pitch = np.concatenate([feature_pitch_period, feature_hnr])
		frame_features_pitch = np.vstack((frame_features_pitch, feature_pitch))
	# print("Num of frames:", len(frame_features_pitch))


	frame_features = np.hstack((frame_features_pitch,frame_features_mfcc,frame_features_mfcc_d,frame_features_mfcc_dd))

	# Remove features of frames which were added by np.zeros()
	num_frames_remove = np.ceil(frame_length_s/2/hop_length_s).astype(int)
	frame_features = frame_features[num_frames_remove:-num_frames_remove,:]

	segment_features = np.empty((0, frame_features.shape[1]*frames_per_segment ))
	for i in range(frames_per_segment//2, len(frame_features)-frames_per_segment//2):
		w = frames_per_segment//2
		feature = np.ravel(frame_features[i-w:i-w+frames_per_segment])

		# Discrad invalid segment
		if np.isnan(feature).any():
			continue

		segment_features = np.vstack((segment_features, feature))

	# print("Num of segments", len(segment_features))

	return segment_features


def get_feature_pitch_period(frame, sample_rate):
	corr = np.correlate(frame, frame, "same")
	corr = corr[len(corr)//2:]

	corr_diff = np.diff(corr)

	try:
		idx_low_first = np.where(corr_diff > 0)[0][0]
		idx_hi_second = np.argmax(corr[idx_low_first:]) + idx_low_first
	except IndexError as e:
		return [np.nan]
		# Noisy?

	pitch_period = sample_rate/idx_hi_second

	return [pitch_period]


def get_feature_hnr(frame, sample_rate, pitch_period):

	def autocorr(frame,t):
		return [np.corrcoef(frame[0:frame.size-t],frame[t:frame.size])[0,1]]

	tau = np.rint(sample_rate/pitch_period).astype(int)

	acf_0 = np.abs(autocorr(frame,0))
	acf_tau = np.abs(autocorr(frame,tau))

	hnr = 10*np.log(acf_tau/(acf_0 - acf_tau))

	return hnr
