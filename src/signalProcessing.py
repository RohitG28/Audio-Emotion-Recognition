import numpy
# import matplotlib.pyplot as plt
import scipy.io.wavfile   #This library is used for reading the .wav file

[fs,signal] = scipy.io.wavfile.read("test.wav") #input wav file ,change here
# fs=sampling frequency,signal is the numpy 2D array where the data of the wav file is written
# length=len(signal) # the length of the wav file.This gives the number of samples ,not the length in time

# window_hop_length = 0.01 #10ms change here
# overlap = int(fs*window_hop_length)
# print("overlap = ",overlap)

# window_size = 0.025 #25 ms,change here
# framesize = int(window_size*fs)
# print("framesize = ",framesize)

# number_of_frames = (length/overlap);
# nfft_length = framesize #length of DFT ,change here
# print("number of frames are = ",number_of_frames)

# frames = numpy.ndarray((number_of_frames,framesize)) # This declares a 2D matrix,with rows equal to the number of frames,and columns equal to the framesize or the length of each DTF

# for k in range(0,number_of_frames):
# 	for i in range(0,framesize):
# 		if((k*overlap+i)<length):
# 			frames[k][i]=signal[k*overlap+i]
# 		else:
# 			frames[k][i]=0
# fft_matrix=numpy.ndarray((number_of_frames,framesize)) #declares another 2d matrix to store  the DFT of each windowed frame
# abs_fft_matrix=numpy.ndarray((number_of_frames,framesize)) #declares another 2D Matrix to store the power spectrum

# for k in range(0,number_of_frames):
# 	fft_matrix[k]=numpy.fft.fft(frames[k]) #computes the DFT
# 	abs_fft_matrix[k]=abs(fft_matrix[k])*abs(fft_matrix[k])/(max(abs(fft_matrix[k]))) # computes the power spectrum

# t=range(len(abs_fft_matrix))  #This code segment simply plots the power spectrum obtained above
# plt.plot(t,abs_fft_matrix)
# plt.ylabel("frequency")
# plt.xlabel("time")
# plt.show()

print(fs,signal)