#!/usr/bin/env python

import librosa
import numpy
from python_speech_features import logfbank, calculate_delta, normalize

y, sr = librosa.load("english.wav", sr=16000)

# calculate log mel features
logmel = logfbank(y, sr)
# add delta
delta = calculate_delta(logmel)
features = numpy.concatenate([logmel, delta], axis=1)
# normalize
features = normalize(features)
print(features.shape)
