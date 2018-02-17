======================
python_speech_features2
======================

This library provides common speech features for ASR including MFCCs and filterbank energies.
It is a fork of `<https://github.com/jameslyons/python_speech_features>`_ and `<https://github.com/ZitengWang/python_kaldi_features>`_

Installation
============

Install from this repository::

	git clone https://github.com/thomasZen/python_speech_features2
	python setup.py install

Usage
=====

Example for creating normalized logmel and delta features. This procedure is tested for CTC-based speech recognition on Tedlium.

.. code-block:: python
	
	import librosa
	import numpy
	from python_speech_features import logfbank, calculate_delta, normalize
	
	y, sr = librosa.load("english.wav", sr=16000)
	logmel = logfbank(y, samplerate=sr)
	delta = calculate_delta(logmel)
	features = numpy.concatenate([logmel, delta], axis=1)
	features = normalize(features)


Changes
=========
Changes compared to `<https://github.com/ZitengWang/python_kaldi_features>`_:

- Added normalize function
- Rewrote delta calculation
- Changed default parameters
- Cleanup and documentation
