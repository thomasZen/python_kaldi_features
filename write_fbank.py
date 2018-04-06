#!/usr/bin/env python

import argparse
from multiprocessing import Pool
import os
import librosa
import numpy
import h5py
from python_speech_features import logfbank, calculate_delta, normalize


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Creates a hdf5 file with a group for each audio file"
    )

    parser.add_argument(
        "directory",
        help="Directory of the data (recursively looks for files there)"
    )

    parser.add_argument(
        "output",
        help="filename of the h5 file"
    )

    parser.add_argument(
        "--fileExtension", default="wav",
        help="file extension of the audio files (e.g. wav, sph, mp3)"
    )

    parser.add_argument(
        "--processes", type=int, default=1,
        help="Processes to create the fbank features with"
    )

    parser.add_argument(
        "--sampleRate", type=int, default=16000,
        help="Sample rate"
    )

    parser.add_argument(
        '--addDelta', action='store_true', default=False,
        help="Wether to add deltas"
    )

    return parser.parse_args()


def process(path):
    y, sr = librosa.load(path, sr=args.sampleRate)

    logmel = logfbank(y, samplerate=sr)
    feature_list = [logmel]

    if args.addDelta:
        delta = calculate_delta(logmel)
        feature_list.append(delta)

    features = numpy.concatenate(feature_list, axis=1)
    features = normalize(features)
    
    name = os.path.basename(path).split(".")[0]
    print("Processed {}".format(name))
    return name, features


if __name__ == "__main__":
    args = parse_args()
    print(args)

    # file_names = glob.glob(args.directory + "/**/*.{}".format(args.fileExtension), recursive=True)
    file_names = [os.path.join(dp, f) for dp, dn, filenames in os.walk(args.directory) for f in filenames if
             f.endswith(args.fileExtension)]

    h5_file = h5py.File(args.output, "w")
    pool = Pool(args.processes)

    for name, features in pool.imap_unordered(process, file_names):
        h5_file.create_dataset(name, data=features)

    h5_file.close()
    print("Done")

