import madmom
import soundfile as sf
import numpy as np
import librosa
import os
import sys

# TODO use other meters, ideally - detect
METER = 4


def convert_to_clave(bar_start_times, bar_durations):
    clave_pattern = np.array([0, 3 / 16, 6 / 16, 10 / 16, 12 / 16]).reshape(1, -1)
    patterns = bar_durations.reshape(-1, 1) * clave_pattern
    return (patterns.T + bar_start_times).T.reshape(-1, 1)


def latinize(source_file_path):
    x, sr = librosa.load(source_file_path)
    proc = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)
    act = madmom.features.beats.RNNBeatProcessor()(source_file_path)
    beat_times = proc(act)
    # we need to skip the last bar as it may be incomplete
    beat_times_to_include = beat_times[0:((len(beat_times) // METER) * METER)]
    reshaped = beat_times_to_include.reshape(-1, METER)
    durations = (reshaped[1:] - reshaped[:-1])[:, 0]
    # just make the last bar same length as the penultimate.
    # This is ugly but covers both the situation when we have the
    # exact amount of beats to get only full bars and when there are additional beats
    durations = np.append(durations, durations[-1])
    start_times = reshaped[:, 0]
    clave_beats = convert_to_clave(start_times, durations)
    clicks = librosa.clicks(clave_beats, sr=sr, length=len(x))
    data = x + clicks
    sf.write(os.path.dirname(__file__) + '/latin_' + os.path.split(source_file_path)[-1], data, sr, subtype='PCM_24')


if __name__ == "__main__":
    file_path = sys.argv[1]
    latinize(file_path)
