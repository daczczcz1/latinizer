import madmom
import soundfile as sf
import numpy as np
import librosa
import os

# noinspection SpellCheckingInspection
filename = "/sourcewavs/bach1short"
meter = 4
# noinspection SpellCheckingInspection
dirname = os.path.dirname(__file__)
PATH = dirname + filename + ".wav"
x, sr = librosa.load(PATH)

proc = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)
act = madmom.features.beats.RNNBeatProcessor()(PATH)
beat_times = proc(act)

# we need to skip the last bar as it may be incomplete
beat_times_to_include = beat_times[0:((len(beat_times) // meter) * meter)]
reshaped = beat_times_to_include.reshape(-1, meter)
bar_durations = (reshaped[1:] - reshaped[:-1])[:, 0]
# just make the last bar same length as the penultimate this is ugly but covers both the situation when we have the
# exact amount of beats to get only full bars and when there are additional beats
bar_durations = np.append(bar_durations, bar_durations[-1])
start_times = reshaped[:, 0]


def convert_to_clave(start_times, durations):
    clave_pattern = np.array([0, 3 / 16, 6 / 16, 10 / 16, 12 / 16]).reshape(1, -1)
    patterns = durations.reshape(-1, 1) * clave_pattern
    return (patterns.T + start_times).T.reshape(-1, 1)


clave_beats = convert_to_clave(start_times, bar_durations)

clicks = librosa.clicks(clave_beats, sr=sr, length=len(x))

data = x + clicks
sf.write('latin_' + filename + '.wav', data, sr, subtype='PCM_24')
