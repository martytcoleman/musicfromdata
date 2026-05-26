"""
EKG Music Generation Test
Converts ekg-waveform-timeseries.csv into music using the existing pipeline.
"""

import os
import sys
import datetime
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
import wave
import subprocess
from midiutil import MIDIFile

# Simple note-name → MIDI number (replaces audiolazy.str2midi broken on Py 3.13)
_NOTE_OFFSETS = {"C":0,"D":2,"E":4,"F":5,"G":7,"A":9,"B":11}
def str2midi(name: str) -> int:
    sharp = "#" in name
    flat  = "b" in name and name[1] == "b"
    if sharp:
        note, octave = name[0], int(name[2:])
        semitone = _NOTE_OFFSETS[note] + 1
    elif flat:
        note, octave = name[0], int(name[2:])
        semitone = _NOTE_OFFSETS[note] - 1
    else:
        note, octave = name[0], int(name[1:])
        semitone = _NOTE_OFFSETS[note]
    return 12 * (octave + 1) + semitone
from utils.numeric_utils import map_value
import config

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
SOUNDFONT_LOW  = os.path.join(BASE_DIR, "soundfont", "Levi_s_Violin.sf2")
SOUNDFONT_HIGH = os.path.join(BASE_DIR, "soundfont", "Bejeweled_3_Percussions__SF2_.sf2")

def midi_to_wav(input_midi: str, output_wav: str, sf2_path: str):
    """Render MIDI to WAV using FluidSynth 2.x (supports -T wav flag)."""
    cmd = ["fluidsynth", "-ni", "-F", output_wav, "-T", "wav", "-r", "44100", sf2_path, input_midi]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if not os.path.exists(output_wav):
        print("FluidSynth stderr:", result.stderr[:400])
        raise RuntimeError(f"FluidSynth failed to create {output_wav}")
    print(f"  WAV saved: {output_wav}")

def join_wavs(file1: str, file2: str, result: str,
              vol1: float = 0.7, vol2: float = 1.0):
    """Mix two WAV files. Each track is RMS-normalised first, then blended
    at the given volumes (violin=0.7, percussion=1.0 so perc cuts through)."""
    def read_wav(path):
        with wave.open(path, "rb") as w:
            params = w.getparams()
            raw = w.readframes(w.getnframes())
        return params, np.frombuffer(raw, dtype=np.int16).astype(np.float32)

    def rms_normalise(d, target=8000.0):
        rms = np.sqrt(np.mean(d ** 2))
        return d * (target / rms) if rms > 0 else d

    p1, d1 = read_wav(file1)
    _, d2   = read_wav(file2)

    d1 = rms_normalise(d1) * vol1
    d2 = rms_normalise(d2) * vol2

    length = max(len(d1), len(d2))
    d1 = np.pad(d1, (0, length - len(d1)))
    d2 = np.pad(d2, (0, length - len(d2)))

    mixed = np.clip(d1 + d2, -32768, 32767).astype(np.int16)
    with wave.open(result, "wb") as w:
        w.setparams(p1)
        w.writeframes(mixed.tobytes())
    print(f"  Mixed WAV: {result}")

# ── 1. Load the EKG waveform CSV ─────────────────────────────────────────────
csv_path = os.path.join(os.path.dirname(__file__), "public", "ekg-waveform-timeseries.csv")
df_ekg = pd.read_csv(csv_path)

# Use Lead II — strongest, most representative EKG signal
time_ms  = df_ekg["time_ms"].values.astype(float)
lead_ii  = df_ekg["II"].values.astype(float)

# ── 2. Interpolate to 10 ms resolution over one beat cycle ──────────────────
beat_ms = 800                                        # our template covers -400 → +400 ms
t_fine  = np.arange(time_ms[0], time_ms[-1] + 10, 10)  # every 10 ms
interp  = interp1d(time_ms, lead_ii, kind="cubic", fill_value="extrapolate")
beat_template = interp(t_fine)                       # ~80 samples per beat

# ── 3. Tile for the full EKG duration (1.5 min ≈ 117 beats at 78 bpm) ───────
#   Heart rate: 78 bpm → beat cycle 769 ms.  Our template is 800 ms ≈ close enough.
n_beats = 117                              # covers the full ~1.5 minute recording
waveform = np.tile(beat_template, n_beats) # ~9360 samples at 10 ms each = 93.6 s

# ── 4. Scale mV values to 0–1000 intensity range ─────────────────────────────
wmin, wmax = waveform.min(), waveform.max()
intensity = ((waveform - wmin) / (wmax - wmin) * 1000).round(2)

# ── 5. Build the rawdata string ──────────────────────────────────────────────
rawdata = "\n".join(str(v) for v in intensity)

# Each sample is 10 ms apart; 93 samples × 10 ms = 930 ms = 1 beat at 78 bpm
samples_per_beat = 93
expected_sec = len(intensity) / samples_per_beat * 60 / 60
print(f"Data points   : {len(intensity)}")
print(f"Intensity range: {intensity.min():.1f} – {intensity.max():.1f}")
print(f"Expected duration: ~{expected_sec:.0f} seconds ({expected_sec/60:.1f} min)")

# ── 6. Run through the music generation pipeline ─────────────────────────────
def music_from_ekg(rawdata: str, output_dir: str = "."):
    filename = os.path.join(output_dir, "ekg_music_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))

    lines = rawdata.strip().split("\n")
    df = pd.DataFrame({"intensity": lines})
    df["time"] = df.index
    df["intensity"] = pd.to_numeric(df["intensity"], errors="coerce")
    df.dropna(subset=["intensity"], inplace=True)
    time      = df["time"].values
    intensity = df["intensity"].values

    # Smooth
    wl = min(51, len(intensity) if len(intensity) % 2 == 1 else len(intensity) - 1)
    intensity = savgol_filter(intensity, wl, 4)

    # Each index step = 10 ms of real EKG time.
    # 93 steps × 10 ms = 930 ms ≈ 1 beat at 78 bpm → samples_per_beat = 93
    samples_per_beat = 93
    t_data           = time / samples_per_beat    # forward time, in beats
    duration_beats   = max(t_data)

    y_data = map_value(intensity, min(intensity), max(intensity), 0, 1)

    note_names = ["C1","C2","G2",
                  "C3","E3","G3","A3","B3",
                  "D4","E4","G4","A4","B4",
                  "D5","E5","G5","A5","B5",
                  "D6","E6","F#6","G6","A6"]
    note_midis = [str2midi(n) for n in note_names]
    n_notes    = len(note_midis)

    midi_data = [int(note_midis[max(0, min(n_notes-1, int(round(map_value(float(y), 0, 1, n_notes - 1, 0)))))]) for y in y_data]

    vel_min, vel_max = 20, 127
    vel_data = [int(round(map_value(float(y), 0, 1, vel_min, vel_max))) for y in y_data]

    bpm          = 60
    duration_sec = duration_beats * 60 / bpm
    print(f"Duration: {duration_sec:.1f} seconds")

    threshold_velocity = sorted(vel_data)[int(0.9 * len(vel_data))]
    print(f"Threshold velocity: {threshold_velocity}")

    low_midi  = MIDIFile(1)
    high_midi = MIDIFile(1)
    low_midi.addTempo(0, 0, bpm)
    high_midi.addTempo(0, 0, bpm)

    for i in range(len(t_data)):
        if vel_data[i] > threshold_velocity:
            high_midi.addNote(0, 0, midi_data[i], float(t_data[i]), 2, vel_data[i])
        else:
            low_midi.addNote(0, 0, midi_data[i], float(t_data[i]), 2, vel_data[i])

    low_mid_path  = filename + "_low.mid"
    high_mid_path = filename + "_high.mid"
    low_wav_path  = filename + "_low.wav"
    high_wav_path = filename + "_high.wav"
    final_wav     = filename + "_final.wav"

    with open(low_mid_path, "wb")  as f: low_midi.writeFile(f)
    with open(high_mid_path, "wb") as f: high_midi.writeFile(f)

    print("Converting MIDI → WAV …")
    midi_to_wav(low_mid_path,  low_wav_path,  SOUNDFONT_LOW)
    midi_to_wav(high_mid_path, high_wav_path, SOUNDFONT_HIGH)

    print("Mixing tracks …")
    join_wavs(low_wav_path, high_wav_path, final_wav)

    print(f"\n✓ Done!  Output: {final_wav}")
    return final_wav


output_dir = os.path.join(os.path.dirname(__file__), "public")
os.makedirs(output_dir, exist_ok=True)
result = music_from_ekg(rawdata, output_dir=output_dir)
