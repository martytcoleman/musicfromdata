"""
EKG → Music  —  12-lead, 4 distinct soundfonts
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Leads I/II/III   → Violin   (limb leads, melodic center)
aVR/aVL/aVF      → Airsynth pad (augmented, ambient sides)
V1/V2/V3         → Saxophone (right precordial, punchy)
V4/V5/V6         → Percussion soundfont melodic (left precordial)
Lead II R-peaks  → Bejeweled percussion (heartbeat drum)

Each group rendered through its own soundfont → guaranteed distinct timbre.
Leads within a group separated by octave shift.
Pan = anatomical electrode position.
MIDI tempo = actual heart rate. QT → reverb. QRS → note length.

Usage:
    python3 ekg_test.py harp
    python3 ekg_test.py violin
    python3 ekg_test.py raga
    python3 ekg_test.py guitar
"""

import os, sys, json, wave, subprocess
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, find_peaks
from scipy.interpolate import interp1d
from midiutil import MIDIFile
from utils.numeric_utils import map_value

# ── Note helpers ───────────────────────────────────────────────────────────────
_NOTE_OFFSETS = {"C":0,"D":2,"E":4,"F":5,"G":7,"A":9,"B":11}
def str2midi(name: str) -> int:
    sharp = "#" in name
    if sharp:
        note, octave = name[0], int(name[2:])
        return 12 * (octave + 1) + _NOTE_OFFSETS[note] + 1
    note, octave = name[0], int(name[1:])
    return 12 * (octave + 1) + _NOTE_OFFSETS[note]

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
PUBLIC     = os.path.join(BASE_DIR, "public")
SF_DIR     = os.path.join(BASE_DIR, "soundfont")
SF_VIOLIN  = os.path.join(SF_DIR, "Levi_s_Violin.sf2")
SF_PAD     = os.path.join(SF_DIR, "Airsynth.sf2")
SF_SAX     = os.path.join(SF_DIR, "Sax-A-Boom_Soundfont.sf2")
SF_PERC    = os.path.join(SF_DIR, "Bejeweled_3_Percussions__SF2_.sf2")

# ── Recording config ───────────────────────────────────────────────────────────
recording = sys.argv[1] if len(sys.argv) > 1 else "harp"

REGISTRY = {
    "harp": {
        "csv":     "harp-ekg-waveform-timeseries.csv",
        "summary": "harp-ekg-clinical-summary.json",
        "label":   "harp",
        "n_beats": 114,
        "notes": ["A3","B3","C#4","E4","F#4","A4","B4","C#5","E5","F#5","A5","B5"],
        # Sinus tachycardia, steady 95–102 bpm. Slight natural deceleration.
        "tempo_curve": [(0, 102), (114, 95)],
    },
    "violin": {
        "csv":     "ekg-waveform-timeseries.csv",
        "summary": "ekg-clinical-summary.json",
        "label":   "binaural_beats",
        "n_beats": 117,
        "notes": ["D3","F3","G3","A3","C4","D4","F4","G4","A4","C5","D5","F5"],
        # Decelerates 90 → 71 bpm over ~80 seconds of binaural beat listening.
        "tempo_curve": [(0, 90), (117, 71)],
    },
    "raga": {
        "csv":     "raga-ekg-waveform-timeseries.csv",
        "summary": "raga-ekg-clinical-summary.json",
        "label":   "raga",
        "n_beats": 137,
        "notes": ["C3","D3","E3","G3","B3","C4","D4","E4","G4","B4","C5","D5"],
        # Progressive parasympathetic deceleration 102 → 80 bpm over ~90 seconds.
        "tempo_curve": [(0, 102), (137, 80)],
    },
    "guitar": {
        "csv":     "guitar-ekg-waveform-timeseries.csv",
        "summary": "guitar-ekg-clinical-summary.json",
        "label":   "guitar",
        "n_beats": 126,
        "notes": ["E3","G3","A3","B3","D4","E4","G4","A4","B4","D5","E5","G5"],
        # Sympathetic arousal: 107 → 118 → 94 → 90 → 105 bpm, variable throughout.
        "tempo_curve": [(0, 107), (32, 118), (63, 94), (95, 90), (126, 105)],
    },
}

cfg = REGISTRY[recording]

# ── Parse clinical metadata ────────────────────────────────────────────────────
with open(os.path.join(PUBLIC, cfg["summary"])) as f:
    meta = json.load(f)

if recording in ("harp", "raga", "guitar"):
    HR_BPM   = meta["heart_rate_bpm"]
    QRS_MS   = meta["interpretation"]["conduction"]["QRS_ms"]
    QT_MS    = meta["interpretation"]["conduction"]["QT_ms"]
    AXIS_DEG = 50
else:
    HR_BPM   = int((meta["rate_bpm"]["start"] + meta["rate_bpm"]["end"]) / 2)
    QRS_MS   = meta["intervals_ms"]["QRS"]
    QT_MS    = meta["intervals_ms"]["QT"]
    AXIS_DEG = meta["axis_degrees"]["QRS"]

BEAT_MS  = 60000.0 / HR_BPM
SPB      = max(1, round(BEAT_MS / 10))   # samples per beat at 10ms resolution
N_BEATS  = cfg["n_beats"]
BASE_DUR = float(np.clip(map_value(float(QRS_MS), 60, 140, 0.8, 2.0), 0.8, 2.0))
REVERB   = int(np.clip(map_value(float(QT_MS), 280, 420, 30, 90), 30, 90))
AXIS_SHIFT = int(np.clip(map_value(float(AXIS_DEG), -90, 90, -15, 15), -15, 15))

print(f"""
Recording : {cfg['label']}   HR={HR_BPM}bpm  QRS={QRS_MS}ms  QT={QT_MS}ms  Axis={AXIS_DEG}°
SPB={SPB}  beats={N_BEATS}  base_dur={BASE_DUR:.2f}  reverb_cc={REVERB}  pan_shift={AXIS_SHIFT}""")

# ── Tempo curve ────────────────────────────────────────────────────────────────
def make_tempo_events(curve):
    """Interpolate a list of (beat, bpm) waypoints into one event every ~5 beats."""
    events = []
    for i in range(len(curve) - 1):
        b0, bpm0 = curve[i]
        b1, bpm1 = curve[i + 1]
        steps = max(2, (b1 - b0) // 5)
        for s in range(steps):
            frac = s / steps
            events.append((b0 + (b1 - b0) * frac,
                           int(round(bpm0 + (bpm1 - bpm0) * frac))))
    events.append((float(curve[-1][0]), int(curve[-1][1])))
    return events

TEMPO_EVENTS = make_tempo_events(cfg["tempo_curve"])

# ── Note palette ───────────────────────────────────────────────────────────────
note_palette = [str2midi(n) for n in cfg["notes"]]
N_NOTES = len(note_palette)

def note_for(y_val, octave_shift=0):
    idx = max(0, min(N_NOTES-1,
              int(round(map_value(float(y_val), 0, 1, N_NOTES-1, 0)))))
    return int(note_palette[idx]) + (octave_shift * 12)

# Fixed low-range palette for perc_mel: stays in bass/tom zone (MIDI 36-57)
# regardless of which recording's scale is active. Amplitude still drives
# which note within this range fires — data is preserved, cymbal zone avoided.
PERC_MEL_NOTES = list(range(36, 58))   # 22 steps: bass drum → low tom range
N_PERC_MEL = len(PERC_MEL_NOTES)

def perc_note_for(y_val):
    idx = max(0, min(N_PERC_MEL-1,
              int(round(map_value(float(y_val), 0, 1, N_PERC_MEL-1, 0)))))
    return PERC_MEL_NOTES[idx]

def vel_for(y_val, scale=1.0):
    return int(np.clip(map_value(float(y_val), 0, 1, 25, 110) * scale, 1, 127))

# ── Load & tile all 12 leads ───────────────────────────────────────────────────
df      = pd.read_csv(os.path.join(PUBLIC, cfg["csv"]))
time_ms = df["time_ms"].values.astype(float)

lead_y = {}
# Resample each template to exactly SPB samples so tiled R-peaks fall on
# integer MIDI beat positions — no cumulative phase drift across the recording.
t_exact = np.linspace(time_ms[0], time_ms[-1], SPB)
for col in ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]:
    raw      = df[col].values.astype(float)
    template = interp1d(time_ms, raw, kind="cubic", fill_value="extrapolate")(t_exact)
    waveform = np.tile(template, N_BEATS)
    smoothed = savgol_filter(waveform, 51, 4)
    mn, mx   = smoothed.min(), smoothed.max()
    lead_y[col] = (smoothed - mn) / (mx - mn) if mx > mn else np.zeros_like(smoothed)

N_SAMPLES = len(next(iter(lead_y.values())))
t_beat    = np.arange(N_SAMPLES) / SPB
print(f"Samples: {N_SAMPLES}  Est. duration: {N_SAMPLES/SPB*BEAT_MS/1000:.0f}s")

# ── Peak/valley event detection ────────────────────────────────────────────────
def detect_events(y, min_dist, prominence):
    """Return sample indices of significant peaks and valleys, sorted by time."""
    peaks,   _ = find_peaks( y, distance=min_dist, prominence=prominence)
    valleys, _ = find_peaks(-y, distance=min_dist, prominence=prominence)
    all_idx = np.sort(np.concatenate([peaks, valleys]))
    return all_idx

# ── 4 soundfont groups ─────────────────────────────────────────────────────────
# Each entry: (lead_name, pan, vel_scale, octave_shift, dur_mult)
# min_dist/prominence control how selectively notes fire (fewer = more sparse)
# At 10ms/sample, one beat ≈ 61-75 samples. P/R/T spaced ~150-250ms apart.
def P(base): return int(np.clip(base + AXIS_SHIFT, 0, 127))

GROUPS = {
    # min_dist in samples (10ms each). At ~91-101 bpm, one beat ≈ 59-66 samples.
    # Target: ~1 note per beat per lead → min_dist ≈ 50-55 samples.
    # Pad fires every ~2 beats (min_dist ~120) for long sustained ambient notes.
    "violin": {
        "sf": SF_VIOLIN, "gain": 1.0, "mix_vol": 1.0,
        "min_dist": 52, "prom": 0.14,   # ~1 note/beat — clean melodic line
        "leads": [
            ("I",    P(64),  1.00,  0, 1.3),
            ("II",   P(76),  0.90, +1, 1.3),
            ("III",  P(52),  0.80, -1, 1.3),
        ],
    },
    "pad": {
        "sf": SF_PAD, "gain": 1.0, "mix_vol": 0.65,
        "min_dist": 115, "prom": 0.18,  # ~1 note every 2 beats — ambient wash
        "leads": [
            ("aVR",  P(16),  0.55,  0, 3.0),
            ("aVL",  P(42),  0.60, +1, 3.0),
            ("aVF",  P(92),  0.60, -1, 3.0),
        ],
    },
    "sax": {
        "sf": SF_SAX, "gain": 1.0, "mix_vol": 0.80,
        "min_dist": 48, "prom": 0.13,   # ~1 note/beat — punchy fills
        "leads": [
            ("V1",   P(112), 0.75,  0, 1.5),
            ("V2",   P( 96), 0.70, -1, 1.5),
            ("V3",   P( 80), 0.65, +1, 1.5),
        ],
    },
    "perc_mel": {
        "sf": SF_PERC, "gain": 4.0, "mix_vol": 0.65,
        "min_dist": 55, "prom": 0.16,   # ~1 note/beat — rhythmic bass accent
        # oct_shift removed — perc_note_for() keeps notes in bass/tom zone (36-57)
        "leads": [
            ("V4",   P(64),  0.70,  0, 2.0),
            ("V5",   P(46),  0.65,  0, 2.0),
            ("V6",   P(28),  0.60,  0, 2.0),
        ],
    },
}

# ── Build one MIDI per group (3 channels each) ─────────────────────────────────
def build_group_midi(gcfg, gname=""):
    group_leads = gcfg["leads"]
    min_dist    = gcfg["min_dist"]
    prom        = gcfg["prom"]
    midi = MIDIFile(len(group_leads))
    for track_idx, (lead, pan, vel_scale, oct_shift, dur_mult) in enumerate(group_leads):
        ch = track_idx
        for beat_pos, bpm in TEMPO_EVENTS:
            midi.addTempo(track_idx, beat_pos, bpm)
        midi.addControllerEvent(track_idx, ch, 0, 10, pan)
        midi.addControllerEvent(track_idx, ch, 0, 91, REVERB)
        y        = lead_y[lead]
        note_dur = BASE_DUR * dur_mult
        events   = detect_events(y, min_dist, prom)
        print(f"    {lead}: {len(events)} events ({len(events)/N_BEATS:.1f}/beat)")
        for i in events:
            t = float(t_beat[i])           # exact position from the data
            v = vel_for(y[i], vel_scale)   # velocity = amplitude, no distortion
            # perc_mel uses fixed bass/tom range; all others use recording's scale
            if gname == "perc_mel":
                p = perc_note_for(y[i])
            else:
                p = note_for(y[i], oct_shift)
            midi.addNote(track_idx, ch, p, t, note_dur, v)
    return midi

# ── Percussion MIDI (Lead II R-peaks only) ────────────────────────────────────
# Fixed note 50 (D3) so the heartbeat timbre is consistent across recordings.
# Only tempo (HR) and velocity (R-peak amplitude) carry physiological variation.
PERC_NOTE   = 50
y_ii        = lead_y["II"]
vel_ii      = np.array([vel_for(v) for v in y_ii])
perc_thresh = int(np.percentile(vel_ii, 82))
perc_midi   = MIDIFile(1)
for beat_pos, bpm in TEMPO_EVENTS:
    perc_midi.addTempo(0, beat_pos, bpm)
for i in range(N_SAMPLES):
    if vel_ii[i] > perc_thresh:
        perc_midi.addNote(0, 0, PERC_NOTE, float(t_beat[i]), 0.4, 127)

# ── Audio helpers ──────────────────────────────────────────────────────────────
def midi_to_wav(mid_path, wav_path, sf2_path, gain=1.0):
    cmd = ["fluidsynth", "-ni", "-g", str(gain),
           "-F", wav_path, "-T", "wav", "-r", "44100", sf2_path, mid_path]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if not os.path.exists(wav_path):
        raise RuntimeError(f"FluidSynth failed:\n{r.stderr[:500]}")
    print(f"  rendered : {os.path.basename(wav_path)}")

def mix_wavs(paths, volumes, out_path):
    def read(p):
        with wave.open(p, "rb") as w:
            params = w.getparams()
            raw = w.readframes(w.getnframes())
        return params, np.frombuffer(raw, dtype=np.int16).astype(np.float32)

    def rms_norm(d, target=8000.0, max_boost=4.0):
        rms = np.sqrt(np.mean(d**2))
        return d * min(target / rms, max_boost) if rms > 0 else d

    params, tracks = None, []
    for p, v in zip(paths, volumes):
        pr, d = read(p)
        params = params or pr
        tracks.append(rms_norm(d) * v)

    length = max(len(t) for t in tracks)
    mixed  = np.clip(
        sum(np.pad(t, (0, length - len(t))) for t in tracks),
        -32768, 32767
    ).astype(np.int16)
    with wave.open(out_path, "wb") as w:
        w.setparams(params)
        w.writeframes(mixed.tobytes())
    print(f"  mixed    : {os.path.basename(out_path)}")

# ── Render ─────────────────────────────────────────────────────────────────────
LABEL = cfg["label"]
wav_paths, mix_vols = [], []

print("\nWriting & rendering MIDI groups…")
for gname, gcfg in GROUPS.items():
    mid_path = os.path.join(PUBLIC, f"ekg_music_{LABEL}_{gname}.mid")
    wav_path = os.path.join(PUBLIC, f"ekg_music_{LABEL}_{gname}.wav")
    midi = build_group_midi(gcfg, gname)
    with open(mid_path, "wb") as f:
        midi.writeFile(f)
    midi_to_wav(mid_path, wav_path, gcfg["sf"], gain=gcfg["gain"])
    wav_paths.append(wav_path)
    mix_vols.append(gcfg["mix_vol"])

# Percussion heartbeat
perc_mid = os.path.join(PUBLIC, f"ekg_music_{LABEL}_perc.mid")
perc_wav = os.path.join(PUBLIC, f"ekg_music_{LABEL}_perc.wav")
with open(perc_mid, "wb") as f:
    perc_midi.writeFile(f)
midi_to_wav(perc_mid, perc_wav, SF_PERC, gain=4.0)
wav_paths.append(perc_wav)
mix_vols.append(0.8)

final = os.path.join(PUBLIC, f"ekg_music_{LABEL}_final.wav")
print("\nMixing 5 layers…")
mix_wavs(wav_paths, mix_vols, final)

# Clean up intermediate files
for p in wav_paths[:-1] + [perc_wav]:
    try: os.remove(p)
    except: pass
for gname in list(GROUPS) + ["perc"]:
    try: os.remove(os.path.join(PUBLIC, f"ekg_music_{LABEL}_{gname}.mid"))
    except: pass

print(f"\n✓  Done → {os.path.basename(final)}")
