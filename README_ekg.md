# EKG to Music: Sonification of 12-Lead Cardiac Recordings

This script converts real 12-lead ECG waveform data into a multi-layered audio piece. The idea was to map each lead group to a distinct instrument and let the actual physiology (heart rate, QRS duration, QT interval, cardiac axis) drive the musical parameters rather than making them up.

---

## What it does

The 12 leads are split into four groups, each rendered through its own soundfont so the timbres stay clearly distinct:

| Leads | Instrument | Why |
|-------|-----------|-----|
| I, II, III | Violin | Limb leads carry the main cardiac axis; violin handles the melodic center |
| aVR, aVL, aVF | Airsynth pad | Augmented leads at the periphery; long ambient pad notes work well here |
| V1, V2, V3 | Saxophone | Right precordial, punchy and forward in the mix |
| V4, V5, V6 | Percussion soundfont (melodic) | Left precordial, rhythmic bass/tom accent |
| Lead II (R-peaks only) | Bejeweled percussion | The literal heartbeat, one hit per detected R-peak |

The clinical data shapes the music directly:
- **MIDI tempo** = actual heart rate in BPM, interpolated over time from the tempo curve
- **Note duration** = derived from QRS width (wider QRS means longer notes)
- **Reverb (CC 91)** = derived from QT interval (longer QT means more reverb/tail)
- **Stereo pan** = anatomical electrode position on the chest, adjusted by cardiac axis deviation
- **Note pitch** = amplitude of the EKG signal at each detected peak/valley, mapped to the recording's scale
- **Velocity** = signal amplitude scaled per lead group

Four recordings are included, each with its own pentatonic/modal scale and tempo curve:

- `harp` - recorded while playing harp; sinus tachycardia around 95-102 bpm
- `binaural` - binaural beat listening session; HR decelerates from 90 down to 71 bpm
- `raga` - raga listening; parasympathetic deceleration from 102 to 80 bpm
- `guitar` - guitar playing; sympathetic arousal with variable HR ranging 90-118 bpm

---

## Requirements

**Python packages** (see `requirements.txt`):
```
numpy
pandas
scipy
midiutil
```

**System dependency:**
```
fluidsynth
```

On macOS:
```bash
brew install fluidsynth
```

On Ubuntu/Debian:
```bash
sudo apt install fluidsynth
```

**Soundfonts** - place these in `soundfont/`:
- `Levi_s_Violin.sf2`
- `Airsynth.sf2`
- `Sax-A-Boom_Soundfont.sf2`
- `Bejeweled_3_Percussions__SF2_.sf2`

**Data files** - place these in `public/`:
- `harp-ekg-waveform-timeseries.csv` + `harp-ekg-clinical-summary.json`
- `ekg-waveform-timeseries.csv` + `ekg-clinical-summary.json`
- `raga-ekg-waveform-timeseries.csv` + `raga-ekg-clinical-summary.json`
- `guitar-ekg-waveform-timeseries.csv` + `guitar-ekg-clinical-summary.json`

---

## Usage

```bash
python3 ekg_test.py harp
python3 ekg_test.py binaural
python3 ekg_test.py raga
python3 ekg_test.py guitar
```

The script prints progress as it processes each lead group and calls FluidSynth to render audio. Output is a single mixed WAV file in `public/`:

```
public/ekg_music_<recording>_final.wav
```

Intermediate per-group WAV and MIDI files are cleaned up automatically after mixing.

---

## How the pipeline works

1. **Load waveform CSV** - reads all 12 leads and resamples each to exactly `SPB` (samples per beat) so tiled R-peaks fall on integer beat positions with no cumulative drift.
2. **Tile the template** - repeats the single-beat template `N_BEATS` times to produce a full-length waveform per lead.
3. **Smooth** - Savitzky-Golay filter (window 51, poly 4) removes noise before peak detection.
4. **Detect events** - `scipy.signal.find_peaks` finds both peaks and valleys above a prominence threshold; these become note-on events.
5. **Build MIDI** - one MIDI file per soundfont group (3 tracks each). Tempo events are written at every ~5-beat interval using the interpolated tempo curve. Pan, reverb CC, and per-note pitch/velocity/duration all come from the data.
6. **Render audio** - FluidSynth converts each MIDI + soundfont pair to a 44100 Hz WAV.
7. **Mix** - the five WAV layers (4 groups + percussion heartbeat) are RMS-normalized and summed with per-group volume weights, then clipped to 16-bit and written as the final output.

---

## File structure

```
musicfromdata/
├── ekg_test.py              # main script (this file)
├── requirements.txt
├── utils/
│   └── numeric_utils.py     # map_value() helper
├── soundfont/               # .sf2 soundfont files
└── public/                  # CSV + JSON data in; WAV out
```
