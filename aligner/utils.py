from torchaudio.transforms import Resample
from scipy.io.wavfile import read, write
import numpy as np
import sys
import torchaudio
import pretty_midi
from parselmouth.praat import call
from scipy.signal import medfilt
import parselmouth
import dtw

def read_wav(wav_path, normalize=False):
    _, wav = read(wav_path)
    wav = wav.astype(np.float32)
    
    if normalize:
        wav = wav / 32768.0

    return wav

def read_sr(wav_path):
    sr, _ = read(wav_path)
    return sr

def resample(wav_path, target_sr, normalize=False):
    MAX_WAV_VALUE = 32768.0
    wav, source_sr = torchaudio.load(wav_path, normalize=False)
    wav = wav / MAX_WAV_VALUE

    if len(wav.size()) > 1 and wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = Resample(source_sr, target_sr, dtype=wav.dtype)(wav)
    if not normalize:
        wav = (wav * MAX_WAV_VALUE).clamp(-MAX_WAV_VALUE, MAX_WAV_VALUE - 1)
    return wav

def preprocess_midi(midi_path, start_sec=None, end_sec=None):
    md = pretty_midi.PrettyMIDI(midi_path)
    md.remove_invalid_notes()
    notes = md.instruments[0].notes

    # GV data error fix: convert silence label 96 -> 0
    for note in notes:
        if note.pitch == 96:
            note.pitch = 0

    # Remove leading notes that are silence (pitch 0 or 96), or end before the start_sec
    while notes and ((notes[0].pitch == 0 or notes[0].pitch == 96) or (notes[0].end < start_sec)):
        notes.pop(0)

    # Remove trailing notes that are silence (pitch 0 or 96), or start after the end_sec
    while notes and ((notes[-1].pitch == 0 or notes[-1].pitch == 96) or (notes[-1].end > end_sec)):
        notes.pop()

    # Align the start of the first note with the audio start time
    notes[0].start = start_sec if start_sec is not None else notes[0].start

    # Align the end of the last note with the audio end time
    notes[-1].end = end_sec if end_sec is not None else notes[-1].end

    # Insert or adjust silence notes between actual notes
    n_notes = len(notes)
    i = 0
    while i < n_notes - 1:
        error = notes[i+1].start - notes[i].end

        if error < 0:
            # Overlapping notes: force start of next note to match end of current note
            notes[i+1].start = notes[i].end
            notes[i+1].end = notes[i+2].start if i + 2 <= n_notes - 1 else notes[i+1].end
        elif 0 < error < 0.3:
            # Very short gap: trim current note to match next note's start
            notes[i].end = notes[i+1].start
        elif error >= 0.3:
            # Long gap: insert a silence note (pitch 0)
            silence_note = pretty_midi.Note(start=notes[i].end, end=notes[i+1].start, pitch=0, velocity=80)
            notes.insert(i+1, silence_note)
            n_notes += 1
        i += 1

    return notes

def hz_to_midi(hz):
    is_0 = hz == 0
    tone = 69 + (12 * np.log2(hz / 440))
    tone[is_0] = 0
    return tone

def midi_to_hz(midi):
    is_0 = midi == 0
    hz = (2 ** ((midi - 69) / 12)) * 440
    hz[is_0] = 0
    return hz

def extract_f0(wav, sampling_rate, hop_length, center=False, pitch_floor=None, voicing_threshold=0.7, octave_cost=0.02, octave_jump_cost=0.5, medfilt_kernel_size=5):

    if not isinstance(wav, np.ndarray):
        wav = wav.squeeze().numpy()
    
    n_frames = int(len(wav) // hop_length)
    time_step = hop_length / sampling_rate
    snd = parselmouth.Sound(wav, sampling_frequency=sampling_rate)
    pitch_floor = (3 / time_step / 4) if pitch_floor is None else pitch_floor
    pitch_object = snd.to_pitch_ac(time_step=time_step, voicing_threshold=voicing_threshold, octave_cost=octave_cost, octave_jump_cost=octave_jump_cost)
    f0 = pitch_object.selected_array['frequency']
    f0 = medfilt(f0, kernel_size=medfilt_kernel_size)

    if center:
        f0 = np.pad(f0, [2, 2])
    else:
        pad_size = (n_frames - len(f0) + 1) // 2
        f0 = np.pad(f0, [[pad_size, n_frames - len(f0) - pad_size]], mode='constant')

    return f0

def align_by_dtw(reference, query):
    # Perform DTW alignment
    alignment = dtw.dtw(query, reference, keep_internals=True)
    # Initialize aligned_query with the same length as reference
    aligned_query = np.zeros(reference.shape[0])
    # Fill aligned_query using the mapping from DTW
    for i in range(len(aligned_query)):
        closest_index = np.argmin(np.abs(alignment.index2 - i))
        aligned_query[i] = query[alignment.index1[closest_index]]

    return aligned_query

def expand_seq(seq, duration):
    def expand(target, n_repeat):
        return target.repeat(n_repeat)
    return np.concatenate([expand(to_expand, n_repeat) for to_expand, n_repeat in zip(seq, duration)], 0)

def correct_octave_error(notes, f0, sampling_rate=22050, hop_length=256, shift_range=(-12,12), plot=False):
    '''
        Assumes silence at head and tail of wav are already removed
    '''

    # Extract F0 from waveform
    # f0 = extract_f0(wav, sampling_rate, hop_length, extractor="praat-parselmouth")
    vocal_pitch = hz_to_midi(f0)
    total_frames = len(vocal_pitch)

    # Extract original note pitches and durations
    note_pitch = np.array([n.pitch for n in notes])
    note_duration = np.array([n.duration for n in notes])

    # Convert note durations from seconds to frames
    note_duration = duration_secs_to_frames(note_duration, sampling_rate, hop_length)
    expanded_note_pitch = expand_seq(note_pitch, note_duration)
    
    min_error = float('inf')
    best_shift = 0
    
    for shift in range(shift_range[0], shift_range[1] + 1):

        # Find the optimal octave shift within the specified range
        shifted_pitch = expanded_note_pitch.copy()
        zero_mask = (shifted_pitch == 0)
        shifted_pitch += shift
        shifted_pitch[zero_mask] = 0

        valid_length = min(total_frames, len(shifted_pitch))

        errors = []
        for i in range(valid_length):
            vocal_p = vocal_pitch[i]
            note_p = shifted_pitch[i]
            if vocal_p == 0 or note_p == 0:
                continue
            errors.append(abs(vocal_p - note_p))
        
        current_error = np.mean(errors)

        if plot:
            print(f"Shift: {shift}, Error: {current_error:.4f}")
            plt.plot(vocal_pitch[:valid_length], label="vocal pitch")
            plt.plot(shifted_pitch[:valid_length], label="note pitch")
            plt.legend()
            plt.show()
            plt.close()
        if current_error < min_error:
            min_error = current_error
            best_shift = shift

    # Apply the best shift to all non-silence notes
    for n in notes:
        if n.pitch != 0:
            n.pitch += best_shift

    print(f"[LOG] all notes shifted by {best_shift}")
    return notes, best_shift

def duration_secs_to_frames(note_duration_sec, sr, hop_length):
    '''
        If the unit of the note duration is "seconds", the unit should be converted to "frames"
        Furthermore, it should be rounded to integer and this causes rounding error
        This function includes error handling process that alleviates the rounding error
    '''
    frames_per_sec = sr / hop_length
    note_duration_frame = note_duration_sec * frames_per_sec
    note_duration_frame_int = note_duration_frame.copy().astype(np.int64)
    errors = note_duration_frame - note_duration_frame_int  # rounding error per each note
    errors_sum = int(np.sum(errors))

    top_k_errors_idx = errors.argsort()[-errors_sum:][::-1]
    for i in top_k_errors_idx:
        note_duration_frame_int[i] += 1

    return note_duration_frame_int
