import numpy as np
from .utils import read_wav, read_sr, resample, preprocess_midi, hz_to_midi, duration_secs_to_frames, extract_f0, expand_seq, align_by_dtw, correct_octave_error
import rle

import matplotlib.pyplot as plt
from numba import njit
from joblib import Parallel, delayed

@njit(cache=True, fastmath=True)
def _find_rightmost_silence(
    vocal: np.ndarray,
    left: int,
    right: int,
    back_limit: int = 4000
) -> tuple:
    """
    Find the right-most silence segment (start, end) ending at or before `right`.
    If the window starts with zeros, search leftward for the true segment start.
    Returns (-1, -1) if not found.
    """
    a = vocal[left:right+1]
    if (a == 0).sum() == 0:
        return -1, -1

    z = (a == 0).astype(np.int8)
    diff = np.diff(np.concatenate((np.zeros(1, np.int8), z, np.zeros(1, np.int8))))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1
    seg_start, seg_end = starts[-1], ends[-1]
    seg_start_abs_pos = left + seg_start
    seg_end_abs_pos = left + seg_end

    # Expand to true leftmost silence if window begins with silence
    if seg_start == 0:
        target_left = max(0, left - back_limit)
        while seg_start_abs_pos > target_left and vocal[seg_start_abs_pos - 1] == 0:
            seg_start_abs_pos -= 1
        if seg_start_abs_pos == 0 and vocal[0] == 0:
            return -1, -1

    if seg_start_abs_pos > 0 and vocal[seg_start_abs_pos-1] != 0:
        return seg_start_abs_pos, seg_end_abs_pos
    return -1, -1

@njit(cache=True, fastmath=True)
def _build_band_edges(dur, k, m, margin_frames=1000):
    """
    For each note (duration = dur[i] frames) build Sakoe-Chiba band:
    j ∈ [cumsum(i) - k·dur[i],  cumsum(i+1) + k·dur[i]], clipped to [0, m-1].
    Returns (N, 2) array of [j_min, j_max] per note.
    """
    cs = np.empty(dur.size + 1, np.int32)
    cs[0] = 0
    cs[1:] = np.cumsum(dur)
    half = np.where(k * dur < margin_frames, margin_frames, k * dur).astype(np.int32)
    j_min = np.maximum(0, cs[:-1] - half)
    j_max = np.minimum(m - 1, cs[1:] + half)
    out = np.empty((dur.size, 2), np.int32)
    out[:, 0] = j_min
    out[:, 1] = j_max
    return out

@njit(cache=True, fastmath=True)
def _pitch_seg_cost(note_val, seg):
    """
    Mean(|note_val - seg|), clipped at 2 semitones (6 MIDI cents).
    """
    raw_diff = np.abs(seg - note_val)
    diff = np.minimum(raw_diff, 6)
    return diff.mean()

@njit(cache=True, fastmath=True)
def _dtw_core(vocal, note, dur, lam_pitch, lam_dur, lam_trans, band):
    """
    Band-limited DTW kernel.
    """
    n, m = note.shape[0], vocal.shape[0]
    INF = 1e9
    cost = np.full((n, m), INF, np.float32)
    ptr = np.zeros((n, m, 2), np.int16)

    cost[0, 0] = abs(vocal[0] - note[0])

    for i in range(n):
        j_start, j_end = band[i, 0], band[i, 1]
        for j in range(j_start, j_end + 1):
            if i == 0 and j == 0:
                continue
            max_v = int(dur[i] * 1.5)
            max_v = min(max_v, j)
            if i == 0 and j - max_v < 0:
                max_v = j
            elif i > 0 and j - max_v <= 0:
                max_v = j - 1
            if max_v <= 0:
                continue

            best = INF
            bu = bv = 0
            u_max = 0 if i == 0 else 1
            win_left = j - max_v
            win_right = j - 1

            if vocal[j] > 0:
                s0, s1 = _find_rightmost_silence(vocal, win_left, win_right)
                unvoiced_cost = 0
            else:
                s0, s1 = -1, -1
                unvoiced_cost = 20

            if s0 != -1:
                max_v = j - s0 + 1

            for u in range(u_max + 1):
                for v in range(1, max_v + 1):
                    prev = cost[i - u, j - v]
                    if prev >= INF:
                        continue
                    if s0 != -1 and u == 0 and note[i-u] != 0 and j - v <= s1:
                        continue
                    dur_cost = abs(dur[i] - np.sum(vocal[j - v + 1 : j + 1] != 0))
                    seg = vocal[j - v + 1 : j + 1]
                    p_cost = _pitch_seg_cost(note[i], seg)
                    tot = prev + lam_pitch * p_cost + lam_dur * dur_cost + unvoiced_cost
                    if u == 1:
                        tot += lam_trans
                    if tot < best:
                        best = tot
                        bu = u
                        bv = v
            cost[i, j] = best
            ptr[i, j, 0] = bu
            ptr[i, j, 1] = bv
    return cost, ptr

class VocalNoteAlign:
    """
    Class for aligning vocal pitch sequences to note pitch using band-limited DTW.
    """
    def __init__(self, lambda_pitch=5, lambda_duration=0.1, lambda_unvoiced=20, lambda_transition=1e-3,
                 sampling_rate=22050, hop_length=256):
        """
        Args:
            lambda_pitch (float): Weight for pitch difference.
            lambda_duration (float): Weight for duration difference.
            lambda_unvoiced (float): Penalty for mismatched silence.
        """
        self.lambda_pitch = lambda_pitch
        self.lambda_duration = lambda_duration
        self.lambda_unvoiced = lambda_unvoiced
        self.lambda_transition = lambda_transition
        self.sampling_rate = sampling_rate
        self.hop_length = hop_length
        self.frame_per_sec = sampling_rate / hop_length

    def __call__(self, wav_path, midi_path):
        return self.multi_process_run(wav_path, midi_path)

    def run(self, vocal_pitch, note_pitch, note_duration):
        """
        Main alignment pipeline: handles silence, DTW, and returns aligned note sequence.
        """
        cost_matrix, pointer_matrix = self.compute_dtw_band(vocal_pitch, note_pitch, note_duration)
        aligned_note_pitch, _note_pitch, _note_duration = self.align(
            vocal_pitch, note_pitch, cost_matrix, pointer_matrix
        )
        aligned_note_pitch[vocal_pitch == 0] = 0
        # Fallback to vanilla DTW if alignment fails
        if np.sum(aligned_note_pitch == -1) > 0:
            aligned_note_pitch = align_by_dtw(
                vocal_pitch, expand_seq(note_pitch, note_duration)
            )
            aligned_note_pitch[vocal_pitch == 0] = 0
        return aligned_note_pitch, _note_pitch, _note_duration, pointer_matrix, cost_matrix

    def multi_process_run(self, wav_path, midi_path, n_jobs=8):
        """
        Main entry for vocal-MIDI alignment. Handles IO, segmentation, and parallel alignment.
        """
        def extract_note_segment(notes, start_sec, end_sec):
            """
            Extract pitches and durations for notes within a given segment (in seconds).
            Trim to start_sec and end_sec, remove unvoiced notes at ends.
            """
            pitch_segment, duration_segment = [], []
            first_note_handled = False
            for note in notes:
                if note.end < start_sec or note.start > end_sec:
                    continue
                if not first_note_handled:
                    note.start = start_sec
                    first_note_handled = True
                pitch_segment.append(note.pitch)
                duration_segment.append(note.end - note.start)
                last_start, last_end = note.start, end_sec
            if duration_segment:
                duration_segment[-1] = last_end - last_start
            pitch_segment = np.array(pitch_segment)
            duration_segment = np.array(duration_segment)
            if pitch_segment.size > 1:
                if pitch_segment[0] == 0:
                    pitch_segment = pitch_segment[1:]
                    duration_segment = duration_segment[1:]
                if pitch_segment[-1] == 0:
                    pitch_segment = pitch_segment[:-1]
                    duration_segment = duration_segment[:-1]
            mask = (pitch_segment != 0)
            return pitch_segment[mask], duration_segment[mask]

        def split_segments(unvoiced_intervals, f0, notes):
            """
            Split the audio into segments based on unvoiced intervals.
            Returns a list of (idx, vocal_pitch, note_pitch, note_duration, start_frame, end_frame).
            """
            segments = []
            for idx in range(len(unvoiced_intervals) - 1):
                start_frame = unvoiced_intervals[idx][1]
                end_frame = unvoiced_intervals[idx + 1][0]
                start_sec = start_frame / self.frame_per_sec
                end_sec = end_frame / self.frame_per_sec
                note_pitch, note_duration = extract_note_segment(notes, start_sec, end_sec)
                note_duration = duration_secs_to_frames(note_duration, self.sampling_rate, self.hop_length)
                vocal_pitch = hz_to_midi(f0[start_frame:end_frame])
                segments.append((idx, vocal_pitch, note_pitch, note_duration, start_frame, end_frame))
            return segments

        def align_segment(segment):
            """
            Align a single segment using DTW and postprocess the aligned notes.
            """
            idx, vocal_pitch, note_pitch, note_duration, *_ = segment
            expanded_note_pitch, aligned_note_pitch, aligned_note_duration, pointer_matrix, cost_matrix = self.run(
                vocal_pitch, note_pitch, note_duration
            )
            # RLE encoding for consecutive identical pitches
            aligned_note_pitch, aligned_note_duration = [np.array(x) for x in rle.encode(expanded_note_pitch)]
            aligned_note_pitch, aligned_note_duration = self.insert_unvoiced_segments(
                aligned_note_pitch, aligned_note_duration, vocal_pitch
            )
            return aligned_note_pitch, aligned_note_duration, expanded_note_pitch, vocal_pitch, pointer_matrix, cost_matrix

        def merge_segments(segments, results, total_frames):
            """
            Merge aligned note sequences for each segment into a global note sequence.
            """
            note_pitch, note_duration = [], []
            curr_seg_end = 0
            for i in range(len(segments)):
                curr_seg_start, curr_seg_end = segments[i][4], segments[i][5]
                if i == 0:
                    prev_seg_end = 0
                else:
                    prev_seg_end = segments[i-1][5]
                # Fill unvoiced gap if any
                gap = curr_seg_start - prev_seg_end
                if gap > 0:
                    note_pitch.append(0)
                    note_duration.append(gap)
                note_pitch.extend(results[i][0])
                note_duration.extend(results[i][1])
            # Tail silence
            if curr_seg_end < total_frames:
                note_pitch.append(0)
                note_duration.append(total_frames - curr_seg_end)
            return note_pitch, note_duration

        # ---- I/O and preprocessing ----
        sr = read_sr(wav_path)
        wav = resample(wav_path, self.sampling_rate, normalize=True).squeeze().numpy() if sr != self.sampling_rate else read_wav(wav_path, normalize=True)
        f0 = extract_f0(wav, self.sampling_rate, self.hop_length, medfilt_kernel_size=5)

        vs, ds = rle.encode(f0)
        cum = np.pad(np.cumsum(ds), (1, 0))
        intervals = [(cum[i], cum[i + 1]) for i in range(len(cum) - 1)]
        unvoiced_duration_threshold = 1 # sec
        unvoiced_intervals = [
            (start, end) for v, (start, end) in zip(vs, intervals)
            if v == 0 and (end - start) > unvoiced_duration_threshold * self.frame_per_sec
        ]
        T = len(f0)
        # Pad for edge cases (start/end)
        if not unvoiced_intervals:
            unvoiced_intervals = [(0, 0), (T, T)]
        else:
            if unvoiced_intervals[0][0] != 0:
                unvoiced_intervals = [(0, 0)] + unvoiced_intervals
            if unvoiced_intervals[-1][1] != T:
                unvoiced_intervals = unvoiced_intervals + [(T, T)]
        notes = preprocess_midi(
            midi_path,
            start_sec=unvoiced_intervals[0][1] / self.frame_per_sec,
            end_sec=unvoiced_intervals[-1][0] / self.frame_per_sec
        )
        trimmed_f0 = f0[unvoiced_intervals[0][1]:unvoiced_intervals[-1][0]]
        notes, _ = correct_octave_error(notes, trimmed_f0, sampling_rate=self.sampling_rate, hop_length=self.hop_length)
        segments = split_segments(unvoiced_intervals, f0, notes)

        # Efficient parallelism for segment alignment
        results = [align_segment(seg) for seg in segments] if len(segments) < 2 else \
                  Parallel(n_jobs=min(n_jobs, len(segments)), backend="threading")(
                      delayed(align_segment)(seg) for seg in segments
                  )
        aligned_note_pitch, aligned_note_duration = merge_segments(segments, results, total_frames=f0.shape[0])
        return np.array(aligned_note_pitch), np.array(aligned_note_duration), results

    def compute_dtw_band(self, vocal_pitch, note_pitch, note_duration, band_width_multiply=40):
        """
        Run band-limited DTW using Numba JIT kernels.
        """
        vocal = np.asarray(vocal_pitch, dtype=np.float32)
        note = np.asarray(note_pitch, dtype=np.float32)
        dur = np.asarray(note_duration, dtype=np.int32).ravel()
        n, m = note.shape[0], vocal.shape[0]
        band = _build_band_edges(dur, band_width_multiply, m)
        cost, ptr = _dtw_core(
            vocal, note, dur,
            self.lambda_pitch,
            self.lambda_duration,
            self.lambda_transition,
            band
        )
        return cost, ptr

    def align(self, vocal_pitch, note_pitch, cost_matrix, pointer_matrix):
        """
        Backtrack through pointer_matrix to produce frame-level aligned note sequence.
        Returns aligned_note_pitch, note_pitch, note_duration.
        """
        n, m = len(note_pitch), len(vocal_pitch)
        aligned_note_pitch = np.full(m, -1.0)
        note_duration = np.zeros([n], dtype=np.int32)
        curr_i, curr_j = n - 1, m - 1
        steps = 0
        max_steps = n * m  # Safety bound for infinite loop
        while steps < max_steps:
            u, v = pointer_matrix[curr_i][curr_j]
            aligned_note_pitch[curr_j - v + 1: curr_j + 1] = note_pitch[curr_i]
            note_duration[curr_i] += v
            curr_i -= u
            curr_j -= v
            steps += 1
            if curr_i == 0 and curr_j == 0:
                aligned_note_pitch[0] = note_pitch[0]
                note_duration[0] += 1
                break
        else:
            print("[WARNING] align(): Max steps reached! Alignment may be broken.")
        return aligned_note_pitch, note_pitch, note_duration

    def insert_unvoiced_segments(
    self,
    note_pitch,
    note_duration,
    vocal_pitch,
    ):
        """
        Align note_pitch/note_duration sequence with frame-level vocal_pitch
        by inserting note=0 (unvoiced) for unvoiced regions.
    
        Parameters
        ----------
        note_pitch : Sequence[int or float]
            Note pitch sequence (0 = unvoiced).
        note_duration : Sequence[int]
            Note durations (in frames), same length as note_pitch.
        vocal_pitch : Sequence[int or float]
            Frame-level vocal pitch. Should have length sum(note_duration).
    
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (New note_pitch, new note_duration) with unvoiced notes correctly inserted and short notes merged.
        """
        new_pitch, new_dur = [], []
        frame_ptr = 0  # current frame position in vocal_pitch
    
        for pitch, dur in zip(note_pitch, note_duration):
            segment = vocal_pitch[frame_ptr : frame_ptr + dur]
            frame_ptr += dur
    
            if pitch == 0:
                # Already unvoiced: preserve as is
                new_pitch.append(0)
                new_dur.append(dur)
                continue
    
            # Find change-points between voiced/unvoiced using numpy diff (vectorized)
            mask = (segment != 0)
            # Get indices where voiced/unvoiced changes, plus [0] and [len]
            if len(mask) == 0:
                continue
            change_idxs = np.flatnonzero(np.diff(mask.astype(int), prepend=mask[0]))  # change points
            # Build block boundaries
            block_starts = np.r_[0, change_idxs]
            block_ends = np.r_[change_idxs, len(mask)]
            for start, end in zip(block_starts, block_ends):
                is_voiced = mask[start]
                length = end - start
                if length <= 0:
                    continue
                new_pitch.append(pitch if is_voiced else 0)
                new_dur.append(length)
    
        # Step 2: Remove/merge very short voiced notes (<=2 frames) into previous note, unless at sequence start
        final_pitch, final_dur = [], []
        duration_buffer = 0
        for p, d in zip(new_pitch, new_dur):
            if duration_buffer > 0:
                d += duration_buffer
                duration_buffer = 0
            if d <= 2 and p != 0:
                # If this is the very first note, keep as is
                if not final_dur:
                    final_pitch.append(p)
                    final_dur.append(d)
                else:
                    if final_pitch[-1] != 0:
                        final_dur[-1] += d  # Merge into previous note
                    else:
                        duration_buffer += d  # Merge into next note (if previous was unvoiced)
            else:
                final_pitch.append(p)
                final_dur.append(d)
        # Handle remaining duration_buffer at the end (rare)
        if duration_buffer > 0 and final_dur:
            final_dur[-1] += duration_buffer
    
        return np.array(final_pitch), np.array(final_dur)
