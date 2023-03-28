import pretty_midi
import numpy as np
import os
import pickle
from parker_chord_encoding import get_chord4_encoded_note_sequence


def get_measures(midi):
    # beat_start = midi.estimate_beat_start()
    beat_start = 0.0
    beats = midi.get_beats(beat_start)

    # For now, using only songs with consistent time signature
    time_sig = midi.time_signature_changes[0]
    beats_per_measure = time_sig.numerator
    measure_starts = np.arange(0, len(beats), beats_per_measure)

    measures = []
    for beat in measure_starts:
        # Want end time to be start of beat following this measure
        if beat + beats_per_measure < len(beats):
            measures.append((beats[beat], beats[beat + beats_per_measure]))
        else:
            measures.append((beats[beat], midi.get_end_time()))

    return measures


def get_encoded_note_sequence(data, linker='-'):
    seq = []
    no_notes = True

    for step in data:
        # If this step is not a rest
        if np.max(step) > 0:
            no_notes = False

            # Get indices of notes being played
            notes_arr = np.nonzero(step)[0]
            for note in notes_arr:
                # Add each note followed by a linker (to represent chords)
                seq.append(int(note))
                seq.append(linker)

            # Remove the trailing linker
            seq[-1] = ':'

        else:
            seq.append('.')
            seq.append(':')

    return seq, no_notes


def get_encoded_note_sequence_max6(data, linker='-'):
    seq = []
    no_notes = True

    for step in data:
        # If this step is not a rest
        if np.max(step) > 0:
            no_notes = False

            # Get indices of notes being played
            notes_arr = np.nonzero(step)[0]

            # If there are more than 6 notes at this time step, quit and return nothing
            if len(notes_arr) > 6:
                return [], True

            for note in notes_arr:
                # Add each note followed by a linker (to represent chords)
                seq.append(int(note))
                seq.append(linker)

            # Remove the trailing linker
            seq[-1] = ':'

        else:
            seq.append('.')
            seq.append(':')

    return seq, no_notes


"""
 Either call get_encoded_note_sequence, or get_encoded_note_sequence_max6
"""
def get_note_sequence(data, max6=True, parker=False):
    if max6:
        return get_encoded_note_sequence_max6(data)
    elif parker:
        return get_chord4_encoded_note_sequence(data)
    return get_encoded_note_sequence(data)


def get_bass_guitar_pairs(bass, guitar, measures, fs, parker=False):
    pairs = []
    b_step_max = 0
    g_step_max = 0

    for measure in measures:
        bass_roll = bass.get_piano_roll(fs=fs, times=np.arange(measure[0], measure[1], 1./fs)).T
        guitar_roll = guitar.get_piano_roll(fs=fs, times=np.arange(measure[0], measure[1], 1./fs)).T

        # Call midi parsing
        bass_measure, bass_no_notes = get_note_sequence(bass_roll, parker=parker)
        guitar_measure, guitar_no_notes = get_note_sequence(guitar_roll, parker=parker)

        if not (bass_no_notes or guitar_no_notes):
            pairs.append((bass_measure, guitar_measure))

    return pairs


"""
Extract the note combinations used from a single piano roll (for the guitar)
 - data (time_steps x notes)
"""
def get_guitar_vocab_single(data):
    pts, notes = np.where(data > 0)
    curr_pt = pts[0]
    s = '.'
    curr_chord = []
    vocab = []

    for i, n in zip(pts, notes):
        if i != curr_pt:
            vocab.append(s.join(curr_chord))
            curr_chord = [str(n)]
            curr_pt = i
        else:
            curr_chord.append(str(n))

    if len(curr_chord) > 0:
        vocab.append(s.join(curr_chord))

    return set(vocab)


"""
Returns indices for guitar instruments from midi
"""
def get_guitar_instr(midi):
    gtrs = []

    for i, inst in enumerate(midi.instruments):
        if 'gtr' in inst.name.lower() or 'guitar' in inst.name.lower():
            gtrs.append(i)

    if len(gtrs) == 0:
        raise LookupError('Guitar does not exist')

    return gtrs


def get_bass_instr(midi):
    for i, inst in enumerate(midi.instruments):
        if 'bass' in inst.name.lower():
            return i

    raise LookupError('Bass does not exist')


def get_guitar_vocab_midi(midi, fs=4):
    guitar_inds = get_guitar_instr(midi)

    if len(guitar_inds) == 0:
        raise Exception('No guitars found')

    vocab = set()

    for i in guitar_inds:
        roll = np.copy(midi.instruments[i].get_piano_roll(fs=fs)).T
        vocab = vocab.union(get_guitar_vocab_single(roll))

    return vocab


def get_vocab_all_midis(dir):
    vocab = set()
    guitarless_instrs = set()

    for subdir, dirs, files in os.walk(dir):
        for f in files:
            midi = pretty_midi.PrettyMIDI(midi_file=dir + '/' + f)
            try:
                vocab = vocab.union(get_guitar_vocab_midi(midi))
            except:
                print(f + ' did not have a guitar track matching naming pattern')
                guitarless_instrs = guitarless_instrs.union(set(midi.instruments))

    return vocab, guitarless_instrs


def get_pairs_all_midis(dir, double_measures=False):
    pairs = []

    for subdir, dirs, files in os.walk(dir):
        for f in files:
            midi = pretty_midi.PrettyMIDI(midi_file=dir + '/' + f)
            try:
                bass_ind = get_bass_instr(midi)
                guitar_inds = get_guitar_instr(midi)

                measures = get_measures(midi)

                if double_measures:
                    measures = get_double_measures(measures)

                song_pairs = []

                for g in guitar_inds:
                    if len(song_pairs) == 0:
                        song_pairs = get_bass_guitar_pairs(midi.instruments[bass_ind], midi.instruments[g], measures, 4)
                    else:
                        song_pairs += get_bass_guitar_pairs(midi.instruments[bass_ind], midi.instruments[g], measures,
                                                            4)
                pairs += song_pairs

            except LookupError:
                print(f + ' did not have the required tracks')

    return pairs


def get_double_measures(measures):
    new_start = 0
    new_measures = []

    for i, m in enumerate(measures):
        start, end = m
        if i % 2 == 0:
            new_start = start
        else:
            new_measures.append((new_start, end))

    return new_measures


midi = pretty_midi.PrettyMIDI('test_midis/GoodTimes.mid')
p_roll = midi.instruments[5]

measures = get_measures(midi)
# d_measures = get_double_measures(measures)
pairs = get_bass_guitar_pairs(p_roll, p_roll, measures, 4, parker=False)

with open('good_times_measures_v7.obj', 'wb') as f:
    pickle.dump(pairs, f)

# print('hi')

# pairs = get_pairs_all_midis('midis', double_measures=True)
# print('hi')
#
# with open('bass_guitar_pairs_double_meas.obj', 'wb') as f:
#     pickle.dump(pairs, f)


# vocab, guitarless = get_vocab_all_midis('midis')
# singles = [str(i) for i in range(127)]
# vocab = vocab.union(set(singles))
#
# vocab_list = list(vocab)
#
# with open('guitar_vocab.obj', 'wb') as f:
#     pickle.dump(vocab_list, f)



# midi = pretty_midi.PrettyMIDI(midi_file='midis/ivegotf.mid')
#
# bass = midi.instruments[1]
# guitar = midi.instruments[6]
#
# measures = get_measures(midi)
# pairs = get_bass_guitar_pairs(bass, guitar, measures, 4)
print('hi')


#
# guitar_vocab = get_guitar_vocab_single(guitar_roll)
# print('hi')
