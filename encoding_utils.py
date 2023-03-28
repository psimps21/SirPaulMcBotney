import itertools
import pickle
import pretty_midi


def get_maj_third(val):
    if val + 3 < 12:
        return val + 3
    else:
        return val + 3 - 12


def get_min_third(val):
    if val + 4 < 12:
        return val + 4
    else:
        return val + 4 - 12


def get_fifth(val):
    if val + 7 < 12:
        return val + 7
    else:
        return val + 7 - 12


def decode_chord(val, rest):
    if val == rest:
        return [-1]

    # SOS/EOS offset
    val -= 2

    if val < 12:
        return [val]
    elif val < 24:
        print(val)
        val = val-12
        return [val, get_maj_third(val), get_fifth(val)]
    elif val < 36:
        val = val-24
        return [val, get_min_third(val), get_fifth(val)]
    elif val < 48:
        print(val)
        val = val-36
        return [val, get_fifth(val)]
    else:
        return [-1]


def seq_to_bass_midi(seq_list, fname, low_a=33, fs=4.0, rest=50):
    bass_midi = pretty_midi.PrettyMIDI()
    bass_program = pretty_midi.instrument_name_to_program('Electric Bass (finger)')
    bass = pretty_midi.Instrument(program=bass_program)

    i = 1.0
    dur = 1/fs

    for seq in seq_list:
        for val in seq[1:]:
            # Account for the 1 EOS
            if val != 1:
                notes = decode_chord(val, rest)
                for note in notes:
                    # Also accounting for the 1 EOS since I subtract 2 from each value in decode_chord
                    if note >= 0:
                        note = pretty_midi.Note(velocity=100, pitch=note+low_a, start=i*dur, end=dur+(i*dur))
                        bass.notes.append(note)

                i += 1

    bass_midi.instruments.append(bass)
    bass_midi.write(fname + '.mid')



def check_interval(r, b):
    if b < r:
        return b + 12 - r
    else:
        return b - r


def check_chord_at_pos(seq):
    r = seq[0]
    min_third = False
    maj_third = False
    fifth = False

    for note in seq[1:]:
        if check_interval(r, note) == 3:
            min_third = True
        elif check_interval(r, note) == 4:
            maj_third = True
        elif check_interval(r, note) == 7:
            fifth = True

    # 12 - 23 for major chords
    if maj_third and fifth:
        return 12 + seq[0]
    # 24 - 35 for minor chords
    elif min_third and fifth:
        return 24 + seq[0]
    elif fifth:
        return 36 + seq[0]

    return -1


def check_chords(notes):
    note_circ = notes + notes
    chord = -1
    for i in range(len(notes)):
        new_chord = check_chord_at_pos(note_circ[i:i+len(notes)])
        temp = note_circ[i:i+len(notes)]

        if new_chord > 0 and chord < 0:
            chord = new_chord
        elif new_chord > 0:
            chord = min(chord, new_chord)

    return chord


# Also converts midi to note
def strip_strs_from_seq(seq):
    new_seq = []

    for e in seq:
        if not isinstance(e, str):
            new_seq.append(midi_2_note(e))

    return new_seq


def encode_chords_measure(seq, delim=':', rest='.'):
    steps = [list(y) for x, y in itertools.groupby(seq, lambda z: z == delim) if not x]

    new_measure = []

    for step in steps:
        new_step = strip_strs_from_seq(step)

        if len(new_step) == 1:
            new_measure.append(new_step[0])
            # new_measure.append(delim)
        elif len(new_step) == 0:
            new_measure.append(rest)
            # new_measure.append(delim)
        else:
            chord = check_chords(new_step)

            if chord < 0:
                return []

            new_measure.append(chord)
            # new_measure.append(delim)

    return new_measure


def encode_chords(data):
    new_data = []

    for bass, guitar in data:
        new_bass = encode_chords_measure(bass)
        new_guitar = encode_chords_measure(guitar)

        if len(new_bass) > 0 and len(new_guitar) > 0:
            new_data.append((new_bass, new_guitar))

    return new_data


"""
 Filters out input pairs that are longer than the specified threshold
"""
def filter_data_seq_threshold(data, threshold=100):
    new_data = []

    for bass, guitar in data:
        if len(guitar) < threshold and len(bass) < threshold:
            new_data.append((bass, guitar))

    return new_data


"""
 Helper function to translate midi values (0-127) to note values (0-12)
"""
def midi_2_note(val):
    # Recenter around 21 as A0
    val_offset = val - 21

    # Get cyclic note number thru mod12
    return val_offset % 12


def load_data(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


"""
 Encodes a single measure given input parameters
  - measure: sequence as list[(int/str)]
  - vmap: dictionary defining the mappings between string values and integers
  - cyclic: if true, then midi values are mapped to real notes represented by half-steps starting at A
  
  This function appends an EOS token (1) to the returned measure
  
  Returns encoded measure as list[int]
"""
def encode_measure(measure, vmap, cyclic=True):
    new_measure = []
    for n in measure:
        if isinstance(n, str):
            new_measure.append(vmap[n])
        elif cyclic:
            new_measure.append(midi_2_note(n)+2)
        else:
            new_measure.append(n+2)

    new_measure.append(1)

    return new_measure


"""
 Encodes the input pairs given parameters
  - data: list[(bass sequence, guitar sequence)]
  - cyclic: if true, then midi values are mapped to real notes represented by half-steps starting at A
  - str_map: mapping to translate string values (e.g. delimiter, rest, linker) to integer values
  
 Returns list[(encoded guitar sequence, encoded bass sequence)]
"""
def encode_data(data, str_map=None, cyclic=True, padded=False, chords=False):
    if str_map is None:
        if cyclic:
            str_map = {
                ':': 14,
                '.': 15,
                '-': 16
            }
        elif padded:
            str_map = {
                ':': 130,
                '.': 131,
                '-': 132,
                '>': 133
            }
        elif chords:
            str_map = {
                '.': 50
            }
        else:
            str_map = {
                ':': 130,
                '.': 131,
                '-': 132
            }

    if chords:
        chord_data = encode_chords(data)

        new_data = []

        for bass, gtr in chord_data:
            new_data.append((encode_measure(gtr, str_map, cyclic=False), encode_measure(bass, str_map, cyclic=False)))

    else:
        new_data = []

        for bass, gtr in data:
            new_data.append((encode_measure(gtr, str_map), encode_measure(bass, str_map)))

    return new_data


def pad_input_seq(seq, max_notes=6, delim=':', linker='-', null_char='>'):
    steps = [list(y) for x, y in itertools.groupby(seq, lambda z: z == delim) if not x]
    new_seq = []

    for step in steps:
        notes = [list(y) for x, y in itertools.groupby(step, lambda z: z == linker) if not x]
        num_links = 0

        for i, note in enumerate(notes):
            new_seq.append(note[0])

            if isinstance(note, str) and note == linker:
                num_links += 1

        while num_links < max_notes - 1:
            new_seq.append(linker)
            new_seq.append(null_char)
            num_links += 1

        new_seq.append(delim)

    return new_seq


def pad_input_seqs(data):
    new_data = []

    for bass, guitar in data:
        new_b = pad_input_seq(bass)
        new_g = pad_input_seq(guitar)

        new_data.append((new_b, new_g))

    return new_data


def get_max_len(data):
    max_len = 0
    for b, g in data:
        a = max(len(b), len(g))

        if a > max_len:
            max_len = a

    return max_len


def save_to_pickle(data, fpath, desc=''):
    data_obj = {'description': desc, 'data': data}
    with open(fpath, 'wb') as f:
        pickle.dump(data_obj, f)


def check_encoding(data):
    seen = {}

    for gtr, bass in data:
        for note in gtr:
            if note not in seen:
                seen[note] = True

        for note in bass:
            if note not in seen:
                seen[note] = True

    return seen


def filter_negatives(seq):
    new_seq = []

    for e in seq:
        if not e == 200:
            if e == -2:
                new_seq.append(201+2)
            else:
                new_seq.append(e+2)

    new_seq.append(1)
    return new_seq


def preprocess_parker_data(seqs):
    new_seqs = []

    for gtr, bass in seqs:
        new_seqs.append((filter_negatives(gtr), filter_negatives(bass)))

    return new_seqs






# 11 test point for chords
# test = load_data('bass_guitar_pairs_v7.obj')
#
#
# data = load_data('good_times_measures_v7.obj')
#
# # test_b, test_g = data[11]
#
# # new_bg = encode_chords(data)
# #
# #
# # max len 531
# enc_data = encode_data(data, cyclic=False, padded=False, chords=True)
# # #
# print(get_max_len(enc_data))
# # #
# seen = check_encoding(enc_data)
# #
# desc = 'Cyclic encoding with chords plus fifths, max_length=45, vocab_size=51, 0/1=SOS/EOS, 50=rest'
# #
# save_to_pickle(enc_data, 'good_times_v7.obj', desc=desc)

# with open('feel_good_measures_v9.obj', 'rb') as f:
#     data = pickle.load(f)
#
# enc_data = preprocess_parker_data(data)
#
# print(get_max_len(enc_data))
#
# desc = 'Parker chord encoding, max_length=50, vocab_size=203, 0/1=SOS/EOS, 203=rest'
# #
# save_to_pickle(enc_data, 'model_inputs/feel_good_v9.obj', desc=desc)


with open('gt_preds_v7.obj', 'rb') as f:
    data = pickle.load(f)

seq_to_bass_midi(data, 'output_midis/good_times_v7')
#
print('hi')
