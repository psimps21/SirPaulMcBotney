import pretty_midi
import numpy as np
import os
import pickle
from tqdm import tqdm

# import sys
# print(sys.executable)

from primePy import primes

TSTEP = 200 # timestep delimiting character
REST = 201 # charater for rest


def GetInterval(root, note):
    """
    finds the interval between a given root note and another note
    Args:
        root: int [0-11] indicating a root note
        note: int [0-11] indicating a note being played

    Return:
        interval: int indicating the number of half steps from the root to the given note
    """

    # If the root note is after the played in in the non cylcic ordering from A to G# (e.g. roog=G(10), note=B(2))
    if note < root:
        return (note + 12) - root # the + 12 translates the not up by one octave

    else:
        return note - root
    

def CheckPairAtPosition(note_seq):
    """
    Args:
        note_seq: a sequence two ordered integers [0-11] indicating the notes being played at a given timestep
    Return:
        chord_type: the type of chord detected
        root_num: the number corresponding to the root of the chord
    """
    root = note_seq[0]

    # loop over all positions in a chord (excluding first)
    for note in note_seq[1:]:
        # boolean indicators for intervals
        # maj_third, min_third, fifth = False, False, False
        maj_third, min_third, fifth, min_seven, maj_seven = False, False, False, False, False

        # Get the interval between the root and current note
        interval = GetInterval(root, note)

        # Update the interval indicators
        if interval == 3:
            min_third = True
        elif interval == 4:
            maj_third = True
        elif interval == 7:
            fifth = True
        elif interval == 10:
            min_seven = True
        elif interval == 11:
            maj_seven = True

    # Use interval indicators to identify present chords

    # If a major 7 pair is present
    if maj_seven:
        return '2-maj7', root # indicates a chord type and its root
    
    # if a minor pair is present
    elif min_seven:
        return '2-min7', root 

    # if a fifth pair is present
    elif fifth :
        return '2-fifth', root 
    
    # If a major pair is present
    elif maj_third:
        return '2-maj3', root 
    
    # if a minor pair is present
    elif min_third :
        return '2-min3', root 
    
    # Ignore all other pairs
    else:
        return None
    
    

def Check3ChordAtPosition(note_seq):
    """
    Args:
        note_seq: a sequence three ordered integers [0-11] indicating the notes being played at a given timestep
    Return:
        chord_type: the type of chord detected
        root_num: the number corresponding to the root of the chord
    """

    root = note_seq[0]

    # loop over all positions in a chord (excluding first)
    for note in note_seq[1:]:
        # boolean indicators for intervals
        # maj_third, min_third, fifth = False, False, False
        maj_third, min_third, dim_fifth, fifth, min_seven, maj_seven = False, False, False, False, False, False

        # Get the interval between the root and current note
        interval = GetInterval(root, note)

        # Update the interval indicators
        if interval == 3:
            min_third = True
        elif interval == 4:
            maj_third = True
        elif interval == 6:
            dim_fifth = True
        elif interval == 7:
            fifth = True
        elif interval == 10:
            min_seven = True
        elif interval == 11:
            maj_seven = True

    # Use interval indicators to identify present chords

    # If a major 7 chord is present
    if maj_third and maj_seven:
        return '3-maj7', root # indicates a major chord and its root
    
    # If a dominant 7 chord is present
    if maj_third and min_seven:
        return '3-dom7', root 
    
    # if a minor 7 chord is present
    elif min_third and min_seven:
        return '3-min7', root 

    # If a major chord is present
    elif maj_third and fifth:
        return '3-maj', root 
    
    # if a minor chord is present
    elif min_third and fifth:
        return '3-min', root 
    
    # if a diminished chord is present
    elif min_third and dim_fifth:
        return '3-dim', root 
    
    # Ignore all other chords
    else:
        return None
    

def Check4ChordAtPosition(note_seq):
    """
    Args:
        note_seq: a sequence four ordered integers [0-11] indicating the notes being played at a given timestep
    Return:
        chord_type: the type of chord detected
        root_num: the number corresponding to the root of the chord
    """

    root = note_seq[0]

    # loop over all positions in a chord (excluding first)
    for note in note_seq[1:]:
        # boolean indicators for intervals
        # maj_third, min_third, fifth = False, False, False
        maj_third, min_third, dim_fifth, fifth, min_seven, maj_seven = False, False, False, False, False, False

        # Get the interval between the root and current note
        interval = GetInterval(root, note)

        # Update the interval indicators
        if interval == 3:
            min_third = True
        elif interval == 4:
            maj_third = True
        elif interval == 6:
            dim_fifth = True
        elif interval == 7:
            fifth = True
        elif interval == 10:
            min_seven = True
        elif interval == 11:
            maj_seven = True

    # Use interval indicators to identify present chords

    # If a major 7 chord is present
    if maj_third and fifth and maj_seven:
        return '4-maj7', root # indicates a major chord and its root
    
    # If a dominant 7 chord is present
    if maj_third and fifth and min_seven:
        return '4-dom7', root 
    
    # if a minor 7 chord is present
    elif min_third and fifth and min_seven:
        return '4-min7', root 
    
    # if a diminished chord is present
    elif min_third and dim_fifth and min_seven:
        return '4-halfdim', root 
    
    # Ignore all other chords
    else:
        return None
    

def EncodeTimeStep(played_notes, lim=4, rest_token=-2):
    """
    Return a number indicating the note, pair, 3 note chord, or 4 note chord being played at a timestep
    Args:
        played_notes: an array containing the midi values for the notes present during the curent time step
    Return:
        token: a number representing the not, pair, of 3 note chord, or 4 note chord curently being played
    """
    # Define the acceptable pairs, three note chords, and four note chords
    pairs = ['maj7','min7','fifth','maj3','min3']
    triplets = ['maj7','dom7','min7','maj','min','dim']
    quads = ['maj7','min7','dom7','halfdim']

    # Def the number of unique pairs, three note chords, and four note chords
    num_notes = 12
    num_pairs = len(pairs)
    num_triplets = len(triplets)
    num_quads = len(quads)

    # Define total number of pairs, three note chords, and four note chords
    total_pairs = num_pairs * 12
    total_triplets = num_triplets * 12
    total_quads = num_quads * 12

    # Map each pair, three note chord, or four note chord to its index
    pair_num_map = {val:inx for inx, val in enumerate(pairs)}
    triplet_num_map = {val:inx for inx, val in enumerate(triplets)}
    quad_num_map = {val:inx for inx, val in enumerate(quads)}

    # Drop midi values below 21
    played_notes = played_notes[played_notes > 20]

    # Translate all notes being played to be 0-11
    played_notes = (played_notes - 21) % 12

    # Get rid of repeated notes, sort the notes and trim to 4
    played_notes = np.sort(np.unique(played_notes))[:lim]

    # if one note is being played return the note number plus 2 (translated bc 0 and 1 and SOS and EOS tokens)
    if len(played_notes) == 1:
        return played_notes.item(0) 

    # A list to track the present chords
    present_chords, count = [], 0

    # While a chord has not been identified and we have note tried all options
        # Allow each note to be the root and check the chords present
    while len(present_chords) == 0 and count+1 != lim:
        # Identify the notes being played
        if len(played_notes) == 2:
            chord_tup = CheckPairAtPosition(played_notes)
        if len(played_notes) == 3:
            chord_tup = Check3ChordAtPosition(played_notes)
        if len(played_notes) == 4:
            chord_tup = Check4ChordAtPosition(played_notes)

        # if an accepted chord is being played add it to the chord list
        if chord_tup is not None:
            chord_info, root_num = chord_tup
            chord_size, chord_type = chord_info.split('-')

            if chord_size == '2':
                # Find the index of the present pair
                pair_inx = pair_num_map[chord_type]

                # map the pair to is appropriate number
                chord_num = (root_num * num_pairs) + pair_inx + num_notes

            elif chord_size == '3':
                # Find the index of the present pair
                triplet_inx = triplet_num_map[chord_type]

                # map the pair to is appropriate number
                chord_num = (root_num * num_triplets) + triplet_inx + num_notes + total_pairs

            else: # if chord_size == '4':
                # Find the index of the present pair
                quad_inx = quad_num_map[chord_type]

                # map the pair to is appropriate number
                chord_num = (root_num * num_quads) + quad_inx + num_notes + total_pairs + total_triplets

            present_chords.append(chord_num)

        # Get next circular rotation of played notes (this effectively sets a new root)
        played_notes = np.roll(played_notes, 1)

        # increment count
        count += 1

    # If the notes being played are note a valid chord then return a rest
    if len(present_chords) == 0:
        return rest_token
    
    # Return the first chord identified
    else:
        return present_chords[0]

def GenerateReverseEncodeTimestepMap(pairs=None, triplets=None, quads = None):
    """
    Generates a reverse mapping of chord numbers to note numbers [0(A)-11(G#)]
    Args:
        pairs - list of all recognized pairs
        triplets - list of all recognized triplets
        quadss - list of all recognized quadss
    """
    # Define the acceptable pairs, three note chords, and four note chords
    if pairs is None:
        pairs = ['maj7','min7','fifth','maj3','min3']

        # The interval (in half steps) of a paired note from its root
        pair_intervals = [11, 10, 7, 4, 3]
    if triplets is None:
        triplets = ['maj7','dom7','min7','maj','min','dim']

        # The intervals (in half steps) of a triplet notes from its root
        triplet_intervals = [[4,11], [4,10], [3,10], [4,7], [3,7], [3,6]]

    if quads is None:
        quads = ['maj7','min7','dom7','halfdim']

        # The intervals (in half steps) of a quad note from its root
        quad_intervals = [[4,7,11], [3,7,10], [4,7,10], [3,6,10]]

    # Def the number of unique pairs, three note chords, and four note chords
    num_notes = 12
    num_pairs = len(pairs)
    num_triplets = len(triplets)

    # Define total number of pairs, three note chords, and four note chords
    total_pairs = num_pairs * 12
    total_triplets = num_triplets * 12
    # total_quads = num_quads * 12

    # Map single notes back to their original number (these stay the sampe)
    note_map = {i:[i] for i in range(12)}

    # Map each pair to a list of its notes
    pair_map = {}

    # Counter for the number of pairs (translated by the number of values used by the single notes)
    pair_count = 0 + num_notes

    # Loop over each possible root note
    for root_num in range(num_notes):
        # Loop over all possible pairs
        for pair_inx in range(len(pairs)):
            # Find the interval (in half steps) of the paired note
            paired_note_interval = pair_intervals[pair_inx]

            # Map the pair itnerval to a note using the root
            # if the interval is out of range
            if root_num + paired_note_interval > 11:
                # note_pair = [root_num, (root_num + paired_note_interval) - 12]
                note_pair = [root_num, (root_num + paired_note_interval)]
            else:
                note_pair = [root_num, (root_num + paired_note_interval)]

            # Map identified pair note numbers to the pair number 
            pair_map[pair_count] = note_pair

            # Increment the pair_count
            pair_count += 1
    # print(pair_map)

    # Map each triplet to a list of its notes
    triplet_map = {}

    # Counter for the number of pairs (translated by the number of values used by the single notes)
    triplet_count = 0  + num_notes + total_pairs

    # Loop over each possible root note
    for root_num in range(num_notes):
        # Loop over all possible pairs
        for triplet_inx in range(len(triplets)):
            # Find the intervals (in half steps) of the triplet notes
            triplet_note_intervals = triplet_intervals[triplet_inx]

            # Map the triple itnervala to notes using the root
            note_triplet = [root_num]
            for interval in triplet_note_intervals:
                # if the interval is out of range
                if root_num + interval > 11:
                    # note_triplet.append((root_num + interval) - 12)
                    note_triplet.append((root_num + interval))
                else:
                    note_triplet.append((root_num + interval))

            # Map identified pair note numbers to the pair number 
            triplet_map[triplet_count] = note_triplet

            # Increment the pair_count
            triplet_count += 1
    # print(triplet_map)

    # Map each triplet to a list of its notes
    quad_map = {}

    # Counter for the number of pairs (translated by the number of values used by the single notes)
    quad_count = 0  + num_notes + total_pairs + total_triplets

    # Loop over each possible root note
    for root_num in range(num_notes):
        # Loop over all possible pairs
        for quad_inx in range(len(quads)):
            # Find the intervals (in half steps) of the triplet notes
            quad_note_intervals = quad_intervals[quad_inx]

            # Map the triple itnervala to notes using the root
            note_quad = [root_num]
            for interval in quad_note_intervals:
                # if the interval is out of range
                if root_num + interval > 11:
                    # note_quad.append((root_num + interval) - 12)
                    note_quad.append((root_num + interval))
                else:
                    note_quad.append((root_num + interval))

            # Map identified pair note numbers to the pair number 
            quad_map[quad_count] = note_quad

            # Increment the pair_count
            quad_count += 1
    # print(quad_map)

    # Combine all maps
    chord_map = {}
    chord_map.update(note_map)
    chord_map.update(pair_map)
    chord_map.update(triplet_map)
    chord_map.update(quad_map)

    # print(list(chord_map.keys()))
    return chord_map


def get_chord4_encoded_note_sequence(data): # old linker linker='-'
# def get_encoded_note_sequence(data): # old linker linker='-'
    seq = []
    no_notes = True

    for step in data:
        # If this step is not a rest
        if np.max(step) > 0:
            no_notes = False

            # Get indices of notes being played
            notes_arr = np.nonzero(step)[0]

            # If more than 4 notes are played at a time
            if len(notes_arr) > 4:
                return [], True

            notes_encoded = EncodeTimeStep(notes_arr)
            
            seq.append(notes_encoded)
            seq.append(TSTEP) # value for timestep delimiter

        else:
            seq.append(REST) # value for rest
            seq.append(TSTEP) # value for timestep delimiter

    return seq, no_notes


def SeqToBassMidi(seq_list, fname, low_a=33, fs=4.0, rest=201):
    bass_midi = pretty_midi.PrettyMIDI()
    bass_program = pretty_midi.instrument_name_to_program('Electric Bass (finger)')
    bass = pretty_midi.Instrument(program=bass_program)

    i = 1.0
    dur = 1/fs

    decode_map = GenerateReverseEncodeTimestepMap()
    
    for seq in seq_list:
        # Check all values after SOS
        for val in seq[1:]:
            # if value is not EOS
            if val != 1:

                # Translate back to zero base
                val -= 2

                if val == rest:
                    notes = -1
                else:
                    notes = decode_map[val]

                for note in notes:
                    if note > -1:
                        note = pretty_midi.Note(velocity=100, pitch=note+low_a, start=i*dur, end=dur+(i*dur))
                        bass.notes.append(note)

                i += 1

    bass_midi.instruments.append(bass)
    bass_midi.write(fname + '.mid')


if __name__ == '__main__':
    # GenerateReverseEncodeTimestepMap()
    with open('fg_preds_v9.obj', 'rb') as f:
        data = pickle.load(f)

    SeqToBassMidi(data, 'output_midis/feel_good_v9')
