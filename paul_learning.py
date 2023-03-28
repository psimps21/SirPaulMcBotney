import pretty_midi
from tqdm import tqdm
from transformer import Transformer, preprocess_data, train
from torch import load
import pickle


def load_data(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)

    return data


def midi_2_note(val):
    # Recenter around 21 as A0
    val_offset = val - 21

    # Get cyclic note number thru mod12
    return val_offset % 12


# Removed the 0's because preprocess data handles that
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


def encode_data(data, str_map=None, cyclic=True):
    if str_map is None:
        if cyclic:
            str_map = {
                ':': 14,
                '.': 15,
                '-': 16
            }
        else:
            str_map = {
                ':': 133,
                '.': 134,
                '-': 135
            }

    new_data = []

    for bass, gtr in data:
        new_data.append((encode_measure(gtr, str_map), encode_measure(bass, str_map)))

    return new_data


# max length: 444
if __name__ == '__main__':
    data = load_data('bass_guitar_pairs_v2.obj')
    enc_data = encode_data(data)
    desc = 'Cyclic note mapping using linker characters, max_length=101, vocab_size=16, 0/1=SOS/EOS, 14/15/16=delim/rest/linker'

    data_obj = {'description': desc, 'data': enc_data}

    with open('bass_guitar_pairs_v3.obj', 'wb') as f:
        pickle.dump(data_obj, f)

    # train_size = 12000
    #
    # train_data = enc_data[:train_size]
    # test_data = enc_data[train_size:]

    # train_source, train_target = preprocess_data(train_data, 16, 16, 101)
    # test_source, test_target = preprocess_data(test_data, 16, 16, 101)

    print('hi')

    model = Transformer(133, 133, 256, 2, 4, 3)
    # model.load_state_dict(load('paul_5e_v2.pkl'))
    #
    # train_loss, test_loss = train(model, train_source, train_target, test_source, test_target, 133, 5, 64, 0.0001, 'paul_10e_v2')

    model = Transformer(133, 133, 256, 2, 4, 3)
    model.load_state_dict(load('paul_20e.pkl'))

    with open('dead_flowers_gtr_meas.obj', 'rb') as f:
        df_gtr = pickle.load(f)

    output_pairs = []
    new_df = encode_data(df_gtr)

    for g, _ in tqdm(new_df):
        result, _ = model.predict(g, max_length=len(g)+2)
        output_pairs.append((g, result))

    print('hi')
