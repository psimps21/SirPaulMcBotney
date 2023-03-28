import pretty_midi
from tqdm import tqdm
from transformer import Transformer, preprocess_data, train
from torch import load, device
import pickle


def load_data(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)

    return data


def encode_measure(measure, vmap):
    new_measure = [0]
    for n in measure:
        if isinstance(n, str):
            new_measure.append(vmap[n])
        else:
            new_measure.append(n+2)

    new_measure.append(1)

    return new_measure


def encode_data(data, str_map=None):
    if str_map is None:
        str_map = {
            ':': 130,
            '.': 131,
            '-': 132
        }

    new_data = []

    for bass, gtr in data:
        new_data.append((encode_measure(gtr, str_map), encode_measure(bass, str_map)))

    return new_data

# max length: 444
if __name__ == '__main__':
    data = load_data('bass_guitar_pairs_v2.obj')
    enc_data = encode_data(data)

    train_size = 12000

    train_data = enc_data[:train_size]
    test_data = enc_data[train_size:]

    train_source, train_target = preprocess_data(train_data, 133, 133, 450)
    test_source, test_target = preprocess_data(test_data, 133, 133, 450)

    model = Transformer(133, 133, 256, 2, 4, 3)
    model.load_state_dict(load('paul_5e_v2.pkl'))

    dev = device('cuda')

    model.to(dev)

    train_source = train_source.to(dev)
    train_target = train_target.to(dev)

    test_source = test_source.to(dev)
    test_target = test_target.to(dev)

    train_loss, test_loss = train(model, train_source, train_target, test_source, test_target, 133, 5, 64, 0.0001, 'paul_10e_v2')

    # model.load_state_dict(load('paul_20e.pkl'))

    # with open('dead_flowers_gtr_meas.obj', 'rb') as f:
    #     df_gtr = pickle.load(f)
    #
    # output_pairs = []
    # new_df = encode_data(df_gtr)
    #
    # for g, _ in tqdm(new_df):
    #     result, _ = model.predict(g, max_length=len(g)+2)
    #     output_pairs.append((g, result))

    print('hi')
