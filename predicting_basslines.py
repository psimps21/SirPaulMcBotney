from tqdm import tqdm
from transformer import Transformer, preprocess_data
from torch import load
import pickle


if __name__ == '__main__':
    version = 5
    vocab = 52
    max_len = 90

    model = Transformer(vocab, vocab, 256, 2, 4, 3)
    # model.load_state_dict(load('paul_15e_v%i.pkl' % version))

    with open('dead_flowers_v%i.obj' % version, 'rb') as f:
        df_gtr = pickle.load(f)

    df, _ = preprocess_data(df_gtr['data'], vocab, vocab, max_len)

    preds = []

    for i in range(len(df)):
        source = [int(j) for j in list(df[i])]
        pred_t, _ = model.predict(source, beam_size=3)
        pred = [int(j) for j in list(pred_t)]

        preds.append(pred)

    with open('df_preds_v%i.obj' % version, 'wb') as f:
        pickle.dump(preds, f)

    print('hi')
