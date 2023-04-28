from tqdm import tqdm
from transformer import Transformer, preprocess_data
from torch import load, device
import pickle
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Bassline prediction script')
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--pred_path', type=str, default=None)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--out_path', type=str, default=None)
    parser.add_argument('--encoder_blocks', type=int, default=2)
    parser.add_argument('--decoder_blocks', type=int, default=4)
    parser.add_argument('--heads', type=int, default=3)
    parser.add_argument('--emb_frac', type=float, default=0.1)
    parser.add_argument('--default_emb', type=bool, default=False)


    args = parser.parse_args()

    with open(args.data_path, 'rb') as f:
        base_data = pickle.load(f)


    if args.default_emb:
        embedding_dim = 256
    else:
        embedding_dim = int(base_data['vocab_size'] * args.emb_frac)
    
    vocab = base_data['vocab_size']
    max_len = base_data['max_len']

    model = Transformer(
        vocab,
        vocab,
        embedding_dim,
        args.encoder_blocks,
        args.decoder_blocks,
        args.heads
    )

    model.load_state_dict(load(args.model_path))
    model.to(device('cuda'))

    with open(args.pred_path, 'rb') as f:
        df_gtr = pickle.load(f)

    df, _ = preprocess_data(df_gtr['data'], vocab, vocab, max_len)

    preds = []

    for i in range(len(df)):
        source = [int(j) for j in list(df[i])]
        pred_t, _ = model.predict(source, beam_size=3)
        pred = [int(j) for j in list(pred_t)]

        preds.append(pred)

    with open(args.out_path, 'wb') as f:
        pickle.dump(preds, f)
