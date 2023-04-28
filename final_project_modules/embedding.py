import argparse
from tqdm import tqdm
from transformer import Transformer, preprocess_data, train
from torch import load, device, Tensor
import pickle


def load_data(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)

    return data


"""
Parameters:
 - X: tensor of shape (n_samples, max_len)
 - model: Transformer object
"""
def get_embeddings(X: Tensor, model: Transformer):
    return model.get_embedding(X)


if __name__ == '__main__':
    # Get arguments from the command line (for easier VM use)
    parser = argparse.ArgumentParser('Training script for Transformer model')
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--pred_path', type=str, default=None)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--out_path', type=str, default=None)
    parser.add_argument('--encoder_blocks', type=int, default=2)
    parser.add_argument('--decoder_blocks', type=int, default=4)
    parser.add_argument('--heads', type=int, default=3)
    parser.add_argument('--emb_frac', type=float, default=0.10) 
    parser.add_argument('--default_emb', type=bool, default=False)
    
    args = parser.parse_args()

    data_obj = load_data(args.data_path)
    input_data = load_data(args.pred_path)

    # Parse data file for config parameters
    data = data_obj['data']
    vocab_size = data_obj['vocab_size']
    max_len = data_obj['max_len']

    if args.default_emb:
        embedding_dim = 256
    else:    
        embedding_dim = int(vocab_size*args.emb_frac)

    # Process data into tensors and add SOS tokens
    source, _ = preprocess_data(input_data['data'], vocab_size, vocab_size, max_len)

    # Declare model
    model = Transformer(vocab_size, vocab_size, embedding_dim, args.encoder_blocks, args.decoder_blocks, args.heads)

    # Send model and data tensors to gpu
    dev = device('cuda')
    model.to(dev)
    source = source.to(dev)

    # Load the model weights
    model.load_state_dict(load(args.model_path))

    # Get embeddings
    embeddings = model.get_embedding(source)

    with open(args.out_path, 'wb') as f:
        pickle.dump(embeddings, f)
