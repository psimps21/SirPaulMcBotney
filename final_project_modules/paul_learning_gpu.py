import argparse
from tqdm import tqdm
from transformer import Transformer, preprocess_data, train
from torch import load, device
import pickle


def load_data(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)

    return data


"""
    Data file should contain an object of the form:
     - data: the input/output sequences
     - vocab_size: the size (including SOS/EOS tokens) of the data's encoding vocab
     - max_len: the max length of any of the input/output (but usually input) sequences including SOS/EOS
     - version: version #
"""
if __name__ == '__main__':
    # Get arguments from the command line (for easier VM use)
    parser = argparse.ArgumentParser('Training script for Transformer model')
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--run_label', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--save_int', type=int, default=5)
    parser.add_argument('--train_split', type=float, default=0.9)
    parser.add_argument('--emb_frac', type=float, default=0.10)
    parser.add_argument('--encoder_blocks', type=int, default=2)
    parser.add_argument('--decoder_blocks', type=int, default=4)
    parser.add_argument('--heads', type=int, default=3)
    parser.add_argument('--default_emb', type=bool, default=False)

    args = parser.parse_args()

    data_obj = load_data(args.data_path)

    # Parse data file for config parameters
    data = data_obj['data']
    vocab_size = data_obj['vocab_size']
    max_len = data_obj['max_len'] + 1

    if args.default_emb:
        embedding_dim = 256
    else:
        embedding_dim = int(vocab_size*args.emb_frac)

    # Make train/test split
    train_size = int(len(data)*args.train_split)
    train_data = data[:train_size]
    test_data = data[train_size:]

    # Process data into tensors and add SOS tokens
    train_source, train_target = preprocess_data(train_data, vocab_size, vocab_size, max_len)
    test_source, test_target = preprocess_data(test_data, vocab_size, vocab_size, max_len)

    # Declare model
    model = Transformer(
            vocab_size, 
            vocab_size, 
            embedding_dim, 
            args.encoder_blocks, 
            args.decoder_blocks, 
            args.heads
        )

    # Send model and data tensors to gpu
    dev = device('cuda')
    model.to(dev)
    train_source = train_source.to(dev)
    train_target = train_target.to(dev)
    test_source = test_source.to(dev)
    test_target = test_target.to(dev)

    for i in range(0, args.epochs, args.save_int):
        # Get the number of epochs to run for
        epochs = min(args.epochs-i, args.save_int)

        print(i+epochs)

        # Load the previous save-state if saving intermediately
        if i != 0:
            model.load_state_dict(load(label + '.pkl'))

        # Create the label for save-state file
        label = args.run_label + '_' + str(i+epochs)

        # Train for epochs epochs and save with label
        train_loss, test_loss = train(
            model, train_source, train_target, test_source, test_target, vocab_size, epochs, 64, 0.0005, label
        )
