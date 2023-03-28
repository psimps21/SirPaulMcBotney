
import pickle
from typing import List, Optional, Tuple, Dict

import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib import ticker

import torch
from torch.nn import Module, Linear, Softmax, ReLU, LayerNorm, ModuleList, Dropout, Embedding, CrossEntropyLoss
from torch.optim import Adam

import time


class PositionalEncodingLayer(Module):

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        X has shape (batch_size, sequence_length, embedding_dim)

        This function should create the positional encoding matrix
        and return the sum of X and the encoding matrix.

        The positional encoding matrix is defined as follow:

        P_(pos, 2i) = sin(pos / (10000 ^ (2i / d)))
        P_(pos, 2i + 1) = cos(pos / (10000 ^ (2i / d)))

        The output will have shape (batch_size, sequence_length, embedding_dim)
        """
        encoded = torch.zeros(X.size()).to(torch.device('cuda'))
        d = self.embedding_dim

        for pos in range(X.size()[1]):
            for i in range(0, d//2):
                encoded[:, pos, 2*i] = np.sin(pos / (10000 ** (2*i / d)))
                encoded[:, pos, 2*i + 1] = np.cos(pos / (10000 ** (2 * i / d)))

        return X + encoded


class SelfAttentionLayer(Module):

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()

        self.linear_Q = Linear(in_dim, out_dim)
        self.linear_K = Linear(in_dim, out_dim)
        self.linear_V = Linear(in_dim, out_dim)

        self.softmax = Softmax(-1)

        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self, query_X: torch.Tensor, key_X: torch.Tensor, value_X: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        query_X, key_X and value_X have shape (batch_size, sequence_length, in_dim). The sequence length
        may be different for query_X and key_X but must be the same for key_X and value_X.

        This function should return two things:
            - The output of the self-attention, which will have shape (batch_size, sequence_length, out_dim)
            - The attention weights, which will have shape (batch_size, query_sequence_length, key_sequence_length)

        If a mask is passed as input, you should mask the input to the softmax, using `float(-1e32)` instead of -infinity.
        The mask will be a tensor with 1's and 0's, where 0's represent entries that should be masked (set to -1e32).

        Hint: The following functions may be useful
            - torch.bmm (https://pytorch.org/docs/stable/generated/torch.bmm.html)
            - torch.Tensor.masked_fill (https://pytorch.org/docs/stable/generated/torch.Tensor.masked_fill.html)
        """
        Q = self.linear_Q(query_X)
        K = self.linear_K(key_X)
        V = self.linear_V(value_X)

        QK = torch.bmm(Q, torch.transpose(K, 1, 2)).to(torch.device('cuda'))

        if mask is not None:
            b_mask = torch.lt(mask, 1).to(torch.device('cuda'))

            QK = torch.masked_fill(QK, b_mask, float(-1*(10**32))).to(torch.device('cuda'))

        QK /= np.sqrt(self.out_dim)

        alpha = self.softmax(QK).to(torch.device('cuda'))

        A = torch.bmm(alpha, V).to(torch.device('cuda'))

        return A, alpha


class MultiHeadedAttentionLayer(Module):

    def __init__(self, in_dim: int, out_dim: int, n_heads: int) -> None:
        super().__init__()

        self.attention_heads = ModuleList([SelfAttentionLayer(in_dim, out_dim) for _ in range(n_heads)])

        self.linear = Linear(n_heads * out_dim, out_dim)

    def forward(self, query_X: torch.Tensor, key_X: torch.Tensor, value_X: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        This function calls the self-attention layer and returns the output of the multi-headed attention
        and the attention weights of each attention head.

        The attention_weights matrix has dimensions (batch_size, heads, query_sequence_length, key_sequence_length)
        """

        outputs, attention_weights = [], []

        for attention_head in self.attention_heads:
            out, attention = attention_head(query_X, key_X, value_X, mask)
            outputs.append(out)
            attention_weights.append(attention)

        outputs = torch.cat(outputs, dim=-1).to(torch.device('cuda'))
        attention_weights = torch.stack(attention_weights, dim=1).to(torch.device('cuda'))

        return self.linear(outputs), attention_weights


class EncoderBlock(Module):

    def __init__(self, embedding_dim: int, n_heads: int) -> None:
        super().__init__()

        self.attention = MultiHeadedAttentionLayer(embedding_dim, embedding_dim, n_heads)

        self.norm1 = LayerNorm(embedding_dim)
        self.norm2 = LayerNorm(embedding_dim)

        self.linear1 = Linear(embedding_dim, 4 * embedding_dim)
        self.linear2 = Linear(4 * embedding_dim, embedding_dim)
        self.relu = ReLU()

        self.dropout1 = Dropout(0.2)
        self.dropout2 = Dropout(0.2)

    def forward(self, X, mask=None):
        """
        Implementation of an encoder block. Both the input and output
        have shape (batch_size, source_sequence_length, embedding_dim).

        The mask is passed to the multi-headed self-attention layer,
        and is usually used for the padding in the encoder.
        """  
        att_out, _ = self.attention(X, X, X, mask)

        residual = X + self.dropout1(att_out)

        X = self.norm1(residual)

        temp = self.linear1(X)
        temp = self.relu(temp)
        temp = self.linear2(temp)

        residual = X + self.dropout2(temp)

        return self.norm2(residual)


class Encoder(Module):

    def __init__(self, vocab_size: int, embedding_dim: int, n_blocks: int, n_heads: int) -> None:
        super().__init__()

        self.embedding_layer = Embedding(vocab_size + 1, embedding_dim, padding_idx=vocab_size)
        self.position_encoding = PositionalEncodingLayer(embedding_dim)
        self.blocks = ModuleList([EncoderBlock(embedding_dim, n_heads) for _ in range(n_blocks)])
        self.vocab_size = vocab_size

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Transformer encoder. The input has dimensions (batch_size, sequence_length)
        and the output has dimensions (batch_size, sequence_length, embedding_dim).

        The encoder returns its output and the location of the padding, which will be
        used by the decoder.
        """

        padding_locations = torch.where(X == self.vocab_size, torch.zeros_like(X, dtype=torch.float64),
                                        torch.ones_like(X, dtype=torch.float64)).to(torch.device('cuda'))
        padding_mask = torch.einsum("bi,bj->bij", (padding_locations, padding_locations)).to(torch.device('cuda'))

        X = self.embedding_layer(X)
        X = self.position_encoding(X)
        for block in self.blocks:
            X = block(X, padding_mask)
        return X, padding_locations


class DecoderBlock(Module):

    def __init__(self, embedding_dim, n_heads) -> None:
        super().__init__()

        self.attention1 = MultiHeadedAttentionLayer(embedding_dim, embedding_dim, n_heads)
        self.attention2 = MultiHeadedAttentionLayer(embedding_dim, embedding_dim, n_heads)

        self.norm1 = LayerNorm(embedding_dim)
        self.norm2 = LayerNorm(embedding_dim)
        self.norm3 = LayerNorm(embedding_dim)

        self.linear1 = Linear(embedding_dim, 4 * embedding_dim)
        self.linear2 = Linear(4 * embedding_dim, embedding_dim)
        self.relu = ReLU()

        self.dropout1 = Dropout(0.2)
        self.dropout2 = Dropout(0.2)
        self.dropout3 = Dropout(0.2)

    def forward(self, encoded_source: torch.Tensor, target: torch.Tensor,
                mask1: Optional[torch.Tensor]=None, mask2: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Implementation of a decoder block. encoded_source has dimensions (batch_size, source_sequence_length, embedding_dim)
        and target has dimensions (batch_size, target_sequence_length, embedding_dim).

        The mask1 is passed to the first multi-headed self-attention layer, and mask2 is passed
        to the second multi-headed self-attention layer.

        Returns its output of shape (batch_size, target_sequence_length, embedding_dim) and
        the attention matrices for each of the heads of the second multi-headed self-attention layer
        (the one where the source and target are "mixed").
        """  
        att_out, _ = self.attention1(target, target, target, mask1)
        residual = target + self.dropout1(att_out)
        
        X = self.norm1(residual)

        att_out, att_weights = self.attention2(X, encoded_source, encoded_source, mask2)

        residual = X + self.dropout2(att_out)
        X = self.norm2(residual)

        temp = self.linear1(X)
        temp = self.relu(temp)
        temp = self.linear2(temp)
        residual = X + self.dropout3(temp)

        return self.norm3(residual), att_weights


class Decoder(Module):

    def __init__(self, vocab_size: int, embedding_dim: int, n_blocks: int, n_heads: int) -> None:
        super().__init__()
        
        self.embedding_layer = Embedding(vocab_size + 1, embedding_dim, padding_idx=vocab_size)
        self.position_encoding = PositionalEncodingLayer(embedding_dim)
        self.blocks = ModuleList([DecoderBlock(embedding_dim, n_heads) for _ in range(n_blocks)])

        self.linear = Linear(embedding_dim, vocab_size + 1)
        self.softmax = Softmax(-1)

        self.vocab_size = vocab_size

    def _lookahead_mask(self, seq_length: int) -> torch.Tensor:
        """
        Compute the mask to prevent the decoder from looking at future target values.
        The mask you return should be a tensor of shape (sequence_length, sequence_length)
        with only 1's and 0's, where a 0 represent an entry that will be masked in the
        multi-headed attention layer.

        Hint: The function torch.tril (https://pytorch.org/docs/stable/generated/torch.tril.html)
        may be useful.
        """
        return torch.tril(torch.ones([seq_length, seq_length])).to(torch.device('cuda'))

    def forward(self, encoded_source: torch.Tensor, source_padding: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """
        Transformer decoder. encoded_source has dimensions (batch_size, source_sequence_length, embedding),
        source_padding has dimensions (batch_size, source_seuqence_length) and target has dimensions
        (batch_size, target_sequence_length).

        Returns its output of shape (batch_size, target_sequence_length, target_vocab_size) and
        the attention weights from the first decoder block, of shape
        (batch_size, n_heads, source_sequence_length, target_sequence_length)

        Note that the output is not normalized (i.e. we don't use the softmax function).
        """

        # Lookahead mask
        seq_length = target.shape[1]
        mask = self._lookahead_mask(seq_length)

        # Padding masks
        target_padding = torch.where(target == self.vocab_size, torch.zeros_like(target, dtype=torch.float64), 
                                     torch.ones_like(target, dtype=torch.float64)).to(torch.device('cuda'))

        target_padding_mask = torch.einsum("bi,bj->bij", (target_padding, target_padding)).to(torch.device('cuda'))
        mask1 = torch.multiply(mask, target_padding_mask).to(torch.device('cuda'))

        source_target_padding_mask = torch.einsum("bi,bj->bij", (target_padding, source_padding)).to(torch.device('cuda'))

        target = self.embedding_layer(target)
        target = self.position_encoding(target)

        att_weights = None
        for block in self.blocks:
            target, att = block(encoded_source, target, mask1, source_target_padding_mask)
            if att_weights is None:
                att_weights = att

        y = self.linear(target)
        return y, att_weights


class Transformer(Module):

    def __init__(self, source_vocab_size: int, target_vocab_size: int, embedding_dim: int, n_encoder_blocks: int,
                 n_decoder_blocks: int, n_heads: int) -> None:
        super().__init__()

        self.encoder = Encoder(source_vocab_size, embedding_dim, n_encoder_blocks, n_heads)
        self.decoder = Decoder(target_vocab_size, embedding_dim, n_decoder_blocks, n_heads)

    def forward(self, source, target):
        encoded_source, source_padding = self.encoder(source)
        return self.decoder(encoded_source, source_padding, target)

    def get_best_inds(self, b, vals):
        vals_flat = vals.flatten()
        best_inds = np.argsort(vals_flat)[-b:]
        lls = vals_flat[best_inds]
        val_inds = [[i//vals.shape[0], i%vals.shape[0]] for i in best_inds]

        return val_inds, lls


    def predict(self, source: List[int], beam_size=1, max_length=12) -> List[int]:
        """
        Given a sentence in the source language, you should output a sentence in the target
        language of length at most `max_length` that you generate using a beam search with
        the given `beam_size`.

        Note that the start of sentence token is 0 and the end of sentence token is 1.

        Return the final top beam (decided using average log-likelihood) and its average
        log-likelihood.

        Hint: The follow functions may be useful:
            - torch.topk (https://pytorch.org/docs/stable/generated/torch.topk.html)
            - torch.softmax (https://pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html)
        """
        self.eval() # Set the PyTorch Module to inference mode (this affects things like dropout)
        
        if not isinstance(source, torch.Tensor):
            source_input = torch.tensor(source).view(1, -1)
        else:
            source_input = source.view(1, -1)
        
        i = 0

        log_likes = np.zeros(beam_size)

        encoded_source, source_padding = self.encoder(source_input)

        encoded_source_beam = encoded_source.repeat(beam_size, 1, 1)
        source_padding_beam = source_padding.repeat(beam_size, 1)

        final_beams = torch.zeros([beam_size, max_length], dtype=int)
        final_lengths = np.zeros(beam_size)
        final_log_likes = np.zeros(beam_size)
        fb_ind = 0

        while i < max_length-1:
            if i == 0:
                step, _ = self.decoder(encoded_source, source_padding, torch.zeros([1, 1], dtype=int))
            else:
                temp1 = encoded_source.repeat(beam_size-fb_ind, 1, 1)
                temp2 = source_padding.repeat(beam_size-fb_ind, 1)
                step, _ = self.decoder(temp1, temp2, beam)

            step_norm = torch.softmax(step, 2)

            # sn_db = step_norm.detach().numpy()

            step_probs = torch.log(step_norm)
            step_probs = step_probs[:, step_probs.size()[1]-1, :]

            # sp_db = step_probs.detach().numpy()

            top_k = torch.topk(step_probs, beam_size-fb_ind, 1)

            if i == 0:
                beam = top_k.indices.reshape(-1,1)
                log_likes = top_k.values.detach().numpy().T
                start = torch.zeros(beam.size(), dtype=int)
                beam = torch.cat((start, beam), 1)
            else:
                curr_log_likes = top_k.values.detach() + log_likes
                best_inds, best_lls = self.get_best_inds(beam_size-fb_ind, curr_log_likes.detach().numpy())
                new_words = []

                ongoing_beams = []
                finished_beams = []

                new_beam = beam.detach().clone()

                for j,b in enumerate(best_inds):
                    new_beam[j, :] = beam[b[0], :]
                    log_likes[j] = best_lls[j]
                    new_words.append(top_k.indices[b[0], b[1]])

                    if new_words[-1] != 1:
                        ongoing_beams.append(j)
                    else:
                        finished_beams.append(j)

                beam = torch.cat((new_beam, torch.tensor(new_words, dtype=int).reshape(-1, 1)), 1)

                for b in finished_beams:
                    final_beams[fb_ind, :beam.size()[1]] = beam[b, :]
                    final_lengths[fb_ind] = i+1
                    final_log_likes[fb_ind] = log_likes[b]
                    fb_ind += 1

                if fb_ind == beam_size:
                    i = max_length
                else:
                    beam = beam[ongoing_beams, :]
                    log_likes = log_likes[ongoing_beams]

            i += 1

        for b in ongoing_beams:
            final_beams[fb_ind, :beam.size()[1]] = beam[b, :]
            final_lengths[fb_ind] = i
            final_log_likes[fb_ind] = log_likes[b]
            fb_ind += 1

        final_lengths += 1
        final_probs = final_log_likes/final_lengths
        best_beam = np.argmax(final_probs)

        return final_beams[best_beam, :int(final_lengths[best_beam])], final_probs[best_beam]


def load_data() -> Tuple[Tuple[List[int], List[int]], Tuple[List[int], List[int]], Dict[int, str], Dict[int, str]]:
    """ Load the dataset.

    :return: (1) train_sentences: list of (source_sentence, target_sentence) pairs, where both source_sentence
                                  and target_sentence are lists of ints
             (2) test_sentences : list of (source_sentence, target_sentence) pairs, where both source_sentence
                                  and target_sentence are lists of ints
             (2) source_vocab   : dictionary which maps from source word index to source word
             (3) target_vocab   : dictionary which maps from target word index to target word
    """
    with open('data/translation_data.bin', 'rb') as f:
        corpus, source_vocab, target_vocab = pickle.load(f)
        test_sentences = corpus[:1000]
        train_sentences = corpus[1000:]
        print("# source vocab: {}\n"
              "# target vocab: {}\n"
              "# train sentences: {}\n"
              "# test sentences: {}\n".format(len(source_vocab), len(target_vocab), len(train_sentences),
                                              len(test_sentences)))
        return train_sentences, test_sentences, source_vocab, target_vocab


def preprocess_data(sentences: Tuple[List[int], List[int]], source_vocab_size,
                    target_vocab_size, max_length):
    
    source_sentences = []
    target_sentences = []

    for source, target in sentences:
        source = [0] + source + ([source_vocab_size] * (max_length - len(source) - 1))
        target = [0] + target + ([target_vocab_size] * (max_length - len(target) - 1))
        source_sentences.append(source)
        target_sentences.append(target)

    return torch.tensor(source_sentences), torch.tensor(target_sentences)


def decode_sentence(encoded_sentence: List[int], vocab: Dict) -> str:
    if isinstance(encoded_sentence, torch.Tensor):
        encoded_sentence = [w.item() for w in encoded_sentence]
    words = [vocab[w] for w in encoded_sentence if w != 0 and w != 1 and w in vocab]
    return " ".join(words)


def visualize_attention(source_sentence: List[int],
                        output_sentence: List[int],
                        source_vocab: Dict[int, str],
                        target_vocab: Dict[int, str],
                        attention_matrix: np.ndarray, title=None):
    """
    :param source_sentence_str: the source sentence, as a list of ints
    :param output_sentence_str: the target sentence, as a list of ints
    :param attention_matrix: the attention matrix, of dimension [target_sentence_len x source_sentence_len]
    :param outfile: the file to output to
    """
    source_length = 0
    while source_length < len(source_sentence) and source_sentence[source_length] != 1:
        source_length += 1

    target_length = 0
    while target_length < len(output_sentence) and output_sentence[target_length] != 1:
        target_length += 1

    # Set up figure with colorbar
    fig = plt.figure()

    ax = fig.add_subplot(111)
    cax = ax.matshow(attention_matrix[:target_length, :source_length], cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.xaxis.set_major_locator(ticker.FixedLocator(range(source_length)))
    ax.xaxis.set_major_formatter(ticker.FixedFormatter(["PAD" if x not in source_vocab else source_vocab[x] for x in source_sentence[:source_length]]))
    ax.yaxis.set_major_locator(ticker.FixedLocator(range(target_length)))
    ax.yaxis.set_major_formatter(ticker.FixedFormatter(["PAD" if x not in target_vocab else target_vocab[x] for x in output_sentence[:target_length]]))

    plt.savefig('figures/' + title + '.jpeg')
    plt.close()


def train(model: Transformer, train_source: torch.Tensor, train_target: torch.Tensor,
          test_source: torch.Tensor, test_target: torch.Tensor, target_vocab_size: int,
          epochs: int = 30, batch_size: int = 64, lr: float = 0.0001, m_name=''):

    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = CrossEntropyLoss(ignore_index=target_vocab_size)

    epoch_train_loss = np.zeros(epochs)
    epoch_test_loss = np.zeros(epochs)

    for ep in range(epochs):

        train_loss = 0
        test_loss = 0

        permutation = torch.randperm(train_source.shape[0])
        train_source = train_source[permutation]
        train_target = train_target[permutation]

        batches = train_source.shape[0] // batch_size
        model.train()
        for ba in tqdm(range(batches), desc=f"Epoch {ep + 1}"):

            optimizer.zero_grad()

            batch_source = train_source[ba * batch_size: (ba + 1) * batch_size]
            batch_target = train_target[ba * batch_size: (ba + 1) * batch_size]

            target_pred, _ = model(batch_source, batch_target)

            batch_loss = loss_fn(target_pred[:, :-1, :].transpose(1, 2), batch_target[:, 1:])
            batch_loss.backward()
            optimizer.step()

            train_loss += batch_loss.item()

        test_batches = test_source.shape[0] // batch_size
        model.eval()
        for ba in tqdm(range(test_batches), desc="Test", leave=False):

            batch_source = test_source[ba * batch_size: (ba + 1) * batch_size]
            batch_target = test_target[ba * batch_size: (ba + 1) * batch_size]

            target_pred, _ = model(batch_source, batch_target)

            batch_loss = loss_fn(target_pred[:, :-1, :].transpose(1, 2), batch_target[:, 1:])
            test_loss += batch_loss.item()

        epoch_train_loss[ep] = train_loss / batches
        epoch_test_loss[ep] = test_loss / test_batches
        print(f"Epoch {ep + 1}: Train loss = {epoch_train_loss[ep]:.4f}, Test loss = {epoch_test_loss[ep]:.4f}")

    torch.save(model.state_dict(), m_name+'.pkl')
    return epoch_train_loss, epoch_test_loss


def get_n_grams(l, n):
    i = 0
    n_grams = []

    while i < len(l)-n+1:
        g = ''
        for j in range(i, i+n):
            g += str(l[j]) + ','
        n_grams.append(g[:-1])
        i += 1

    return n_grams


def get_gram_dict(grams):
    g_dict = {}

    for g in grams:
        if g not in g_dict:
            g_dict[g] = 1
        else:
            g_dict[g] += 1

    return g_dict


def p_score_n(predicted, target, n):
    score = 0

    p_grams = get_n_grams(predicted, n)
    t_grams = get_n_grams(target, n)

    p_gram_counts = get_gram_dict(p_grams)
    t_gram_counts = get_gram_dict(t_grams)

    for p, ct in p_gram_counts.items():
        if p in t_gram_counts:
            d1 = ct
            d2 = t_gram_counts[p]
            score += min(ct, t_gram_counts[p])

    return score / (len(predicted) - n + 1)


def strip_list(l):
    new_l = []
    going = True

    for v in l:
        if v == 1:
            going = False
        if v != 0 and going:
            new_l.append(v)

    return new_l


def bleu_score(predicted: List[int], target: List[int], N: int = 4) -> float:
    """
    *** For students in 10-617 only ***
    (Students in 10-417, you can leave `raise NotImplementedError()`)

    Implement a function to compute the BLEU-N score of the predicted
    sentence with a single reference (target) sentence.

    Please refer to the handout for details.

    Make sure you strip the SOS (0), EOS (1), and padding (anything after EOS)
    from the predicted and target sentences.
    
    If the length of the predicted sentence or the target is less than N,
    the BLEU score is 0.
    """
    score = 1

    pred = strip_list(predicted)
    targ = strip_list(target)

    if len(pred) < N or len(targ) < N:
        return 0

    for k in range(1, N+1):
        temp = (p_score_n(pred, targ, k))
        temp = temp**(1/N)
        score *= temp

    brev = np.exp(1 - float(len(targ))/len(pred))

    return score*min(1, brev)


if __name__ == "__main__":
    # train_sentences, test_sentences, source_vocab, target_vocab = load_data()

    with open('dead_flowers_v5.obj', 'rb') as f:
        data = pickle.load(f)
    
    df, _ = preprocess_data(data['data'], 52, 52, 90)

    model = Transformer(52, 52, 256, 2, 4, 3)
    model.load_state_dict(torch.load('paul_15e_v5.pkl', map_location=torch.device('cpu')))

    for i in range(len(df)):
        source = [int(j) for j in list(df[i])]

        pred_t, ll = model.predict(source, beam_size=3)

        print('hi')


    # P3 CODE
    # p3_source = test_source[:8]
    # p3_target = test_target[:8]
    #
    # for i in range(8):
    #     source = [int(j) for j in list(p3_source[i])]
    #     target = [int(j) for j in list(p3_target[i])]
    #
    #     pred_t, ll = model.predict(source, beam_size=3)
    #
    #     temp = list(pred_t)
    #     pred_t = [int(j) for j in list(pred_t)]
    #
    #     pred_sentence = decode_sentence(pred_t, target_vocab)
    #     true_sentence = decode_sentence(target, target_vocab)
    #     orig_sentence = decode_sentence(source, source_vocab)
    #
    #     print('Orig: ' + orig_sentence)
    #     print('Pred: ' + pred_sentence)
    #     print('True: ' + true_sentence)
    #     print('Avg. Log-Likelihood: %f \n' % ll)
    a = None

    # P4 Code
    # p4_source = train_source[:3]
    # p4_target = train_target[:3]
    #
    # for i in range(3):
    #     model.train()
    #
    #     pred, attn = model(p4_source[i].reshape(1, -1), p4_target[i].reshape(1, -1))
    #
    #     source_sentence = [int(j) for j in list(p4_source[i])]
    #
    #     pred_sentence = torch.argmax(pred, 2)
    #     output_sentence = [int(j) for j in list(pred_sentence[0])]
    #
    #     for j in range(3):
    #         title = 'Train sentence %i, head %i' % (i, j)
    #         visualize_attention(source_sentence, output_sentence, source_vocab, target_vocab, attn[0, j].detach(), title=title)
    a = None

    # P5 Code
    # ll_beam = np.zeros((100, 8))
    # p5_source = test_source[:100]
    #
    # for i in range(100):
    #     source = [int(j) for j in list(p5_source[i])]
    #
    #     for j in range(1,9):
    #         pred, ll = model.predict(source, beam_size=j)
    #         ll_beam[i, j-1] = ll
    #
    # np.savetxt('p5_ll_beam.csv', ll_beam, delimiter=',')
    a = None
    # P6 Code
    # model_labels = [[1,1,1], [1,1,3], [2,2,1], [2,2,3], [2,4,3]]
    # bleu_k = np.zeros((5,4))
    #
    # for j, m in enumerate(model_labels):
    #     m1 = Transformer(len(source_vocab), len(target_vocab), 256, m[0], m[1], m[2])
    #     m_str = [str(i) for i in m]
    #
    #     m_label = ''.join(m_str)
    #
    #     m1.load_state_dict(torch.load(m_label + '.pkl'))
    #
    #     for i in range(1000):
    #         source = [int(j) for j in list(test_source[i])]
    #         target = [int(j) for j in list(test_target[i])]
    #         pred, _ = m1.predict(source, beam_size=3)
    #
    #         pred = [int(j) for j in list(pred)]
    #
    #         for k in range(1, 5):
    #             temp = bleu_score(pred, target, k)
    #             bleu_k[j, k-1] += temp
    #
    # bleu_k /= 1000
    #
    # np.savetxt('bleu_k.csv', bleu_k, delimiter=',')
    a = None
    # model = Transformer(len(source_vocab), len(target_vocab), 256, 1, 1, 1)
    #
    # train_loss, test_loss = train(model, train_source, train_target, test_source, test_target, len(target_vocab), m_name='111')
    #
    # np.savetxt('111.csv', np.vstack((train_loss, test_loss)), delimiter=',')
    #
    #
    # model = Transformer(len(source_vocab), len(target_vocab), 256, 1, 1, 3)
    # train_loss, test_loss = train(model, train_source, train_target, test_source, test_target, len(target_vocab),
    #                               m_name='113')
    # np.savetxt('113.csv', np.vstack((train_loss, test_loss)), delimiter=',')
    #
    #
    # model = Transformer(len(source_vocab), len(target_vocab), 256, 2, 2, 1)
    # train_loss, test_loss = train(model, train_source, train_target, test_source, test_target, len(target_vocab),
    #                               m_name='221')
    # np.savetxt('221.csv', np.vstack((train_loss, test_loss)), delimiter=',')
    #
    #
    # model = Transformer(len(source_vocab), len(target_vocab), 256, 2, 2, 3)
    # train_loss, test_loss = train(model, train_source, train_target, test_source, test_target, len(target_vocab),
    #                               m_name='223')
    # np.savetxt('223.csv', np.vstack((train_loss, test_loss)), delimiter=',')
    #
    #
    # model = Transformer(len(source_vocab), len(target_vocab), 256, 2, 4, 3)
    # train_loss, test_loss = train(model, train_source, train_target, test_source, test_target, len(target_vocab),
    #                               m_name='243')
    # np.savetxt('243.csv', np.vstack((train_loss, test_loss)), delimiter=',')
