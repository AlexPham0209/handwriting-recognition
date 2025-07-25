import itertools
import torch
from torch.nn.functional import log_softmax, softmax

def ctc_greedy_decoding(out, idx_to_char):
    out = out.permute(1, 0, 2)
    out = softmax(out, dim=2)
    out = torch.argmax(out, dim=-1)[0].tolist()
    out = list(filter(lambda x: x != 0, [letter for letter, _ in itertools.groupby(out)]))
    return ''.join([idx_to_char[elem] for elem in out])
