import itertools
import torch
from torch.nn.functional import log_softmax, softmax

def ctc_greedy(out, idx_to_char):
    out = out.permute(1, 0, 2)
    out = softmax(out, dim=-1)
    out = torch.argmax(out, dim=-1)[0].tolist()
    out = list(filter(lambda x: x != 0, [letter for letter, _ in itertools.groupby(out)]))
    return ''.join([idx_to_char[elem] for elem in out])

def ctc_beam_search(out, idx_to_char, beam_size=3):
    out = out.permute(1, 0, 2)[0]
    out = log_softmax(out, dim=-1)
    T, C = out.shape

    candidates = [([], (0,0))]

    # Go through all time steps
    for t in range(T):
        new_candidates = []
        
        # Go through the best candidates
        for candidate, score in candidates:

            # Iterate through all tokens in lexicon
            for c in range(C):
                new_candidate = list(candidate) + [c]
                new_score = score * out[t, c].item()

                new_candidates.append((new_candidate, new_score))

        new_candidates = sorted(new_candidates, key=lambda x: x[1], reverse=True)[:beam_size]
        candidates = new_candidates
    
    out, _ = candidates[0]
    out = list(filter(lambda x: x != 0, [letter for letter, _ in itertools.groupby(out)]))
    return ''.join([idx_to_char[elem] for elem in out])

def log_sum_exp(out):
  """
  Stable log sum exp.
  """
  if torch.all(out == -torch.inf):
      print('wowwie')
      return -torch.inf

  a_max = torch.max(out)
  lsp = torch.log(torch.exp(out - a_max).sum(dim=-1))
  return a_max + lsp

log_sum_exp(torch.tensor([1, 2, 3, 5]))