import itertools
import collections
import torch
from torch.nn.functional import log_softmax, softmax

def ctc_greedy(out, idx_to_char):
    out = out.permute(1, 0, 2)
    out = softmax(out, dim=-1)
    out = torch.argmax(out, dim=-1)[0].tolist()
    out = list(filter(lambda x: x != 0, [letter for letter, _ in itertools.groupby(out)]))
    return ''.join([idx_to_char[elem] for elem in out])

def ctc_beam_search(out, idx_to_char, beam_size=8, blank=0):
    out = out.permute(1, 0, 2)[0]
    out = log_softmax(out, dim=-1)
    T, S = out.shape

    # (Candidate, total, blank, non_blank)
    candidates = [(tuple(), (0, -torch.inf))]

    # Go through all time steps
    for t in range(T):
        new_candidates = make_new_beam()
        
        for s in range(S):
            p = out[t, s]

            for candidate, (p_b, p_nb) in candidates:
                if s == blank:
                    n_p_b, n_p_nb = new_candidates[candidate]
                    n_p_b = torch.logsumexp(torch.tensor([n_p_b, p_b + p, p_nb + p]), dim=-1)
                    new_candidates[candidate] = (n_p_b, n_p_nb)
                    continue
                
                end_t = candidate[-1] if candidate else None
                next_candidate = candidate + (s,)
                n_p_b, n_p_nb = new_candidates[next_candidate]

                if s != end_t:
                    n_p_nb = torch.logsumexp(torch.tensor([n_p_nb, p_b + p, p_nb + p]), dim=-1)
                else:
                    n_p_nb = torch.logsumexp(torch.tensor([n_p_nb, p_b + p]), dim=-1)

                new_candidates[next_candidate] = (n_p_b, n_p_nb)

                if s == end_t:
                    n_p_b, n_p_nb = new_candidates[candidate]
                    n_p_nb = torch.logsumexp(torch.tensor([n_p_nb, p_nb + p]), dim=-1)
                    new_candidates[candidate] = (n_p_b, n_p_nb)

        candidates = sorted(new_candidates.items(),
            key=lambda x : torch.logsumexp(torch.tensor([x[1][0], x[1][1]]), dim=-1),
            reverse=True
        )
        candidates = candidates[:beam_size]
    
    out, _ = list(candidates[0])
    out = list(filter(lambda x: x != 0, [letter for letter, _ in itertools.groupby(out)]))
    return ''.join([idx_to_char[elem] for elem in out])

# def log_sum_exp(out):
#   """
#   Stable log sum exp.
#   """
#   if torch.all(out == -torch.inf):
#       print('wowwie')
#       return -torch.inf

#   a_max = torch.max(out)
#   lsp = torch.log(torch.exp(out - a_max).sum(dim=-1))
#   return a_max + lsp

def make_new_beam():
  fn = lambda : (-torch.inf, -torch.inf)
  return collections.defaultdict(fn)
