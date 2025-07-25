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
    candidates = [([], 1)]

    for i in range(out.shape[1]):
        new_candidates = []
        
        for candidate, score in candidates:
            top_k_prob, top_k_idx = torch.topk(score + log_softmax(out[i], dim=-1), beam_size, dim=-1)
            
            for i in range(beam_size):
                new_candidate = list(candidate) + [top_k_idx[i].item()]
                new_score = top_k_prob[i].item()

                new_candidates.append((new_candidate, new_score))
            
        new_candidates = sorted(new_candidates, key=lambda x: x[1], reverse=True)[:beam_size]
        candidates = new_candidates
    
    out, _ = candidates[0]
    out = list(filter(lambda x: x != 0, [letter for letter, _ in itertools.groupby(out)]))
    return ''.join([idx_to_char[elem] for elem in out])
    