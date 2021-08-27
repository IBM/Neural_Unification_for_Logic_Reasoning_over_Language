import torch
import torch.nn.functional as F


# sentences is a list of sentence_embedding
# sentence_embedding  a tensor of Size [s, d], where s is the number of tokens in this sentence
# H max number of sentences in the Batch
# W max number of tokens in Batch sentences
# d transformer embedding dim
def padding_sentences(sentences):
    w = max([s.size()[0] for s in sentences])
    padded_sentences = []
    masks = []
    for s in sentences:
        # from [s, d] to [w, d]
        pad_size = w-s.size()[0]
        mask = torch.ones(s.size()[0])
        if pad_size > 0:
            masks += [F.pad(mask, (0, pad_size), value=0).unsqueeze(0)]
            padded_sentences += [F.pad(s, (0, pad_size), value=0).unsqueeze(0)]
        else:
            masks += [mask.unsqueeze(0)]
            padded_sentences += [s.unsqueeze(0)]

    return torch.cat(padded_sentences), torch.cat(masks)


# sentences is a list of sentence_embedding
# sentence_embedding  a tensor of Size [s, d], where s is the number of tokens in this sentence
# H max number of sentences in the Batch
# W max number of tokens in Batch sentences
# d transformer embedding dim
def padding_sentence_embeddings(sentences):
    w = max([s.size()[0] for s in sentences])
    padded_sentences = []
    masks = []
    for s in sentences:
        # from [s, d] to [w, d]
        pad_size = w-s.size()[0]
        mask = torch.ones(s.size()[0])
        if pad_size > 0:
            masks += [F.pad(mask, (0, pad_size), value=0).unsqueeze(0)]
            padded_sentences += [F.pad(s, (0, 0, 0, pad_size), value=0).unsqueeze(0)]
        else:
            masks += [mask.unsqueeze(0)]
            padded_sentences += [s.unsqueeze(0)]

    return torch.cat(padded_sentences), torch.cat(masks)


def grad_stats(m):
    s = 0.0
    mean = 0.0
    i = 0
    for p in m.parameters():
        if p.grad is not None:
            s += torch.max(torch.abs(p.grad)).item()
            mean += torch.mean(p).item()
            i += 1

    if i != 0:
        return s/i
    else:
        return 0.0


def weight_stats(m):
    s = 0.0
    i = 0
    for p in m.parameters():
            s += torch.mean(torch.abs(p)).item()
            i += 1
    if i != 0:
        return s/i, i
    else:
        return 0.0, 0


def nearest(w, weights):
        return torch.argmin(torch.mean(torch.abs(weights - w), dim=1)).item()


def nearest_indices(w, weights, indices):
    m_idx = -1
    x = None
    for idx in indices:
        y = torch.mean(torch.abs(weights[idx] - w)).item()
        if x is None:
            x = y
            m_idx = idx
        elif x > y:
            x = y
            m_idx = idx
    return m_idx


def get_first_segment(segments):
    idx = 0
    for i in range(len(segments)):
        if segments[i] == 0:
            idx = i
        else:
            return idx
    return idx


def decode_nearest(ws, weights, tokenizer, mask, segments=None):
    token_ids = []
    if segments is not None:
        idx = get_first_segment(segments)
        indices = []
        for i in range(idx):
            w = ws[i]
            indices.append(nearest(w, weights))

    for i in range(len(ws)):
        w = ws[i]
        if mask[i].item() != 0.0:
            if segments is None:
                token_ids.append(nearest(w, weights))
            else:
                if i <= idx:
                    token_ids.append(nearest(w, weights))
                else:
                    token_ids.append(nearest_indices(w, weights, indices))
    return tokenizer.decode(token_ids)