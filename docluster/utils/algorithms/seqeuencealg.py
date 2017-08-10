def n_gram(sequence, n_groups):
    return zip(*[sequence[i:] for i in range(n_groups)])
