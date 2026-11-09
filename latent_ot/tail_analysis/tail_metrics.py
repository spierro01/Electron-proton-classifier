"""Fixed-cut signal tail fractions from three ordered classifier logits."""
import numpy as np


def discriminant(logits, class_index):
    x = np.asarray(logits,dtype=np.float64)
    if x.ndim!=2 or x.shape[1]!=3 or class_index not in (0,1,2) or not np.isfinite(x).all():
        raise ValueError('Expected finite three-class logits and a valid class index.')
    j,k = [i for i in range(3) if i!=class_index]
    return x[:,class_index] - np.logaddexp(x[:,j],x[:,k]) + np.log(2.)


def tail_counts(scores, cut):
    scores=np.asarray(scores)
    if scores.ndim!=1 or not scores.size or not np.isfinite(scores).all() or not np.isfinite(cut):
        raise ValueError('Expected a nonempty finite score vector and a finite cut.')
    n=scores.size
    k=int(np.count_nonzero(scores<=cut))
    return k,n,k/n
