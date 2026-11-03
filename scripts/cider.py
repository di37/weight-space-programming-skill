"""CIDEr-D (Vedantam et al. 2015), the COCO caption metric, computed as the official pycocoevalcap scorer does.

Pure Python, no Java. Words are lower-cased alphanumeric runs; the official PTB tokenizer also strips punctuation, so
scores are close to, but not bit-identical with, leaderboard numbers. IDF comes from the references of the scored set,
so score a whole split with one CiderD; a set of one image scores 0. Scores are on the official scale (a good
captioner averages about 1.0 on COCO; papers print 100 times that).
"""
import math
import re
from collections import Counter

N, SIGMA = 4, 6.0
WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def tokens(text):
    return WORD.findall(str(text).lower())


def ngrams(words):
    return Counter(tuple(words[i:i + k]) for k in range(1, N + 1) for i in range(len(words) - k + 1))


class CiderD:
    def __init__(self, references):
        """references: one list of reference captions per image of the scored set."""
        cooked = [[ngrams(tokens(r)) for r in refs] for refs in references]
        self.df = Counter(g for refs in cooked for g in {g for ref in refs for g in ref})
        self.log_images = math.log(max(len(cooked), 1))

    def _vector(self, counts):
        vec, norm = [{} for _ in range(N)], [0.0] * N
        for gram, tf in counts.items():
            weight = tf * (self.log_images - math.log(max(1.0, self.df[gram])))
            vec[len(gram) - 1][gram] = weight
            norm[len(gram) - 1] += weight * weight
        # The official scorer counts length in bigrams (its `n == 1` test is on the 0-based n-gram index). Kept, so
        # the length penalty and therefore the scores match it.
        length = sum(tf for gram, tf in counts.items() if len(gram) == 2)
        return vec, [math.sqrt(x) for x in norm], length

    def _similarity(self, hyp, ref):
        (vh, nh, lh), (vr, nr, lr) = hyp, ref
        penalty = math.exp(-((lh - lr) ** 2) / (2 * SIGMA ** 2))
        total = 0.0
        for n in range(N):
            val = sum(min(w, vr[n].get(g, 0.0)) * vr[n].get(g, 0.0) for g, w in vh[n].items())
            if nh[n] and nr[n]:
                val /= nh[n] * nr[n]
            total += val * penalty
        return total / N

    def score(self, hypothesis, references):
        """CIDEr-D of one caption against its image's references."""
        if not references:
            return 0.0
        hyp = self._vector(ngrams(tokens(hypothesis or "")))
        sims = [self._similarity(hyp, self._vector(ngrams(tokens(r)))) for r in references]
        return 10.0 * sum(sims) / len(references)
