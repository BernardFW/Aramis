# coding: utf-8
"""
Trigram computation utils. Although the algorithm are pretty different, the
code here is inspired from PostgreSQL's pg_trgm module and should give similar
or identical results.
"""
import re
from collections import deque
from typing import Iterable, Optional, Tuple, TypeVar

RE_WHITESPACES = re.compile(r'[\W.,;?!\'"«»\-_\s]+')

T = TypeVar("T")


def make_trigrams(
    i: Iterable[T],
) -> Iterable[Tuple[Optional[T], Optional[T], Optional[T]]]:
    """
    Compute all trigrams of an iterable and yield them. You probably want
    to do something like:

    >>> t = set(make_trigrams('hi there'))
    """
    q = deque([None, None, None])

    def nxt():
        q.append(x)
        q.popleft()
        return tuple(c if c is not None else " " for c in q)

    for x in i:
        yield nxt()

    if q[-1] is not None:
        x = None
        yield nxt()


class Trigram(object):
    """
    This represents a "compiled" trigram object. It is able to compute its
    similarity with other trigram objects.
    """

    def __init__(self, string):
        self._string = string
        self._trigrams = set(make_trigrams(self._string))

    def __repr__(self):
        return f"Trigram({repr(self._string)})"

    def similarity(self, other: "Trigram") -> float:
        """
        Compute the similarity with the provided other trigram.
        """
        if not len(self._trigrams) or not len(other._trigrams):
            return 0

        count = float(len(self._trigrams & other._trigrams))
        len1 = float(len(self._trigrams))
        len2 = float(len(other._trigrams))

        return count / (len1 + len2 - count)

    def __mod__(self, other: "Trigram") -> float:
        """
        Shortcut notation using modulo symbol.
        """
        return self.similarity(other)
