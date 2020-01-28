from collections import defaultdict, deque
from itertools import product
from typing import Dict, List, NamedTuple, Sequence, Set, Tuple, Union

from .lexer import Lexer, Option, OptionWord, Token
from .rules import Nomination, RuleInfo
from .weights import Weights


class Interpretation(NamedTuple):
    """
    For a given token, gives the list of possible interpretations
    """

    # Token referred to
    token: Token

    # The first tuple holds a list of all possible interpretations while each
    # tuple inside holds a list of potential nominations to be used
    # consecutively.
    nominations: Tuple[Tuple[Nomination, ...], ...]


class NoMatch:
    """
    Special object to indicate the lack of match
    """

    def __repr__(self):
        return "NoMatch()"


WordMatch = Union[NoMatch, Nomination]


class Parser:
    """
    Core engine of NLU parsing
    """

    def __init__(
        self, lexer: Lexer, rules: Sequence[RuleInfo], weights: Weights = Weights()
    ):
        self.lexer = lexer
        self.rules = {i.name: i for i in rules}
        self.weights = weights

    def nominate(self, tokens: Sequence[Token]) -> Sequence[Interpretation]:
        """
        Provided a list of tokens, ask the rules to nominate each word that
        they deem interesting.

        Notes
        -----
        This function is way too complex for me to like it, unfortunately I
        don't really see how to simplify it.

        The first step is to take all words and pass them through the rules to
        know which ones are nominated.

        Then all of that is aggregated in a succession of interpretations, one
        for each token from the input, in the same order.

        Parameters
        ----------
        tokens
            List of tokens to be nominated

        Returns
        -------
        All the nominations for the provided tokens
        """

        words = [w for t in tokens for o in t.options for w in o.words]
        nominations: Dict[
            Token, Dict[Option, Dict[OptionWord, List[Nomination]]]
        ] = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [])))

        for rw in self.rules.values():
            nom: Nomination
            for nom in rw.rule.nominate_words(words):
                nominations[nom.word.option.token][nom.word.option][nom.word].append(
                    nom
                )

        out = deque()

        for token in tokens:
            token_nom = deque()

            for option in token.options:
                word_nom: List[List[WordMatch]] = [[NoMatch()]] * len(option.words)

                for i, word in enumerate(option.words):
                    word_nom[i].extend(nominations[token][option][word])

                token_nom.extend(
                    p
                    for p in product(*word_nom)
                    if not all(isinstance(w, NoMatch) for w in p)
                )

            out.append(
                Interpretation(
                    token=token, nominations=tuple(tuple(x) for x in token_nom)
                )
            )

        return tuple(out)
