from collections import defaultdict, deque
from dataclasses import dataclass
from itertools import product
from math import sqrt
from typing import Dict, List, NamedTuple, Sequence, Text, Tuple

from scipy.optimize import OptimizeResult, shgo

from .lexer import Lexer, Option, OptionWord, Token
from .rules import NoMatch, Nomination, RuleInfo, WordMatch
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


@dataclass(frozen=True)
class Match:
    """
    Output of the optimization. The score is a number between 0 and 1, 1 being
    a perfect match and 0 is a no-go.
    """

    # 0 to 1 score (1 is the best)
    score: float

    # Sequence of matches, in the same order as input tokens
    matched: Sequence[WordMatch]


class _Optimizer:
    """
    Helper class that exists solely to easily provide callables to optimization
    functions from SciPy.
    """

    def __init__(self, interpretations: Sequence[Interpretation], parser: "Parser"):
        self.interpretations = interpretations
        self.parser = parser

    @property
    def bounds(self) -> List[Tuple[float, float]]:
        """
        Value bounds for the interpretations

        Returns
        -------
        Couples that include the indices for different interpretations
        """

        return [(-0.1, len(x.nominations) + 1) for x in self.interpretations]

    def optimize(self) -> Match:
        """
        Starts the optimization process and translates into a Match()
        """

        opt: OptimizeResult = shgo(func=self.score, bounds=self.bounds)

        if not opt["success"]:
            return Match(
                score=0.0,
                matched=[NoMatch() for _ in range(0, len(self.interpretations))],
            )

        score = opt["fun"]
        max_score = sqrt(sum(r.weight ** 2 for r in self.parser.rules.values()))

        return Match(
            score=max([0, (max_score - score) / max_score]),
            matched=self._get_selection(opt["x"]),
        )

    def score(self, x, *_) -> float:
        """
        Generates the score for a given of indices

        Notes
        -----
        There is an "extra" component of the score which helps the minimizer
        fall on round values (because our function takes integers but the
        optimizer looks for float values).

        Parameters
        ----------
        x
            Indices for each interpretation
        _
            Trash
        """

        selection = self._get_selection(x)
        extra = sqrt(sum((v - int(v)) ** 2 for v in x))
        return self.parser.score(selection) + extra

    def _get_selection(self, x) -> Sequence[Nomination]:
        """
        Given a set of indices, returns the corresponding selection of
        arguments.

        Notes
        -----
        Since the optimizer doesn't seem to care testing only integer values,
        the values found in `x` are changed to the nearest index value.

        Parameters
        ----------
        x
            Coordinates of interpretations to be tried

        Returns
        -------
        The nominations matching the input array
        """

        selection = []

        for i, pos in enumerate(x):
            if pos < 0:
                pos = 0

            if pos >= len(self.interpretations[i].nominations):
                val = [NoMatch()]
            else:
                val = self.interpretations[i].nominations[int(pos)]

            selection.extend(val)

        return selection


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

    def _make_nominations(
        self, tokens: Sequence[Token]
    ) -> Dict[Token, Dict[Option, Dict[OptionWord, List[Nomination]]]]:
        """
        Runs all the tokens through the rules in order to extract the
        nominations indexed in a dictionary that allows to find the nominations
        by token/option/word.

        Parameters
        ----------
        tokens
            List of the tokens you want to analyze

        Returns
        -------
        Indexed tokens
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

        return nominations

    def _order_nominations(
        self,
        nominations: Dict[Token, Dict[Option, Dict[OptionWord, List[Nomination]]]],
        out: deque,
        tokens: Sequence[Token],
    ) -> None:
        """
        Given the generated nominations list and tokens sequence, generates
        for each token the different nomination sequences to be considered.

        Parameters
        ----------
        nominations
            Indexed nominations
        out
            Output list to be filled. Each new row corresponds to a given
            token.
        tokens
            List of tokens for which the nomination options have to be
            generated
        """

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

        out = deque()
        nominations = self._make_nominations(tokens)
        self._order_nominations(nominations, out, tokens)

        return tuple(out)

    def score(self, nominations: Sequence[WordMatch]) -> float:
        """
        Applies all the grammar rules and computes the resulting score.
        It is considered that every grammar rule is an euclidean coordinate
        in its own dimension and the resulting score is the distance from the
        origin point.

        Parameters
        ----------
        nominations
            Sequence of nominations to be evaluated

        Returns
        -------
        A score for this sequence. The lowest score means that the sequence
        makes more sense.
        """

        total = []

        rule: RuleInfo
        for rule in self.rules.values():
            score = rule.rule.evaluate(nominations)

            if score is None:
                score = self.weights.rule_miss_penalty

            total.append(score * rule.weight)

        return sqrt(sum(s ** 2 for s in total))

    def optimize(self, interpretations: Sequence[Interpretation]) -> Match:
        """
        Runs the optimization algorithms, see the _Optimizer class for more
        information.

        Parameters
        ----------
        interpretations
            Sequence of interpretations to optimize

        Returns
        -------
        Returns the match object, holding all information about what was
        matched. If the score is low (or 0) it means that the match was not
        a success while the closest it is to 1 the best the match is.
        """

        optimizer = _Optimizer(interpretations, parser=self)
        return optimizer.optimize()

    def parse(self, text: Text) -> Match:
        """
        Runs all parsing operations in order.

        See Also
        --------
        aramis.lexer.Lexer.process
            The function in charge of turning words into tokens
        aramis.parser.Parser.nominate
            The part in charge of nominating tokens
        aramis.parser.Parser.optimize
            Optimization function

        Parameters
        ----------
        text
            Text to be analyzed

        Returns
        -------
        Match found
        """

        tokens = self.lexer.process(text)
        interpretations = self.nominate(tokens)
        return self.optimize(interpretations)
