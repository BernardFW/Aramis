from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional, Sequence, Text, Union

from .lexer import OptionType, OptionWord


@dataclass(frozen=True)
class RuleInfo:
    """
    Meta-information about how you want to insert that rule in the parser.
    """

    # The rule instance
    rule: "Rule"

    # How much will the scores of that rule will be multiplied by during the
    # parser's scoring
    weight: float

    # Name of the rule for reference in results
    name: Text


@dataclass(frozen=True)
class Flag:
    """
    Flag applied by a rule onto a matching word (through a Nomination object)
    """

    # Rule instance
    rule: "Rule"

    # Meta-data that the rule wants to keep along for later use (by example
    # communicate the interpreted text to the parent)
    data: Any = None


@dataclass
class Nomination:
    """
    A nomination is how a rule says that it things that this word could be
    matched by it.
    """

    word: OptionWord = field(repr=False)
    flag: Flag


class NoMatch:
    """
    Special object to indicate the lack of match
    """

    def __repr__(self):
        return "NoMatch()"


WordMatch = Union[NoMatch, Nomination]


@dataclass(eq=False, frozen=True)
class WordMatcher:
    """
    Utility class that can be compared (using ==) to an OptionWord and lets
    you know if it's a match.
    """

    # The text you want to compare to. It's always going to be compared to the
    # lowercase version of the option.
    word: Text

    # Do you want to have only verbatim+neighbor options or only stem options?
    stem: bool = False

    def __eq__(self, other):
        if isinstance(other, Nomination):
            other = other.word

        if not isinstance(other, OptionWord):
            return False

        if self.stem != (other.option.type == OptionType.stem):
            return False

        if other.word_lower != self.word:
            return False

        return True


class Rule(ABC):
    def __repr__(self):
        return f"{self.__class__.__name__}()"

    def nominate_words(self, words: Iterator[OptionWord]) -> Iterator[Nomination]:
        """
        Given a list of words, nominate those which could be interesting for
        this rule.

        Notes
        -----
        Not all rules need to nominate words. By example global optimization
        rules like minimizing the edit distance don't need to do that. However,
        rules looking for a specific meaning, typically intents and grammar,
        have to nominate which words they would like to see considered as
        potentially meaningful.

        That's why there is a default implementation that returns an empty
        list but if you implement a grammar class then you should definitely
        override this function.

        Parameters
        ----------
        words
            An iterator over words to be examined

        Returns
        -------
        An iterator over words nominated
        """

        return []

    @abstractmethod
    def evaluate(self, words: Sequence[WordMatch]) -> Optional[float]:
        """
        Evaluates a sequence of words in and determines a score of how much
        the rule is respected. The lower the better.

        Reasonable acceptable value should be between 0 and 1, although there
        is no upper limit.

        If the rule does not accept this combination, it should return None
        instead.

        Parameters
        ----------
        words
            Currently evaluated sequence of nominated words

        Returns
        -------
        Either a score (0 or more, preferably up to 1, closer to 0 means better
        match)
        """

        raise NotImplementedError

    @classmethod
    def info(cls, name: Text, weight: float, *args, **kwargs) -> "RuleInfo":
        """
        Generates the meta-info object for the rule. It's just some sugar.
        """

        # noinspection PyArgumentList
        return RuleInfo(rule=cls(*args, **kwargs), weight=weight, name=name)


class SausageRule(Rule):
    """
    Test rule that implements a very simple grammar.
    """

    def nominate_words(self, words: Iterator[OptionWord]) -> Iterator[Nomination]:
        """
        We're looking for just 2 words: aimer (stemmed, meaning "to love") and
        saucisse (stemmed, meaning "sausage").

        All other words will be ignored.
        """

        allowed = [
            WordMatcher("aimer", stem=True),
            WordMatcher("saucisse", stem=True),
        ]

        for word in words:
            if word in allowed:
                yield Nomination(word=word, flag=Flag(self))

    def evaluate(self, words: Sequence[WordMatch]) -> Optional[float]:
        """
        Makes sure that our two keywords are found in order in the sentence.
        The closest they are the better the match.

        Parameters
        ----------
        words
            Sequence of words to evaluate

        Returns
        -------
        Returns a score of likeliness if the sentence makes sense, nothing
        otherwise.
        """

        like_pos = None
        sausage_pos = None

        like = WordMatcher("aimer", stem=True)
        sausage = WordMatcher("saucisse", stem=True)

        for i, word in enumerate(words):
            if like == word:
                if like_pos is not None:
                    return

                like_pos = i

            if sausage == word:
                if sausage_pos is not None:
                    return

                sausage_pos = i

        if like_pos is None or sausage_pos is None:
            return

        diff = sausage_pos - like_pos

        if diff < 0:
            return
        elif 1 <= diff <= 2:
            return 0
        elif diff == 3:
            return 0.25
        elif diff == 4:
            return 0.5
        else:
            return 1.0


class MaximizeMatch(Rule):
    """
    The goal here is to maximize the amount of matched words. We should make
    sure that as many words as possible are matched and take the decision to
    ignore words only if this unlocks significant grammatical sense.
    """

    def evaluate(self, words: Sequence[WordMatch]) -> Optional[float]:
        total = len(words)
        matching = 0

        for word in words:
            if not isinstance(word, NoMatch):
                matching += 1

        return 1.0 - float(matching) / float(total)


class MaximizeSimilarity(Rule):
    """
    Maximizes the similarity of matched words. If two words are in competition
    then use the one that is spelled the closest to what the user wrote.
    """

    def evaluate(self, words: Sequence[WordMatch]) -> Optional[float]:
        total = 0
        cnt = 0

        for word in words:
            if isinstance(word, Nomination):
                # noinspection PyUnresolvedReferences
                total += 1 - word.word.option.score
                cnt += 1

        if cnt == 0:
            return None

        return float(total) / float(cnt)
