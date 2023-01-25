from collections import deque
from dataclasses import InitVar, dataclass, field
from enum import Enum
from itertools import chain
from os import getenv
from typing import NamedTuple, Optional, Sequence, Text, Tuple

from .langs import Lang
from .trigram import Trigram
from .weights import Weights


class Neighbor(NamedTuple):
    """
    Neighboring match for a word (something suggested by the spell checker
    which is more or less close to our word).
    """

    words: Sequence[Text]
    sim: float


class Token:
    """
    Token extracted from a text.
    """

    def __init__(self, lexer: "Lexer", word: Text):
        self.lexer = lexer
        self.word = word
        self.neighbors: Optional[Sequence[Neighbor]] = None
        self.stems: Optional[Sequence[Text]] = None
        self._options: Optional[Sequence["Option"]] = None

    def __repr__(self):
        return f"Token({self.word!r})"

    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return hash(self.word)

    @property
    def is_word(self) -> bool:
        """
        Indicates if this token is recognized as a word by the language or if
        it is something else (a number, punctuation, etc).
        """

        return bool(self.lexer.lang.get_word_re().match(self.word))

    @property
    def weights(self) -> Weights:
        """
        Shortcut to access the weight constants
        """

        return self.lexer.weights

    @property
    def options(self) -> Sequence["Option"]:
        """
        Generates a list of options for the current token. It's a list of all
        the words that the spellchecking things that are likely to be this
        word, including the original word itself.

        The first option of the returned sequence is the most likely to be the
        right one, followed by decreasingly likely options.
        """

        if self._options is None:
            self._options = self._make_options()

        return self._options

    def _make_options_verbatim(self) -> Sequence["Option"]:
        """
        The word itself, verbatim
        """

        return [
            Option(
                OptionType.verbatim, self.weights.option_verbatim, self, (self.word,)
            )
        ]

    def _make_options_stem(self) -> Sequence["Option"]:
        """
        Options that are stems of the word ("words" -> "word",
        "doing" -> "to do", etc).
        """

        out = deque()

        for stem in self.stems or []:
            option = Option(OptionType.stem, self.weights.option_stem, self, (stem,))
            out.append(option)

        return out

    def _make_options_neighbors(self) -> Sequence["Option"]:
        """
        Options of words with a similar spelling
        """

        out = deque()

        for neighbor in self.neighbors or []:
            option = Option(
                OptionType.neighbor, neighbor.sim, self, tuple(neighbor.words)
            )
            out.append(option)

        return out

    def _make_options(self) -> Sequence["Option"]:
        """
        Combines together all kinds of options and returns them.
        """

        return [
            *sorted(
                chain(
                    self._make_options_verbatim(),
                    self._make_options_stem(),
                    self._make_options_neighbors(),
                ),
                key=lambda o: o.score,
                reverse=True,
            )
        ]

    def explore(self) -> None:
        """
        Explores the current word by looking at different spellings from the
        spell checker and also looking at its stems.
        """

        self.neighbors = []
        self.stems = []

        if not self.is_word:
            return

        initial = Trigram(self.word)

        for sug in self.lexer.hunspell.suggest(self.word):
            if sug == self.word:
                continue

            self.neighbors.append(
                Neighbor(words=self.lexer.lang.split(sug), sim=(initial % Trigram(sug)))
            )

        self.stems.extend(self.lexer.hunspell.stem(self.word))


class OptionType(Enum):
    """
    Represents the type of option generated, in case you want to filter which
    options are interesting for you.
    """

    # The word verbatim
    verbatim = "verbatim"

    # A word with a neighbor writing
    neighbor = "neighbor"

    # A possible stem for this word
    stem = "stem"


@dataclass(frozen=True)
class Option:
    """
    An option is one of possible options that are derived from a token. Each
    option consists of one or several words. By example it could be someone
    that said "helo" which is being fixed to "hello", or it could be a
    "hithere" which is being split into "hi there".
    """

    type: OptionType
    score: float
    token: "Token"
    raw_words: InitVar[Tuple[Text, ...]]
    words: Tuple["OptionWord"] = field(init=False)

    def __post_init__(self, raw_words: Tuple[Text, ...]):
        object.__setattr__(
            self, "words", tuple(OptionWord(self, word) for word in raw_words)
        )


@dataclass(frozen=True)
class OptionWord:
    """
    A word part of an option

    See Also
    --------
    The Option class right above
    """

    option: Option = field(hash=False)
    word: Text
    word_lower: Text = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "word_lower", self.word.lower())


class Lexer:
    """
    Lexing the content, aka transforming texts into a series of tokens that are
    ready to be given to the parser.

    Examples
    --------
    A typical usage would be:

    >>> from aramis.langs import fr_FR
    >>> lex = Lexer(fr_FR)
    >>> print(lex.process("J'aime les frites"))

    Notes
    -----
    By default it's going to look for dictionaries in /usr/share/hunspell,
    you can change this by setting the value of the HUNSPELL_DATA_DIR
    environment variable.

    You can override the dictionary name from the Lang class, if you need to
    do so. By default it's going to look for the dictionary that has the name
    of the current lang (makes sense I guess).
    """

    def __init__(self, lang: Lang, weights: Weights = Weights()):
        self.lang = lang
        self.weights = weights
        self._hunspell = None

    @property
    def hunspell_data_dir(self):
        """
        Returns the location of the Hunspell data dir. The default
        """

        return getenv("HUNSPELL_DATA_DIR", "/usr/share/hunspell")

    @property
    def hunspell(self) -> "Hunspell":
        """
        Returns the (cached) Hunspell instance
        """

        if not self._hunspell:
            from hunspell_cffi import Hunspell, ffi, lib

            class Hunspell2(Hunspell):
                """
                We had to override Hunspell-CFFI because they didn't implement stem()
                """

                def stem(self, word: str) -> Sequence[str]:
                    """
                    "De-conjugate" words
                    """

                    if not isinstance(word, str):
                        raise ValueError("Expected a string")

                    sl = ffi.new("char***")
                    n = lib.Hunspell_stem(self.hun, sl, word.encode("utf-8"))

                    try:
                        if not n:
                            return []

                        return [ffi.string(sl[0][i]).decode("utf-8") for i in range(n)]
                    finally:
                        lib.Hunspell_free_list(self.hun, sl, n)

            self._hunspell = Hunspell2(
                path=self.hunspell_data_dir,
                language=self.lang.get_hunspell_dict_name(),
            )

        return self._hunspell

    def normalize(self, text: Text) -> Text:
        """
        Normalizes a text according the lang's rules. The goal is to make it
        easy to tokenize (by example by making sure that all tokens are
        separated by spaces) and that various keyboard mishaps or different
        conventions for numbers or dates are accounted for.

        Notes
        -----
        That's mostly sugar, that's the Lang class that provides the regular
        expressions to normalize the text.

        Parameters
        ----------
        text
            Text to be normalized

        Returns
        -------
        The same text but normalized.
        """

        for rule, replace in self.lang.get_usual_typos():
            text = rule.sub(replace, text)

        return text

    def tokenize(self, text: Text, explore: bool = True) -> Sequence[Token]:
        """
        Splits the string into tokens. Optionally, explores the possible
        spelling mistakes of those words.

        Notes
        -----
        This is just some sugar, it's the Lang class that actually decides
        how to split up the sentence, as it's also the one deciding how to
        normalize it.

        Parameters
        ----------
        text
            Text to be split up.
        explore
            Activates the spellchecking, which helps matching words in spite of
            spelling mistakes.

        Returns
        -------
        A sequence of tokens, potentially explored
        """

        out = tuple(Token(word=w, lexer=self) for w in self.lang.split(text))

        if explore:
            self.explore(out)

        return out

    def explore(self, tokens: Sequence[Token]) -> None:
        """
        Starts the exploration process of every provided token.

        Notes
        -----
        Later this might provide some parallelization of this processing,
        however for now it does not seem to be efficient.

        Similarly, the hunspell library seems to provide a bunch of bulk_*()
        methods which will give you the ability to process several words at the
        same time, however they don't return the same results as the regular
        non-bulk methods and don't seem to provide a lot of performance
        improvement (quite the opposite actually) so I'm just guessing that
        they are mode for really big texts.

        Parameters
        ----------
        tokens
            Tokens to be explored. They will be mutated.
        """

        for token in tokens:
            token.explore()

    def process(self, text: Text) -> Sequence[Token]:
        """
        Utility function to run the full lexing process on a text and receive
        the tokens as output.

        Notes
        -----
        The tokens will be explored before being returned.

        Parameters
        ----------
        text
            Text that you want to lex

        Returns
        -------
        A sequence of the tokens
        """

        norm = self.normalize(text)
        return self.tokenize(norm)
