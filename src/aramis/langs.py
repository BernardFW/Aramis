import re
from abc import ABC, abstractmethod
from typing import Callable, NamedTuple, Sequence, Text, Tuple, Union

UsualTypos = Sequence[Tuple[re.compile, Union[Text, Callable]]]

LOCALE_RE = re.compile(r"([a-zA-Z]{2,3})[-_]([a-zA-Z]{2,3})")


class Locale(NamedTuple):
    """
    Neutral representation of a locale, which can then be derived to different
    representations for different uses.
    """

    lang: Text
    region: Text

    @property
    def unix_locale(self):
        """
        Conventional Unix locale representation, that's quite popular and
        also used by Hunspell.
        """

        return f"{self.lang.lower()}_{self.region.upper()}"

    @classmethod
    def parse(cls, locale: Text) -> "Locale":
        """
        Parsing of a generic locale string to produce a Locale object

        Parameters
        ----------
        locale
            Locale string. Could be something like `fr_FR` or `fr-fr`.

        Returns
        -------
        Processed Locale object.
        """

        m = LOCALE_RE.match(locale)

        if not m:
            raise ValueError("Provided value is not a valid locale")

        return Locale(
            lang=m.group(1).lower(),
            region=m.group(2).lower(),
        )


class Lang(ABC):
    """
    If you want to implement a new language for Aramis then you need to
    implement this class.
    """

    @abstractmethod
    def get_usual_typos(self) -> UsualTypos:
        """
        Generates a list of regular expression and replacement patterns that
        will normalize the text against usual typos, mistakes and conventions.
        """

        raise NotImplementedError

    @abstractmethod
    def get_locale(self) -> Locale:
        """
        Returns the locale that this instance is currently configured for.
        """

        raise NotImplementedError

    @abstractmethod
    def split(self, text: Text) -> Sequence[Text]:
        """
        Splits a _normalized_ string into a sequence of words.

        Parameters
        ----------
        text
            Text to be split
        """

        raise NotImplementedError

    @abstractmethod
    def get_hunspell_dict_name(self) -> Text:
        """
        Returns the Hunspell dictionary name that you want.
        """

        raise NotImplementedError

    @abstractmethod
    def get_word_re(self) -> re.compile:
        """
        Returns a compiled regular expression that can match what you would
        define as a word in the parsed language.

        That's useful in order to know which tokens should be spell-checked
        versus those who should stay as-is.
        """

        raise NotImplementedError


class BasicLang(Lang):
    """
    Simple implementation of the Lang interface, you just provide all the
    values to the constructor and they will be returned when needed.
    """

    def __init__(self, locale: Text, usual_typos: UsualTypos, word_re: re.compile):
        self.locale = Locale.parse(locale)
        self.usual_typos = usual_typos
        self.word_re = word_re

    def get_usual_typos(self) -> UsualTypos:
        return self.usual_typos

    def get_locale(self) -> Locale:
        return self.locale

    def get_hunspell_dict_name(self) -> Text:
        return self.get_locale().unix_locale

    def get_word_re(self) -> re.compile:
        return self.word_re

    def split(self, text: Text) -> Sequence[Text]:
        return text.split(" ")


_NASTY_NUM_CHAR_FR = re.compile(r"[.\-\s]")
_INITIAL_CHAR = re.compile(r"(\w)\s*\.\s*")
_DATE_SEP = re.compile(r"\s*[./]\s*")
fr_FR = BasicLang(
    locale="fr_FR",
    word_re=re.compile(r"([a-zéàèùâêîôûëïüÿç]’)?[a-zéàèùâêîôûëïüÿç\-]+", re.IGNORECASE),
    usual_typos=[
        # The goal of this regexp is to handle the different formats of french
        # numbers, both numeric values and phone numbers. It's not perfect, use
        # cases can be found in unit tests, look out for regressions.
        # - 42 000 000 €
        # - 1 053,43$
        # - 1.543.556,43 $
        # - 06 11 78 04 60
        # - 05-61-78-77-85
        (
            re.compile(
                r"((\d{1,3}(\s*\.\s*\d{3})*(\s*,\s*\d+)?|\d{1,3}(\s+\d{3})*(\s*,\s*\d+)?|\d+)(\s*[€$%])|\d+(\s*([.\-]\s*)?\d+(?!\d)){3,})"
            ),
            lambda m: _NASTY_NUM_CHAR_FR.sub("", m.group(0)),
        ),
        # Normalizes ellipsis
        (re.compile(r"\.\.\."), "…"),
        # Put space around comas (but not when in numbers)
        (
            re.compile(r"([a-zéàèùâêîôûëïüÿç]\s*)(,)(\s*\w)?", re.IGNORECASE),
            r"\1 \2 \3",
        ),
        # Put spaces after/before punctuation
        (re.compile(r"(\w|\))\s*([!?;…./])(\s*\w)?"), r"\1 \2 \3"),
        # Normalizes spaces inside parenthesis
        (re.compile(r"\(([^)]+)\)"), r"( \1 )"),
        # Replaces ' contractions by fancy ’
        (re.compile(r"(\w)\s*'\s*(\w)"), r"\1’\2"),
        # Normalize t-il/t'il
        (re.compile(r"(t)’(il|elle)"), r"\1-\2"),
        # Normalizes spaces
        (re.compile(r"\s+"), " "),
        # Name initials ("M.L. Blindon")
        (
            re.compile(
                r"([A-ZÉÀÈÙÂÊÎÔÛËÏÜŸÇ]\s*\.\s*)+[A-ZÉÀÈÙÂÊÎÔÛËÏÜŸÇ][a-zéàèùâêîôûëïüÿç]"
            ),
            lambda m: _INITIAL_CHAR.sub(r"\1. ", m.group(0)),
        ),
        # Fix together dates
        (
            re.compile(
                r"(\s|^)(\d{2}\s*/\s*\d{2}\s*/\s*(\d{2}|\d{4})(?!\d)\s*|\d{2}\s*\.\s*\d{2}\s*\.\s*(\d{2}|\d{4})(?!\d)\s*)"
            ),
            lambda m: _DATE_SEP.sub(r"/", m.group(0)),
        ),
        # Strip start/end white spaces
        (re.compile(r"(^\s+|\s+$)"), ""),
    ],
)
