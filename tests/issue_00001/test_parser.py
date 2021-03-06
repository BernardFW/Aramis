from pytest import fixture

from aramis.lexer import Lexer, OptionType
from aramis.langs import fr_FR
from aramis.parser import Parser, Interpretation
from aramis.rules import (
    SausageRule,
    WordMatcher,
    MaximizeMatch,
    MaximizeSimilarity,
    NoMatch,
)


@fixture(name="lex")
def make_lex():
    return Lexer(fr_FR)


@fixture(name="parser")
def make_parser(lex):
    return Parser(
        lex,
        rules=[
            SausageRule.info("sausage", 1.0),
            MaximizeMatch.info("max-match", 1.0),
            MaximizeSimilarity.info("max-sim", 1.0),
        ],
    )


def test_word_matcher(lex):
    word = lex.process("J'aime")[0]

    word_opt = word.options[0].words[0]
    assert WordMatcher("j’aime") == word_opt


def test_word_matcher_in(lex):
    word = lex.process("J'aime")[0]

    word_opt = word.options[0].words[0]
    assert word_opt in [WordMatcher("j’aime")]


def test_word_matcher_stem(lex):
    word = lex.process("J'aime")[0]
    word_opt = None

    for option in word.options:
        if option.type == OptionType.stem:
            word_opt = option.words[0]
            break

    assert WordMatcher("aimer", stem=True) == word_opt


def test_word_matcher_stem_in(lex):
    word = lex.process("J'aime")[0]
    word_opt = None

    for option in word.options:
        if option.type == OptionType.stem:
            word_opt = option.words[0]
            break

    assert word_opt in [WordMatcher("aimer", stem=True)]


def test_nominate(parser: Parser, lex: Lexer):
    like: Interpretation
    nope: Interpretation
    sausage: Interpretation

    tokens = lex.process("J'aime les saucisses")
    like, nope, sausage = parser.nominate(tokens)

    assert len(like.nominations) == 1
    assert len(like.nominations[0]) == 1
    assert like.nominations[0][0].word.option.token.word == "J’aime"
    assert isinstance(like.nominations[0][0].flag.rule, SausageRule)

    assert len(nope.nominations) == 0

    assert len(sausage.nominations) == 1
    assert len(sausage.nominations[0]) == 1
    assert sausage.nominations[0][0].word.option.token.word == "saucisses"
    assert isinstance(sausage.nominations[0][0].flag.rule, SausageRule)


def test_optimize(parser: Parser):
    match = parser.parse("j'aime les saucisses")
    assert match.score > 0.8
    assert match.matched[0].word.option.token.word == "j’aime"
    assert isinstance(match.matched[1], NoMatch)
    assert match.matched[2].word.option.token.word == "saucisses"

    match = parser.parse("j'aime les bananes")
    assert match.score == 0.0

    match = parser.parse("nous aimons les bonnes saucisses")
    assert match.score > 0.5

    match = parser.parse("les saucisses je n'aime pas du tout")
    assert match.score == 0.0

    match = parser.parse("nous aimons les saucisses et les bananes")
    assert match.score >= 0.5
