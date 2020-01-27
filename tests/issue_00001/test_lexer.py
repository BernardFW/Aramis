from aramis.lexer import Lexer, Token, Neighbor
from aramis.langs import fr_FR


def test_normalize_fr_fr():
    lex = Lexer(fr_FR)

    assert (
        lex.normalize("J'ai perdu mes codes d'acc à mon site")
        == "J’ai perdu mes codes d’acc à mon site"
    )
    assert (
        lex.normalize("auriez vous une solution a me proposer svp ?")
        == "auriez vous une solution a me proposer svp ?"
    )
    assert lex.normalize(
        "BONJOUR JE VOUDRAIS SOUSCRIRE POUR UNE MISE EN PLACE AVANT LE "
        "01/01/19 MERCI D'AVANCE"
    ) == (
        "BONJOUR JE VOUDRAIS SOUSCRIRE POUR UNE MISE EN PLACE AVANT LE "
        "01/01/19 MERCI D’AVANCE"
    )
    assert lex.normalize(
        "J'ai installé le plugin e-transaction sur woocommerce "
        '(dont mon client à souscris "ANABISHOP" ) mais j\'obtiens un '
        "message erreur."
    ) == (
        "J’ai installé le plugin e-transaction sur woocommerce "
        '( dont mon client à souscris "ANABISHOP" ) mais j’obtiens un '
        "message erreur ."
    )
    assert lex.normalize(
        "Encore aujourd’hui, tous les techniciens sont en réunion… "
        "cela fait 20 minutes que je suis en attente téléphonique !"
    ) == (
        "Encore aujourd’hui , tous les techniciens sont en réunion … "
        "cela fait 20 minutes que je suis en attente téléphonique !"
    )
    assert (
        lex.normalize("Encore aujourd’hui, tous les techniciens sont en réunion...")
        == "Encore aujourd’hui , tous les techniciens sont en réunion …"
    )
    assert lex.normalize(
        "Nous estimons que nous aurons envrion 30 000€ de chiffre "
        "d'affaire annuel par ce bais ainsi que plus de 2000 "
        "transactions annuels."
    ) == (
        "Nous estimons que nous aurons envrion 30000€ de chiffre "
        "d’affaire annuel par ce bais ainsi que plus de 2000 "
        "transactions annuels ."
    )
    assert (
        lex.normalize("Mon numéro est le 06.11.78.04.60")
        == "Mon numéro est le 0611780460"
    )
    assert (
        lex.normalize("J'ai payé 42. 10 de plus qu'annoncé.")
        == "J’ai payé 42 . 10 de plus qu’annoncé ."
    )
    assert lex.normalize("Cdt, M.L. Blidon") == "Cdt , M. L. Blidon"
    assert (
        lex.normalize("de pâtisserie ( création depuis 2017).")
        == "de pâtisserie ( création depuis 2017 ) ."
    )
    assert (
        lex.normalize("Votre produit est il adapté à cet effet?")
        == "Votre produit est il adapté à cet effet ?"
    )
    assert lex.normalize("Accès refusé !") == "Accès refusé !"
    assert (
        lex.normalize("J'ai découvert votre offre : je vends de produits")
        == "J’ai découvert votre offre : je vends de produits"
    )
    assert (
        lex.normalize(
            "Je souhaite augmenter ma notoriété ,je souhaite creer un site internet"
        )
        == "Je souhaite augmenter ma notoriété , je souhaite creer un site internet"
    )
    assert lex.normalize("vetements 100% basques.") == "vetements 100% basques ."
    assert lex.normalize("vetements 100 % basques.") == "vetements 100% basques ."


def test_tokenize():
    lex = Lexer(fr_FR)

    assert [x.word for x in lex.tokenize("Accès refusé !")] == ["Accès", "refusé", "!"]


def test_explore():
    lex = Lexer(fr_FR)

    b: Token
    (b,) = lex.tokenize("bonjour", explore=True)

    assert b.word == "bonjour"
    assert b.stems == [
        "bonjour",
    ]
    assert b.neighbors == [
        Neighbor(["bonjours"], 0.7),
        Neighbor(["bon", "jour"], 0.5454545454545454),
        Neighbor(["bon-jour"], 0.5454545454545454),
    ]
