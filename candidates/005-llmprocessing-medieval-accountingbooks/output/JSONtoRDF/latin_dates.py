# Deterministic resolution of Latin date expressions to ISO dates (Julian calendar).
#
# Author: Maximilian Vogeltanz, University of Graz
#
# WHY THIS EXISTS
# ---------------
# LLMs reliably identify WHICH words in an entry are the date, and reliably fail to
# compute WHAT date those words denote. Measured on L343 (JSONtoRDF output, qwen3.5-397b):
# of 1083 entries where a date expression is present, the model agreed with the
# arithmetic in 362 — a third. Moveable feasts were wrong by up to two weeks
# ("dominica oculi" 1455 encoded as 02-25 instead of 03-09), fixed feasts by a day or
# two ("marie magdalene" as 07-21 instead of 07-22).
#
# So the split is: the model picks the phrase, this module computes the date. Nothing
# here consults a model, and the same phrase always yields the same date.
#
# This is a library extracted from encode_latin_dates_standalone.py, which worked on
# TEI. Two differences follow from the new input being the plain-text bk:entry:
#
#   * The inline markup alternatives (<lb/>, <hi rend="superscript">) are gone, but
#     their FLATTENED residue is not: TEI's `feria 3<hi rend="superscript">a</hi>`
#     arrives as `feria 3a`, `lvi<hi>o</hi>` as `lvio`, `lxx<hi>me</hi>` as `lxxme`.
#     The numeric feria forms actually outnumber the spelled-out ones in L343
#     (191 vs 164), so these are load-bearing, not edge cases.
#   * get_preceding_year() is NOT ported. It scanned backwards through <fw>/<head> for
#     a roman numeral and carried a known bug (it took the last match whether or not
#     that match was a year). The pipeline no longer needs it: the JSON that feeds
#     the JSONtoRDF step carries a per-object "year" field, so the caller passes that as
#     fallback_year and an `Anno <roman>` inside the entry overrides it.
#
# WHAT THIS MODULE DELIBERATELY DOES NOT DO
# -----------------------------------------
# It does not decide which of several date expressions in an entry is the transaction
# date. "tempus solucionis mediam partem purificacionis et mediam Geori martyris"
# contains two payment deadlines and no transaction date at all; resolve() will happily
# compute the first one. That selection is the model's job. DEADLINE_MARKERS is exported
# so a caller can at least flag the risk — see find_all() and the report.

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta

__all__ = [
    "DateExpr", "resolve", "resolve_phrase", "find_all", "normalize",
    "julian_easter", "feast_date", "parse_anno", "roman_to_int", "weekday",
    "advent_sundays", "DEADLINE_MARKERS", "UNCERTAIN_FEASTS", "FIXED_FEASTS",
    "EASTER_OFFSETS", "FERIA_WEEKDAY",
]


# ──────────────────────────── Calendar arithmetic ────────────────────────────
# Everything is Julian. The ledgers end well before the 1582 reform, so unlike the TEI
# script there is no Gregorian branch: a Gregorian fallback could only ever fire on a
# misparsed year, and would then produce a plausible-looking wrong date. Refusing is
# better than guessing.

def julian_easter(year: int) -> date:
    """Easter Sunday in the Julian calendar (Meeus' Julian algorithm)."""
    a = year % 19
    b = year % 4
    c = year % 7
    d = (19 * a + 15) % 30
    e = (2 * b + 4 * c + 6 * d + 6) % 7
    day = 22 + d + e
    if day > 31:
        return date(year, 4, day - 31)
    return date(year, 3, day)


def julian_day(year: int, month: int, day: int) -> int:
    """Julian Day Number, Julian calendar reckoning."""
    a = (14 - month) // 12
    y = year + 4800 - a
    m = month + 12 * a - 3
    return day + ((153 * m + 2) // 5) + 365 * y + y // 4 - 32083


def weekday(d: date) -> int:
    """0 = Sunday … 6 = Saturday, for a date read as Julian."""
    return int((julian_day(d.year, d.month, d.day) + 1.5) % 7)


def advent_sundays(year: int) -> list[date]:
    """The four Advent Sundays, first to fourth."""
    dec24 = date(year, 12, 24)
    fourth = dec24 - timedelta(days=weekday(dec24))   # Sunday on/before 24 Dec
    first = fourth - timedelta(days=21)
    return [first + timedelta(days=7 * i) for i in range(4)]


# ──────────────────────────── Feast tables ────────────────────────────
# Offsets from Easter Sunday, in days.
#
# CORRECTED against both earlier sources. Septuagesima stood at -70 in
# encode_latin_dates_standalone.py AND in systemprompt_RDF_datevalidation_v2.txt;
# it is the 9th Sunday before Easter, i.e. -63. The prompt additionally had
# Sexagesima at -63 (the script had -56, which is right). The Latin names count
# inclusively and approximately — they are not the offsets.
EASTER_OFFSETS = {
    "septuagesima":        -63,
    "sexagesima":          -56,
    "quinquagesima":       -49,   # Estomihi
    "ash wednesday":       -46,
    "invocavit":           -42,   # Quadragesima, 1st Sunday of Lent
    "reminiscere":         -35,
    "oculi":               -28,
    "laetare":             -21,
    "judica":              -14,
    "palm sunday":          -7,
    "cena domini":          -3,   # Gründonnerstag
    "good friday":          -2,
    "easter sunday":         0,
    "easter monday":         1,
    "quasimodogeniti":       7,
    "misericordia domini":  14,
    "jubilate":             21,
    "cantate":              28,
    "rogate":               35,   # Vocem Iocunditatis
    "ascension":            39,
    "exaudi":               42,
    "pentecost":            49,
    "pentecost monday":     50,
    "trinity":              56,
    "corpus christi":       60,
}

# Fixed feasts, (month, day), diocese of Passau usage.
FIXED_FEASTS = {
    "circumcision":                  (1, 1),
    "epiphany":                      (1, 6),
    "erhardi":                       (1, 8),
    "wilhelmi episcopi":             (1, 10),
    "felicis in pincis":             (1, 14),
    "sebastiani et fabiani":         (1, 20),
    "angnetis":                      (1, 21),
    "vincencii":                     (1, 22),
    "conversionis pauli":            (1, 25),
    "juliani episcopi":              (1, 27),
    "purificacionis":                (2, 2),    # Lichtmess
    "blasii":                        (2, 3),
    "agathe":                        (2, 5),
    "scolastice":                    (2, 10),
    "valentini":                     (2, 14),
    "kathedra petri":                (2, 22),
    "mathie apostoli":               (2, 24),   # Matthias — not Matthaeus, see below
    "gregori pape":                  (3, 12),
    "benedicti":                     (3, 21),   # Depositio — the house is Cistercian and
                                                # lives under his rule. 11 July is the
                                                # Translatio and has its own key.
    "annunciacionis":                (3, 25),
    "ambrosii":                      (4, 4),    # Ambrose of Milan, depositio — the German
                                                # usage; 7 December is his ordination.
    "geori":                         (4, 24),   # UNCERTAIN, see below
    "marci ewangeliste":             (4, 25),
    "philippi et jacobi":            (5, 1),
    "invencionis sancte crucis":     (5, 3),
    "floriani":                      (5, 4),
    "urbani":                        (5, 25),
    "barnabe apostoli":              (6, 11),
    "anthoni confessoris":           (6, 13),   # UNCERTAIN, see below
    "viti":                          (6, 15),
    "gervasii et prothasii":         (6, 19),
    "x milia martyrum":              (6, 22),
    "johannis baptiste":             (6, 24),
    "johannis et pauli":             (6, 26),
    "petri et pauli":                (6, 29),
    "petri martyris":                (6, 30),   # UNCERTAIN, see below
    "visitacionis":                  (7, 2),
    "processi et martiniani":        (7, 2),
    "udalrici":                      (7, 4),
    "kiliani":                       (7, 8),
    "translacionis benedicti":       (7, 11),
    "margrethe":                     (7, 13),   # 2026-08-26, was 07-12 — see below
    "divisio apostolorum":           (7, 15),
    "allexi":                        (7, 17),
    "marie magdalene":               (7, 22),
    "jacobi apostoli":               (7, 25),
    "vincula petri":                 (8, 1),
    "stephani":                      (8, 2),    # UNCERTAIN, see below
    "invencionis stephani":          (8, 3),
    "translacionis valentini":       (8, 4),    # UNCERTAIN, see below
    "oswaldi regis":                 (8, 5),    # "regis" is required in the variant
    "sixti pape":                    (8, 6),    # UNCERTAIN, see below
    "laurencii":                     (8, 10),
    "ypoliti":                       (8, 13),
    "assumpcionis":                  (8, 15),
    "bernhardi":                     (8, 20),   # Bernhard v. Clairvaux — Aldersbach is Cistercian
    "bartholomei":                   (8, 24),
    "augustini":                     (8, 28),
    # Decollatio Johannis Baptistae — a SECOND Johannes feast, 29 August, two months from
    # the Nativitas on 24 June. It must precede "johannis baptiste" in _VARIANTS or the
    # shorter key swallows it and the entry lands in June.
    "decollacionis johannis":        (8, 29),
    "felicis et adaucti":            (8, 30),
    "egidi":                         (9, 1),
    "nativitatis marie":             (9, 8),
    "exaltacionis crucis":           (9, 14),
    "lamperti":                      (9, 17),
    "mathei apostoli":               (9, 21),
    "mauricii":                      (9, 22),
    "michahelis":                    (9, 29),
    "jeronimi":                      (9, 30),
    "francisci":                     (10, 4),
    "dionysii":                      (10, 9),
    "augustini episcopi":            (10, 11),  # Translatio S. Augustini
    "cholmanni":                     (10, 13),
    "galli abbatis":                 (10, 16),
    "luce ewangeliste":              (10, 18),
    "xi milium virginum":            (10, 21),
    "simonis et jude":               (10, 28),
    "omnium sanctorum":              (11, 1),
    "animarum":                      (11, 2),
    "leonhardi":                     (11, 6),
    "martini":                       (11, 11),
    "briccii":                       (11, 13),
    "emundi episcopi":               (11, 16),
    "elisabet":                      (11, 19),
    "cecilie":                       (11, 22),
    "clementis":                     (11, 23),
    "katherine":                     (11, 25),
    "andree apostoli":               (11, 30),
    "nicolai":                       (12, 6),
    "concepcionis":                  (12, 8),
    "lucie":                         (12, 13),
    "thome apostoli":                (12, 21),
    "christmas":                     (12, 25),
    "stephani protomartyris":        (12, 26),
    "johannis apostoli":             (12, 27),
    "innocentum":                    (12, 28),
    "silvestri":                     (12, 31),
}

# Values carried over from encode_latin_dates_standalone.py that its own TODOs, or the
# standard reference works, put in doubt. They are kept AS THEY WERE so that porting
# does not silently change any date; each needs a decision from the sources, and the
# report prints how often each actually fires.
UNCERTAIN_FEASTS = {
    "petri martyris": "Petrus Martyr (Verona) is 29 April. 30 June is Commemoratio "
                      "S. Pauli. Both earlier sources say 30 June — likely a misreading "
                      "of the manuscript, needs checking against the original. Does not "
                      "occur in the ground truth, so it cannot be settled from there.",
    "sixti pape": "6 August is standard, but Grotefend records a Salzburg Sixtus on "
                  "6 April. Not in the ground truth.",
    "translacionis valentini": "4 August, the Translatio of Valentin of Raetia, the second "
                  "patron of Passau — the entry says 'patavie' and so means him, not "
                  "Valentine of Rome on 14 February, which stays the bare 'valentini' key. "
                  "The date is taken from Passau usage and has NOT been confirmed against "
                  "the ground truth or the diocesan calendar; it fires once.",
}

# Settled against the ground truth on 2026-08-25, and kept here so the question is not
# reopened from the reference works alone:
#   geori    = 04-24. The GT encodes "in die Geori martyris" as 1455-04-24 directly, and
#              "feria 6a ante Geori martyris" as 1456-04-23 — Georgii 1456 is a Saturday
#              only if the feast is the 24th; on the 23rd that entry would be 16 April.
#              Grotefend's 23 April does not hold for this house.
#   stephani = 08-02. "in die sancti Stepfani" is 1455-08-02 in L343 and 1458-08-02 in
#              L344; "in die sancti Stepfani patavie" is likewise 1455-08-02, so the
#              Passau epithet names the patrocinium and does not move the day.
#   anthoni  = 06-13, i.e. Antonius of Padua, not Antonius Abbas in January: the GT dates
#              "dominica ante Anthoni confessoris" into June 1456.
#
# Decided on 2026-08-26 AGAINST the ground truth, the only such entry:
#   margrethe = 07-13, was 07-12. The GT's 20 July (L343 T58, T92) is ruled out by the
#              chronology of the surrounding entries (L343 T593, L344 T269). 12 vs 13 July
#              the corpus cannot separate, so decided on provenance: 07-12 came from the
#              v2 prompt's feast table, which also had Septuagesima and Sexagesima wrong.
#              13 July is the German/Passau usage.
#
# Settled on 2026-08-26 and removed from UNCERTAIN_FEASTS:
#   wilhelmi episcopi = 01-10, Wilhelm of Bourges (d. 1209). The doubt was that no Wilhelm
#              was attested on that day; he is. It fires 13 times with a date marker
#              ("in die wilhelmi episcopi anno liio") and 0 times without.

ADVENT = {"first advent": 0, "second advent": 1, "third advent": 2, "fourth advent": 3}


def _is_sunday_feast(key: str) -> bool:
    """Does this feast always fall on a Sunday?

    Decided by the offset rather than by a list: every Sunday feast sits a whole number
    of weeks from Easter Sunday. That keeps Ascension (+39, a Thursday), Corpus Christi
    (+60, a Thursday), Ash Wednesday (-46) and Good Friday (-2) out without naming them.
    Fixed feasts drift through the week and are never Sunday feasts."""
    if key in ADVENT:
        return True
    off = EASTER_OFFSETS.get(key)
    return off is not None and off % 7 == 0


def feast_date(key: str, year: int) -> date | None:
    """The date of a canonical feast key in the given year, or None if unknown."""
    if key in EASTER_OFFSETS:
        return julian_easter(year) + timedelta(days=EASTER_OFFSETS[key])
    if key in FIXED_FEASTS:
        month, day = FIXED_FEASTS[key]
        return date(year, month, day)
    if key in ADVENT:
        return advent_sundays(year)[ADVENT[key]]
    return None


# ──────────────────────────── Spelling variants ────────────────────────────
# canonical key -> alternatives as they occur in the manuscripts. Order within the list
# does not matter; order of the LIST does — the combined alternation is tried in this
# sequence, so a more specific spelling must precede a shorter one that would also match
# ("augustini episcopi" before "augustini", "invencionis stephani" before "stephani").
#
# The `[- ]?` in some patterns absorbs the manuscript line-break hyphenation that
# survives into bk:entry ("penthe- costes", "transla- cionis"); normalize() removes it
# up front, so the patterns only need it where a hyphen may also be genuine.
#
# "evangelista" is spelt four ways across the ledgers and all four attach to three
# different feasts, so the sub-pattern is named once rather than repeated: `w` for `v`
# is the ordinary scribal alternation, and the missing `e` ("evangliste") is a suspension.
_EWANG = r"e[wv]ang(?:e)?liste\.?"

# Stephanus attracts every spelling the ph/pf/f alternation allows, across three feasts,
# so it too is named once. "Steppfani" doubles the p on top of the pf.
_STEPH = r"ste(?:ph|pf|pff|ppf|pph|ff)ani"

_VARIANTS: list[tuple[str, str]] = [
    # ---- moveable, most specific first ----
    ("quasimodogeniti",      r"quasi\s*modo\s*geniti|quasimodogeniti"),
    ("misericordia domini",  r"miser[i]?c[o]?rdia(?:\s+domini)?"),
    ("rogate",               r"vocem\s+iocunditatis|rogacionum"),
    # The accusative -am is what "ante/post" governs and the scribes write it out:
    # "feria quarta ante Septuagesimam" (L344 T7).
    ("septuagesima",         r"lxx\s*me\b|septuagesim(?:a[m]?|e)"),
    ("sexagesima",           r"lx\s*me\b|sexagesim(?:a[m]?|e)|exurge"),
    ("quinquagesima",        r"\bl\s*me\b|quinquagesim(?:a[m]?|e)|esto\s+mic?hi|festo\s+micha"),
    ("invocavit",            r"invocavit"),
    ("reminiscere",          r"reminiscere"),
    ("oculi",                r"oculi"),
    ("laetare",              r"l[ae]tare"),
    ("judica",               r"[ji]udica"),
    ("jubilate",             r"[ji]ubilate"),
    ("cantate",              r"cantate"),
    ("exaudi",               r"exaudi"),
    ("palm sunday",          r"palmarum"),
    ("cena domini",          r"cen[ae][m]?(?:\s+domini)?"),
    # parasceve / parasceven / parascheve / parascaphe — the h migrates and the v hardens.
    ("good friday",          r"parasc(?:h)?[ae](?:ph|v)e[sn]?"),
    ("ash wednesday",        r"c[iy]nerum"),
    ("ascension",            r"ascen[sc]+ion(?:is|em)(?:\s+domini)?"),
    ("pentecost",            r"pe?n?t[h]?ecoste[sn]?"),
    ("corpus christi",       r"corporis\s+christi"),
    ("trinity",              r"trinitatis"),
    ("easter sunday",        r"pasc[h]?(?:ae|e|a)"),
    # ---- advent ----
    # The bare "dominica adventus" is read as the FIRST Advent Sunday. Evidence from L343:
    # the scribe writes "dominica prima adventus" 9 times and never once names a second,
    # third or fourth — nor uses gaudete/rorate. Having a form for the others and never
    # using it makes the short form an abbreviation of the one he does write. Grotefend
    # reads it the same way. 3 occurrences.
    ("first advent",         r"dominica\s+(?:(?:prima|1a)\s+)?adventus(?:\s+domini)?"),
    ("second advent",        r"dominica\s+(?:secunda|2a)\s+adventus(?:\s+domini)?"),
    ("third advent",         r"dominica\s+(?:tertia|tercia|3a)\s+adventus(?:\s+domini)?|dominica\s+gaudete"),
    ("fourth advent",        r"dominica\s+(?:quarta|4a)\s+adventus(?:\s+domini)?|dominica\s+rorate"),
    # ---- fixed, most specific first ----
    # Both word orders occur: "Invencionem sancti Steffani" 71×, "Steffani Invencionem" 4×.
    ("invencionis stephani", rf"inven(?:t|c)ion(?:em|is|e)\s+(?:sancti\s+)?{_STEPH}(?:\s+in\s+patavia)?"
                             rf"|{_STEPH}\s+inven(?:t|c)ion(?:em|is|e)"),
    ("stephani protomartyris", rf"{_STEPH}\s+protomartyris"),
    # "In die stef" — a suspension. The \b keeps it off "Stefan Aichhoren" and the like;
    # "stephani" is in _AMBIGUOUS, so an ungated mention is refused in a free-text scan.
    ("stephani",             rf"(?:sancti\s+)?(?:{_STEPH}(?:\s+(?:in\s+patavia|patavie))?|stef\b\.?)"),
    ("invencionis sancte crucis", r"invencion(?:is|em)\s+sancte\s+crucis"),
    ("exaltacionis crucis",  r"exaltacion(?:is|em)\s+(?:sanct[ei]\s+)?crucis|angaria[m]?\s+crucis"),
    ("nativitatis marie",    r"nativitatis\s+(?:beate\s+)?(?:vir?ginis(?:\s+gloriose)?"
                             r"|marie(?:\s+vir?ginis)?(?:\s+gloriose)?)"),
    ("annunciacionis",       r"annun[cti]{1,4}a?c?ionis\s+(?:vir?ginis\s+gloriose|marie(?:\s+vir?ginis)?(?:\s+gloriose)?)"
                             r"|annuntiatio(?:nem)?\s+beatae?\s+maria[e]?\s+vir?ginis"),
    ("visitacionis",         r"visitacionis(?:\s+(?:vir?ginis\s+gloriose|marie))?"),
    ("purificacionis",       r"purificacionis(?:\s+(?:vir?ginis\s+gloriose|marie(?:\s+vir?ginis)?))?|liecht?messe?"),
    ("concepcionis",         r"concepcionis(?:\s+vir?ginis)?(?:\s+marie)?(?:\s+gloriose)?"),
    ("assumpcionis",         r"assumpcion(?:is|em)(?:\s+(?:vir?ginis\s+gloriose|marie(?:\s+vir?ginis)?))?"),
    # A bare "nativitatis" is read as CHRISTMAS. Counted over L342-L346: of 255 mentions
    # only 14 carry no qualifier, and every Marian one ("nativitatis marie", "nativitatis
    # beate virginis") and every Baptist one ("Nativitatis sancti Johannis Baptiste") is
    # qualified without exception. The scribe writes the qualifier for the two he has to
    # distinguish and omits it for the Nativity that needs no distinguishing — the same
    # argument as for the bare "dominica adventus" above. "in vigilia nativitatis pro
    # offertorio" fits Christmas Eve and nothing else.
    #
    # This is why the Baptist's Nativity is claimed HERE, out of alphabetical place and
    # ahead of "christmas": the ordered alternation would otherwise let the bare form
    # swallow it, and 24 June would be encoded as 25 December.
    ("johannis baptiste",    r"nativitatis\s+(?:sancti\s+)?johannis\s+[wb]aptiste"),
    ("christmas",            r"nat(?:alis|ivitatis)\s+(?:christi|domini)|nativitatis"
                             r"|weihnacht\w*|christtag"),
    ("epiphany",             r"ep[iy]p[p]?[hf]ani(?:a|e)[ms]?(?:\s+domini)?"),
    ("circumcision",         r"circumcision(?:is|em)(?:\s+domini)?"),
    ("omnium sanctorum",     r"omnium\s+sanctorum|aller\s*heyligen\s*tag\w*"),
    ("animarum",             r"animarum|aller\s*seelen\s*tag\w*"),
    ("petri et pauli",       r"(?:beatorum\s+)?(?:apostolorum\s+)?petri\s+et\s+pauli(?:\s+apostolorum)?"),
    ("decollacionis johannis", r"deco[l]{1,2}a(?:c|t)ion(?:is|em)\s+(?:sancti\s+)?johannis\s+[wb]aptiste"),
    ("johannis et pauli",    r"(?:beatorum\s+martyrum\s+)?johannis\s+et\s+pauli(?:\s+martyr(?:um|is))?"),
    ("johannis baptiste",    r"(?:sancti\s+)?johannis\s+[wb]aptiste"),
    ("johannis apostoli",    rf"(?:sancti\s+)?johannis\s+(?:apostoli(?:\s+et\s+{_EWANG})?|{_EWANG})"),
    ("kathedra petri",       r"kathedr[ae]\s+(?:sancti\s+)?petri"),
    ("vincula petri",        r"vincula\s+(?:sancti\s+)?petri(?:\s+apostoli)?"),
    ("petri martyris",       r"petri\s+martyris"),
    ("conversionis pauli",   r"conversion(?:is|em)\s+(?:sancti\s+)?pauli"),
    ("translacionis benedicti", r"translacion(?:is|e|em)\s+sancti\s+benedicti(?:\s+abbat[t]?is)?"),
    ("divisio apostolorum",  r"divisio(?:nis)?\s+apostolorum"),
    # "xim virginum": the m of "milium" survives as a flattened superscript on the numeral
    # and the rest of the word is gone. "xim" must precede the bare "xi" in the alternation.
    ("xi milium virginum",   r"(?:xi\s*cim|xim|xi|undecim)\s+(?:milium\s+)?virginum"),
    ("x milia martyrum",     r"(?:x|decem)\s+mili(?:um|a|e)[m]?\s+martyrum"),
    ("augustini episcopi",   r"augustini\s+episcopi"),
    ("augustini",            r"(?:sancti\s+)?augustini"),
    ("michahelis",           r"(?:sancti\s+)?michah?el(?:is|em)|michahel\.|michelstag"),
    ("martini",              r"(?:sancti\s+)?martini(?:\s+episcopi)?|martinstag"),
    ("leonhardi",            r"(?:sancti\s+)?l(?:eo|i)n[h]?ard[io](?:\s+confessoris)?"),
    ("bernhardi",            r"(?:sanctissimi\s+patris\s+nostri\s+)?(?:sancti\s+)?bernhardi(?:\s+abbat[t]?is)?"),
    ("bartholomei",          r"(?:sancti\s+)?[wb]artholomei(?:\s+apostoli)?"),
    # "in patavia" names the Passau patrocinium, the same construction as with Stephanus,
    # and like it does not move the day.
    ("andree apostoli",      r"(?:sancti\s+)?andre[e]?(?:\s+apostoli)?(?:\s+(?:in\s+patavia|patavie))?"),
    ("thome apostoli",       r"(?:sancti\s+)?thome(?:\s+apostoli)?"),
    ("jacobi apostoli",      r"(?:sancti\s+)?jacobi(?:\s+apostoli)?(?:\s+in\s+partibus\s+inferioribus)?"),
    ("philippi et jacobi",   r"philippi\s+et\s+jacobi(?:\s+apostolorum)?"),
    # The genitive separates two different apostles, and the scribes keep them apart:
    # "Mathei" = Matthaeus, 21 September; "Mathie" = Matthias, 24 February. Counted over
    # L342-L345: 68 "Mathei apostoli" + 2 "Mathei ... et Ewangeliste" against 18 "Mathie
    # apostoli". L343 T664 settles the reading — "de Angaria Cynerum ... feria quarta post
    # mathie apostoli" is a Lenten Ember term, seven months away from Matthaeus.
    ("mathie apostoli",      r"(?:sancti\s+)?mathie\s+apostoli"),
    ("mathei apostoli",      rf"(?:sancti\s+)?mathei(?:\s+(?:apostoli|{_EWANG})"
                             rf"(?:\s+et\s+{_EWANG})?)?"),
    ("marci ewangeliste",    rf"marci(?:\s+{_EWANG})?"),
    ("luce ewangeliste",     rf"luce\s+{_EWANG}"),
    ("marie magdalene",      r"marie\s+magdalene"),
    ("katherine",            r"(?:sancte\s+)?kath(?:e|a)rine(?:\s+vir?ginis)?"),
    ("margrethe",            r"marg[ae]?ret[h]?e(?:\s+vir?ginis)?"),
    ("cecilie",              r"cecilie(?:\s+vir?ginis)?"),
    ("lucie",                r"(?:angaria[m]?\s+)?lucie(?:\s+vir?ginis)?"),
    ("angnetis",             r"a[n]?gnetis(?:\s+vir?ginis)?"),
    ("agathe",               r"agat[h]?e(?:\s+vir?ginis)?(?:\s+martyris)?"),
    ("scolastice",           r"sc[h]?olastic(?:ae|e)(?:\s+vir?ginis)?"),
    ("elisabet",             r"eli[sz]abet[h]?"),
    ("innocentum",           r"innocentum|kindlein\w*tag\w*"),
    ("laurencii",            r"(?:sancti\s+)?laurencii(?:\s+martyris)?"),
    ("vincencii",            r"(?:beati\s+|sancti\s+)?vincencii(?:\s+mart[yi]ris)?"),
    ("clementis",            r"clementis(?:\s+pape\s+et)?\s+martyris"),
    ("lamperti",             r"lamperti(?:\s+(?:martyris|episcopi(?:\s+et\s+(?:martyris|confessoris))?))?"),
    # The bare "viti" ("in die viti", "feria quarta ante viti") needs the \b: without the
    # epithet the alternation would otherwise be free to match inside a longer word during
    # a free-text scan. Veit is also a personal name here, hence the _AMBIGUOUS entry.
    ("viti",                 r"\bviti(?:\s+martyris)?"),
    ("ypoliti",              r"ypoliti(?:\s+martyr(?:um|is))?"),
    ("blasii",               r"blasii(?:\s+episcopi)?"),
    ("nicolai",              r"(?:sancti\s+)?nicolai(?:\s+episcopi)?"),
    ("erhardi",              r"erhardi(?:\s+episcopi)?"),
    ("juliani episcopi",     r"juliani\s+episcopi"),
    ("wilhelmi episcopi",    r"wilhelmi\s+episcopi(?:\s+et\s+confessoris)?"),
    ("briccii",              r"briccii(?:\s+episcopi)?"),
    ("ambrosii",             r"ambrosii(?:\s+episcopi)?(?:\s+confessoris)?"),
    ("emundi episcopi",      r"(?:ae|e)(?:d)?mundi(?:\s+episcopi)?"),
    # "Ulrici" is the same bishop of Augsburg with the syncopated German form of the name;
    # it occurs alongside "Vdalrici" in L344 ("dominica post ulrici").
    ("udalrici",             r"[uv](?:da)?lrici(?:\s+episcopi)?"),
    ("mauricii",             r"(?:sancti\s+)?maurici[i]?(?:\s+et\s+sociorum\s+eius)?"),
    ("jeronimi",             r"jeronimi(?:\s+confessoris)?"),
    ("gregori pape",         r"gregori[i]?(?:\s+pape)?"),
    ("gervasii et prothasii", r"gervasii\s+et\s+p(?:ro|or)t[h]?asii"),
    ("floriani",             r"floriani(?:\s+mart[yi]ris)?"),
    # After "translacionis benedicti", which claims the July feast.
    ("benedicti",            r"(?:sancti\s+)?benedicti(?:\s+abbat[t]?is)?"),
    ("sixti pape",           r"\bsixti(?:\s+pape)?"),
    ("francisci",            r"francisci(?:\s+confessoris)?"),
    ("dionysii",             r"d[iy]on[iy]si[i]?"),
    ("simonis et jude",      r"s[iy]monis\s+et\s+[ji]ude(?:\s+apostolorum)?"),
    ("sebastiani et fabiani", r"(?:sebastiani\s+et\s+fabiani|fabiani\s+et\s+sebastiani)"
                              r"(?:\s+mart[yi]rum)?"),
    # Both Felix feasts REQUIRE their qualifier: a bare "felicis" is not a feast here but
    # the adjective, as in "antecessori nostro felicis memorie".
    ("felicis in pincis",     r"felicis\s+in\s+pincis"),
    ("felicis et adaucti",    r"felicis\s+et\s+(?:ad\s+)?(?:ad)?aucti(?:\s+mart[yi]rum)?"),
    ("oswaldi regis",        r"oswaldi\s+regis(?:\s+et\s+mart[yi]ris)?"),
    ("kiliani",              r"kiliani"),
    ("egidi",                r"egidi[i]?(?:\s+abbat[t]?is)?"),
    ("barnabe apostoli",     r"[wb]arnabe(?:\s+apostoli)?"),
    ("galli abbatis",        r"galli(?:\s+abbatis)?"),
    ("cholmanni",            r"(?:solucionis\s+)?cholmanni|colmani"),
    ("processi et martiniani", r"processi\s+et\s+martiniani"),
    ("anthoni confessoris",  r"(?:sancti\s+)?anthoni(?:\s+confessoris)?"),
    ("allexi",               r"al[l]?exi[i]?"),
    # The Passau Translatio must precede the bare key, which stays Valentine of Rome.
    ("translacionis valentini", r"translacion(?:is|e|em)\s+(?:sancti\s+)?valentini"
                                r"(?:\s+(?:in\s+patavia|patavie))?"),
    ("valentini",            r"valentini(?:\s+mart[yi]ris)?"),
    ("urbani",               r"(?:beati\s+)?[uv]rbani"),
    ("geori",                r"(?:sancti\s+)?geori[i]?(?:\s+mart[yi]ris)?"),
    ("silvestri",            r"silvestri"),
]

# Feast names that are also common personal names in these ledgers.
#
# Whether "Geori" is the man Georg or the 24th of April is a SELECTION question, not an
# arithmetic one, and it cannot be settled by any rule over the text. So the gate below
# applies only where nobody has selected yet — resolve()/find_all(), which scan free text
# and take the first hit. It does NOT apply to resolve_phrase(): there the model has
# already said "this phrase is the date", and a rule that overrides that answer fires
# exactly where the model was free to abstain. DateExpr.ambiguous carries the flag
# through instead, so the report can single these out for spot-checking without the
# module refusing them.
_AMBIGUOUS = {
    "geori", "martini", "leonhardi", "andree apostoli", "erhardi",
    "wilhelmi episcopi", "michahelis", "johannis baptiste", "johannis apostoli",
    "elisabet", "valentini", "augustini", "bernhardi", "viti",
    # Added 2026-09-01 with the epithet made optional: "Lamperti", "Thome" and "Anthoni"
    # used to be reachable only as "Lamperti martyris", "Thome apostoli", "Anthoni
    # confessoris", and the epithet was itself the guarantee that a feast was meant.
    # Without it the guarantee has to come from the date marker instead.
    "lamperti", "thome apostoli", "anthoni confessoris",
    # Same on 2026-09-01 for these, now reachable without their epithet. "misericordia
    # domini" is not a personal name but belongs here for the same reason: on its own the
    # word is ordinary Latin, and only the date marker makes it the Sunday.
    "marci ewangeliste", "mathei apostoli", "gregori pape", "benedicti",
    "misericordia domini", "cena domini",
    # The ground truth carries "fratris Stepfani prioris nostri" and "Iudicis nostri
    # Stepfani Hollerbekch" — both men, both undated.
    "stephani",
}


# ──────────────────────────── Grammar ────────────────────────────
# Weekday names. Both word and flattened-superscript numeral forms; the numerals are the
# more frequent of the two in the RDF entry texts.
# The `-ta`/`-da`/`-cia` forms are the same flattened superscript as the bare `a`, just
# with more of the ending written out (`quin<hi>ta</hi>` -> `5ta`). 7 occurrences in
# L342-L346; without them the weekday is skipped and the feast resolves on its own day.
#
# The numeral may also be ROMAN, with the same flattened superscript: `feria v<hi>a</hi>`
# arrives as `feria va`. Counted over L342-L346: 41 × "feria via", 37 × "feria va", 2 each
# for the bare "feria v" and "feria vi". The lower ordinals are not attested in that form
# but cost nothing, since "feria" has to stand next to the numeral for any of this to fire.
#
# Feria septima is Saturday, the same day as "sabbato" (Grotefend: feria VII = sabbatum).
# The count is closed — feria II is Monday, feria VI is Friday — so VII admits no other
# reading. Marginal in the corpus (753 × "sabbato", 1 entry with "feria viia"), but the
# ordinals are still listed one by one rather than matched as roman numerals generally,
# so an uninterpreted form stays refused instead of quietly becoming a day.
_FERIA_ORD = {
    "secunda": 1, "2": 1, "2a": 1, "2da": 1,
    "ii": 1, "iia": 1, "iida": 1,
    "tertia": 2, "tercia": 2, "3": 2, "3a": 2, "3cia": 2, "3tia": 2,
    "iii": 2, "iiia": 2, "iiicia": 2, "iiitia": 2,
    "quarta": 3, "4": 3, "4a": 3, "4ta": 3,
    "iiii": 3, "iiiia": 3, "iiiita": 3, "iv": 3, "iva": 3, "ivta": 3,
    "quinta": 4, "5": 4, "5a": 4, "5ta": 4,
    "v": 4, "va": 4, "vta": 4,
    "sexta": 5, "6": 5, "6a": 5, "6ta": 5,
    "vi": 5, "via": 5, "vita": 5,
    "septima": 6, "7": 6, "7a": 6, "7ma": 6,
    "vii": 6, "viia": 6, "viita": 6,
}
FERIA_WEEKDAY = {"sabbato": 6, "sabbatho": 6, "dominica": 7, "dominicam": 7,
                 # "domica"/"domicam" are transcription variants that occur in the corpus
                 "domica": 7, "domicam": 7}
for _o, _v in _FERIA_ORD.items():
    FERIA_WEEKDAY[f"feria {_o}"] = _v
    FERIA_WEEKDAY[f"{_o} feria"] = _v

_ORD_RE = "|".join(sorted(_FERIA_ORD, key=len, reverse=True))

# "dominica 2a post trinitatis" — the SECOND Sunday after Trinity, not the first. A plain
# cardinal count, so it cannot reuse _FERIA_ORD, whose values are weekday numbers
# (feria secunda = Monday = 1, but "dominica 2a" means the 2nd Sunday).
_NTH = {"prima": 1, "1": 1, "1a": 1, "i": 1, "ia": 1,
        "secunda": 2, "2": 2, "2a": 2, "ii": 2, "iia": 2,
        "tertia": 3, "tercia": 3, "3": 3, "3a": 3, "iii": 3, "iiia": 3,
        "quarta": 4, "4": 4, "4a": 4, "iiii": 4, "iiiia": 4, "iv": 4, "iva": 4}
_NTH_RE = "|".join(sorted(_NTH, key=len, reverse=True))
_FERIA_RE = rf"(?:feria\s+(?:{_ORD_RE})|(?:{_ORD_RE})\s+feria|sabbath?o|domi(?:ni)?cam?)"

_FEAST_RE = "|".join(f"(?:{alt})" for _key, alt in _VARIANTS)

# The full expression. Everything before <feast> is optional.
#
# Two feria slots, and the distinction between them matters:
#   <feria> + <direction>  "feria quarta post dominicam Iudica" — the Wednesday after
#                          Judica. Fully supported.
#   <feria_bare>           "feria quinta penthecostes" — the Thursday WITHIN Pentecost
#                          week. A different rule, and one this module refuses rather
#                          than implements. It has to be captured all the same: without
#                          the slot the regex would skip the weekday, match "penthecostes"
#                          alone and return Pentecost Sunday — silently off by four days.
#                          Refusing shows up in the report; guessing would not.
# The one bare form that is safe is "dominica <sunday feast>", where the weekday merely
# restates what the feast already is ("dominica oculi" = Oculi).
# "in die", "in festo", "diem", or a bare "in". This appears in TWO places, and the second
# one is why: "Sabbato ante diem palmarum" puts it AFTER the direction, where a single
# leading slot cannot reach it. That one omission accounted for 22 of the 123 phrases the
# first v5 run could not resolve — by far the largest single cause.
_DIE = r"(?:in\s+)?(?:die[m]?|fest(?:um|o|is))\s+|in\s+"

_EXPR = re.compile(
    # NOTE: a leading "de" is deliberately NOT consumed here. "de angaria crucis",
    # "de festo penthecostes et laurencii" name the TERM a payment is for, not the day it
    # was made, and several of them name more than one term at once. Refusing the whole
    # expression is the point — see _TERM_MARKER and test_term_markers_stay_unresolved.
    rf"(?P<lead>\b(?:{_DIE}))?"
    r"(?P<vigilia>\b(?:in\s+)?(?P<previgilia>pre|pro)?vigilia\s+(?:sancti\s+)?)?"
    # `post post` / `ante ante` and a stray "+" are not sloppy transcription: they are what
    # is left when a <del>…</del> or <add>+ </add> in the TEI is flattened into bk:entry.
    # "infra octavam X" — the named weekday falling inside the octave of X, i.e. the first
    # such weekday after X. That is the same arithmetic as "post", so it is treated as a
    # third direction rather than as its own rule.
    # "dominica die ante Steffani", "sabbato die post bartholomei" — a pleonastic "die"
    # between the weekday and the direction, 11 occurrences.
    rf"(?:(?P<feria>{_FERIA_RE})\s+(?:die\s+)?(?:(?P<nth>{_NTH_RE})\s+)?"
    rf"(?:(?P<direction>post|ante)(?:\s+(?:post|ante))?|(?P<infra>infra\s+octava[ms]?))"
    rf"\s+(?:(?:\+|pro)\s*)?)?"
    rf"(?P<feria_bare>{_FERIA_RE}\s+)?"
    # "feria via videlicet in die scolastice" — the weekday and the feast are in apposition,
    # naming the same day, so the weekday adds no offset.
    r"(?P<videlicet>videlicet\s+)?"
    r"(?P<octava>\b(?:in\s+)?octava[ms]?\s+)?"
    rf"(?P<festum>\b(?:{_DIE}))?"
    r"(?P<sancti>\bsanct[eio]\s+)?"
    rf"(?P<feast>{_FEAST_RE})"
    # "Anno domini etc. lvo"; and a bare ordinal year after the feast ("dominica Iudica
    # lviio"), which is only accepted with the ordinal "o" so that a plain roman numeral —
    # an amount, most of the time — is never mistaken for a year.
    # After an explicit "Anno" the token may carry any flattened ordinal ending; parse_anno
    # decides which letters belong to the number and refuses what it cannot place.
    # "de anno etc. lviiio" — the same partitive "de" as above, here in front of the year.
    # "Anno quo supra" points back at a year already stated, which is the fallback the
    # caller passes anyway. Consumed only so the phrase matches in full; sets no roman
    # group, so the year and year_source stay the fallback's.
    r"(?:(?P<anno_supra>\s+(?:de\s+)?ann[oi]\b\s+(?:quo|ut|ubi)\s+supra\b(?:\s+etc\.?)?)"
    r"|(?P<anno>\s+(?:de\s+)?ann[oi]\b(?:\s+domini)?(?:\s+etc\.?)?\s*(?P<roman>[mdclxvij]+[a-z]{0,3})\b)"
    # A bare ordinal year with no "Anno" ("dominica Iudica lviio") stays strict: the ending
    # must be there, so a plain roman numeral — an amount, most of the time — is never
    # mistaken for a year.
    r"|(?P<anno2>\s+(?P<roman2>[mdclxvij]+(?:cio|tio|cii|tii|mo|to|no|do|vo|io|mi|ti|ni|di|o))\b))?",
    re.IGNORECASE,
)

_SUNDAY_WORD = re.compile(r"domi(?:ni)?cam?\s*\Z", re.IGNORECASE)

# A phrase opening with "de" names the accounting TERM the money is for, not the day it
# changed hands — "de angaria crucis et lucie" names two Ember terms at once, and no single
# day can be both. These are refused, and named as such so the report does not file them
# under "phrase is not a single date expression", where they look like a lexicon gap.
# "de anno lviiio" is the exception: a bare year statement, handled by _YEAR_ONLY.
_TERM_MARKER = re.compile(r"\s*de\s+(?!ann[oi]\b)", re.IGNORECASE)

# A phrase consisting of nothing but a stated year — see resolve_phrase().
_YEAR_ONLY = re.compile(
    r"\s*(?:de\s+)?ann[oi]\b(?:\s+domini)?(?:\s+etc\.?)?\s*(?P<roman>[mdclxvij]+[a-z]{0,3})\s*",
    re.IGNORECASE)

# The same, but naming no numeral: "Anno quo supra" alone is the fallback year, at year
# precision, and taken from the caller rather than from the entry.
_YEAR_AS_ABOVE = re.compile(
    r"\s*(?:de\s+)?ann[oi]\b\s+(?:quo|ut|ubi)\s+supra\b(?:\s+etc\.?)?\s*", re.IGNORECASE)

# Individual patterns, anchored, for identifying WHICH key a match belongs to.
_KEY_RES = [(key, re.compile(rf"(?:{alt})\Z", re.IGNORECASE)) for key, alt in _VARIANTS]

# Phrases that mark a date as something other than the day of the transaction: a payment
# deadline, or a settlement day recorded after the fact. Exported for the report — this
# module never acts on them, because deciding which date an entry is "about" is exactly
# the judgement that belongs to the model.
DEADLINE_MARKERS = re.compile(
    r"\b(?:terminus\s+solucionis|tempus\s+solucionis|solucionem\s+faciet"
    r"|solvet|solvere\s+debet|acta\s+sunt\s+hec|proxime\s+(?:futur|ventur)\w*)\b",
    re.IGNORECASE,
)

# Constructions seen in the corpus that are NOT handled, listed so the report can name
# them rather than reporting a bare "unresolved":
#   * "feria quinta penthecostes"  — weekday inside a feast week, no ante/post
#   * "dominica infra octavam nativitatis marie" — Sunday within an octave
#   * "post festum X" with no weekday — genuinely no single day; year precision is right
#   * date ranges ("de … usque ad …")
UNSUPPORTED_HINTS = (
    # A direction with no weekday: "ante Epiphaniam domini", "post festum Geori martyris".
    # These name a stretch of time, not a day, and the bare year is the right outcome.
    (re.compile(r"\A(?:ante|post)\b", re.I), "a direction with no weekday — names no single day"),
    # "tempore vindemie" (the vintage), "tempore stifte": a season or a rent term.
    (re.compile(r"\A(?:per\s+idem\s+)?tempor[ei]\b", re.I), "'tempore X' — a season or term, not a day"),
    (re.compile(rf"{_FERIA_RE}\s+(?:{_FEAST_RE})", re.I), "weekday within a feast week (no ante/post)"),
    (re.compile(r"\binfra\s+octava[ms]?\b", re.I), "'infra octavam' — day within an octave"),
    (re.compile(r"\busque\s+ad\b", re.I), "date range"),
)


# ──────────────────────────── Text normalisation ────────────────────────────
_HYPHEN = re.compile(r"(\w)-\s+(\w)")
_WS = re.compile(r"\s+")
# Editorial signs flattened out of the TEI that have no reading of their own. "Ø" marks an
# entry in the manuscript (206 occurrences in L344, nearly all at the start of one) but
# lands mid-phrase often enough to matter: "feria 6a ante dominicam Ø Vocem Iocunditatis"
# is the same expression as the three other Vocem Iocunditatis entries, which all resolve.
# Removing it here covers every position at once, rather than adding a slot to the grammar
# for each place it happens to turn up.
_EDITORIAL = re.compile(r"[Ø∅]")


def normalize(text: str) -> str:
    """Undo what the TEI→JSON conversion left behind: manuscript line-break hyphenation
    ("penthe- costes" → "penthecostes", 29 occurrences in L343), editorial signs, and
    collapsed whitespace.

    Note that spans returned by resolve() refer to the NORMALIZED text. A caller that
    needs to check a model-supplied span against the raw bk:entry should normalize both
    sides before comparing."""
    return _WS.sub(" ", _EDITORIAL.sub(" ", _HYPHEN.sub(r"\1\2", text))).strip()


# ──────────────────────────── Year ────────────────────────────
_ROMAN = {"m": 1000, "d": 500, "c": 100, "l": 50, "x": 10, "v": 5, "i": 1, "j": 1}


def roman_to_int(s: str) -> int | None:
    """Roman numeral to int. Returns None on anything that is not a clean numeral —
    a stray letter, an abbreviation stroke, mojibake. Refusing matters more than
    parsing here: a misparsed year would LICENSE a wrong date rather than block one."""
    s = s.strip().lower()
    if not s or not all(c in _ROMAN for c in s):
        return None
    total, prev = 0, 0
    for ch in reversed(s):
        v = _ROMAN[ch]
        if v < prev:
            total -= v
        else:
            total += v
            prev = v
    return total


# The ordinal ending written as a superscript on the year numeral, flattened by the
# TEI→JSON conversion exactly like `feria 5ta`: `lxv<hi>to</hi>` arrives as `lxvto`
# (sexagesimo quinTO), `l<hi>mo</hi>` as `lmo` (quinquagesiMO), `lxvii<hi>mi</hi>` as
# `lxviimi` (the genitive, "anni ... septimi"). 240 of 1073 year statements in
# L342-L345 carry one of these; before 2026-08-26 all of them were unreadable.
#
# They cannot simply be stripped, because their first letter is often a roman numeral
# too: in `lvio` the `i` belongs to the number (56), in `limo` it does not (51). So the
# endings are tried SHORTEST FIRST and the first plausible reading wins — which prefers
# the longest numeral, and leaves the plain `-o` (829 cases) untouched.
_ORDINAL_ENDINGS = ("", "o", "mo", "to", "no", "do", "vo", "io", "mi", "ti", "ni", "di",
                    "cio", "tio", "cii", "tii")   # ter-CIO: `lxiiicio` = 63, 20 occurrences


def parse_anno(roman: str, fallback_year: int | None, tolerance: int = 3) -> int | None:
    """Turn `Anno lvio` into 1456, using the century of fallback_year.

    Guarded twice. The numeral must parse cleanly, and the result must land within
    `tolerance` years of the fallback — a ledger covering 1455–57 cannot suddenly
    produce 1493. Failing either check returns None, and the caller falls back to the
    year the JSON already knows. This is what makes reading the year out of the entry
    safe: the failure mode is 'use the known year', not 'invent one'.

    The tolerance is also what makes trying several splits of the ordinal ending safe:
    a wrong split does not yield a wrong year, it yields no year."""
    if fallback_year is None:
        return None
    token = roman.strip().lower()
    century = (fallback_year // 100) * 100
    for ending in _ORDINAL_ENDINGS:
        if ending and not token.endswith(ending):
            continue
        n = roman_to_int(token[:len(token) - len(ending)] if ending else token)
        if n is None or not 0 < n < 100:
            continue
        # A ledger may run across a century boundary in its last entries.
        for cand in (century + n, century + n + 100, century + n - 100):
            if abs(cand - fallback_year) <= tolerance:
                return cand
    return None


# ──────────────────────────── Resolution ────────────────────────────
@dataclass(frozen=True)
class DateExpr:
    span: str            # the matched phrase, in normalized form
    start: int           # offset into the normalized text
    end: int
    feast: str           # canonical key
    feria: str | None    # e.g. "feria quarta"
    direction: str | None  # "post" | "ante"
    vigilia: bool
    octava: bool
    gated: bool          # the phrase carries an explicit date marker
    ambiguous: bool      # the feast name is also a personal name in these ledgers
    year: int
    year_source: str     # "entry" | "fallback"
    iso: str             # YYYY-MM-DD, or plain YYYY when precision == "year"
    precision: str = "day"   # "day" | "year"


def _identify(feast_text: str) -> str | None:
    for key, rx in _KEY_RES:
        if rx.match(feast_text):
            return key
    return None


def _build(m: re.Match, fallback_year: int | None,
           gate_ambiguous: bool = True) -> tuple[DateExpr | None, str | None]:
    key = _identify(m.group("feast"))
    if key is None:
        return None, "matched a feast alternative but could not identify the key"

    # A bare weekday before the feast names a day INSIDE the feast's week:
    # "feria 3a pasche" is the Tuesday of Easter week, "feria 2a Rogacionum" the Monday
    # after Rogate. Counting is inclusive from the Sunday, so feria secunda = +1.
    #
    # This only has a meaning where the feast IS a Sunday, which the offsets decide for
    # us: every Sunday feast sits a whole number of weeks from Easter. Ascension (+39)
    # and Corpus Christi (+60) are Thursdays and fall out of the test by themselves, so
    # "feria tertia ascensionis" stays refused rather than being invented.
    week_offset, verify_weekday = 0, None
    bare = m.group("feria_bare")
    if bare:
        bw = FERIA_WEEKDAY.get(_WS.sub(" ", bare.lower()).strip())
        if bw is None:
            return None, f"unknown weekday '{bare.strip()}'"
        if m.group("videlicet"):
            # "feria via videlicet in die scolastice" — the weekday and the feast are in
            # apposition and name the same day, so the weekday adds no offset. But it is
            # also a second, independent statement of that day, so it is CHECKED below
            # rather than discarded: if the two disagree, one of them is wrong and the
            # entry should not resolve quietly.
            week_offset, verify_weekday = 0, bw
        elif bw == 7:
            # "dominica oculi" — the weekday only restates what the feast already is.
            week_offset = 0
        elif _is_sunday_feast(key):
            week_offset = bw
        else:
            return None, (f"unsupported construction: weekday within a feast week "
                          f"('{bare.strip()} {m.group('feast')}')")

    gated = bool(m.group("lead") or m.group("vigilia") or m.group("feria")
                 or bare or m.group("octava") or m.group("festum"))
    ambiguous = key in _AMBIGUOUS and not gated
    if ambiguous and gate_ambiguous:
        return None, f"'{m.group('feast')}' is also a personal name and carries no date marker"

    year, source = fallback_year, "fallback"
    # roman  = after an explicit "Anno"; roman2 = a bare ordinal year ("dominica Iudica lviio")
    roman = m.group("roman") or m.group("roman2")
    if roman:
        parsed = parse_anno(roman, fallback_year)
        if parsed is not None:
            year, source = parsed, "entry"
    if year is None:
        return None, "no year available (neither in the entry nor supplied)"

    d = feast_date(key, year)
    if d is None:
        return None, f"no date known for feast '{key}'"

    if m.group("vigilia"):
        # The vigil is the eve, feast − 1. The PRE-vigil is the day before the eve,
        # feast − 2: "in previgilia Nativitatis Christi" is 23 December, not the 24th,
        # which has its own name and which the scribe writes when he means it. This was
        # silently −1 until 2026-09-01, i.e. wrong on all 8 occurrences in L344
        # (7 × previgilia, 1 × provigilia — the same word, the same day).
        d -= timedelta(days=2 if m.group("previgilia") else 1)
    if m.group("octava"):
        # The octave day is counted INCLUSIVELY from the feast, so feast + 7, on the same
        # weekday as the feast: "in octava pasche" is Quasimodogeniti (Easter +7), "in
        # octava nativitatis domini" is 1 January. Corrected from +8 on 2026-08-26; the
        # ground truth carried the same off-by-one and was corrected with it.
        d += timedelta(days=7)
    if week_offset:
        d += timedelta(days=week_offset)

    feria = m.group("feria")
    # "dominica infra octavam nativitatis marie" = the Sunday inside the octave = the first
    # Sunday after the feast. The delta==0 rule below then does the rest: where the feast
    # itself falls on that weekday, the day meant is the one a week later, since the feast's
    # own day is not "inside its octave".
    direction = "post" if m.group("infra") else m.group("direction")
    if feria and direction:
        wd = FERIA_WEEKDAY.get(_WS.sub(" ", feria.lower()).strip())
        if wd is None:
            return None, f"unknown weekday '{feria}'"
        cur = weekday(d)
        # "post": forward to the next such weekday; "ante": back to the previous one.
        # A feast falling on the named weekday itself moves a full week, which is what
        # "the Wednesday after X" means when X is already a Wednesday.
        delta = ((wd - cur) % 7) if direction.lower() == "post" else -((cur - wd) % 7)
        if delta == 0:
            # The feast already falls on the named weekday. "ante X" means before X, and X
            # is not before X — the scribe had "in die X" for that and uses it constantly —
            # so the day meant is the one a week away. L343 has 24 such entries, 7 of them
            # "Sabbato ante Agathe" in 1457, where Agatha is itself a Saturday.
            #
            # This diverges from encode_latin_dates_standalone.py, which returned the feast
            # day (a plain %7 gives 0). The ground truth appeared to back the old behaviour:
            # its one case of this shape, "dominica ante Anthoni confessoris" in 1456 where
            # 13 June is a Sunday, encodes 1456-06-13. But the GT also encodes "dominica
            # ante petri et pauli apostolorum" as 1457-06-29 — and that feast fell on a
            # WEDNESDAY, so the Sunday before it cannot be the feast itself. There the GT
            # demonstrably dropped the "dominica ante" rather than applying a convention.
            # The Anthoni case has the identical shape and is most likely the same slip.
            #
            # So the GT does not settle this either way (it is right in 51 of 56 comparable
            # entries, and these are among its 5 errors). Decided on the semantics.
            delta = 7 if direction.lower() == "post" else -7
        # "dominica 2a post trinitatis" — each further ordinal is one more week out.
        nth = m.group("nth")
        if nth:
            extra = (_NTH[nth.lower()] - 1) * 7
            delta += extra if direction.lower() == "post" else -extra
        d += timedelta(days=delta)

    if verify_weekday is not None and weekday(d) != verify_weekday % 7:
        return None, (f"the phrase names both a weekday and a day ('{m.group(0).strip()}') "
                      f"and they disagree: {d.isoformat()} is not that weekday")

    return DateExpr(
        span=m.group(0).strip(), start=m.start(), end=m.end(), feast=key,
        feria=feria, direction=direction,
        vigilia=bool(m.group("vigilia")), octava=bool(m.group("octava")),
        gated=gated, ambiguous=ambiguous,
        year=year, year_source=source, iso=d.isoformat(),
    ), None


def find_all(text: str, fallback_year: int | None = None) -> tuple[list[DateExpr], list[str]]:
    """Every resolvable date expression in the text, plus the reasons matches were
    rejected. Both lists are what the report needs: several hits mean the entry names
    more than one date and the model's choice matters; the reasons name what the
    module is missing."""
    norm = normalize(text)
    found, reasons = [], []
    for m in _EXPR.finditer(norm):
        expr, why = _build(m, fallback_year)
        if expr is not None:
            found.append(expr)
        elif why:
            reasons.append(why)
    return found, reasons


def resolve(text: str, fallback_year: int | None = None) -> tuple[DateExpr | None, str | None]:
    """The first resolvable date expression in a free-text entry.

    'First' is a positional heuristic and nothing more — use this for reporting and for
    the TEI encoder, not for choosing which date an accounting entry is about. When the
    model has named the phrase, call resolve_phrase() instead."""
    found, reasons = find_all(text, fallback_year)
    if found:
        return found[0], None
    if reasons:
        return None, reasons[0]
    for rx, hint in UNSUPPORTED_HINTS:
        if rx.search(normalize(text)):
            return None, f"unsupported construction: {hint}"
    return None, "no date expression found"


def resolve_phrase(phrase: str, fallback_year: int | None = None) -> tuple[DateExpr | None, str | None]:
    """Resolve a phrase the model picked out of bk:entry.

    Stricter than resolve() in one way and looser in another.

    Stricter: the expression must span the whole phrase, so a model that hands over a
    sentence instead of a date phrase is refused rather than silently resolved on some
    fragment of it.

    Looser: the personal-name gate does not apply. By naming the phrase the model has
    already ruled that "Geori martyris" is a date here and not the man Georg — that is
    the selection half of the task, and it is the half a rule cannot do. The result
    carries .ambiguous so the report can still list these for spot-checking."""
    norm = normalize(phrase)

    # A phrase that is ONLY a year: "Anno lvio", "Anno etc. lviio". The entry states its
    # own year and names no day, so the answer is that year at year precision — taken from
    # the entry, not from the JSON. Both happen to agree in L343, but that is luck: the
    # JSON year comes from the rubric heading, and where the two disagree the entry is the
    # better witness. Resolving it here makes the outcome right by construction.
    y = _YEAR_ONLY.fullmatch(norm)
    if y:
        year = parse_anno(y.group("roman"), fallback_year)
        if year is None:
            return None, f"year not readable or implausible: {norm!r}"
        return DateExpr(span=norm, start=0, end=len(norm), feast="", feria=None,
                        direction=None, vigilia=False, octava=False, gated=True,
                        ambiguous=False, year=year, year_source="entry",
                        iso=f"{year:04d}", precision="year"), None

    if _YEAR_AS_ABOVE.fullmatch(norm):
        if fallback_year is None:
            return None, "no year available (neither in the entry nor supplied)"
        return DateExpr(span=norm, start=0, end=len(norm), feast="", feria=None,
                        direction=None, vigilia=False, octava=False, gated=True,
                        ambiguous=False, year=fallback_year, year_source="fallback",
                        iso=f"{fallback_year:04d}", precision="year"), None

    m = _EXPR.match(norm)
    if not m or m.end() < len(norm):
        if _TERM_MARKER.match(norm):
            return None, ("names an accounting term, not a day ('de …'): the entry says "
                          "which term the payment is FOR, and may name several at once")
        # Name the known-unsupported shapes rather than filing them as a lexicon gap.
        for rx, hint in UNSUPPORTED_HINTS:
            if rx.search(norm):
                return None, f"unsupported construction: {hint}"
        return None, f"phrase is not a single date expression: {norm!r}"
    return _build(m, fallback_year, gate_ambiguous=False)
