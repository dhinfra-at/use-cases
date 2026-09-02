# Author: Maximilian Vogeltanz, University of Graz, 2026
#
# Deterministic resolution of the amounts in the Aldersbach ledgers — the counterpart to
# latin_dates.py, and built for the same reason.
#
# WHY THIS IS NOT THE MODEL'S JOB
# -------------------------------
# Measured on L343 (2204 bookings with money and no "minus"): the numbers the JSONtoRDF model wrote
# agreed with a deterministic reading of the entry text in 75.9 % of bookings. 7.0 % differ
# by EXACTLY the j-stroke, every time in the same direction — the model reads "viij̸" as
# 8.5 where the material means 7.5, because it treats the j as another i. That is a
# systematic misreading of one scribal convention, not arithmetic weakness, and it is
# exactly the shape of error a lexicon fixes and a prompt does not.
#
# So the work splits the way it does for dates: the model SELECTS the numeral phrase out of
# bk:entry (a judgement, checkable against the text), and this module COMPUTES the number.
#
# THE NUMERAL SYSTEM
# ------------------
# Additive, not subtractive: nine is "viiii", forty "xxxx", four "iiii". Where the short
# forms do appear they still mean 9, 40, 4.
#
#     i = 1   v = 5   x = 10   l = 50   c = 100   d = 500   m = 1000
#
# Two conventions carry the weight:
#
#   * THE HALVING STROKE.  "j̸" means "and a half". It adds 0.5 to the numeral in front of
#     it and carries NO value of its own — the j is a scribal form, not another i:
#         j̸ = 0.5   ij̸ = 1.5   iij̸ = 2.5   iiij̸ = 3.5   vj̸ = 5.5   vij̸ = 6.5
#         viij̸ = 7.5
#     Settled 2026-08-26 against the material. This is the reading convertlatin.py has
#     always had; the JSONtoRDF model does not, which is where the 7.0 % come from.
#     "x̸" behaves the same way and adds 9.5.
#
#   * C AS A MULTIPLIER.  One to four i's immediately before C mean hundreds, not the
#     letter C = 100 in sequence:
#         iC = 100   iiC = 200   iiiC = 300   iiiiC = 400
#     So "iC lxxxij̸" = 100 + 81.5 = 181.5, and "iiC vi" = 206.
#     The stroke combines with the multiplier the same way it combines with a plain
#     numeral — it counts HUNDREDS, not units:
#         ij̸C = 1.5 hundreds = 150   iij̸C = 250
#     So "ij̸C xxxiiii" = 150 + 34 = 184.

from __future__ import annotations

import re
from dataclasses import dataclass

# ──────────────────────────── Units ────────────────────────────
# Value of one unit in denarii. Only the units that have a fixed rate appear here;
# everything else resolves to a quantity but has no denarius value, and to_denarii()
# returns None for it. Florins (Rhenish, Hungarian) are the reason that matters: their
# rate against the local currency floated through this period, so converting them would
# invent precision the source does not have.
RATES = {
    "lbdwien": 240.0,   # 1 lb = 8 s = 240 d
    "lbdrat": 240.0,
    "lbdpat": 240.0,
    "s": 30.0,          # 1 s = 30 d
    "d": 1.0,
    "gr": 7.5,          # grossus
    "hr": 0.5,          # hallensis / Heller
    "ob": 0.5,          # obulus
}

# Surface form -> canonical unit key. Longest-first matching is applied when the pattern is
# built, so "lb. d. Wien." wins over "lb.".
_UNIT_FORMS = {
    "lb.d.wien.": "lbdwien", "lb.wien.d.": "lbdwien", "lb.wien.": "lbdwien",
    "lb.wn.d.": "lbdwien", "lb.d.": "lbdwien", "lb.": "lbdwien",
    "lb.rat.": "lbdrat", "lb.pat.": "lbdpat",
    "s.d.wien.": "s", "s.d.": "s", "s.": "s", "ß.": "s", "s.rat.": "s",
    "d.wien.": "d", "d.": "d", "t.wien.": "d", "t.": "d",
    "gr.": "gr", "hall.": "hr", "hall": "hr", "hr.": "hr",
    "ob.": "ob", "obulus": "ob", "obolus": "ob",
    "f.rhen.": "frhen", "fl.rhen.": "frhen", "f.rhen": "frhen", "fl.": "fl", "f.": "fl",
    "f.ung.": "fung",
    "ort": "ort", "ortt": "ort",
    "diebus": "tag", "dies": "tag",
}

# "ort" is a quarter of whatever currency it follows ("i lb. i ort" = 1.25 lb), so it is
# never a unit in its own right and is folded into the preceding term instead.
_FRACTION_UNITS = {"ort": 0.25}

# ──────────────────────────── Numerals ────────────────────────────
_STROKE = "̸"  # the combining long solidus overlay on its own
_HALF = "j̸"        # j with a combining long solidus overlay
_NINE_HALF = "x̸"

_LETTERS = {"m": 1000, "d": 500, "c": 100, "l": 50, "x": 10, "v": 5, "i": 1}
# Subtractive spellings, tolerated though the material is additive.
_SUBTRACTIVE = [("iiiic", "cccc"), ("iiic", "ccc"), ("iic", "cc"), ("ic", "c"),
                ("ix", "viiii"), ("xl", "xxxx"), ("iv", "iiii")]

_HUNDREDS = re.compile(rf"^(i{{1,4}})({re.escape(_HALF)})?c(.*)$", re.IGNORECASE)
_CLEAN = re.compile(r"^[mdclxvi]+$", re.IGNORECASE)


def roman_to_number(token: str) -> tuple[float | None, str | None]:
    """One numeral token -> (value, reason it could not be read).

    Applies the halving stroke and the hundreds multiplier described at the top of the
    file. Returns (None, reason) rather than a guess whenever the token is not a clean
    numeral — a misparse here would silently produce a wrong quantity, which is worse than
    a gap that shows up in the fallback report.
    """
    tok = (token or "").strip()
    if not tok:
        return None, "empty numeral"

    # The hundreds multiplier is read BEFORE the stroke is stripped, because a stroke in
    # front of the C belongs to the multiplier ("ij̸C" = 1.5 hundreds) and not to the units.
    m = _HUNDREDS.match(tok)
    if m:
        hundreds = (len(m.group(1)) + (0.5 if m.group(2) else 0.0)) * 100
        rest = m.group(3)
        if not rest:
            return hundreds, None
        tail, why = _plain(rest)
        return (None, why) if tail is None else (hundreds + tail, None)

    return _plain(tok)


def _plain(token: str) -> tuple[float | None, str | None]:
    """A numeral with no hundreds multiplier: additive letters plus any stroke."""
    tok = token.strip()
    extra = 0.0
    if _HALF in tok:
        extra += 0.5 * tok.count(_HALF)
        tok = tok.replace(_HALF, "")
    if _NINE_HALF in tok:
        extra += 9.5 * tok.count(_NINE_HALF)
        tok = tok.replace(_NINE_HALF, "")
    # A bare stroke ("j̸ sch.") is the whole numeral: half of something.
    tok = tok.strip(" .")
    if not tok:
        return (extra, None) if extra else (None, "empty numeral")

    low = tok.lower()
    if not _CLEAN.match(low):
        return None, f"not a clean numeral: {token!r}"

    for src, dst in _SUBTRACTIVE:
        low = low.replace(src, dst)
    return _additive(low) + extra, None


def _additive(s: str) -> float:
    return float(sum(_LETTERS.get(c, 0) for c in s.lower()))


def misread(numeral_text: str) -> float | None:
    """The value the JSONtoRDF model produces for a numeral, under ITS reading.

    The model applies two rules that are each defensible and wrong together: it takes the
    j for a final i (`vij` = 7 is the ordinary Latin spelling, and heavily represented in
    what it learned), and then adds 0.5 for the stroke on top. Hence `viij̸` = 8.5 where
    the material means 7.5, and `ij̸C` = 200.5 where it means 150. Measured on L343 the
    prompt does not talk it out of this: the model reads correctly only where counting the
    j would spell an impossible numeral (`iiiij̸` -> `iiiii`), which is why the rate is
    ~87 % there and ~0 % wherever `…j` happens to be well formed.

    Reproducing the misreading is what makes a deterministic correction safe. A quantity is
    replaced only when it equals this prediction EXACTLY, so a term the model read some
    other way — or a sum it selected differently — never matches and is left untouched.

    Returns None for a numeral without a stroke (nothing to predict) or one that cannot be
    read at all.
    """
    if _STROKE not in (numeral_text or ""):
        return None
    total = 0.0
    for tok in numeral_text.split():
        # Strip the overlay, then let the j count as the i the model takes it for.
        plain = tok.replace(_STROKE, "").replace("j", "i").replace("J", "i")
        value, _why = roman_to_number(plain)
        if value is None:
            return None
        total += value + 0.5 * tok.count(_STROKE)
    return total


_NUMCHARS = set("mdclxvij") | {_STROKE}


def numeral_runs(text: str) -> list[tuple[str, float, float | None]]:
    """Every numeral in the text as (run, value, the value the model would read).

    Needs no known unit, unlike find_all(), so it also reaches the numerals of commodities
    and services ("viij̸ sch.", "ij̸e") — the stroke is misread there just the same, but that
    vocabulary is open and cannot be tabulated. The third item is None without a stroke.
    """
    toks = normalize(text).split()
    pure = [bool(t) and all(c.lower() in _NUMCHARS for c in t) for t in toks]
    runs, i = [], 0
    while i < len(toks):
        head = ""
        for ch in toks[i]:
            if ch.lower() not in _NUMCHARS:
                break
            head += ch
        # A numeral that is merely the START of a token counts only when it carries a
        # stroke ("ij̸e"); otherwise "lb." would read as the numeral "l".
        if not head or (not pure[i] and _STROKE not in head):
            i += 1
            continue
        parts, nxt = [head], i + 1
        while pure[i] and nxt < len(toks) and pure[nxt]:
            parts.append(toks[nxt])
            nxt += 1
        run = " ".join(parts)
        value, _why = _term_value(run)
        if value is not None:
            runs.append((run, value, misread(run)))
        i = max(nxt, i + 1)
    return runs


# ──────────────────────────── Amounts ────────────────────────────
@dataclass(frozen=True)
class Amount:
    span: str          # the matched text, normalised
    start: int         # offset into the normalised text
    end: int
    quantity: float    # the value, expressed in `unit`
    unit: str          # canonical unit key — for arithmetic and reporting only; the
                       # bk:unit written to the RDF stays the model's decision
    parts: tuple       # ((value, unit), ...) as written, before folding
    subtracted: bool   # a "minus" term was folded in

    @property
    def denarii(self) -> float | None:
        return to_denarii(self.quantity, self.unit)


def to_denarii(quantity: float, unit: str) -> float | None:
    """Value in denarii, or None for a unit with no fixed rate (the florins)."""
    rate = RATES.get(unit)
    return None if rate is None else quantity * rate


# ──────────────────────────── Text ────────────────────────────
_WS = re.compile(r"\s+")
# The currency abbreviations are written with spaces the encoder does not care about
# ("lb. d. Wien."). Collapsing them lets one pattern match every spelling, and it is why
# convertlatin.py does the same dance around its money regex.
_GLUE = [("lb. wien. d.", "lb.wien.d."), ("lb. d. wien.", "lb.d.wien."),
         ("lb. wien.", "lb.wien."), ("lb. wn. d.", "lb.wn.d."), ("lb. rat.", "lb.rat."),
         ("lb. pat.", "lb.pat."), ("s. d. wien.", "s.d.wien."), ("s. rat.", "s.rat."),
         ("d. wien.", "d.wien."), ("t. wien.", "t.wien."), ("fl. rhen.", "fl.rhen."),
         ("f. rhen.", "f.rhen."), ("f. ung.", "f.ung."), ("lb. d.", "lb.d."),
         ("s. d.", "s.d.")]


def normalize(text: str) -> str:
    """Collapse whitespace and glue multi-word currency abbreviations together.

    Case is preserved — only the spacing changes — so offsets stay meaningful for a report
    that wants to quote the entry.
    """
    out = _WS.sub(" ", (text or "").replace("<lb/>", " ")).strip()
    low = out.lower()
    for src, dst in sorted(_GLUE, key=lambda p: -len(p[0])):
        start = 0
        while True:
            i = low.find(src, start)
            if i < 0:
                break
            out = out[:i] + out[i:i + len(src)].replace(" ", "")[:len(dst)] + out[i + len(src):]
            low = out.lower()
            start = i + len(dst)
    return out


# Letters may follow the stroke as well as precede it, so that "ij̸C" is ONE token. Without
# the trailing part the pattern would start matching at the "C", read "C xxxiiii" as 134 and
# drop the "ij̸" in silence — a wrong number, where the multiplier rule gives 184.
_NUM = (rf"(?:[mdclxvi]*(?:{re.escape(_HALF)}|{re.escape(_NINE_HALF)})[mdclxvi]*"
        rf"|[mdclxvi]+)")
_UNIT_RE = "|".join(re.escape(u) for u in sorted(_UNIT_FORMS, key=len, reverse=True))
# One "<numeral> <unit>" term. The numeral may be written as several tokens ("xxx ix").
_TERM = re.compile(rf"\b((?:{_NUM})(?:\s+(?:{_NUM}))*)\s*\.?\s*({_UNIT_RE})",
                   re.IGNORECASE)
_MINUS = re.compile(r"\bminus\b", re.IGNORECASE)
# A numeral, optionally followed by ONE word that is not a currency abbreviation — the
# shape a commodity or service measure takes ("ii schaf.", "i equa", "iiii kubel"), and
# also a bare "ii". See the note in resolve_phrase.
# The trailing word must be a WHOLE token: without that, the numeral run would greedily
# swallow the "l" out of "lb." and read "v lb.d." as 55.
_BARE = re.compile(rf"^((?:{_NUM})(?:\s+(?:{_NUM}))*)(?:\s+(\S+?))?\.?$", re.IGNORECASE)


def _term_value(numeral_text: str) -> tuple[float | None, str | None]:
    """Sum the tokens of one numeral ("xxx ix" -> 39)."""
    total = 0.0
    for tok in numeral_text.split():
        v, why = roman_to_number(tok)
        if v is None:
            return None, why
        total += v
    return total, None


def find_all(text: str) -> tuple[list[Amount], list[str]]:
    """Every amount term in the text, plus the reasons terms were rejected.

    Terms are returned AS WRITTEN — no "minus" folding and no merging of "iii lb. xxx d."
    into one figure, because deciding which terms belong to one encoded node is exactly
    the judgement this module does not make. resolve_phrase() does the folding, on the
    phrase the model has already delimited.
    """
    norm = normalize(text)
    found, reasons = [], []
    for m in _TERM.finditer(norm):
        unit = _UNIT_FORMS[m.group(2).lower()]
        value, why = _term_value(m.group(1))
        if value is None:
            if why:
                reasons.append(why)
            continue
        found.append(Amount(span=m.group(0), start=m.start(), end=m.end(),
                            quantity=value, unit=unit,
                            parts=((value, unit),), subtracted=False))
    return found, reasons


def resolve_phrase(phrase: str) -> tuple[Amount | None, str | None]:
    """Resolve the amount phrase the model copied out of bk:entry.

    Stricter than find_all() in the way latin_dates.resolve_phrase() is: the terms must
    cover the whole phrase, so a model that hands over half a sentence is refused rather
    than resolved on some fragment of it.

    One bk:quantity holds one number, so the phrase must resolve to one:

        "v lb. d."              -> 5 lb        a single term
        "i lb. i ort"           -> 1.25 lb     "ort" is a quarter of the term it follows
        "vi lb. minus lx d."    -> 1380 d      a subtraction, normalised to denarii
        "iii lb. xxx d."        -> refused     two amounts, two bk:Money

    The last case is the point. A phrase naming several units is not one amount that
    happens to need a fraction — the encoding gives each unit its own bk:Money, which is
    what the JSONtoRDF prompt has always asked for. Folding it into "3.125 lb" would invent a
    precision the material never writes: the only fractions in these ledgers are the half
    from the stroke and the quarter from "ort".
    """
    norm = normalize(phrase)
    if not norm:
        return None, "empty phrase"

    terms, covered = [], 0
    for m in _TERM.finditer(norm):
        if norm[covered:m.start()].strip(" .,") not in ("", "minus", "et", "und"):
            return None, f"phrase carries text that is not an amount: {norm!r}"
        value, why = _term_value(m.group(1))
        if value is None:
            return None, why
        negative = bool(_MINUS.search(norm[covered:m.start()]))
        terms.append((value, _UNIT_FORMS[m.group(2).lower()], negative))
        covered = m.end()

    if not terms:
        # No currency unit anywhere. Commodities and services measure themselves with
        # ordinary nouns — "ii schaf.", "i equa", "iiii kubel" — and that vocabulary is
        # open, so it cannot be tabulated the way the currencies are. For this module such
        # a phrase is simply a numeral: the number is what it computes, and bk:unit is the
        # model's decision either way. A bare "ii" takes the same path.
        #
        # Reached only after the term parse has found nothing, so no currency phrase can
        # end up here. And the permissiveness cannot leak into free text: resolve_phrase
        # sees a segment the model has already delimited — find_all() stays strict.
        m = _BARE.fullmatch(norm)
        if m:
            value, why = _term_value(m.group(1))
            if value is None:
                return None, why
            return Amount(span=norm, start=0, end=len(norm), quantity=round(value, 2),
                          unit=None, parts=((value, None),), subtracted=False), None
        return None, f"no amount found in {norm!r}"
    if norm[covered:].strip(" .,"):
        return None, f"phrase carries text that is not an amount: {norm!r}"

    if terms[0][1] == "ort":
        return None, "phrase starts with a fraction word"

    # "ort" is a quarter of the term it follows, not an amount of its own, so fold it in
    # before anything else is decided. "i lb. i ort" is ONE amount of 1.25 lb.
    folded: list[tuple[float, str, bool]] = []
    for value, unit, negative in terms:
        if unit == "ort":
            prev_value, prev_unit, prev_neg = folded[-1]
            share = value * _FRACTION_UNITS["ort"]
            folded[-1] = (prev_value - share if negative else prev_value + share,
                          prev_unit, prev_neg)
            continue
        folded.append((value, unit, negative))

    subtracted = any(neg for _v, _u, neg in folded)
    units = {u for _v, u, _neg in folded}

    if len(folded) == 1:
        value, unit, _neg = folded[0]
        return Amount(span=norm, start=0, end=len(norm), quantity=round(value, 2),
                      unit=unit, parts=((value, unit),), subtracted=False), None

    if not subtracted:
        # Several units and nothing to subtract: this is not one number but several, and
        # the encoding gives each its own bk:Money. Refusing keeps the module out of a
        # judgement that belongs to whoever delimited the phrase.
        return None, (f"phrase names {len(folded)} amounts in {len(units)} unit(s) "
                      f"— encode one bk:Money per unit")

    # A subtraction: normalise to denarii, as the JSONtoRDF step has always done. Every term must have
    # a fixed rate, which is what rules out mixing a florin into the arithmetic.
    total = 0.0
    for value, unit, negative in folded:
        rate = RATES.get(unit)
        if rate is None:
            return None, (f"no fixed rate for {unit} — this currency cannot be "
                          f"normalised to denarii")
        total += -value * rate if negative else value * rate

    return Amount(span=norm, start=0, end=len(norm), quantity=round(total, 2), unit="d",
                  parts=tuple((v, u) for v, u, _neg in folded), subtracted=True), None
