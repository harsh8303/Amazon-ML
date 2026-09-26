"""Text normalisation for business names and addresses.

Everything here is rule-based and country-agnostic in structure: dictionaries only
canonicalise spelling variants (abbreviations, legal forms, state names), they never
filter on the country label. Non-Latin scripts are romanised with unidecode and every
token also gets a phonetic "consonant skeleton" so that e.g. "marketing" and the
Devanagari romanisation "maarkettiNg" both become "mrktng".
"""
import re
from functools import lru_cache

import numpy as np
from unidecode import unidecode

# ---------------------------------------------------------------- legal forms
LEGAL = {
    "inc": "inc", "incorporated": "inc", "incorporation": "inc",
    "corp": "corp", "corporation": "corp", "corpn": "corp",
    "co": "co", "company": "co", "cie": "co",
    "llc": "llc", "pllc": "llc",
    "ltd": "ltd", "limited": "ltd", "ltda": "ltd", "lt": "ltd",
    "pvt": "pvt", "private": "pvt", "pte": "pvt", "prv": "pvt", "priv": "pvt",
    "llp": "llp", "lp": "lp", "plc": "plc", "pc": "pc",
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "sci": "sci", "eurl": "eurl",
    "sa": "sa", "snc": "snc", "scop": "scop", "selarl": "selarl",
}
# legal words written in Indic scripts romanise to these skeletons
LEGAL_SKEL = {"prvt": "pvt", "pvt": "pvt", "pr": "pvt", "lmtd": "ltd", "ltd": "ltd",
              "l": "ltd", "lp": "llp", "lk": "llc", "nk": "inc", "krp": "corp",
              "krprsn": "corp", "kmpn": "co"}

# ------------------------------------------------------------- address terms
ADDR = {
    # english street types
    "street": "st", "str": "st", "saint": "st", "st": "st",
    "road": "rd", "rd": "rd", "avenue": "ave", "av": "ave", "ave": "ave", "aven": "ave",
    "drive": "dr", "dr": "dr", "drv": "dr", "lane": "ln", "ln": "ln",
    "court": "ct", "ct": "ct", "crt": "ct", "place": "pl", "pl": "pl",
    "boulevard": "blvd", "blvd": "blvd", "bd": "blvd", "bld": "blvd", "boul": "blvd",
    "circle": "cir", "cir": "cir", "trail": "trl", "trl": "trl",
    "highway": "hwy", "hwy": "hwy", "parkway": "pkwy", "pkwy": "pkwy", "pky": "pkwy",
    "terrace": "ter", "ter": "ter", "square": "sq", "sq": "sq",
    "north": "n", "south": "s", "east": "e", "west": "w",
    "northeast": "ne", "northwest": "nw", "southeast": "se", "southwest": "sw",
    "suite": "ste", "ste": "ste", "apartment": "apt", "apt": "apt", "unit": "unit",
    "floor": "fl", "fl": "fl", "building": "bldg", "bldg": "bldg",
    "mount": "mt", "mt": "mt", "fort": "ft", "ft": "ft", "point": "pt", "pt": "pt",
    "center": "ctr", "centre": "ctr", "ctr": "ctr", "expressway": "expy",
    "way": "way", "wy": "way", "plaza": "plz", "plz": "plz",
    # indian address words
    "number": "no", "no": "no", "num": "no", "hno": "no", "h": "h",
    "near": "nr", "nr": "nr", "opposite": "opp", "opp": "opp", "behind": "bh",
    "sector": "sec", "sec": "sec", "nagar": "ngr", "ngr": "ngr",
    "colony": "col", "col": "col", "village": "vill", "vill": "vill", "vpo": "vill",
    "district": "dist", "dist": "dist", "distt": "dist", "tehsil": "teh", "teh": "teh",
    "post": "po", "po": "po", "marg": "marg", "chowk": "chowk", "cross": "crs",
    "main": "main", "phase": "ph", "ph": "ph", "industrial": "ind", "indl": "ind",
    "area": "area", "estate": "est", "complex": "cmplx", "plot": "plot",
    "flat": "flat", "door": "door", "shop": "shop", "gali": "gali", "ward": "ward",
    # french street types
    "rue": "r", "r": "r", "chemin": "ch", "ch": "ch", "route": "rte", "rte": "rte",
    "impasse": "imp", "imp": "imp", "quai": "q", "q": "q", "allee": "all", "all": "all",
    "cours": "crs", "crs": "crs", "faubourg": "fbg", "fbg": "fbg", "residence": "res",
    "res": "res", "lieu": "ld", "ld": "ld", "lieudit": "ld", "bis": "bis", "ter_": "ter",
    "sainte": "ste_", "ste_": "ste_",
}
NULL_TOKENS = {"null", "none", "nan", "na", "n/a", "unknown"}
NAME_STOP = {"and", "of", "the", "de", "la", "le", "les", "du", "des", "et", "d", "l", "a", "an"}

US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia",
    "kansas": "ks", "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn", "mississippi": "ms",
    "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv",
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny",
    "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok",
    "oregon": "or", "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc",
    "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt",
    "virginia": "va", "washington": "wa", "west virginia": "wv", "wisconsin": "wi",
    "wyoming": "wy", "district of columbia": "dc", "puerto rico": "pr",
}
IN_STATES = {
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br",
    "chhattisgarh": "cg", "chattisgarh": "cg", "goa": "ga", "gujarat": "gj", "haryana": "hr",
    "himachal pradesh": "hp", "jharkhand": "jh", "karnataka": "ka", "kerala": "kl",
    "madhya pradesh": "mp", "maharashtra": "mh", "manipur": "mn", "meghalaya": "ml",
    "mizoram": "mz", "nagaland": "nl", "odisha": "od", "orissa": "od", "punjab": "pb",
    "rajasthan": "rj", "sikkim": "sk", "tamil nadu": "tn", "telangana": "ts",
    "tripura": "tr", "uttar pradesh": "up", "uttarakhand": "uk", "uttaranchal": "uk",
    "west bengal": "wb", "paschim banga": "wb", "paschim bangal": "wb", "delhi": "dl",
    "jammu and kashmir": "jk", "jammu kashmir": "jk", "ladakh": "la", "puducherry": "py",
    "pondicherry": "py", "chandigarh": "ch", "andaman and nicobar islands": "an",
    "dadra and nagar haveli": "dn", "daman and diu": "dd", "lakshadweep": "ld",
}

_DIGRAPHS = [("ph", "f"), ("bh", "b"), ("kh", "k"), ("gh", "g"), ("th", "t"), ("dh", "d"),
             ("sh", "s"), ("ch", "c"), ("ck", "k"), ("q", "k"), ("c", "k"), ("z", "j"),
             ("w", "v"), ("x", "ks")]
_VOWELS = re.compile(r"[aeiouyh]")
_REPEAT = re.compile(r"(.)\1+")


@lru_cache(maxsize=2_000_000)
def skeleton(tok: str) -> str:
    """Consonant skeleton: robust to vowel spelling and Indic romanisation."""
    s = tok
    for a, b in _DIGRAPHS:
        s = s.replace(a, b)
    s = _REPEAT.sub(r"\1", s)
    k = _REPEAT.sub(r"\1", _VOWELS.sub("", s))
    return k if k else s[:1]


def _state_key(s):
    return "".join(skeleton(t) for t in s.split())


STATE_SKEL = {}
for _d in (US_STATES, IN_STATES):
    for _full, _code in _d.items():
        STATE_SKEL[_state_key(_full)] = _code
STATE_CODES = set(US_STATES.values()) | set(IN_STATES.values())

_NONASCII = re.compile(r"[^\x00-\x7f]")
_LATIN_EXT = re.compile(r"[À-ɏ]")
_NON_LATIN = re.compile(r"[^\x00-\x7fÀ-ɏ -⁯]")
_DOMAIN = re.compile(r"^\s*(?:www\.)?([a-z0-9\-]+)\.(com|in|net|org|co|biz|info|fr|us|io|co\.in)\s*$")
_DOTTED = re.compile(r"(?<=\b[a-z])\.(?=[a-z]\b)")
_PUNCT = re.compile(r"[^a-z0-9 ]+")
_SPACES = re.compile(r"\s+")
_NUM = re.compile(r"\d+")


def to_ascii(s: str) -> str:
    if s is None:
        return ""
    if _NONASCII.search(s):
        s = unidecode(s)
    return s.lower()


def norm_name(raw: str):
    """Returns (core_tokens, legal_tokens, is_domain, is_nonlatin)."""
    raw = raw or ""
    nonlatin = bool(_NON_LATIN.search(raw))
    s = to_ascii(raw).strip()
    is_dom = False
    m = _DOMAIN.match(s)
    if m:
        s = m.group(1).replace("-", " ")
        is_dom = True
    elif " " not in s and len(s) >= 9 and s.endswith(("com", ".in", ".net", ".org")):
        # concatenated domain with the dot dropped: "creativesolutionscom"
        s = re.sub(r"\.?(com|in|net|org)$", "", s)
        is_dom = True
    s = s.replace("&", " and ").replace("'", "")
    s = _DOTTED.sub("", s)          # l.l.c -> llc
    s = s.replace(".", " ")
    s = _PUNCT.sub(" ", s)
    toks = [t for t in s.split() if t not in NULL_TOKENS]
    core, legal = [], []
    for t in toks:
        if t in NAME_STOP:
            continue
        if t in LEGAL:
            legal.append(LEGAL[t])
        elif nonlatin and skeleton(t) in LEGAL_SKEL:
            legal.append(LEGAL_SKEL[skeleton(t)])
        else:
            core.append(t)
    # dedupe while keeping order (noise often repeats tokens: "Inc Inc", "Group Group")
    core = list(dict.fromkeys(core))
    legal = list(dict.fromkeys(legal))
    return core, legal, is_dom, nonlatin


def norm_address(raw: str):
    """Returns (tokens, numbers, state_code, components)."""
    raw = raw or ""
    s = to_ascii(raw)
    s = s.replace("&", " and ").replace("'", "")
    comps_out, toks, state = [], [], ""
    for comp in s.split(","):
        c = _DOTTED.sub("", comp).replace(".", " ")
        c = _PUNCT.sub(" ", c)
        ct = [t for t in c.split() if t not in NULL_TOKENS]
        if not ct:
            continue
        key = "".join(skeleton(t) for t in ct if not t.isdigit())
        if key in STATE_SKEL:
            state = STATE_SKEL[key]
            continue
        if len(ct) == 1 and ct[0] in STATE_CODES and len(comps_out) > 0:
            state = ct[0]
            continue
        ct = [(t.lstrip("0") or "0") if t.isdigit() else ADDR.get(t, t) for t in ct]
        comps_out.append(" ".join(ct))
        toks.extend(ct)
    nums = _NUM.findall(" ".join(toks))
    # also join digit groups split by '-' or '/' e.g. "29-04" -> "2904"
    joined = re.findall(r"\d+(?:[-/]\d+)+", s)
    nums += [re.sub(r"\D", "", j).lstrip("0") for j in joined]
    nums = [x for x in nums if x]
    return toks, list(dict.fromkeys(nums)), state, comps_out


class Segmenter:
    """Unigram Viterbi word segmentation of concatenated tokens ("raamvallalarprivate"
    -> "raam vallalar private") with a vocabulary learnt from Source-1 names."""

    def __init__(self, counts, max_len=20):
        total = float(sum(counts.values()))
        self.cost = {w: np.log(total / c) for w, c in counts.items() if len(w) >= 2 or w in ("a",)}
        self.max_len = max_len
        self.unk = np.log(total) + 5.0

    def split(self, text):
        n = len(text)
        best = [0.0] + [float("inf")] * n
        back = [0] * (n + 1)
        for i in range(1, n + 1):
            for j in range(max(0, i - self.max_len), i):
                w = text[j:i]
                c = self.cost.get(w)
                if c is None:
                    c = self.unk * (i - j)  # unknown chars are very expensive
                v = best[j] + c
                if v < best[i]:
                    best[i], back[i] = v, j
        out, i = [], n
        while i > 0:
            out.append(text[back[i]:i])
            i = back[i]
        return out[::-1]


def resegment(nm, legal, seg, nonlatin=False):
    """Split long out-of-vocabulary tokens that are concatenations of known words
    ("raamvallalarprivate", "greatmanagementcom"); return (nm, legal, nsk).
    A split is accepted only if every piece is a known Source-1 word of >= 3 letters, so
    ordinary typos ("consuihancy") and romanised Indic words are left untouched."""
    if nonlatin:
        return None
    core, leg = [], legal.split() if legal else []
    changed = False
    for t in nm.split():
        parts = [t]
        if len(t) >= 8 and t not in seg.cost:
            base = t[:-3] if t.endswith("com") and len(t) > 6 else t
            cand = seg.split(base)
            if len(cand) <= 5 and all(len(w) >= 3 and w in seg.cost for w in cand):
                parts, changed = cand, True
            elif base != t and base in seg.cost:
                parts, changed = [base], True
        for w in parts:
            if w in NAME_STOP:
                continue
            if w in LEGAL:
                leg.append(LEGAL[w])
            else:
                core.append(w)
    if not changed:
        return None
    core = list(dict.fromkeys(core))
    leg = list(dict.fromkeys(leg))
    return " ".join(core), " ".join(leg), " ".join(skeleton(t) for t in core)


def process_record(name, addr):
    core, legal, is_dom, nonlatin = norm_name(name)
    atoks, nums, state, comps = norm_address(addr)
    return (
        " ".join(core),
        " ".join(legal),
        " ".join(skeleton(t) for t in core),
        is_dom,
        nonlatin,
        " ".join(atoks),
        " ".join(nums),
        state,
        "|".join(comps),
    )


def process_batch(args):
    names, addrs = args
    return [process_record(n, a) for n, a in zip(names, addrs)]
