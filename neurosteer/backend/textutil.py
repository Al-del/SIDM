import re

STOP = set("""a about above after again against all also am an and any are as at be because been before being below
between both but by can could did do does doing down during each even few for from further had has have having he her
here hers him his how i if in into is it its itself just like may me might more most much must my no nor not now of off
on once only or other our out over own same she should so some such than that the their them then there these they
this those through to too under until up very was we were what when where which while who whom why will with would
you your yours one two many often still yet rather way thing things get gets""".split())


def stem(w):
    for suf, n in (("ies", "y"), ("ing", ""), ("ed", ""), ("es", ""), ("s", "")):
        if w.endswith(suf) and len(w) - len(suf) >= 3 and not w.endswith("ss"):
            return w[: -len(suf)] + n
    return w


def words(text):
    return re.findall(r"[a-z][a-z'-]*", (text or "").lower())


def content_words(text):
    return [w for w in words(text) if w not in STOP and len(w) > 2]


def stems(text):
    return {stem(w) for w in content_words(text)}


def jaccard(a, b):
    sa, sb = stems(a), stems(b)
    return round(len(sa & sb) / len(sa | sb), 3) if sa | sb else 1.0


def only_in(a, b):
    sb, out = stems(b), []
    for w in content_words(a):
        if stem(w) not in sb and w not in out:
            out.append(w)
    return out


def overlap_alignment(units, sentence):
    s = stems(sentence)
    if not units or not s:
        return None
    hit = sum(u.get("weight", 1.0) for u in units if any(stem(w.lower()) in s for w in u.get("words") or [u["word"]]))
    total = sum(u.get("weight", 1.0) for u in units) or 1.0
    return round(min(1.0, hit / total), 3)
