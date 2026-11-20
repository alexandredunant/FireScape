"""Compare stylometric features of the author's papers against the review draft."""
import re
import statistics as st
import sys
from pathlib import Path

CONNECTIVES = ["However", "Despite", "Thus", "Therefore", "In contrast", "Similarly",
               "In sum", "Moreover", "Furthermore", "Additionally", "Indeed", "Hence",
               "Notably", "Importantly", "Crucially", "Yet"]
HEDGES = r"\b(may|might|could|likely|unlikely|appears? to|tends? to|suggests?|plausibly|perhaps|possibly)\b"
AI_VOCAB = r"\b(crucial|pivotal|landscape|underscores?|highlights?|nuanced|robust|leverag\w*|delve|multifaceted|interplay|critically|key|central|fundamental\w*|reframe\w*|lens)\b"
ORDINALS = r"(?:^|\. )(First|Second|Third|Finally|Firstly|Secondly)\b[ ,]"
COUNTED = r"(?:^|\. )(Two|Three|Four|Five|Six) [a-z]+ (?:follow|emerge|are|explain|stand|matter|shape|differ|recur)"
PASSIVE = r"\b(is|are|was|were|been|be|being)\s+(\w+ly\s+)?\w+(ed|en)\b"


def clean_pdf(text: str) -> str:
    """Cut references/back matter and drop hyphenation and line wrapping."""
    m = re.search(r"\n\s*(REFERENCES|References)\s*\n", text)
    if m:
        text = text[: m.start()]
    text = re.sub(r"-\n(\w)", r"\1", text)
    # drop parenthetical citations so punctuation counts reflect prose only
    for _ in range(2):
        text = re.sub(r"\s?\([^()]*\b(19|20)\d\d[a-z]?\b[^()]*\)", "", text)
    return text


def clean_tex(text: str) -> str:
    """Strip LaTeX to prose: body only, no comments, floats, citations or commands."""
    text = text.split(r"\begin{document}", 1)[-1]
    text = re.split(r"\\bibliography\{|\\end\{document\}", text)[0]
    text = re.sub(r"(?<!\\)%.*", "", text)
    text = re.sub(r"\\begin\{(figure|table|figure\*|table\*|tabular|equation|align)\}.*?\\end\{\1\}", "", text, flags=re.S)
    text = re.sub(r"\\(begin|end)\{abstract\}", "", text)
    text = re.sub(r"\s?\\citep(\[[^\]]*\])*\{[^}]*\}", "", text)
    text = re.sub(r"\\(cite[pt]?|ref|label|eqref|autoref|url)\*?(\[[^\]]*\])*\{[^}]*\}", "X", text)
    text = re.sub(r"\\(section|subsection|subsubsection|paragraph)\*?\{([^}]*)\}", r"\n\n", text)
    text = re.sub(r"\\(textit|emph|textbf|mbox|textsc)\{([^}]*)\}", r"\2", text)
    text = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^}]*\})?", "", text)
    text = text.replace("~", " ").replace("{", "").replace("}", "")
    text = text.replace("---", "\u2014").replace("--", "\u2013")
    return text


def paragraphs(text: str) -> list[str]:
    paras = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", text)]
    return [p for p in paras if len(p.split()) >= 30]


def sentences(paras: list[str]) -> list[str]:
    out = []
    for p in paras:
        p = re.sub(r"\b(e\.g|i\.e|et al|cf|approx|Fig|Sect|Eq|vs)\.", r"\1", p)
        out += [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z(])", p) if len(s.split()) >= 3]
    return out


def per10k(n: int, words: int) -> float:
    return round(n * 10000 / words, 1)


def metrics(paras: list[str]) -> dict:
    sents = sentences(paras)
    body = " ".join(paras)
    words = len(body.split())
    lens = sorted(len(s.split()) for s in sents)
    conn = {c: len(re.findall(rf"(?:^|[.;] ){c}\b", body)) for c in CONNECTIVES}
    return {
        "words": words,
        "paragraphs": len(paras),
        "words/paragraph (median)": int(st.median(len(p.split()) for p in paras)),
        "sentence length mean": round(st.mean(lens), 1),
        "sentence length median": st.median(lens),
        "sentence length p90": lens[int(0.9 * len(lens))],
        "sentence length max": lens[-1],
        "% sentences > 45 words": round(100 * sum(l > 45 for l in lens) / len(lens), 1),
        "we/our/us per 10k": per10k(len(re.findall(r"\b(we|We|our|Our|us)\b", body)), words),
        "passive constructions per 10k": per10k(len(re.findall(PASSIVE, body)), words),
        "hedges per 10k": per10k(len(re.findall(HEDGES, body, re.I)), words),
        "sentence-initial connectives per 10k": per10k(sum(conn.values()), words),
        "  top connectives": ", ".join(f"{k}:{v}" for k, v in sorted(conn.items(), key=lambda x: -x[1]) if v)[:120],
        "% paragraphs opening 'The'": round(100 * sum(p.startswith("The ") for p in paras) / len(paras), 1),
        "em dashes per 10k": per10k(body.count("\u2014"), words),
        "semicolons per 10k": per10k(body.count(";"), words),
        "colons per 10k": per10k(body.count(":"), words),
        "parentheses per 10k": per10k(body.count("("), words),
        "ordinal scaffolding (First, Second...)": len(re.findall(ORDINALS, body)),
        "counted announcements (Three X follow...)": len(re.findall(COUNTED, body)),
        "AI-flag vocabulary per 10k": per10k(len(re.findall(AI_VOCAB, body, re.I)), words),
    }


def main() -> None:
    s = Path(sys.argv[1])
    corpora = {
        "Frontiers 2021": paragraphs(clean_pdf((s / "frontiers.txt").read_text())),
        "NHESS 2025": paragraphs(clean_pdf((s / "nhess.txt").read_text())),
        "review v7": paragraphs(clean_tex(Path(sys.argv[2]).read_text())),
    }
    res = {k: metrics(v) for k, v in corpora.items()}
    keys = list(next(iter(res.values())).keys())
    print(f"{'metric':45s}" + "".join(f"{k:>18s}" for k in res))
    for key in keys:
        if key.startswith("  "):
            continue
        print(f"{key:45s}" + "".join(f"{str(res[k][key]):>18s}" for k in res))
    for k in res:
        print(f"\n{k} connectives: {res[k]['  top connectives']}")


if __name__ == "__main__":
    main()
