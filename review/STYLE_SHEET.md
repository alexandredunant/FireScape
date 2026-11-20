# Style sheet: Dunant register

Derived from two first-author papers, re-measured 2026-11-19 (style only, never content):
- Dunant (2021), *Are We Missing the Target? A Bias-Variance Perspective on Multi-Hazard Risk Assessment*, Frontiers in Earth Science 9:685301. A Perspective: argumentative, didactic.
- Dunant et al. (2025), *Impacts from cascading multi-hazards using hypergraphs: a case study from the 2015 Gorkha earthquake in Nepal*, NHESS 25:267. A research paper: measured, methodologically humble.

**Target register for the review: NHESS.** First person for methodological and interpretive choices, passive for description and results. No organising analogy (the bias-variance paper is style-only, and a carried analogy sits too close to content).

**Reference passage:** the Introduction of `review_paper_v7.tex` as restyled on 2026-11-19 (approved by the author). When in doubt, match its texture.

## What went wrong in v7 before the 2026-11-19 restyle
The draft had drifted into a flattened register: a short topic sentence, a run of numbers, then an aphoristic closing line ("These are the same positions where fuel is now filling in."). Sentences were uniformly short, sources were anonymised ("One assessment couples…"), connectives had almost disappeared, and a quarter of paragraphs opened with "The". The author writes longer, subordinated sentences that carry the reasoning, names the people whose work is being discussed, and grades interpretation with modals.

## Measured targets (per 10,000 words unless stated)

| Feature | Frontiers 2021 | NHESS 2025 | v7 before | **Target** |
|---|---|---|---|---|
| Median sentence length (words) | 25 | 24 | 20 | **23–27** |
| 90th-percentile sentence length | 50 | 41 | 33 | **38–45** |
| Sentences > 45 words | 11% | 7% | 0.6% | **4–9%** |
| we / our / us | 28 | 106 | 17 | **40–70** (a review uses fewer than a methods paper) |
| Hedges (may, could, likely, suggests, appears to, tends to) | 49 | 59 | 28 | **40–55** |
| Sentence-initial connectives | 12 | 24 | 1.7 | **10–20** |
| Paragraphs opening with "The" | 0% | 4% | 25% | **< 10%** |
| Non-citation parentheses | 159 | 144 | 12 | **40–80** |
| Median paragraph length (words) | 177 | 154 | 118 | **130–170** |

Measure with `scripts/style_metrics.py DIR review/review_paper_v7.tex`, where `DIR` holds `nhess.txt` and `frontiers.txt` (`pdftotext` output of the two papers; not stored in the repository). Citations are stripped before counting.

## Sentence construction
- Build reasoning inside the sentence through subordination: *Because we lack…, we instead use…*; *…, whereas…*; *…, which…*; *Even so, …*.
- Long sentences are part of the register. Allow 45–60 words when the sentence carries a single line of reasoning; split only when it carries two unrelated claims or loses its subject. Avoid anything above ~70 words.
- Simple sentences carry definitions and openings ("A graph comprises a set of nodes connected by edges").
- No aphoristic paragraph closers that restate the paragraph as a slogan. A paragraph may end on an interpretation, but phrase it as one: *We therefore treat X as…*, *These results suggest that…*.
- Use *for example*, *for instance*, *such as*, *as well as*, *respectively* to carry examples and pairings inside the sentence.

## Person and voice
- `we` / `our` for methodological and interpretive choices: *Here we treat*, *We therefore treat*, *We take*, *We found no*, *We stress that*, *We hesitate to*, *In this review we*.
- Passive and impersonal for description, measurement and results.

## Citations
- **Narrative citations are permitted and expected** where the study is the grammatical agent: *Billing et al. (2026) coupled…*, *Conedera et al. (2024) showed that…* (use `\citet{}`). This replaces anonymous agents such as *One assessment…*, *One study…*, *Analysis of… concluded…*.
- Use parenthetical `\citep{}` for supporting evidence, lists of sources, and claims that are not about a single study.
- Attribute accurately: a review is cited as reviewing (*As reviewed by \citet{}*), never as having run the experiments it summarises; a preprint is identified as one.
- Aim for roughly one narrative citation per 150–250 words where a single study carries the point; do not convert every citation.
- No direct quotation of sources (unchanged).

## Connectives
Signpost at genuine turns, usually sentence-initial or after the subject: **However, Despite, Thus, Therefore, In contrast, Similarly, Even so, Taken together, At the same time, In sum**. Use at real turns only; roughly one every paragraph or two. *Notably*, *Importantly*, *Crucially* and *We stress* stay rare (one or two in the whole paper; author instruction of 2026-10-06).

## Narrative, not a list of arguments (author instruction, 2026-10-06)
- No counted announcements: *Three features…*, *Two conclusions follow*, *Our argument has four parts*.
- No ordinal scaffolding: *First,… Second,… Finally,…*, *The first is… The second is…*.
- Join paragraphs by the logic that connects them: cause (*because*, *so*), consequence (*It follows that*), contrast (*by contrast*, *yet*), scale or sequence in the physical process (*Once a fire is alight*, *Further downstream*).
- Orient through the argument itself, not through a roadmap. *Below we…* and *We close by…* only in the Introduction's structure paragraph and in methods.
- Max 5 subsections per section.

## Hedging, calibrated to claim strength
- Strong where evidence is strong; modal where it is not. Keep *breaks down*, *shows* where the evidence warrants them.
- Vocabulary: *may, could, likely, appears to, tends to, suggests, is plausibly*.
- Modal density rises in interpretation and discussion, falls in reporting results.
- Graduated, not binary: *performs much less well at smaller scales*, not *fails*. Soften negative findings rather than dramatising them.

## Paragraph openings
Vary them, roughly balanced:
1. **Assertion**: *European fire science has largely been shaped by where burnt area is concentrated.*
2. **Problem-posed**: *There remain, however, some important challenges and limitations.*
3. **Linking**: picks up what the previous paragraph left open (*These limits appear first in the atmosphere and in observation.*).
4. **Contrastive**: *Climate alone, however, does not determine…*; *Other findings, however, complicate…*
Fewer than one paragraph in ten opens with "The".

## Parentheses
Use parentheses for acronym definitions at first use (*European Forest Fire Information System (EFFIS)*), short clarifications (*the Alpine Space cooperation area (which spans the Alps and their surrounding foothills)*), asides that qualify a metric, and pointers to sections and figures. Not for numbers in apposition.

## Numbers
- Numbers embedded as grammatical units: *the 2015 Mw 7.8 Gorkha earthquake*, not *the Gorkha earthquake (Mw 7.8, 2015)*.
- Quantifiers precede: *approximately*, *about*, *roughly*, *of the order of*.

## Uncertainty and limitations
- Both a dedicated treatment and inline caveats.
- Where a claim rests on fewer than three independent cases, state the number, and say what a single case can and cannot show: *As this rests on a single event, it shows that sub-threshold triggering can occur rather than how often it does.*
- Mark imported mechanisms as imported at the point of use.
- Acknowledge without defensiveness, then redirect to what would resolve it.

## Headings
2–10 words, descriptive noun phrases or gerunds. Colons for subtopics. No interrogatives in the review.

## Unchanged constraints
- No em dashes for parentheticals; parentheses or commas.
- No bullet or numbered lists in the discussion or conclusions.
- No self-promotional novelty claims.
- Avoid AI-flag vocabulary: *crucial, pivotal, landscape, underscore, highlight, nuanced, robust, leverage, delve, multifaceted, interplay, lens*.
