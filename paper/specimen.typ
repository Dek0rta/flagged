#import "common.typ": *

// One real ELL essay from PERSUADE 2.0, before and after the LLM grammar fix (data from analysis/site_data.py).
#let sp = json("/site/data.json").specimen
#let render(ops) = ops.map(((op, t)) => {
  if op == "eq" { t } else if op == "del" { text(fill: rgb("#8A9099"), strike(stroke: 0.9pt + rgb("#C8102E"), t)) } else { highlight(fill: rgb("#F5F04A"), extent: 1pt, t) }
}).join(" ")
#let verdict(d) = {
  let x = sp.scores.at(d)
  let flag(v) = if v > x.cut { text(fill: rgb("#C8102E"), weight: "bold")[AI] } else { text(fill: rgb("#1F7A5A"))[human] }
  ([#x.label], [#pc(x.original, d: 1) (#flag(x.original))], [#pc(x.grammar, d: 1) (#flag(x.grammar))])
}

#figure(
  block(width: 100%, inset: 10pt, radius: 4pt, stroke: 0.5pt + rgb("#D9DCD6"))[
    #set align(left)
    #set text(size: 9.5pt)
    #set par(justify: false)
    #render(sp.diff.grammar.at(1))
    #v(6pt)
    #table(
      columns: (1fr, auto, auto),
      align: (left, right, right),
      table.hline(stroke: 0.5pt),
      table.header([*Detector*], [*As written*], [*After LLM grammar fix*]),
      table.hline(stroke: 0.3pt),
      ..("desklib_2025", "fakespot_2025", "openai_2019").map(verdict).flatten(),
      table.hline(stroke: 0.5pt),
    )
  ],
  kind: image,
  caption: [Second paragraph of a #sp.words\-word essay by a grade-#sp.grade English learner (PERSUADE 2.0, prompt "#sp.prompt"), with the edits an LLM made when asked to fix only grammar: deletions struck through, insertions highlighted. The table gives each detector's probability that the whole essay is AI-written, and its verdict at the threshold that flags 5% of native essays. A dictionary spellcheck found nothing to change in this essay.],
) <fig-specimen>
