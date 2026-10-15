// Flagged — paper source. Numbers come from results/summary.json and results/*.csv (analysis/).
// Build: uv run python paper/build.py

#import "common.typ": *

#set document(title: "Fixed by AI, Flagged as AI", author: "Zakhar Sukhanov")
#set page(paper: "us-letter", margin: (x: 1in, y: 0.9in), numbering: "1")
#set text(font: "Libertinus Serif", size: 10.5pt)
#set par(justify: true, leading: 0.62em)
#set heading(numbering: "1.1")
#show heading.where(level: 1): it => block(above: 1.4em, below: 0.7em, text(size: 12pt, weight: "bold", it))
#show heading.where(level: 2): it => block(above: 1.1em, below: 0.6em, text(size: 10.5pt, weight: "bold", style: "italic", it))
#show figure.caption: it => text(size: 9pt, it)
#set figure(gap: 0.6em)
#show figure: set block(above: 1.2em, below: 1.2em)
#set table(stroke: none, inset: (x: 4pt, y: 2.6pt))
#show link: it => text(fill: rgb("#1e4fa0"), it)

#align(center)[
  #text(size: 17pt, weight: "bold")[Fixed by AI, Flagged as AI: A Matched Audit of AI-Text Detectors \ on #thousands(s.n.native + s.n.ell + s.n.ellipse) Student Essays]
  #v(0.6em)
  #text(size: 11pt)[Zakhar Sukhanov]
  #v(0.1em)
  #text(size: 9.5pt)[Independent researcher · #link("mailto:zexa1993@outlook.com")[zexa1993\@outlook.com] · #link("https://www.linkedin.com/in/zakhar-sukhanov-1ba2313b6/")[LinkedIn] · #link("https://github.com/Dek0rta")[GitHub]]
  #v(0.1em)
  #text(size: 9.5pt)[Draft of #datetime.today().display("[day] [month repr:long] [year]")]
]

#v(0.8em)
#block(inset: (x: 2.2em))[
  #text(size: 9.5pt)[
    *Abstract.* #include "abstract.typ"
  ]
]

#include "body.typ"

#bibliography("refs.yml", style: "apa", title: "References")
