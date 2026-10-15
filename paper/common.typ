// Shared data and helpers for main.typ, abstract.typ and body.typ.

#let s = json("/results/summary.json")
#let main = csv("/results/main.csv", row-type: dictionary)

#let num(x, d: 2) = {
  let v = calc.round(float(x), digits: d)
  if d == 0 { return str(int(v)) }
  let t = str(v)
  if not t.contains(".") { t += "." }
  t + "0" * (d - t.split(".").at(1).len())
}
#let pc(x, d: 0) = if x == none { "–" } else { num(100 * float(x), d: d) + "%" }
#let thousands(n) = {
  let t = str(n)
  let out = ""
  for (i, c) in t.clusters().enumerate() {
    if i > 0 and calc.rem(t.len() - i, 3) == 0 { out += "," }
    out += c
  }
  out
}
#let label = (
  openai_2019: "OpenAI RoBERTa (2019)", hc3_2023: "HC3 RoBERTa (2023)", radar_2023: "RADAR (2023)",
  superannotate_2024: "SuperAnnotate (2024)", fakespot_2025: "Fakespot/Mozilla (2025)",
  desklib_2025: "Desklib (2025)", logppl: "Log-perplexity", binoculars: "Binoculars",
  fastdetect_qwen: "Fast-DetectGPT (Qwen)", fastdetect_neo: "Fast-DetectGPT (Neo)",
)
#let short = (
  openai_2019: "OpenAI '19", hc3_2023: "HC3 '23", radar_2023: "RADAR '23", superannotate_2024: "SuperAnn. '24",
  fakespot_2025: "Fakespot '25", desklib_2025: "Desklib '25", logppl: "Log-ppl", binoculars: "Binoculars",
  fastdetect_qwen: "Fast-Detect",
)
#let row(det, policy) = main.find(r => r.detector == det and r.policy == policy)
#let f5_toefl(det) = s.fpr5.at(det).fpr_toefl
