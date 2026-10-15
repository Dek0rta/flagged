#import "common.typ": *

#let e(mode, group, det) = s.edits.at(mode + "|" + group + "|" + det, default: none)
#let ba(mode, group, det) = {
  let x = e(mode, group, det)
  if x == none { "–" } else { [#pc(x.before, d: 1) → #pc(x.after, d: 1)] }
}
#let edit_dets = s.detectors.filter(d => e("grammar", "ell", d) != none)
#let fs = (m, g) => e(m, g, "fakespot_2025")

== Length explains part of the TOEFL result

The TOEFL essays are short (median #s.median_words.toefl words, against #s.median_words.native for native PERSUADE essays). To separate length from writer group, I cut 1,000 native and 1,000 ELL essays to TOEFL-like length: each keeps its first $k$ words, with $k$ drawn from the TOEFL length distribution, and is trimmed back to the last full sentence. Shortening raises false positives for both groups with seven of the nine detectors; only the log-perplexity and Fast-DetectGPT scores do not rise (Table 3). For the Fakespot detector, for example, the share of native essays flagged rises from #pc(fs("truncate", "native").before, d: 1) to #pc(fs("truncate", "native").after, d: 1), and for RADAR it jumps from #pc(e("truncate", "native", "radar_2023").before, d: 1) to #pc(e("truncate", "native", "radar_2023").after, d: 1). At the same length, ELL essays are flagged no more often than native ones. For most detectors the shortened essays are still flagged less often than the TOEFL essays (Fakespot: #pc(fs("truncate", "native").after, d: 1) against #pc(f5_toefl("fakespot_2025"), d: 1)). Length therefore accounts for part of the original gap, and the rest belongs to the TOEFL essays themselves: they are timed answers built from stock phrases, and they are also unusually clean (a dictionary spellchecker finds nothing to correct in them; see below).

== An LLM grammar fix turns human essays into "AI" essays

Students are told to proofread, and many now do it with writing tools that use language models. I took 200 ELL essays, 200 native essays on the same prompts and the 91 TOEFL essays, and asked an open model (Qwen2.5-7B-Instruct, run locally) to "correct the spelling, grammar and punctuation mistakes ... Do not change word choice, sentence structure, tone or meaning, and do not add or remove content." The edits are light: the median share of words kept in order is #num(e("grammar", "ell", edit_dets.at(0)).sim) for ELL essays, #num(e("grammar", "native", edit_dets.at(0)).sim) for native essays and #num(e("grammar", "toefl", edit_dets.at(0)).sim) for TOEFL essays. As a control without any language model, I corrected only misspelled lowercase words with a dictionary spellchecker (pyspellchecker), which changes about #pc(1 - e("spell", "ell", edit_dets.at(0)).sim) of the words in ELL essays.

The LLM correction raises false positives for every detector and both groups (Table 3). With every threshold set to flag 5% of native essays, the share of ELL essays flagged rises by a factor of #num(s.ratio_grammar_ell.at(0), d: 0) to #num(s.ratio_grammar_ell.at(1), d: 0) across the #s.detectors.len() detectors, and of native essays by #num(s.ratio_grammar_native.at(0), d: 0) to #num(s.ratio_grammar_native.at(1), d: 0). The Fakespot detector flags #pc(fs("grammar", "ell").before, d: 1) of the ELL essays before correction and #pc(fs("grammar", "ell").after, d: 1) after it; Desklib goes from #pc(e("grammar", "ell", "desklib_2025").before, d: 1) to #pc(e("grammar", "ell", "desklib_2025").after, d: 1). The dictionary correction raises rates by more than two points for only #s.n_up_spell_ell detectors on ELL essays and for none on native essays. So the detectors are not reacting to the absence of spelling errors as such: they react to text that has passed through a language model, even when it changed only a few words. For the Fakespot and SuperAnnotate classifiers, LLM-corrected essays in which fewer than 5% of words changed are flagged at least as often as more heavily corrected ones (#link("https://github.com/Dek0rta/flagged/blob/main/results/grammar_by_edit.csv")[grammar_by_edit.csv]), although these lightly edited essays also started out cleaner.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto, auto),
    align: (left, right, right, right, right, right, right),
    table.hline(),
    table.header(
      table.cell(rowspan: 2, align: bottom)[*Detector*],
      table.cell(colspan: 2)[*Cut to TOEFL length*], table.cell(colspan: 2)[*Dictionary spellcheck*], table.cell(colspan: 2)[*LLM grammar fix*],
      [*Native*], [*ELL*], [*Native*], [*ELL*], [*Native*], [*ELL*],
    ),
    table.hline(stroke: 0.5pt),
    ..edit_dets.map(d => (
      short.at(d),
      ba("truncate", "native", d), ba("truncate", "ell", d),
      ba("spell", "native", d), ba("spell", "ell", d),
      ba("grammar", "native", d), ba("grammar", "ell", d),
    )).flatten(),
    table.hline(),
  ),
  caption: [Share of the same human essays flagged before → after each edit, with thresholds set to flag 5% of native calibration essays. Length: 1,000 essays per group. Spellcheck and LLM fix: the same 200 essays per group.],
)

#figure(
  image("/figures/edits.pdf", width: 100%),
  caption: [The same human essays before (dot) and after (arrow tip) each edit, with thresholds set to flag 5% of native calibration essays: 200 ELL essays, 200 native essays and the 91 TOEFL essays.],
) <fig-edits>

== The 2023 remedy now backfires

#cite(<liang2023>, form: "prose") showed that asking ChatGPT to "enhance the word choices to sound more like that of a native speaker" made their detectors flag far fewer TOEFL essays, and concluded that such prompting can mitigate the bias. I applied the same prompt with the local model. Unlike the grammar fix, it rewrites heavily (median share of words kept in order #num(e("native", "ell", edit_dets.at(0)).sim) for ELL essays). With today's detectors it has the opposite effect: the share of ELL essays flagged rises for all #s.detectors.len() detectors, by a factor of #num(s.ratio_native_ell.at(0), d: 0) to #num(s.ratio_native_ell.at(1), d: 0), and Desklib flags #pc(e("native", "ell", "desklib_2025").after, d: 0) of the rewritten essays. Detectors trained on modern LLM output recognize the rewriting itself.
