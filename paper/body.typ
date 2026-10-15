#import "common.typ": *

= Introduction

Schools and universities now run student writing through AI-text detectors, tools that estimate whether a passage was written by a large language model (LLM). For the student involved, a false positive is not a statistic: it can mean an academic-integrity case, a lower grade or, in admissions, an essay read with suspicion. The best-known evidence on who bears these errors is a 2023 Stanford study by #cite(<liang2023>, form: "prose"). They ran seven detectors on 91 TOEFL essays by non-native writers and 88 essays by US eighth graders. The detectors flagged #pc(0.6122) of the TOEFL essays as AI-written on average and almost none of the US essays. The study is now the standard citation for the claim that detectors are biased against non-native English writers, and detector companies answer it directly: Pangram reports no false positives on the same TOEFL set @emi2024 and GPTZero reports an "ESL de-biasing" step @gptzero2023.

The comparison behind that claim changes many things at once. The TOEFL essays are short answers written under time pressure on test prompts and collected from a test-preparation forum; the US essays are longer school assignments on other topics. Writer background, length, genre and source all differ, so the data cannot say which of them the detectors respond to. The companies' replies are tested on the same 91 essays.

The question matters to me directly. I am a high school student from Russia applying to US universities this year, and I am writing my application essays in English, my second language. If detectors really treat non-native writing as machine-written, essays like mine are at risk; if they respond to something else, students and admissions readers should know what that is. So I asked the question again with a design that holds topic and setting fixed. I use the PERSUADE 2.0 corpus @crossley2024, in which US students in grades 6–12 who are and are not classified as English language learners (ELL) wrote on the same 15 prompts, and the ELLIPSE corpus @crossley2023, in which ELL students' essays carry expert ratings of English proficiency. I score #thousands(s.n.native + s.n.ell + s.n.ellipse + s.n.toefl + s.n.college) human essays and #thousands(s.n.ai) LLM-written essays on the same prompts with #s.detectors.len() open detectors released between 2019 and 2025, including the classifier behind Mozilla's Fakespot browser add-on @fakespot2025 and two recent zero-shot methods @hans2024 @bao2024. Then I change the human essays directly: I cut them to TOEFL length, and I let an LLM fix only their spelling and grammar, the way a student might with a writing assistant.

The findings are:

+ *The original result replicates on the original data.* At their default settings, the six trained classifiers flag #pc(s.toefl_default_range.at(0))–#pc(s.toefl_default_range.at(1)) of Liang et al.'s TOEFL essays (mean #pc(s.toefl_default_mean)).
+ *It does not hold for English learners writing on the same prompts as native peers.* With prompt, grade and length held fixed, #s.or_ell_sig_lower of #s.detectors.len() detectors flag ELL essays significantly _less_ often than native-speaker essays, and none flags them more often.
+ *What the detectors respond to is the text.* Short length raises false positives for nearly every detector. Published "model" college-admission essays, the most heavily edited human texts in the study, are flagged at up to #pc(s.college_default_range.at(1)). For #s.n_rho_pos of #s.detectors.len() detectors, English learners' essays look more AI-like the higher their rated proficiency, and three trained classifiers flag native speakers' essays more often the higher human raters scored them; the newest classifier shows the opposite trend.
+ *A language model's light touch is enough.* Letting an LLM fix only spelling and grammar, which leaves most words in place, multiplies the share of human essays flagged for all #s.detectors.len() detectors; correcting spelling with a dictionary does not. The rewording that Liang et al. found to reduce false positives now backfires.

@fig-specimen shows the last finding on one real essay.

#include "specimen.typ"

The practical message for students, teachers and admissions readers is that a false flag says more about the text than about who wrote it, and that a student who lets an AI tool fix the grammar of an essay they wrote can get it flagged.

= Related work

*Detectors.* Trained detectors fine-tune an encoder on human and machine text: the GPT-2 output detector @solaiman2019, models trained on ChatGPT answers @guo2023, adversarially trained RADAR @hu2023, and recent classifiers trained on the RAID benchmark @dugan2024 @superannotate2024 @desklib2025 @fakespot2025. Zero-shot detectors instead read the text with a language model and test whether it is unusually predictable: DetectGPT @mitchell2023 and Fast-DetectGPT @bao2024 compare the text's likelihood to that of alternative tokens, and Binoculars @hans2024 normalizes perplexity by the cross-perplexity between two related models.

*Fairness.* #cite(<liang2023>, form: "prose") linked the bias they found to perplexity: TOEFL essays with simpler vocabulary are more predictable, and rewriting them with richer word choice lowered the false-positive rate. Vendors' reports of fixed bias evaluate their own products @emi2024 @gptzero2023. An independent comparison of commercial detectors on human text from several genres found very low error rates for the best tool but did not study non-native writing @jabarian2025. Closest to this paper, #cite(<park2026>, form: "prose") compare scientific manuscripts by non-native authors before and after professional English editing and find that editing moves detector scores, in different directions for different detectors. I study student essays rather than research papers, compare learners and native speakers on identical prompts, relate flags to human ratings of writing quality, and decompose the original TOEFL result into length and text-type effects.

= Data

*Native and ELL essays (PERSUADE 2.0).* The corpus contains about 25,000 argumentative essays by US students in grades 6–12 on 15 prompts, collected before the release of ChatGPT, with each writer's ELL status, grade and a holistic score from 1 to 6 assigned by trained raters @crossley2024. I keep all #thousands(s.n.ell) essays by ELL students and a random sample of #thousands(s.n.native) by non-ELL students (hereafter "native"), each with at least 100 words. The native sample is split in half at random: #thousands(s.n_native_calib) essays are used only to set detector thresholds, and the rest are used for evaluation.

*ELL proficiency (ELLIPSE).* #thousands(s.n.ellipse) essays by ELL students in grades 8–12, each rated by trained raters from 1 to 5 for overall proficiency and for cohesion, syntax, vocabulary, phraseology, grammar and conventions @crossley2023. Some ELLIPSE essays also appear in PERSUADE; the analyses keep the two corpora separate.

*Liang et al.'s essays.* The 91 TOEFL essays (median #s.median_words.toefl words) and the 70 "authentic" US college-admission essays released with #cite(<liang2023>, form: "prose"). The admission essays are published examples from two admissions-advice websites ("college essay examples that worked"), so they are polished, often professionally edited texts rather than a random sample of applicants' writing.

*AI essays.* #thousands(s.n.ai) essays written by #s.n_ai_models LLM configurations (including ChatGPT, GPT-4, Claude, PaLM, Llama 2 and Mistral variants) on the same PERSUADE prompts, sampled up to 200 per source from the DAIGT v2 training set of a 2023 Kaggle competition @daigt2023. They are used to check that each detector works on this kind of text and to report its true-positive rate.

= Detectors and thresholds

I run #s.detectors.len() detectors locally (Tables 1 and 2). Six are trained classifiers that output a probability that the text is AI-written: the 2019 GPT-2 output detector @solaiman2019, a RoBERTa model trained on the HC3 corpus @guo2023, RADAR @hu2023, SuperAnnotate's RoBERTa-large detector @superannotate2024, the RoBERTa detector released with Mozilla's Fakespot add-on @fakespot2025, and Desklib's DeBERTa-v3-large detector, which led the RAID leaderboard when released @desklib2025. Three are zero-shot scores computed from open language models: log-perplexity under Qwen2.5-1.5B @qwen2024; Binoculars @hans2024 with Qwen2.5-1.5B as observer and Qwen2.5-1.5B-Instruct as performer; and analytic Fast-DetectGPT @bao2024 with Qwen2.5-1.5B as both sampling and scoring model. Texts are truncated to 512 tokens. The zero-shot detectors use smaller models than their original papers because of hardware limits, which lowers their accuracy (Table 2).

A detector's false-positive rate depends on where its threshold is set, so I report three settings. *Default* uses a trained classifier's own decision rule, P(AI) > 0.5, which is how such a classifier is used out of the box. *1% and 5% FPR* set the threshold so that 1% or 5% of the native calibration essays are flagged, which puts all ten detectors on the same footing; rates on the held-out native essays then measure how well that calibration transfers.

I summarize group differences in three ways: the share of each group's essays above the threshold, with Wilson 95% intervals; the ratio of the ELL rate to the native rate; and a logistic regression of being flagged on ELL status with fixed effects for prompt and controls for grade and log length, with and without the essay's holistic score.

= Results

== The original result replicates on the original essays

#let dets = s.detectors
#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto, auto),
    align: (left, right, right, right, right, right, right),
    table.hline(),
    table.header([*Detector*], [*AUROC*], [*Native*], [*ELL*], [*TOEFL*], [*College*], [*AI caught*]),
    table.hline(stroke: 0.5pt),
    ..dets.filter(d => row(d, "default") != none).map(d => {
      let r = row(d, "default")
      (label.at(d), num(r.auroc_native_vs_ai), pc(r.fpr_native, d: 1), pc(r.fpr_ell, d: 1), pc(r.fpr_toefl), pc(r.fpr_college), pc(r.tpr_ai))
    }).flatten(),
    table.hline(),
  ),
  caption: [Trained classifiers at their default threshold (P(AI) > 0.5). AUROC: how well the detector separates native PERSUADE essays from AI essays on the same prompts (1 = perfectly). Native, ELL, TOEFL, College: share of human essays flagged as AI. AI caught: share of AI essays flagged.],
)

All six trained classifiers separate human from AI essays on these prompts better than chance, and the three most recent do so almost perfectly (Table 1). On Liang et al.'s TOEFL essays they reproduce the original finding: at the default threshold they flag #pc(s.toefl_default_range.at(0))–#pc(s.toefl_default_range.at(1)) of them, against a mean of #pc(0.6122) for the seven commercial detectors of 2023. The same tools flag published college-admission essays at very different rates, from #pc(s.college_default_range.at(0)) to #pc(s.college_default_range.at(1)).

#figure(
  image("/figures/groups_fpr5.pdf", width: 100%),
  caption: [Share of human essays flagged as AI by each detector when its threshold is set to flag 5% of native calibration essays. Lines are 95% intervals.],
) <fig-groups>

== English learners writing on the same prompts are flagged less, not more

#let f5 = s.fpr5
With every detector calibrated to flag 5% of native essays, the held-out native essays are flagged at close to 5% and ELL essays on the same prompts at lower rates for #s.n_ell_lower of #dets.len() detectors (@fig-groups). The regression holds prompt, grade and length fixed: the odds ratio for ELL status is significantly below one for #s.or_ell_sig_lower of #dets.len() detectors and significantly above one for none (Table 2). The ELLIPSE essays, all by English learners, are flagged at or near the native rate. Without any threshold, the probability that a random ELL essay scores as more AI-like than a random native essay on the same prompt is below one half for #s.auroc_ell_below_half of #dets.len() detectors.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto, auto),
    align: (left, right, right, right, right, right, right),
    table.hline(),
    table.header([*Detector*], [*Native*], [*ELL*], [*ELLIPSE*], [*TOEFL*], [*AI caught*], [*Odds ratio, ELL (95% CI)*]),
    table.hline(stroke: 0.5pt),
    ..dets.map(d => {
      let r = f5.at(d)
      let o = s.or_ell.at(d)
      (label.at(d), pc(r.fpr_native, d: 1), pc(r.fpr_ell, d: 1), pc(r.fpr_ellipse, d: 1), pc(r.fpr_toefl),
       pc(r.tpr_ai), [#num(o.at(0)) (#num(o.at(1))–#num(o.at(2)))])
    }).flatten(),
    table.hline(),
  ),
  caption: [All detectors with the threshold set to flag 5% of native calibration essays. Odds ratio: being flagged, ELL vs native, from a logistic regression with prompt fixed effects, grade and log word count (PERSUADE held-out essays).],
)

== Writing quality matters, but not in the same way for every detector

If the detectors responded to non-native writing as such, the weakest English learners should look most AI-like. For most detectors the opposite holds. In ELLIPSE, the score of #s.n_rho_pos of #s.detectors.len() detectors is positively rank-correlated with the writer's rated overall proficiency (Spearman's $rho$ from #num(s.rho_pos_min) to #num(s.rho_pos_max)); for the Desklib classifier it is flat ($rho$ = #num(s.ellipse_rho.desklib_2025)). Among native speakers in PERSUADE, the HC3, SuperAnnotate and Fakespot classifiers flag essays more often the higher their holistic score: Fakespot flags #pc(s.native_by_score.fakespot_2025.at("1"), d: 1) of essays scored 1 and #pc(s.native_by_score.fakespot_2025.at("6"), d: 1) of essays scored 6, and SuperAnnotate #pc(s.native_by_score.superannotate_2024.at("1"), d: 1) and #pc(s.native_by_score.superannotate_2024.at("6"), d: 1) (@fig-quality). Desklib, the newest and most accurate classifier, shows the reverse: it flags #pc(s.native_by_score.desklib_2025.at("1"), d: 1) of native essays scored 1 and #pc(s.native_by_score.desklib_2025.at("6"), d: 1) of those scored 6. The older classifiers and the zero-shot scores show no clear trend. Writing quality is therefore not a universal driver of false positives; which human texts a detector confuses with AI depends on what human text it was trained on. Controlling for the holistic score in the regression moves the ELL odds ratios toward one for most detectors but does not remove the gap.

#figure(
  image("/figures/quality.pdf", width: 100%),
  caption: [Share of PERSUADE essays flagged (5%-FPR threshold) by holistic score assigned by trained human raters, for four detectors. Points with fewer than 30 essays are omitted.],
) <fig-quality>

#include "edits.typ"

= Discussion

#include "discussion.typ"
