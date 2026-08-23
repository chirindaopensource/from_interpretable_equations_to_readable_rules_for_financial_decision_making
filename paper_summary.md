# How Simple Can It Get? From Interpretable Equations to Readable Rules for Financial Decision Making

**A Formal Summary**

*Source paper: Adia Lumadjeng, Ilker Birbil, and Erman Acar (University of Amsterdam)*

---

## Executive Summary

In regulated financial domains, a model that cannot be explained cannot be deployed. The prevailing response has been interpretable-by-design classifiers, whose reasoning is legible from the model itself. This paper identifies a gap that such designs do not close: structural transparency does not guarantee practical readability. An interpretable-by-design classifier trained on loan default produces a single equation spanning forty-two features — nothing is hidden, yet no regulator could reasonably read it.

The authors therefore invert the usual research direction. Rather than constructing transparent models that retain accuracy, they begin from an already-interpretable model and ask how far its representation can be simplified before something important is lost. Starting from a fitted single-term monomial classifier, they derive four progressively simpler representations — a pruned monomial, a directional if–then rule, an integer scorecard, and a tally — each removing a different, identifiable component of the original model's information.

The decisive methodological move is that the equation *is* the predictive model, not a post-hoc explanation of one. Each simplified rule can therefore be evaluated as the classifier it displays, making the cost of every simplification directly measurable rather than assumed.

Three findings stand out. First, the reductions do not lie on a single accuracy–readability trade-off: pruning weakly contributing features is nearly free, removing effect magnitudes is often cheap, but binarizing continuous feature values is expensive. Second, predictive performance and fidelity to the original model can diverge — a simpler rule may remain an effective classifier while no longer faithfully reproducing the model it claims to represent. Third, some of these losses are predictable in advance: the authors derive a worst-case bound on the perturbation caused by pruning and a closed-form prediction of the directional rule's rank fidelity, computable immediately after training. A human assessment confirms that simplification improves perceived ease, but shows that which representation practitioners actually prefer depends on professional background.

---

## Body

### 1. Problem Setting and Motivation

Credit default and fraud detection are imbalanced binary classification problems in which decisions must be explainable to a customer, auditor, or supervisor. The interpretability literature has largely worked forward — building models that stay transparent while retaining accuracy. The authors argue this leaves a question unaddressed: given an already-interpretable model, how far can its *representation* be simplified while preserving predictive performance, fidelity to the original model, and gaining genuine human readability?

This distinction matters specifically for interpretable-by-design methods. The authors note that the original ECSEL work evaluates the fitted model at full size while using pruning only to produce more legible displayed equations — it never evaluates how those simplified displays perform as classifiers, nor how faithfully they preserve the fitted model's behaviour. This paper makes that transition explicit: the rule shown to the reader is also the rule whose predictions are evaluated.

### 2. The Model

The starting point is ECSEL, which learns a signomial equation serving simultaneously as classifier and explanation. The authors restrict it to its simplest single-term form (`K = 1`), a monomial:

$$z(x) = \alpha \prod_{j=1}^{m} x_j^{\beta_j}, \qquad P(y = 1 \mid x) = \sigma(z(x)).$$

Because exponents are real-valued, every feature is scaled to a fixed positive range $[\ell, u]$ with $0 < \ell$, fit on the training split alone. Taking logarithms linearises the model:

$$\log|z(x)| = \log|\alpha| + \sum_{j=1}^{m} \beta_j \log x_j.$$

This decomposition exposes exactly three interpretable components — **support** (which features participate), **direction** (the signs of $\beta_j$ and $\alpha$), and **magnitude** ($|\beta_j|$) — to which the authors add a fourth axis, whether **feature values** are continuous or binarized. These four components can be removed independently, which is what makes the cost of each simplification separable rather than conflated into a single vague notion of "readability."

### 3. The Controlled Simplifications

| Representation | Support | Values | Direction | Magnitude |
|---|---|---|---|---|
| Full monomial | All | Continuous | ✓ | Exact |
| Pruned monomial | Reduced | Continuous | ✓ | Exact |
| Directional rule | Reduced | Continuous | ✓ | — |
| Scorecard | Reduced | Binary | ✓ | Quantized |
| Tally | Reduced | Binary | ✓ | — |

**Pruned monomial.** Retains the $r$ features with the largest $|\beta_j|$, zeroing the rest. The authors justify this criterion with a worst-case bound: with $L := \max\{|\log \ell|, |\log u|\}$ and $S$ the removed set,

$$|s_{\mathrm{full}}(x) - s_{\mathrm{pruned}}(x)| \le L \sum_{j \in S} |\beta_j| \quad \text{for every } x.$$

Crucially, the authors are careful about what this establishes. It is a *selection criterion*, not a performance guarantee: among all ways of discarding $m - r$ features, dropping the smallest magnitudes minimises the worst-case log-score perturbation, but nothing is promised about predictive performance or ranking. The support size $r$ is therefore chosen empirically on validation data.

**Scorecard and tally.** Each retained feature becomes a binary condition at its training-set median, with the direction set by the sign of $\alpha\beta_j$. The tally counts satisfied conditions equally; the scorecard assigns integer points $p_j = \max(1, \mathrm{round}(P_{\max}|\beta_j|/\max_k|\beta_k|))$, the floor of one ensuring every retained feature stays visible on the card. The tally is exactly the scorecard with all points set to one.

**Directional rule.** Replaces every exponent with its sign, giving $s_{\mathrm{dir}}(x) = \sum_{j} \mathrm{sign}(\beta_j) \log x_j$. It preserves support, direction, and continuous values while discarding relative strength entirely.

### 4. Predicting Fidelity Before Simplification

The paper's sharpest theoretical contribution concerns the directional rule, whose cost is least predictable from its form alone. Writing $w = (\log x_1, \dots, \log x_m)$ with covariance $\Sigma$, and $d_j = \mathrm{sign}(\beta_j)$ on the retained set, the correlation between full and directional log-scores is

$$\rho = \frac{\beta^\top \Sigma d}{\sqrt{\beta^\top \Sigma \beta}\sqrt{d^\top \Sigma d}},$$

which holds without distributional assumptions. If $w$ is elliptically distributed, Greiner's classical relation converts this into a predicted rank fidelity, $\tau = \tfrac{2}{\pi}\arcsin(\rho)$.

The practical significance is that both $\beta$ and $\Sigma$ are available immediately after training. The expected agreement between the directional rule and the original classifier can therefore be estimated **before the simplified rule is constructed or evaluated**, requiring no held-out data.

### 5. Experimental Design

Four public financial datasets span an order of magnitude in size and positive-class rates from 0.17% to 22.1%: **Creditcard** (284,807 rows, 30 features, 0.17%), **FraudEcom** (151,112, 6, 9.4%), **Loan** (395,492, 42, 10.1%), and **Default** (30,000, 26, 22.1%). They also differ in feature type — named continuous attributes, small integer counts, and anonymized principal components respectively — a difference that proves consequential.

Protocol discipline is strict: 60/20/20 stratified splits over five random seeds; scaling to $[0.01, 10.01]$ fit on training data only; hyperparameters via Optuna; pruning level and all decision thresholds selected on validation; the test split used only for final evaluation. PR-AUC is the primary metric given class imbalance, tie-corrected Kendall's $\tau_b$ measures fidelity, and each non-probabilistic representation is recalibrated with isotonic regression and audited by expected calibration error.

### 6. Findings

**Pruning is nearly free.** The largest test PR-AUC reduction is 0.008 (FraudEcom); Loan and Creditcard lose 0.001 and 0.002. Fidelity stays high, $\tau_b$ between 0.70 and 0.85. On Loan, forty-two features reduce to seven in the representative display with virtually no predictive loss.

**Binarizing feature values is costly.** On Loan, PR-AUC falls from 0.885 (pruned) to 0.329 (scorecard) and 0.300 (tally). On Creditcard both point-based forms collapse to roughly 0.018 and 0.017, near the 0.002 base rate — attributed to thresholding anonymized principal components, which discards most of the information carried in their continuous values. Continuous feature values evidently carry more predictive information than precise exponent magnitudes.

**Performance and fidelity diverge.** On Default, the directional rule essentially matches the full monomial (PR-AUC 0.381 versus 0.380) despite rank fidelity of only 0.744. On Loan, fidelity falls to 0.568 while predictive performance remains far above the point-based forms. A simplified rule can therefore remain an effective classifier without faithfully reproducing the model it purports to represent — the paper's most consequential negative result for practitioners who treat readable explanations as trustworthy proxies.

**Simplified scores calibrate reliably.** After isotonic recalibration, ECE remains low across all representations and datasets, showing that ranking loss does not preclude reliable probability estimates.

**Fidelity prediction largely holds.** On Loan, Default, and Creditcard the prediction is accurate across fifteen fitted models: mean absolute error 0.017, maximum error 0.039, over observed fidelities from 0.25 to 0.83. FraudEcom is the instructive exception (MAE 0.236): its dominant features are low-cardinality count variables producing many tied scores. Pearson correlation is insensitive to tie structure whereas $\tau_b$ is not, so the failure is one of the elliptical distributional assumption, not of the correlation formula.

**Human assessment.** Of 36 respondents, 34 were analysed — 14 with finance/risk backgrounds and 20 AI/ML researchers. Objective comprehension was high across all four forms (92–100%), so differences concern perceived ease and preference rather than basic understanding. All three simplified rules were perceived as easier than the pruned monomial. Preferences, however, split sharply by background: finance/risk respondents chose the directional rule in 57% of assessments versus 29% for point-based forms, while AI/ML researchers chose point-based forms in 72% versus 18% for the directional rule. The pruned monomial was rarely preferred by either group.

**Robustness.** Comparing post-hoc pruning against in-training iterative hard thresholding, both routes retain nearly identical feature sets — Jaccard overlap 1.00 on Loan, FraudEcom, and Creditcard, and 0.83 on Default — with comparable predictive performance. The extracted structure therefore reflects learned data structure rather than a pruning artifact.

### 7. Limitations

The authors are explicit about scope. The study covers only single monomials ($K = 1$) and four financial datasets. The readable forms use deliberately fixed simplifications, such as median binarization, to isolate the cost of removing specific information rather than to optimise each representation — so the results characterise the cost of these simplifications, not the best achievable performance of each rule form. The human assessment is exploratory, with 36 respondents, and measures stated preferences rather than use in real decisions.

---

## Conclusion

The paper's central contribution is conceptual as much as empirical: it reframes readable explanations as **measurable reductions of an interpretable model** rather than as representations that may be assumed faithful because they are easy to read. By making the displayed rule identical to the evaluated classifier, the authors convert a question usually settled by intuition into one settled by measurement.

The resulting picture resists the familiar "accuracy–readability trade-off" framing. Different simplifications remove different information, and their cost depends on where a given dataset's predictive signal resides. Removing weakly contributing features is nearly free everywhere. Removing effect magnitudes is often cheap, and — uniquely among the simplifications studied — its effect on fidelity can be anticipated from the fitted model before the rule is built. Binarizing continuous values consistently incurs the largest predictive losses, severely so when features are latent components rather than meaningful financial variables.

The divergence between predictive performance and fidelity is the finding with the most direct governance implications. A scorecard or directional rule that classifies well may nonetheless rank cases differently from the model it was derived from. For institutions presenting simplified rules to customers, auditors, or supervisors as accounts of a deployed model, accuracy alone is not evidence of faithfulness; the two must be measured separately.

Finally, the human assessment cautions against a single notion of readability. Simplification reliably improved perceived ease, but finance/risk practitioners and AI/ML researchers preferred opposite representations — suggesting that the appropriate readable form is a function of audience, not a property of the rule. Consistent with the authors' own ethics statement, simplification should complement rather than replace domain validation, fairness assessment, and regulatory review.
