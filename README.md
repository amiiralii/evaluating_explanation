# evaluating_explanation
What's a good explanation? in this repository we're looking for ways to evaluate different XAI methods, in a way that fits best with the requirements of search based software engineering.


## Expectations of an XAI method in SBSE

1. Attribution — which decision variables/objectives explain why this solution was favored (absorbs: decision rationale, sensitivity explanation)
2. Contrastive / counterfactual generation — why this solution rather than an alternative; what would have to change for another solution to be preferred; why an infeasible target can't be reached; why the search stopped here rather than continuing (absorbs: contrastive explanation, counterfactual explanation, infeasibility explanation, alternative-solution relevance, stopping justification)
3. Trade-off characterization — what was gained and sacrificed relative to the local Pareto neighborhood (SBSE-native; no classifier-XAI equivalent — single-outcome predictions have no trade-off structure to characterize)
4. Faithfulness — does the cited attribution/contrast/trade-off actually drive the outcome, verifiably (e.g., under perturbation)?
5. Specificity — is it quantified and falsifiable (a bound, a threshold, a magnitude), rather than vague ("this value was big")?
6. Parsimony — is it small enough for a human to act on (e.g., 2–3 variables), rather than an unfiltered dump of every contributing factor?
7. Consistency / stability — does the same input produce the same explanation, and do similar inputs produce similar explanations?
8. Calibration — does it signal its own reliability boundary (when not to trust it), rather than presenting uniform confidence regardless of context?
9. interactivity — can the practitioner query/challenge the explanation (what-if), or is it necessarily a static, one-shot output?

