# evaluating_explanation
What's a good explanation? in this repository we're looking for ways to evaluate different XAI methods, in a way that fits best with the requirements of search based software engineering.


## Expectations of an XAI method in SBSE

1. Attribution — which decision variables/objectives explain why this solution was favored (absorbs: decision rationale, sensitivity explanation)
2. Contrastive / counterfactual generation — why this solution rather than an alternative; what would have to change for another solution to be preferred;
3. Trade-off characterization — what was gained and sacrificed relative to the local Pareto neighborhood (SBSE-native; no classifier-XAI equivalent — single-outcome predictions have no trade-off structure to characterize)
4. Faithfulness — does the cited attribution/contrast/trade-off actually drive the outcome, verifiably (e.g., under perturbation)?
5. Specificity — is it quantified and falsifiable (a bound, a threshold, a magnitude), rather than vague ("this value was big")?
6. Parsimony — is it small enough for a human to act on (e.g., 2–3 variables), rather than an unfiltered dump of every contributing factor?
7. Consistency / stability — does the same input produce the same explanation, and do similar inputs produce similar explanations?
8. Calibration — does it signal its own reliability boundary (when not to trust it), rather than presenting uniform confidence regardless of context?
9. interactivity — can the practitioner query/challenge the explanation (what-if), or is it necessarily a static, one-shot output?

--------

## Experiments to evaluate xai methods based on criteria above
+ **Global** / C1, C4: build model using all features, come up with an explanation method. extract k most important features. build another model one more time on the k suggested feature. one model per xai method. the best accuracy presents the xai method that can best describe the global space
+ **Local** / C2: build model, run it to get the best possible outcome, extract explanations. pick 20 random points. based on explanation on them, decide one feature to change on the random point to generate an artificial suggestion. feed the new point to the same model and see if the change increase the efficiency. record the improvement as a metric
+ **Global & Local** / C2, C5: build model, run it to get the best possible outcome, extract explanations. Now, based on the explanation of the best config, try creating an ideal point, without looking at the best config. feed it to the model to see if the new fake config is actually eficient, on the same model. (importance of range)
+ **Local** / C3: for multi-objective problems, when suggesting the best config, the XAI method should indicate which objective is most sacrificed.
+ **Local** / C7: build model, pick a random point and feed it. extract the explanation for it. then apply small changes to the point and feed it to the model again. does the explanations remain the same?(considering the predictions still being close)
+ **Global** / C6: threshhold is 7 (based on that psychology work). does the output of explanation contain more than 7?
+ **Local** / C8: can the output of explanation suggest a level of confidence in the output's quality or explanation?
+ **Local** / C9: How does explanation output interact with the "what-it" questions? (no idea / level of change / direction of change)
