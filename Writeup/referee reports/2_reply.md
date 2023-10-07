---
geometry: margin=1in
---

We thank the referee for another detailed and critical report, to which points we respond in the order that they appear (with some minor rearrangement to assist with overall organisation). Despite the referee's stated wishes to be constructive, we find in several places that the referee first materially misrepresents what we have written, and then finds fault with our discussion based primarily on these misrepresentations. In others, their suggested actions would, if taken, actively detract from our analysis, or are not substantively relevant to its substance. On top of this, the referee impugns our motivations by accusing us of inappropriate conduct, with no substantiation. If the referee genuinely wishes to be constructive, as they assert, we ask that they restrict themselves to comments which are actually salient to the scientific analysis that we have presented.

# Comments that were not, or not properly, taken into account

> - The properties of the time series are not given. The frequency resolution is given as 8 nHz (in the caption of Fig 2), same as Kepler. However, in lines 1725-1726 the authors note that the TESS data have 'shorter duration compared to the nominal Kepler mission.' It is absolutely necessary to give precise information about the times series (lines 159-161 are much too vague).

We have added a statement that the time series spans an effective length of two years. The value we have given for the resolution of the data is found from the lower bound provided by the Fourier uncertainty principle, rather than simply the inverse total duration (as suggested by the referee's calculation for Kepler). Since the referee seems to take issue with this, we have also rewritten the figure caption for fig. 2.

> The window effect and its consequence has to be studied: do aliases perturb the analysis?

No. (plot of window function)

> The power spectrum density should be shown in a broad frequency range (broader than current figures 1 and 2).

We have expanded the plot range as requested.

> - Many paragraphs can be omitted since they introduced unnecessary information (for one example, the lines 199-204 dealing with clump stars - unneeded since Zvrk cannot be a clump star).

We would appreciate if the referee could specifically describe these "many paragraphs" and why they are superfluous, as otherwise we cannot guess what they mean given such a lack of description. We point out that we only mention clump stars in two short sentences: one only regarding the visual appearance of the echelle diagram (rather than assigning evolutionary identification), and one specifically saying that Zvrk is not a clump star. The indicated lines are where we first classify Zvrk as either a first-ascent RG, or AGB star, and are clearly not superfluous.

We note also that the referee is fortunate to be able to tell immediately from experience that Zvrk's seismology cannot be that of a clump star (as we point out, overriding the judgement of automated analysis tools). Other readers, and for that matter several authors of this paper, are not so fortunate. In editing this paper, we had removed many points of discussion from Sections 2 and 4 for length, owing to feedback that we received about some expository material being obvious. It is clear from the referee's reactions to Section 4, to which we respond below, that such obviousness was clearly not universal. We therefore by the same token suspect that what the referee finds obvious and unnecessary may also not be so to non-seismologists reading the paper. As the referee may note by the nature of Section 4, this paper is clearly not intended to be read solely by seismologists. We ask that the referee keep this in mind when recommending passages for deletion.

> - The lack of relevance of Section 2.4 is not because of its content, but from the inappropriate location of this section.

We would appreciate if the referee could explain how an inappropriate position of this section invalidates its relevance irrespective of content, as that is not clear to us.

In any case, as written, Section 2.4 must come after both Section 2.3, which introduces the spectra used in discussion, and Section 2.2, which introduces our ASAS-SN and custom TESS photometry. Sections 2.2 and 2.3 in turn must both also come after Section 2.1, as our analysis there relies on seismic rotation rates and $\nu_\mathrm{max}$. Thus, Section 2.4 is constrained by these logical dependencies to be placed in its current position. Since the referee appears to think otherwise, we would appreciate explicit guidance regarding alternative locations.

# New issues

> - Concerning the definition provided for S/N shows: what matters here is the ratio of the seismic signal compared to the background. The granulation background, even if noisy due to incoherent signal, follows rules that are deterministic. Hence, the denominator is not pure noise. In fact, the authors do mention the "noise" (with quote marks introduced by them) due to granulation. Many papers use the expression 'height to background ratio'; the authors should do so.

We remind the referee that describing this ratio as a "signal-to-noise" is also well-established usage in the astronomical literature. Nonetheless, in accordance with their wishes, we have adopted the phrasing 'height-to-background ratio' instead.

> The model of the background must be given

We now describe that the background for all three panels of fig. 1 were found using the running-window median filter implemented in `lightkurve`.

> Please provide correct notation, with a single symbol, for the background (and not BG as in Eq. (2))

This seems more an issue with aesthetic preference rather than correctness, but we now use B for the background in eq. 2 in deference to the referee's wishes.

> - Dividing physical quantities by their units is very bad practice (e.g. nu/muHz, Fig. 1). The x- and y-axes of Figures 5, 10, 11 are correctly presented, but all other figures and tables suffer from this inconsistent shortcut, which must be corrected.

Regarding dividing out units in our axis labels: we note that doing so is no mere ``shortcut", but is rather a very well-established convention both within and outside of astronomy. We refer the referee to e.g. §5.4.1 of the BIPM SI Brochure (9th ed., 2019); §7.1 of the corresponding NIST Guide to the SI (NIST Special Publication 811, 2020); as well as §2.1, para. 2 of the IAU Style Manual (G.A. Wilkins, Comm. 5, in IAU Transactions XXB, 1989). In all cases, the convention that we have used (i.e. writing "quantity / unit" rather than "quantity [unit]" or variations thereof) is recommended precisely on the grounds of reducing ambiguity, as all plotted quantities are then understood to be dimensionless.

# Quadrupole modes

> The way the authors determine the quadrupole modes is not properly given. As a result, the mode parameters have ambiguous meaning and statistical tests remain inconclusive.

We have already stated clearly how we have arrived at this mode identification: by identifying the double-ridged feature as rotationally split dipole modes, placing the radial modes within the permitted range of values of $\epsilon_p$ for first-ascent giants, noting that the visibility of dipole modes requires quadrupole modes also to be visible, and identifying unassigned peaks as variously zonal or sectoral quadrupole modes. We have described our fitting and statistical testing procedure in exhaustive detail. We would appreciate clarification from the referee about what precisely they find remains inconclusive or ambiguous.

> - The authors claim that the 'linewidths are not physically relevant for the discussion.' This is incorrect, and contradicts the result that leaving individual linewidths as free parameters hampers any fit.

The physical interpretation of linewidths is as damping rates/mode lifetimes. Nowhere in our subsequent analysis do we discuss the physics underlying the excitation or damping of modes; neither do we use the linewidths as a constraint on Zvrk's physical properties. Thus, this statement as we have written it is correct, contrary to the referee's misrepresentation. If the referee disagrees, we ask them to point out any instances of our physical use of the linewidths in the paper, so that we may remove them.

It is true we found that permitting the mode-wise linewidths to vary freely interfered with the fitting procedure. However, this only indicates that the fitted linewidths may not be relied upon for physical interpretation, and, as described, we have made no attempt to do so. We will appreciate if the referee would explicitly explain what they find to be contradictory about this.

> - As already noted, the degeneracy of the mode identification is total: ALL peaks identified as quadrupole modes can be confused with either radial or dipole modes.

As can be seen in fig. 2, this is factually incorrect. In particular, none of our identified zonal ($m = 0$) quadrupole modes may be misidentified as radial or dipole modes.

> The word 'degeneracy' needs to appear in the text

We have added it.

> and the robustness of any statistical tests against degeneracy must be presented.

As noted in the our last reply, we have chosen a methodology for our statistical tests *specifically* to prevent the quadrupole modes from being misinterpreted as more significant than they actually are, in the event that they should overlap with radial or dipole modes. This directly addresses their concern of degeneracy. The referee has not materially engaged with this preemptive good-faith effort on our part. We now provide literature references describing the principles of operation for likelihood-ratio tests of this kind. This may not satisfy the referee, but they do not offer any constructive alternatives.

> - The crucial role of the linewidths is evident from the fact that, for 10 quadrupole modes with a claimed identification, 6 overlaps with other dipole or radial modes in total confusion, 3 overlap with a frequency difference below the mean linewidth, 1 overlap with a frequency difference of about 2 x linewidth.

This observation is noted. We are not sure what the referee wants us to do with this information, and would appreciate clarification. As above, the referee also neglects to mention that the three zonal quadrupole modes nearest to $\nu_\mathrm{max}$ can be seen to unambiguously exhibit power in fig. 2, and do not overlap modes of any other degree.

In any case, we refer the referee to Kamiaka et al. 2018 for a clear description of why fitting frequencies, linewidths, and heights all individually (rather than pooling across multiple modes through some model) is not necessarily the ideal thing to do when modes overlap --- doing so systematically biases the resulting estimates of not only their linewidths and heights, but also underlying separations, and associated derived quantities (e.g. their fig. 3). Correspondingly, while we agree that having a single pooled linewidth across all modes will return unreliable estimates of the linewidth, the referee's observation indicates clearly that, given such a configuration, this is in fact preferable to letting them all vary freely in the fit, as we describe having previously attempted. We have added this discussion to the main text. As we already note, we do not treat the linewidth as anything but a nuisance parameter. 

> - Quadrupole heights in a given multiplet seem to be defined as a function of a mean height modulated by the inclination and the azimuthal order. If I am right, then the fit relies on the assumption that |m|= 2 modes in a given multiplet have the same height. This must be clearly stated.

We now do so.

> This assumption certainly contributes to locking the fit on the initial guess. In real spectra, the heights of modes with same |m| are often quite different. Therefore, the authors have also to test a fit with independent individual heights.

Such a test is not only a methods paper in itself, but effectively already has been done. The fitted rotation rates and inclinations of Kuszlewicz+ 2019, MNRAS, 488, 572, who model multiplets with symmetric heights as we do, were found by Gehan+ 2021 (their section 4) to produce results which were consistent with an asymmetric-height model, in the regime of symmetric rotational splittings (as considered here by ansatz).

# Differential rotation

> The authors claim the detection of a weak radial differential rotation. As shown above, the degeneracy of the oscillation spectrum must be considered before deriving any conclusion.

As we have discussed above, we have made methodological decisions explicitly to avoid the systematic pitfalls that would be encountered when fitting peaks individually, of the kind described in Kamiaka et al. 2018, as well as specifically to ensure that the significance of overlapping modes is not overstated. The referee has not offered any explicit guidance as to what exactly about this methodology they deem to be inadequate, or any specific and meaningful changes they wish to be made.

> Even in the absence of degeneracy, the frequency difference reported (and illustrated by Fig. 3) is a very small fraction of the mode widths. This hampers any conclusion in terms of differential rotation.

We remind the referee that oscillation mode frequencies are generally constrained far more precisely than suggested by their associated linewidths. A mode in the power spectrum can both have a very wide profile, and yet also be well-constrained in position, without any logical contradiction. The relevant property to consider here is instead the width of the posterior distribution, as we have done.

> Did the authors test any possible pollution via the window effect?

See our reply above regarding the window function.

> Moreover, the analysis in Appendix B demonstrates the irrelevance of testing differential rotation.

This is justification post hoc that we cannot rely on in the paper itself. Moreover, we remind the referee that suppressing the reporting of negative results is itself a statistical bias.

> Figure B3 shows a fully insecure analysis: such exploded posterior distributions mask any firm result. The sentence in lines 1861- 1863 confirms that the analysis is unsuccessful.

The referee perhaps confuses the basis of our claim for differential rotation, which is illustrated in fig. 3, with our attempts to constrain its location within the stellar structure, which is the focus of Appendix B. We agree that Appendix B yields negative results. However, our failure there to completely localise this differential rotation does not change the conclusion that we first drew from fig. 3. We now try to make this distinction clearer in the text.

> To sum up, from a biased, inappropriate, negative analysis, the authors conclude a 'weak evidence for differential rotation.' This claim has a serious consequence: readers are likely to be unconvinced.

We take seriously any accusations of bias or scientific misconduct. These are severe allegations that referee must substantiate.

Our negative result in the appendix is precisely why we report the evidence for differential rotation as being weak. We are not sure what the referee wants us to do here, or indeed what else we can do here. Clearly we cannot report a strong detection; in frequentist terms, we likely cannot reject a notional null hypothesis that the pooled dipole and quadrupole rotational splittings are equivalent. By the same token, however, it would also not be accurate to claim that we find evidence for solid-body rotation in the envelope, as that would require them to agree; again in frequentist terms, a double-hypothesis equivalence-interval test between the two would also fail to reject a null hypothesis that the two are inequivalent. We chose the specific phrasing of "weak evidence" to describe this situation, in light of the fact that fig. 3 illustrates a preference for inequivalence, and therefore differential rotation, from a Bayesian perspective. We would be happy to rephrase this if the referee has suggestions for alternatives which fully encapsulate the situation that we have described.

# Section 4

> The name of the Section, Discussion, seems to indicate that results obtained in the previous Sections will be discussed. But new concepts are introduced and developed over 5 pages, in an unclear way. They all address different aspects of the rotation of Zvrk.

We have renamed the section, and the one after it. We would appreciate if the referee could specify concretely what exactly they find unclear in it, so that we can clarify those points.

> - The flow of Section 4 is as confusing as the layout of the whole papers: the order of presentation is often illogical, most often incomplete, and therefore unclear.

We would appreciate the referee being actionably specific as to how and (more importantly) where we have been unclear, incomplete, or illogical, so that we can remedy these apparent faults.

> Again, the organization of the paper is faulty, with arguments of Section 4.3 already used in Section 4.1 (line 988). 

As above, the position of this section is constrained by logical dependencies. The bulk of Section 4.3 in particular concerns magnetorotational braking in an engulfment scenario, and so must come after Section 4.1 (introducing and justifying consideration of such a scenario) and Section 4.2 (justifying the consideration of magnetorotational braking in a red giant). The nonbraking scenario is only very briefly considered in Section 4.3 as a counterfactual introduction to the rest of the section, and to substantiate a claim made in Section 4.1 regarding why scenario (I) alone is not tenable.

- Concepts are not properly introduced, with many omissions and shortcuts. The reader is often lost, for instance when concepts are discussed without any precise definition, or even without any contextual elements.

Again, we would appreciate if the referee could point out specific omissions or elisions, concepts lacking introduction, or which definitions/what context they feel we should add to the section, so that we can actually add them.

> The introduction of four scenarios does not help clarify anything: the discussion of scenario I is rapidly mixed with scenario III. The discussion of scenario II introduces elements of scenario IV: this disorder underlines the lack of logic of the layout. The readers need a clear discussion, in a systematic way, instead of complex speculations.

This is a blatant misrepresentation of our discussion. We have indeed investigated these scenarios systematically, and in isolation as far as we are able. After introducing our four scenarios, our first paragraph explains why scenario (I) is not tenable in isolation. The next paragraph explains why scenario (III) is not tenable in isolation. Thus, the discussion of these two is not mixed, contrary to the referee's misrepresentation.

The next few paragraphs develop why any astrophysical configuration which can explain Zvrk's rotation in scenario (II) in isolation, while remaining consistent with Gaia radial velocities, must lead to an engulfment (scenario II implies IV also). Finally, the remainder of the section examines scenario (IV) in isolation, and in particular the properties that a companion in such an engulfment must have, and conclude that companion masses compatible with Zvrk's rotational constraints cannot produce the observed lithium abundance, which is likely explained by tidally induced rotational mixing during inspiral (scenario IV suggests II also). Thus, we conclude not only that none of the four scenarios in isolation is adequate, but specifically that two of them are not tenable, and the other two must happen together, for full consistency with all observational constraints.

We think that this organisation could be made clearer with the introduction of subsubsection headings, which we have added. Otherwise, should the referee still find this layout illogical (misrepresentations aside), we would appreciate if they would provide clear, concrete, and constructive suggestions for improvement, rather than simply dismissing it as they have done.

> - Section 4.1: the large amount of lithium should be used to reach more direct results. If the engulfment of a single planet cannot work for obvious quantitative reasons, then there is no need to consider this hypothesis in isolation. More recent references should be used. In their answer, the authors 'find that a planet-engulfment scenario is preferred to tidal spin-up or mass transfer, for example.' At lines 1143-1145, we read 'Zvrk would have had to ingest the lithium supply of ~ 10^3 gas giants to generate this chemical signature from engulfment alone.' Then the process that provides the equivalent of 999 engulfments should be preferred?

The referee's response illustrates well why it was important to systematically investigate the astrophysical implications of each of the scenarios that we have denumerated, as we have done. Specifically, these remarks are not only incompatible with our findings in this section, but also misrepresent it by taking a single sentence out of context.

By the time this sentence appears, we have already shown that tidal spin-up of the kind required to produce Zvrk's rotational configuration will lead eventually to engulfment of the perturber owing to dissipative instability. Moreover, a companion massive enough to spin Zvrk up to its current rotation rate by tidal effects alone before merging will be spectroscopically and photometrically consistent with existing samples of stellar-mass merger remnants, which Zvrk is not, as we discuss at the start of the section. Therefore, while a planetary-mass engulfment scenario is still preferred, this sentence serves to advance the argument that it alone cannot be solely responsible for Zvrk's lithium abundance.

> None of the four scenarios in isolation is adequate.

Correct. As above, this is a *result* that we are presenting.

> Nothing is said about possible interactions.

As above, this misrepresents our analysis, which in fact finds that some interaction between scenarios (II) and (IV) is mandatory. (II) alone cannot yield a stable orbital configuration undetectable to Gaia, and (IV) alone cannot explain the observed lithium abundance.
 
> The mention of the Cameron-Fowler process is not put into context (by the way, more recent work on Li should be cited, at least for examining surface abundance variations as a function of stellar evolution, as for instance Lagarde et al. (2015, A&A 580, A141)

We have added the requested citation.

> - Section 4.2 is also unclear: the first sentence introduces magnetic activity in *dwarves*, and no link with magnetic activity in red giants is given.

We have added a reference linking this to magnetic activity in red giants.

> None of the terms of Eq (9) are defined.

We have added definitions where appropriate.

> - Section 4.3 is missing important information. Fig 12 is totally unclear: the reader lacks basic information needed to understand Fig 12.

As far as we can determine, all quantities appearing in these figures are defined and explained in both the figure caption and the main text. If the referee finds any missing, we would appreciate they point them out to us, so that we can add them. Otherwise, we would also appreciate if the referee could explain why they find Fig. 12 unclear (as it is just a plot of the results of our numerical integration of magnetorotational braking) or what basic information they find lacking, which we should add.

> Moreover, terms are not defined, as vSP13.

This is provided in the figure caption as a reference to van Saders and Pinsonneault 2013. We have made this clearer.

> Unscientific terminology are introduced: what are more aggressive/pessimistic scenarios?

We now use alternative adjectives to denote stronger/weaker braking.

> This section should be removed from the current paper, and can well constitute the basis of a new paper, for the [above] reasons:

> As written above, this Section can and should be removed from the paper.

We would appreciate if the referee would justify this request. The interpretation of observational results is a fundamental part of science, and up to this section, all of the presented analysis has so far been the distillation of raw observations into a sparse set of derived quantities, with no astrophysical interpretation. The removal of this section would therefore be of very significant detriment to the scientific value of the paper, contrary to the referee's advertised goal of improving its quality.

> Pay attention to avoid imprecise sentences as
> - 'Since we have only measured Zvrk's present state' (line 950)
> - '...departures from the existing theory of single-star evolution (line 959)': do you mean 'standard model'? 'Theory' is inappropriate'

We have revised these sentences to be more precise.

# Appendix B

> The new method introduced here should be tested on a high-quality spectrum, as explained by the authors (lines 1724-1726).

We agree. The method presented here is a greatly simplified version of one currently in development that also fully accounts for coupling of dipole modes to an internal g-mode cavity. A test of the full method must both include and supersede this special case of no coupling, and lies in the scope of a different paper, not this one.

> - Lines 1651-1656: The sentence should be reworded or skipped.

We are not sure what about the sentence the referee wishes us to reword or remove, and would appreciate clarification.

> - Figure B1: the y-caption of Fig B1a is incorrect; it should be different from Fig B1c

It is indeed different. Note that $K$ is marked, and shown, as a function of $r$ in B1a, but of $t$ in B1c and B1d, through the reparameterisation that we describe in the text.

> - Figure B3 shows whye the analysis is not satisfying, with exploded posterior distributions. This figure shows the weakness of the conclusion in terms of 'differential rotation', as indicated by the authors in lines 1861-1863. Why persist in announcing a weak detection of differential rotation?

See above for our response to identical points.