---
geometry: margin=1in
---

We thank the referee for another critical and detailed review, to which points we respond in the order that they appear (with some minor rearrangement --- unfortunately necessary to assist with the overall organisation of such a long reply). Unless otherwise stated, line, section, and equation numbers refer to those of the reviewed, rather than updated, manuscript. 

We appreciate the reminder of the constructive nature of the review process and its goal of enhancing the paper's quality. While we strive to understand and act upon all feedback, there were however several instances where we require further clarification. We also noticed some discrepancies between our original content and certain interpretations in the feedback. We believe addressing these might further enhance the accuracy and value of our work.

<!--Since the referee has seen fit to remind us that the reviewing process is intended to be constructive, with the stated goal of improving the quality of the paper, we will give them the benefit of the doubt, and assume that they have simply forgotten to actually provide actionable feedback of this kind in the many places where we request clarification. We must note with disappointment, however, that the referee has either missed or chosen not to engage with several similar requests for clarification in our last reply. Despite the referee's stated wishes to be constructive, we also find in multiple instances that the referee first materially misrepresents what we have written, and then finds fault with our discussion based primarily on these material misrepresentations. In multiple others, their suggested actions would (as we explain) proactively devalue our work, if taken. On top of this, we are dismayed that the referee impugns our methods and intentions without substantiation. If the referee genuinely wishes to be constructive, as they assert, we must ask that they restrict themselves to comments which are actually salient to the scientific merit of what we have presented.-->

# Comments that were not, or not properly, taken into account

> *- The properties of the time series are not given. The frequency resolution is given as 8 nHz (in the caption of Fig 2), same as Kepler. However, in lines 1725-1726 the authors note that the TESS data have 'shorter duration compared to the nominal Kepler mission.' It is absolutely necessary to give precise information about the times series (lines 159-161 are much too vague).*

We have added a statement that the time series spans an effective length of 3 years with a 1 yr gap. We have also rewritten the figure caption for fig. 2.

> *The window effect and its consequence has to be studied: do aliases perturb the analysis?*

We appreciate the point being raised and recognize its value. However, we would like to highlight that this particular point was not mentioned in the referee's original report.

![Window function of the full time series in the time (left panel) and frequency (right panel) domains.](window.png)

We plot in fig. 1 the window function of the full timeseries. The left panel shows it in the time domain, while the right panel, which we will focus on, shows it in the frequency domain. In addition to the nominal central peak, we see also sidelobes, the most prominent of which recur at frequencies close to the nominal recurring downlink frequency of once every 13 days (i.e. $0.89\ \mu$Hz). However, the low prominence of these sidelobes compared to the main peak is such that they cannot explain the multiple peaks of similar relative height that are visible in the power spectrum. Moreover, these sidelobes are also widely separated enough that they cannot have been misidentified as producing either the small separation between modes of alternating even degree (around $0.35\ \mu$Hz), nor our putative identified rotational splittings (around $0.1\ \mu$Hz). Other families of sidelobes and their aliases can be seen to be of far lower prominence, and are thus not of observational concern. As such, we believe our proposed mode identification is not the result of misinterpreting aliases in the observational window function. We have added the above discussion to the manuscript, along with a simplified version of the figure, in the form of a new Appendix A.

> *The power spectrum density should be shown in a broad frequency range (broader than current figures 1 and 2).*

We have expanded the plot range still further, as requested.

> *- Many paragraphs can be omitted since they introduced unnecessary information (for one example, the lines 199-204 dealing with clump stars - unneeded since Zvrk cannot be a clump star).*

We note that we had previously requested similar clarification for an identical point in our last reply. Regarding clump stars specifically, we only mention clump stars in two short sentences: one only regarding the visual appearance of the echelle diagram (rather than assigning evolutionary identification), and one specifically saying that Zvrk is not a clump star. The lines that the referee highlights are where we first classify Zvrk as either a first-ascent RG, or AGB star, and are therefore not superfluous. If the referee has specific paragraphs in mind otherwise, we would appreciate detailed references so that we can address them accordingly. 

Additionally, we recognize that seasoned experts in seismology may have immediate insights that not all readers or authors might share (such as the referee's immediate understanding, from seismology alone, that Zvrk cannot be a clump star --- as we point out, overriding the judgement of automated analysis tools). Our intent is for the paper to be accessible to a broader audience, not just seismology experts. Feedback on how to maintain this balance while ensuring clarity would be greatly appreciated.

<!--As an aside: we note also that the referee is fortunate to be able to tell immediately by experience from seismology alone that Zvrk cannot be a clump star (as we point out, overriding the judgement of automated analysis tools). Other readers, and for that matter most authors of this paper, are not so well-equipped. In editing this paper, we had removed many introductory points from Sections 2 and 4 for length, owing to feedback that we received about such material being obvious. It is clear from the referee's reactions to Section 4, to which we respond below, that this obviousness was not universal. We therefore by the same token believe that what the referee finds obvious and unnecessary here will also not be so to non-seismologists reading the paper. As the referee may note by the nature and contents of Section 4 (notwithstanding their persistent requests for us to delete it, which we also address), this paper is clearly not intended to be read solely by seismologists.-->

> *- The lack of relevance of Section 2.4 is not because of its content, but from the inappropriate location of this section.*

The placement of Section 2.4 is a result of our careful consideration of multiple constraints required to ensure logical consistency and coherence of the paper as a whole. Section 2.4 cannot be deleted, as otherwise Occam's razor would preclude us from considering nonstandard evolutionary scenarios for a red giant in Section 4, in lieu of well-behaved scenarios under alternative object classifications. However, as written, Section 2.4 must come after both Section 2.3, which introduces the spectra used in discussing its lithium and H$\alpha$ lines, and Section 2.2, which introduces our ASAS-SN and custom TESS photometry also used in the discussion of its rotation. Sections 2.2 and 2.3 in turn must both also come after Section 2.1, as our analysis there relies on seismic rotation rates and $\nu_\mathrm{max}$. Thus, Section 2.4 is constrained by these logical dependencies to be placed in its current position at earliest. From Section 3 onwards, all analysis and discussion relies on the first-ascent-giant evolutionary classification; since this is justified in Section 2.4, it also cannot be placed later in the paper. If the referee has alternative recommendations on structuring this section, kindly provide specific suggestions, and we'll gladly review them.

# New issues

> *- Concerning the definition provided for S/N shows: what matters here is the ratio of the seismic signal compared to the background. The granulation background, even if noisy due to incoherent signal, follows rules that are deterministic. Hence, the denominator is not pure noise. In fact, the authors do mention the "noise" (with quote marks introduced by them) due to granulation. Many papers use the expression 'height to background ratio'; the authors should do so.*

We have adopted the phrasing 'height-to-background ratio' for the caption of fig. 1, and its description in the main text.

> *The model of the background must be given*

We now describe that the background for all panels of fig. 1 were found using the running-window median filter implemented in `lightkurve`. We have already specified our parameterisation of the background model used in nested sampling.

> *Please provide correct notation, with a single symbol, for the background (and not BG as in Eq. (2))*

We now use $B$ for the background in eq. 2 in deference to the referee's wishes.

> *- Dividing physical quantities by their units is very bad practice (e.g. nu/muHz, Fig. 1). The x- and y-axes of Figures 5, 10, 11 are correctly presented, but all other figures and tables suffer from this inconsistent shortcut, which must be corrected.*

Regarding dividing out units in our axis labels and table headings: doing so is a very well-established convention both within and outside of astronomy. We refer the referee to e.g. §5.4.1 of the BIPM SI Brochure (9th ed., 2019); §7.1 of the corresponding NIST Guide to the SI (NIST Special Publication 811, 2020); as well as §2.1, para. 2 of the IAU Style Manual (G.A. Wilkins, Comm. 5, in IAU Transactions XXB, 1989). In all cases, the convention that we have used (i.e. writing "quantity / unit" rather than "quantity [unit]" or variations thereof) is recommended precisely on the grounds of reducing ambiguity, as all depicted quantities are then understood to be formally dimensionless.

# Quadrupole modes

> *The way the authors determine the quadrupole modes is not properly given. As a result, the mode parameters have ambiguous meaning and statistical tests remain inconclusive.*

We do not understand what the referee means by their assertion that our procedure is "not properly given", as we have already stated clearly how we have arrived at this mode identification. We have detailed our approach in the manuscript: identifying the double-ridged feature as rotationally split dipole modes, placing the radial modes within the permitted range of values of $\epsilon_p$ for first-ascent giants from Kepler, noting that the visibility of dipole modes requires quadrupole modes also to be visible, and identifying unassigned peaks as variously zonal or sectoral quadrupole modes. Our fitting and statistical testing methodology was also elaborated upon to ensure transparency. As explained in our last reply, the methodology for the latter was furthermore designed not to unfairly overstate the significance of overlapping modes.

> *- The authors claim that the 'linewidths are not physically relevant for the discussion.' This is incorrect, and contradicts the result that leaving individual linewidths as free parameters hampers any fit.*

The physical interpretation of linewidths is as damping rates/mode lifetimes. Nowhere in our subsequent analysis do we discuss the physics underlying the excitation or damping of modes; neither do we use the linewidths as a constraint on Zvrk's physical properties. Thus, this statement as we have written it appears correct. It is true we found that permitting the mode-wise linewidths to vary freely interfered with the fitting procedure. However, this only indicates that the fitted linewidths may not be relied upon for physical interpretation, and, as described, we have made no attempt to do so.

We point out that, even in high-quality Kepler data of main-sequence p-mode oscillators, where one would imagine the fitting procedure to be ideally robust, the lowest-amplitude modes will still return unphysically large linewidths when these are left to vary freely in the fit. It is for this reason that the linewidths are often instead constrained to vary slowly as a function of frequency, rather than permitted to each vary freely (e.g. Appourchaux et al. 2016, A&A, 595C, 2; Kuszlewicz et al. 2019, MNRAS, 488, 572; Hall et al. 2021, NatAs, 5, 707). While we agree that having a single pooled linewidth across all modes will return unreliable estimates of the linewidth, doing so is closer to this well-established observational practice than having an independent linewidth per mode. Since we are not using the linewidths physically, more sophisticated techniques are not needed. Conversely, systematic biases in the mode heights and rotational widths are reported by Kamiaka et al. 2018, who do permit the linewidths to each vary freely, when peaks are close to overlapping. Using a constant linewidth thus also avoids these induced systematics in the specific case of overlapping peaks. We have added this discussion to the main text.

> *- As already noted, the degeneracy of the mode identification is total: ALL peaks identified as quadrupole modes can be confused with either radial or dipole modes.*

We thank the referee for this observation, but are not sure what they want us to do with this information, and would appreciate clarification. We would like to highlight that our identified zonal (m=0) quadrupole modes appear distinct and cannot easily be mistaken for radial or dipole modes. We look forward to the referee's guidance on this matter.

> *The word 'degeneracy' needs to appear in the text*

We have added it.

> *and the robustness of any statistical tests against degeneracy must be presented.*

As pointed out in the our last reply, we have chosen a methodology for our statistical tests *specifically* to prevent the quadrupole modes from being misinterpreted as more significant than they actually are, in the event that they should overlap with radial or dipole modes. This directly, and preemptively, addresses the referee's concern of degeneracy. Regarding its robustness, we now provide literature references describing the properties and principles of operation for likelihood-ratio tests of this kind. We believe a more detailed and general examination of hypothesis-testing procedures than this lies outside the scope of this paper, which concerns Zvrk specifically.

It is possible that the referee believes that our identification of quadrupole modes at all is spurious (despite the physical arguments we have already provided in support of their identification). We now also perform a likelihood-ratio test of our nominal model of the power spectrum, compared to one where the quadrupole modes have been removed altogether. We find that this gives a test statistic $\Lambda = 28$, removing 11 degrees of freedom in our model for the power spectrum, corresponding to a probability $p = 0.003$ that the power spectrum favouring a model including quadrupole modes can be explained by coincidence alone (roughly a 3$\sigma$ detection).

In performing this calculation, we have also recomputed all the test statistics and false-alarm probabilities for each of the modes individually using more sophisticated optimisation schemes than gradient descent, in order to avoided local optima in the likelihood function. It turns out that such trapping in local optima had given substantially lower p-values in the earlier revision than ought to have actually been the case, and we apologise for this oversight. This recomputation results in a rejection of one radial mode, two dipole modes, and one more quadrupole mode than in the last revision of the paper, but still does not materially change our results.

> *- The crucial role of the linewidths is evident from the fact that, for 10 quadrupole modes with a claimed identification, 6 overlaps with other dipole or radial modes in total confusion, 3 overlap with a frequency difference below the mean linewidth, 1 overlap with a frequency difference of about 2 x linewidth.*

See above for our response to a substantively identical point.

> *- Quadrupole heights in a given multiplet seem to be defined as a function of a mean height modulated by the inclination and the azimuthal order. If I am right, then the fit relies on the assumption that |m|= 2 modes in a given multiplet have the same height. This must be clearly stated.*

We now do so.

> *This assumption certainly contributes to locking the fit on the initial guess. In real spectra, the heights of modes with same |m| are often quite different. Therefore, the authors have also to test a fit with independent individual heights.*

We point out that symmetric-height models have been extensively used in the literature with little issue. For example, the fitted rotation rates and inclinations of Kuszlewicz+ 2019, MNRAS, 488, 572, who model multiplets with symmetric heights as we do, were found by Gehan+ 2021 (their section 4) to produce results which were consistent with an asymmetric-height technique, in the regime of approximately symmetric rotational splittings (as we also consider here by ansatz). Moreover, we have in this paper resorted to technique development only insofar as it was necessary to analyse Zvrk given a lack of alternatives. Existing asymmetric-height techniques, e.g. as employed in Gehan+ 2021, are specific to dipole modes, so a corresponding technique for quadrupole modes would first need to be developed. In this case, a viable alternative, in the form of symmetric-height models for quadrupole modes, already exists.

# Differential rotation

> *The authors claim the detection of a weak radial differential rotation. As shown above, the degeneracy of the oscillation spectrum must be considered before deriving any conclusion.*

As we have discussed above, we have made methodological decisions explicitly to avoid the systematic pitfalls that would be encountered when fitting mode linewidths individually when peaks overlap, as well as specifically to ensure that the statistical significance of overlapping modes is not overstated. 

We also note that we have claimed a weak detection of differential rotation, rather than detection of weak differential rotation. Several of the referee's comments appear to concern the latter rather than the former.

> *Even in the absence of degeneracy, the frequency difference reported (and illustrated by Fig. 3) is a very small fraction of the mode widths. This hampers any conclusion in terms of differential rotation.*

Oscillation mode frequencies are generally constrained far more precisely than suggested by their associated linewidths. A mode in the power spectrum can both have a very wide profile, and yet also be well-constrained in position. The relevant property to consider here is instead the width of the posterior distribution, as we have done.

> *Did the authors test any possible pollution via the window effect?*

See our reply above regarding the window function. In the main text, we have also added a reference to this new appendix in our discussion of quadrupole modes specifically.

> *Moreover, the analysis in Appendix B demonstrates the irrelevance of testing differential rotation.*

This is justification post hoc that we cannot rely on in the paper itself. We also wish to avoid deliberately suppressing the reporting of negative results (i.e. publication bias).

> *Figure B3 shows a fully insecure analysis: such exploded posterior distributions mask any firm result. The sentence in lines 1861- 1863 confirms that the analysis is unsuccessful.*

The referee perhaps confuses the basis of our claim for differential rotation, which derives from differing quadrupole, dipole, and photometric rotation rates (e.g. fig. 3), with our attempts to constrain its sense and location in the radial direction within the stellar structure, which is the focus of Appendix B. We agree that Appendix B yields negative constraints on the sense or location of this shear. However, our failure there to completely localise this differential rotation does not change the conclusion that we first drew from these differing averaged rotation rates. We now try to make this distinction clearer in the text, and thank the referee for demonstrating this potential source of confusion.

> *To sum up, from a biased, inappropriate, negative analysis, the authors conclude a 'weak evidence for differential rotation.' This claim has a serious consequence: readers are likely to be unconvinced.*

If the referee wishes to claim that we have been biased or have engaged in inappropriate conduct, we would appreciate them explaining how. These are serious allegations that referee cannot make lightly, and must substantiate. Otherwise, we fail to see how remarks of this kind are intended to be constructive, or raise the quality of the paper.

Our negative result in the appendix is precisely why we report the evidence for differential rotation as being weak. We are not sure what the referee wants us to do here, or indeed what else we can do here. Clearly we cannot report a strong detection; in frequentist terms, we likely cannot reject a notional null hypothesis that the pooled dipole and quadrupole rotational splittings and photometric surface rotation rate are equivalent. Equally, however, it would also not be accurate to claim that we find evidence for solid-body rotation in the envelope, as that would require them all to agree; again in frequentist terms, a multiple-hypothesis equivalence-interval test between them would also fail to reject a null hypothesis that they are *in*equivalent. We chose the specific phrasing of "weak evidence" to describe this situation, in light of the fact that fig. 3 illustrates a strong preference for inequivalent rotational splittings, and therefore differential rotation, from a Bayesian perspective. We will be happy to rephrase this if the referee has suggestions for alternatives which fully encapsulate the situation that we have described.

# Section 4

> *The name of the Section, Discussion, seems to indicate that results obtained in the previous Sections will be discussed. But new concepts are introduced and developed over 5 pages, in an unclear way. They all address different aspects of the rotation of Zvrk.*

We point out that these two are not mutually exclusive. One may discuss observational results obtained in previous sections by considering their astrophysical implications, and using concepts, outside of the primary observational context. Indeed, this is done often. We are not sure why the referee thinks this is not permissible.

Since the referee appears to principally object to the name of the section, we will be happy to rename it if they could provide constructive alternatives. If this statement is intended to improve the paper, we would also appreciate if the referee could specify concretely what exactly we have been unclear about, so that we can clarify those points.

> *- The flow of Section 4 is as confusing as the layout of the whole papers: the order of presentation is often illogical, most often incomplete, and therefore unclear.*

If this statement is intended to improve the paper, we would appreciate the referee substantiating this by being actionably specific as to how and (more importantly) where our ordering has been unclear, incomplete, or illogical, as we would be happy to remedy these faults. For that matter, we would also appreciate the referee doing so regarding the layout of rest of the paper, on which they have up to now likewise provided no constructive suggestions.

> *Again, the organization of the paper is faulty, with arguments of Section 4.3 already used in Section 4.1 (line 988).*

This materially misrepresents the content of Section 4.3. Moreover, as above for Section 2, the organisation of this section is constrained by logical dependencies. Section 4.3 concerns magnetorotational braking in an engulfment scenario, and so must come after Section 4.1 (introducing and justifying consideration of such a scenario) and Section 4.2 (justifying the consideration of magnetorotational braking in this red giant). Thus, Section 4.3 can appear no earlier in the paper. It is also the final subsection before the conclusion, and so can appear no later. The nonbraking scenario, which is also alluded to in Section 4.1, is only very briefly considered in Section 4.3 as a counterfactual introduction to the rest of the section concerning magnetic braking. If the referee finds any of this faulty, we would appreciate if they could at least explain why, or better still, provide constructive alternatives.

> *- Concepts are not properly introduced, with many omissions and shortcuts. The reader is often lost, for instance when concepts are discussed without any precise definition, or even without any contextual elements.*

If this statement is intended to improve the paper, we would appreciate if the referee could substantiate this criticism by pointing out specific omissions or elisions, concepts lacking introduction, or which definitions/what context we should add to the section, as we would then be happy to add them.

> *The introduction of four scenarios does not help clarify anything: the discussion of scenario I is rapidly mixed with scenario III. The discussion of scenario II introduces elements of scenario IV: this disorder underlines the lack of logic of the layout. The readers need a clear discussion, in a systematic way, instead of complex speculations.*

We believe that the referee misrepresents the organisation of our argument. We have indeed investigated these scenarios systematically, and in isolation as far as we are able. After introducing our four scenarios, our first paragraph explains why scenario (I) is not tenable in isolation. The next paragraph explains why scenario (III) is not tenable in isolation. As such, the discussion of these two scenarios is not mixed, contrary to the referee's misrepresentation.

The next few paragraphs develop why any astrophysical configuration which can explain Zvrk's rotation in scenario (II) in isolation, while remaining consistent with Gaia radial velocities, must unavoidably lead to an engulfment (scenario II implies IV also). Finally, the remainder of the section examines scenario (IV) in isolation, and in particular the properties that a companion in such an engulfment must have, and conclude that companion masses compatible with Zvrk's rotational constraints cannot produce the observed lithium abundance, which is likely explained by tidally induced rotational mixing during inspiral (scenario IV suggests II also).

Thus, we conclude not only that none of the four scenarios in isolation is adequate, but specifically that two of them are not tenable, and the other two must happen together, for full consistency with all observational constraints. We believe that the manner in which this discussion was laid out has been indeed been systematic and methodical.

We think that this organisation could be made clearer with the introduction of subsubsection headings, which we have now added. We have also renumbered the scenarios in keeping with the order in which they are presented (so that (II) is now (III) and vice versa), again in the interests of clarity. Otherwise, should the referee still find this layout unclear or illogical (misrepresentations aside), we would appreciate if they would provide clear, concrete, and constructive suggestions for improvement, rather than simply dismissing it as they have done.

> *- Section 4.1: the large amount of lithium should be used to reach more direct results. If the engulfment of a single planet cannot work for obvious quantitative reasons, then there is no need to consider this hypothesis in isolation. More recent references should be used. In their answer, the authors 'find that a planet-engulfment scenario is preferred to tidal spin-up or mass transfer, for example.' At lines 1143-1145, we read 'Zvrk would have had to ingest the lithium supply of ~ 10^3 gas giants to generate this chemical signature from engulfment alone.' Then the process that provides the equivalent of 999 engulfments should be preferred?*

The referee's response not only misrepresents us by taking a single sentence of the paper out of context, but also requests that we make changes which are not logically justifiable.

By the time this sentence appears, we have already shown that tidal spin-up of the kind required to produce Zvrk's rotational configuration, in an orbital configuration that evades detection by Gaia, will quickly lead to engulfment of the perturber owing to dissipative tidal instability. Thus, we have established that an engulfment scenario is the preferred explanation for Zvrk's rotation.

However, it is at this point not yet possible to rule out an engulfment scenario as also being the source of Zvrk's lithium abundance, as the amount of mass engulfed is not yet known. For this, we needed to derive bounds on possible engulfment masses that would produce the observed rotational configuration, through tidal-dissipation calculations including stellar spin-up during inspiral. The result of this calculation gives us a *planet*-mass engulfment scenario to explain Zvrk's rotation specifically. It is only in this context that this sentence then appears, to advance the argument that (planet) engulfment alone cannot be all that happened to Zvrk, given its lithium abundance.

The referee's request that we skip the above steps and simply not consider scenario (IV) in isolation, but jump directly to discussing (IV) and (II) together, would thus break the chain of logic that both justifies and requires us to consider (IV) and (II) together in the first place. Were we to do as they ask, we would purposely introduce a circular argument into the paper. This clearly is counterproductive to improving its quality. We also point out that this would directly contradict the referee's immediately preceding request that we systematically investigate the astrophysical implications of each of the scenarios that we have denumerated.

> *None of the four scenarios in isolation is adequate.*

The referee's observation is correct. As above, this is a conclusion that we arrived at by systematically considering the implications of each.

> *Nothing is said about possible interactions.*

As above, this misrepresents our analysis, which in fact finds that some interaction between scenarios (II) and (IV) is unavoidable. (II) alone cannot yield a stable orbital configuration undetectable to Gaia, and (IV) alone cannot explain the observed lithium abundance.
 
> *The mention of the Cameron-Fowler process is not put into context (by the way, more recent work on Li should be cited, at least for examining surface abundance variations as a function of stellar evolution, as for instance Lagarde et al. (2015, A&A 580, A141)*

We have added the requested citation. If this statement is intended to improve the paper, we would appreciate if the referee could specify what other context ought to be provided here regarding this physical mechanism, beyond referring to the original work in which it is developed (which we have already cited).

> *- Section 4.2 is also unclear: the first sentence introduces magnetic activity in *dwarves*, and no link with magnetic activity in red giants is given.*

We thank the referee for pointing out this missing link to magnetic activity in red giants. We have added back a sentence providing such a connection.

> *None of the terms of Eq (9) are defined.*

We thank the referee for pointing out missing definitions, and have added them where appropriate.

> *- Section 4.3 is missing important information. Fig 12 is totally unclear: the reader lacks basic information needed to understand Fig 12.*

As far as we can determine, all quantities appearing in these figures are defined and explained in both the figure caption and the main text. If the referee finds any missing, we would appreciate they point them out to us, as we would be happy to remedy this. Otherwise, if this statement is intended to improve the paper, we would also appreciate if the referee could explain why Fig. 12 is unclear (as it just directly plots the results of our numerical integration of magnetorotational braking) or what basic information they find lacking, which we would also be happy to add.

> *Moreover, terms are not defined, as vSP13.*

We thank the referee for pointing this out. This is actually already provided in the figure caption as a reference to van Saders and Pinsonneault 2013. However, we have now made this clearer.

> *Unscientific terminology are introduced: what are more aggressive/pessimistic scenarios?*

We thank the referee for pointing out the potentially confusing nature of these words. We now use alternative adjectives to denote stronger/weaker braking.

> *This section should be removed from the current paper, and can well constitute the basis of a new paper, for the [above] reasons:*

> *As written above, this Section can and should be removed from the paper.*

We observe that none of the referee's comments about this section specifically concern making a case for removing it, nor splitting it off into a new paper; the referee's only engagements with its substance (rather than its presentation) are built on misrepresentations of its content. This being the case, we would appreciate if the referee would justify this request. 

We must however point out that the interpretation of observational results is a fundamental part of science, and up to this section, all of the presented analysis has so far been the taking of raw observations, and their distillation into increasingly sparse sets of derived quantities, with little astrophysical interpretation. Such interpretation is, by design, deferred to this section. As we described in our last reply, the provision of this interpretation is necessary to connect all of the preceding observations into a coherent explanation of Zvrk as a physical object, and in that sense this interpretation is the most important part of the paper. The referee has either missed this point, or refuses to engage with it. By recommending its removal, the referee is in effect asking that we write a different paper, of severely diminished astrophysical (as opposed to observational) interest. Likewise, the arguments presented in this section rely on many parts of Sections 2 and 3. Were it to be separated out into a different paper, that paper would have no intrinsic evidentiary basis, and thus would be essentially unreadable without either significant repetition of the contents of our preceding sections, or forcing the reader to consult this one side-by-side. Thus, following the referee's suggestion to remove this section would be of very significant detriment to the scientific value of our work (whether in one paper or two), which we find difficult to reconcile with the referee's advertised goal of improving its quality. 

> *Pay attention to avoid imprecise sentences as*  
> *- 'Since we have only measured Zvrk's present state' (line 950)*
> *- '...departures from the existing theory of single-star evolution (line 959)': do you mean 'standard model'? 'Theory' is inappropriate'*

We thank the referee for pointing out our imprecision. We have revised these sentences to be more precise.

# Appendix B

> *The new method introduced here should be tested on a high-quality spectrum, as explained by the authors (lines 1724-1726).*

We agree. However, we have only introduced this method insofar as it was necessary to analyse Zvrk, given a lack of viable alternatives. The method presented here is a greatly simplified version of one currently in development that also fully accounts for coupling of dipole modes to an internal g-mode cavity. A test of the full method must both include and supersede this special case of no coupling, and this lies in the scope of a different paper, rather than this one (which concerns Zvrk specifically).

> *- Lines 1651-1656: The sentence should be reworded or skipped.*

We have reworded by including a reference to a paper describing how mixed-mode kernels respond nonlinearly to small structural perturbations --- a defect not shared by these pure p-mode kernels.

> *- Figure B1: the y-caption of Fig B1a is incorrect; it should be different from Fig B1c*

It is already different. Note that $K$ is labelled, and shown, as a function of $r$ in fig. B1a, but of $t$ in B1c and B1d, through the reparameterisation that we describe in the text.

> *- Figure B3 shows why the analysis is not satisfying, with exploded posterior distributions. This figure shows the weakness of the conclusion in terms of 'differential rotation', as indicated by the authors in lines 1861-1863. Why persist in announcing a weak detection of differential rotation?*

See above for our response to identical points.