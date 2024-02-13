We thank the referee for their third report, and also for taking the time to prepare it over the holiday season. We respond to their points in the order in which they appear. Again, unless otherwise specified, section and equation numbers will refer to those of the reviewed rather than updated manuscript.

1. Linewidths and degeneracies.

> The authors persist in claiming that 'linewidths are not physically
> relevant for the following discussion,' despite the fact that linewidths
> are crucial for deriving a relevant fit. This is clear from lines
> 312-316, which show that no fit can be derived with linewidths fitted on
> a per-mode basis.

The referee suggests that no asteroseismic constraints may be derived when linewidths are left to vary individually, which is not true. The lines which the referee cites read: "when linewidths were fitted on a per-mode basis, the posterior distributions for modes with low SNR were very poorly constrained, and therefore permitted to take on unphysically large values at low amplitudes, given our uninformative prior". That is to say: fits *could* be derived, but the posterior distributions in the linewidths of, specifically, low-amplitude modes --- several of which we already mark in Appendix B not to be individually significant anyway --- were so unconstrained as to assign nontrivial probability to large linewidths that are known a priori to be physically unmeaningful. In turn, this is because we had deliberately chosen an uninformative prior for use in the fitting procedure, rather than a physically motivated one.

We have adjusted the paragraph to make it clearer that it is primarily the lowest-amplitude modes that suffer from these considerations. We have already marked these modes as not being individually significant. Being of lowest amplitude, they already contribute the least to our constraint on the rotational splittings, and (through having many-times-larger uncertainties) on the derived stellar properties and structure. The high-amplitude modes closer to $\nu_\text{max}$ were still easily fitted without issue, with physically reasonable linewidths being returned. As can be seen from our table of detection significances, these modes already dominate the constraints we have on parameters that are shared between modes in our model, such as the linewidths and rotational splittings, as well as on other derived quantities.

Moreover, while a constant linewidth is more physically motivated than independent uninformative priors on individual modes, we still make no attempt to assign physical interpretations to our fitted linewidths, and so even if our fitted linewidths were to be systematically incorrect, this has no bearing on any of our subsequent astrophysical discussion. We have additionally rephrased the text to make this last point clearer.

> The authors have also shown that the fit does not
> converge when individual heights are left free.

The convergence of the fit when individual heights are left to vary freely does not fall at all within the scope of our paper, or earlier correspondence, given that we had opted from the beginning not to permit the heights of individual multiplet components to vary independently in the first place. If this statement appears in the manuscript, it would be factually incorrect, in which case we would appreciate the referee pointing out its location so that we can delete it.

> This means that authors
> had to significantly reduce the number of degrees of freedom, so that
> the fitting process could (artificially) divide single peaks in two
> modes: a quadrupolar component and a radial or dipolar component.

We do not currently know how it will be possible to accommodate the referee's objections --- that our data model has too few degrees of freedom --- while still satisfying existing best practices in statistical analysis, which (as far as we understand) instead explicitly penalise having too many degrees of freedom in a fitted model.

To simplify subsequent discussion, let us consider the referee's own chosen illustrative example of fitting two modes to a single peak in the power spectrum. In such a case, the underlying cause of degeneracy in the posterior distributions --- and thus strong posterior correlations between the properties of the two fitted modes --- is too many degrees of freedom to be fitted against this single peak in a well-posed fashion (i.e. the fit is underdetermined), rather than too few. All parameters for these two degenerate modes will then have deterministically overestimated posterior uncertainties, compared to if only a single mode were to have been fitted to the single peak in the data. This is easily generalisable to more complicated underdetermined configurations.

If indeterminacy of this kind should result from fitting individual modes to our power spectrum, then it does not make sense to report the uncertainties from such an underdetermined model anyway, rather than applying the standard remedy of using a more parsimonious model with fewer free parameters, as we have already done. Conversely, even if our power spectrum already does not fully determine all of the parameters of our current power-spectral model to begin with, *increasing* the number of degrees of freedom in the model, to comply with the referee's objections, can be seen to be an incorrect remedy. Further, if our existing fit to the power spectrum is already the result of purely artificial separation of single peaks into multiple components (as the referee claims), then it follows that our currently reported uncertainties would already be *larger* than we would obtain with a properly-constrained model for the power spectrum, free of degeneracies, rather than being underestimates as the referee suggests below.

In the simple example of two modes fitted against one peak, a reduction of dimensionality might be achieved by only fitting one mode to such a peak. However, deleting modes from the fitted list is not the only correct way of reducing the number of free parameters in a power spectral model. For example, a dipole-mode triplet for a star seen equator-on will have its $m=0$ mode's parameters be unconstrained by the power spectrum, but fitting a triplet with an inclination parameter controlled by relative mode heights is no less valid a solution than simply deleting the $m=0$ mode from the fitted model altogether when fitting individual peaks. For the actual power spectrum, we thus employ both the already well-established practice of forward-modelling entire multiplets, as well as approximate the linewidths as being close to constant with frequency.

> So, on this point, the authors need to properly account for the effect
> of degeneracy, and explicitly alert readers that their identification of
> the oscillation spectrum is possible, but uncertain. They also have to
> revise the uncertainties.

We agree with the referee that any possible indeterminacy from overlapping peaks in the power spectrum ought to be handled as rigorously as possible. However, it is clear, from the above discussion, that intentionally inflating our uncertainties, e.g. by reporting posterior uncertainties of a purposely underdetermined model of the power spectrum with deliberately more degrees of freedom than may be constrained by the data, is not necessarily more correct than what we have already done.

We think that the referee's overall objections originate, fundamentally, from their being not fully confident in our chosen data model and proposed mode identification. In our first reply, we had noted that all quantities and uncertainties emerging from our seismic data analysis are ultimately derived from conditional distributions associated with our specific choice of power-spectrum model and mode identification. If we wished to be rigorous in accounting for our uncertainty in selecting both of these, we ought then instead to report quantities and uncertainties associated with the full posterior distributions, which would be a combination of conditional distributions from all possible mode identifications and models, weighted by probabilities assigned to these mode identifications and models. Such full uncertainties certainly would be larger than we report here, but this would potentially also come at the expense of e.g. multimodal posterior distributions, and accounting for this would not be as simple as only revising our uncertainties by increasing them.

In that same reply we had noted that, as far as we are able to determine in the asteroseismic literature, only one identification is ordinarily assigned to modes before further analysis is performed. To our knowledge, no statistical machinery exists yet in the asteroseismic literature by which one may assign probabilities to different data models and mode identifications, nor to denumerate them exhaustively rather than only marginalising over a limited subset of hypotheses. Developing the general machinery required for both tasks lies outside the scope of this work, which concerns Zvrk specifically. On the other hand, without this machinery, we believe that we would introduce errors into the analysis by reporting anything other than such conditional quantities and uncertainties as we already do.

In light of all this, we have opted instead to not only caution the reader explicitly that all quantities we report are conditioned on our choice of data model and mode identification (as the referee suggests), but also to highlight upfront the fundamental methodological limitation that, for lack of existing options, we are unable to assign any probability to this mode identification in particular, compared to others. In our conclusion, we now highlight that this may serve as motivation for future work actually developing such techniques.

2. Differential vs. Solid-Body Rotation

> The authors persist in claiming a 'weak evidence for rotational shear.'
> All the rotation rates agree within 1-sigma (Figure 8), so that deriving
> any evidence of differential rotation, even weak, is not at all a
> significant finding. A 1-sigma difference (and certainty less than 1
> sigma, since sigma is underestimated) cannot support a detection. Nearly
> flat rotation (which is an interesting result) is as probable as
> differential rotation.

> Before publication, the authors need to remove claims of a weak
> detection of differential rotation or rotational shear.

The referee's assertion, of solid-body rotation being equally likely to the scenarios we have considered, is very interesting, and warrants further detailed investigation.

We have now performed additional tests in Appendix C comparing both of our differential-rotation scenarios with the case of solid-body rotation. This now allows us to quantitatively assess the strength of any claim of differential rotation through a likelihood-ratio test against a "null hypothesis" scenario of solid-body rotation, which we now also more faithfully characterise using the machinery of RLS inversions. Moreover, by construction such a likelihood-ratio test does not depend on the width of the posterior distribution, even if misestimated. While this is (as the referee describes) not conclusive, *both* scenarios are nonetheless favoured over the solid-body scenario, with better than $1\sigma$ support. In any case, even if the tests had gone the other way, a failure to reject the null hypothesis still would not have entitled us to accept it and claim solid-body rotation to be equally plausible.

In including these additional tests, we have substantially modified the discussion in Appendix C.2.2, and our evaluation of the likelihood of differential rotation in §3.3 (now §3.4; see below). We believe that this new comparison of differential and solid-body rotation scenarios is both more rigorous, and fairer, than our previous methodology. We thank the referee for the suggestion.

3. Organisation

> The authors should move Section 2.4 toward the beginning of a (new) section "Discussion," for the sake of clarity.

We understand that the referee's suggestion may be motivated by the fact that all of the rest of §2 is concerned strictly with the gathering and first-order analysis of observational data. Conversely, §2.4 contains second-order analysis of the kind mostly also performed in §3. This being the case, we agree that it makes sense to split §2.4 off from the rest of §2. However, we still believe §2.4 has to come before the rest of §3, since all following analysis relies on the evolutionary identification discussed therein. Rather than placing it into its own section altogether (which would be very short), we have decided instead to transfer it over to the beginning of §3 (to now serve as the new §3.1). Since this does not materially affect the content of the section --- merely its organisation --- we have opted not to bold out the text in the revised manuscript.

> In Section 4.3, Equation 10 remains cryptic, with undefined terms.

All terms appearing in eq. 10 are already defined in the text. However, we had placed a few definitions several sentences after the equation itself; it is likely that this distance is what has caused the referee to miss them, and it is also likely that a reader might miss them in a similar fashion. To avoid this, we now place these definitions immediately after the equation.

> Figure 12 is not analyzed in a clear way (not to mention inconsistent presentation: at line 1241, the vertical line indicates a notional rotation period of 100 d, but the caption indicates that this line corresponds to $\nu_\text{max}$.

We now make clear in the text that the vertical line indicates the currently observed $\nu_\text{max}$, at which we impose in our simulations a rotation rate of 100 d. We have also restructured our description and discussion of the figure in an effort to improve clarity.

> I reiterate my recommendation to split the paper in two parts, to allow
> for a clearer presentation of all elements introduced in Section 4. If
> not split off, a short paragraph introducing Section 4 should be added.

We have taken the referee's suggestion of writing up an introductory paragraph for the section.