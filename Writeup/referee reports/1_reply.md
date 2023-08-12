---
geometry: margin=1in
---
We thank the referee for a detailed and critical report, to which points we respond in the order that they appear. Where necessary, excerpts appearing verbatim in our corrected manuscript are marked out in **bold face**.

Before we respond to their specific comments, there are some overarching features of the referee's notes, and our replies to them, that we must first remark upon. A substantial majority of the referee's comments appear to ultimately derive from a single misunderstanding, owing to a plotting error on our part in the construction of the first panel of Fig. 1. Had it correctly depicted the data as shown, the referee judges, correctly, that our seismic detection would have been insecure. As we note below, many of the referee's subsequent comments --- including their dismissal of our mode identification as spurious, rejection of our statistical analysis as invalid, discounting of our reported uncertainties as unrealistic, and suggested elimination of our further astrophysical interpretation without regard for their non-seismic observational substantiation --- appear all to be coloured by this single misinterpretation. Correspondingly, we believe our rectification of this error (and also, hopefully, this misunderstanding) renders these comments substantively no longer applicable. We mark the relevant points below with an asterisk (*). Moreover, since our seismic analysis turns out not to be as invalid as the referee initially understood, we hope that the referee will now also see fit to engage materially with the substance of the non-seismic portions of our observations, analysis, and interpretation, which they appear not to have done in their first report.

We do not wish to downplay or minimise the referee's concerns, which we feel remain justifiable from a perspective of principled skepticism. Where applicable, we have supplemented our existing analysis with additional statistical tests, to more quantitatively assess and compare the strengths of the various claims that we have made. In addition, we have tried our best to address the referee's other concerns, pertaining to the overall structure and organisation of the work. It is our hope that the referee agrees that this has improved the clarity and readability of the revised work.

# Data Analysis

> 1) The Fourier spectrum in Figure 1 shows no peak with a SNR above 2.5. The application of the H0 test then indicates that noise is enough to explain all peaks. Taken in isolation, this would signal that the rest of the work is not credible.
> However, the comparison of Fig 1 with Fig 2, (a bit challenging because of imprecise captions for both figures) may indicate that the raw spectrum is not shown in Fig 1, but a smoothed spectrum. If the raw spectrum was used, then the data should be considered as consistent with pure noise. If not, then the use of the label S/N is incorrect and misleading. A proper definition of the y-axis must be given; a proper statistical analysis is needed to assess the presence of solar-like oscillations.
> [editor: showing a broader range of frequencies should help show the level of power excess as well]

There was a plotting error made in the construction of this figure. The quantity shown in the two Kepler panels of Fig. 1 is the "signal-to-noise periodogram", where the power spectrum is divided by a smooth background model, so that the expectation value of the usual multicomponent red noise is normalised to 1 (as a form of prewhitening). Indeed, one can see that the background level between peaks does take on mean S/N values of 1 in the Kepler panels. However, this can be seen to take values closer to 0.2 on average in the first panel, showing our TESS target. This is because the quantity shown in that panel is actually the exact same raw (unsmoothed) Lomb-Scargle power spectrum, in the same units, as the filled-circle data points of Fig. 2 --- i.e. the background had not been divided out when making this panel. The referee may verify that several features can be seen to take identical values in both figures, e.g. the shape of the raw data underlying the dipole doublet between 7.5 and 8 $\mu$Hz.

We agree that, were we to take SNR as originally shown at face value, these peaks would have been consistent with being noise (although one would then also question why the background was not normalised to 1). However, while the SNR of the TESS data is indeed lower than in the Kepler data, the SNR of the most prominent peaks that we have fitted is closer to 10, so these peaks really are significant. We thank the referee for catching this error --- clearly a very important one, if their reaction is any indication --- and apologise for any confusion that it may have caused.

We have now rectified the figure by showing the background-divided PS in the first panel, consistently with the other two panels, and by extending the frequency range shown to match that in Fig. 2.

> 2) The comparison of Zvrk with two stars observed by Kepler confirms that noise is dominating. For such similar bright stars observed over the course of years, the oscillation and background oscillation power spectra should be the same, with no influence of the instrumental noise, and no influence of the observation duration (except in terms of frequency resolution, but not in terms of power spectral density). Here, regardless the exact definition of S/N plotted on the y-axis (which is not given, but is the same for the three stars), it looks like the spectrum of Zvrk is dominated by noise. A quantitative discussion about the actual detection of solar-like oscillations is needed, prior to a serious seismic analysis. The method by Viani et al (2019) seems either inappropriate (it was developed for low-noise synthetic data), or is used in an improper way.

(*)

The SNR of the TESS data is indeed lower than Kepler (~10 vs. 40), but this is consistent with both rotation being known to distribute oscillation power between multiplet components (rather than concentrating it in a single peak), as well as rapid rotation being known to suppress oscillations. We should thus expect a priori that a rotating mode identification would conversely be preferred in the event of unusually low observed S/N. Since the two Kepler stars shown are not rotating, this difference of S/N further motivates our mode identification including rotation. We have added this discussion to the manuscript.

The actual SNR range of our data does render the method of Viani+ 2019 suitable for use. However we also note that we have in any case only used it as an initial guess, as our final value of $\Delta\nu$ is constructed using the fitted radial mode frequencies.

> 3) In the Bayesian analysis for assessing the seismic parameters is irrelevant, the authors have used such tight priors that the pseudo-blind Bayesian analysis solution is fixed to the hand-made solution. The (psuedo) fit of the quadrupole multiplets is based on peaks with a very tiny height.

(*)

Our choice of relatively narrow priors compared to $\Delta\nu$ is consistent with existing practice in fitting p-modes from red giants (e.g. as used in DIAMONDS, FAMED, pbjam, TACO). The usual interpretation of this is to treat the resulting posterior distributions as being in fact conditional distributions (i.e. conditioned on the mode identification). In any case, the blue shaded regions in Fig. 2 show the widths of the prior intervals, which can clearly be seen to span more than just the frequency bins containing the handmade solution around each putative $m=0$ mode.

We agree that our identified quadrupole modes are lower in amplitude than the others in our identification (as a result of both mode visibility, and the distribution of power across more multiplet components). We now report odds ratios from a H1 test in the appendix, as statistical justification for their not being spurious.

> 4) The entire analysis is based on unrealistic uncertainties. The authors claim a precision of 10 nHz for Dnu (Table 1) and for individual frequencies (Table A1), which corresponds to the frequency resolution of the data (inverse of ~3 yr). Such a precision level is not reached for longer, higher-quality data observed by Kepler.

(*)

As far as we are able to determine, this statement about Kepler data is factually incorrect. For example, DIAMONDS reports frequency uncertainties of less than 5 nHz for p-modes of red giants --- see e.g. Tables B3, B5, B8, B11 etc. of Corsaro et al. 2015, A&A, 579, A83 --- which are significantly less evolved (and therefore exhibit both lower intrinsic relative amplitudes, and far shorter mode lifetimes), but with similar apparent magnitude, compared to Zvrk. DIAMONDS uses nested sampling, as we also do, so methodologically this is also an apples-to-apples comparison.

> In general, the treatment of uncertainties is systematically unrealistic (e.g.: Z0 in Table 2, Gamma at line 344). The uncertainty on the inclination is also significantly underestimated (line 342)

(*)

We agree that our reported statistical uncertainties on $\Gamma$ will underestimate the systematic error on it. However, we do not use $\Gamma$ anywhere in our analysis; it is only fitted as a nuisance parameter.

For $Z_0$ in particular, the second modelling group also obtained a 10% statistical uncertainty. This makes the two pipelines 2$\sigma$ inconsistent with each other --- i.e. the statistical error in $Z_0$ does *not* dominate systematic error, unlike mass and radius. We now note this in the main text. However, and again, we do not use $Z_0$ directly in our analysis. In general we expect the composition returned from seismology not to be representative of Zvrk's chemically anomalous nature, and have used this modelling exercise only to yield a reasonable guess for the near-polytropic structure of its convective envelope, and for order-of-magnitude estimates of various timescales of its evolution up the RGB. We have tried to make our existing description of this clearer in the text.

> 5) Lines 329-331 illustrates a problem of analysis, where an a priori is considered as a result.

We agree that this ought not to have been done, and have deleted the offending sentence. We thank the referee for pointing this out.

> 6) Basic information is often hidden in the paper and/or comes too late. The magnitude of the star should be provided very early, so that one understands that the poor signal in the Fourier spectrum is not related to a low luminosity. The actual frequency resolution should be given (either the value 0.007 µHz in the caption of Fig 2, or the description of the time series in Section 2, is incorrect).

(*)

Easy to fix; would appreciate other suggestions.

# Seismic Analysis

> As far as I know, seismology alone is not able to unambiguously distinguish between RGB and AGB stars. Using $\epsilon_p$ for stating that the star is first-ascent giant is incorrect. A realistic relative uncertainty of Dnu implies a large uncertainty in $\epsilon_p$ and precludes any identification.

(*)

From the same fit against radial modes that produces our estimate of $\Delta\nu$, we obtain $\delta\epsilon = 0.05$, which is sufficient for such discrimination. (FILL WITH ACTUAL VALUE)

> 1) A simple sanity check strongly suggests that the identification/analysis of the quadrupole multiplets is an artifact. As shown in Fig 2, the quadrupole multiplets are artificially created by a confusion of l=2, m=-2 modes with dipole (l=1, m=+1) modes, and l=2, m=+2 with radial modes. From the figure, it is clear that the quality of the spectrum hampers the clear identification of quadrupole modes, and is by far not enough for deriving any information about them.

(*)

We already note this feature of our mode identification --- i.e. that the quadrupole triplets straddle the radial and dipole modes --- in the manuscript. Our significance testing should hopefully reassure the referee that these peaks are not spurious. Moreover, as we describe below, identification of dipole modes requires these quadrupole modes also to be identified in this fashion.

> 2) The remark above indicates that, contrary to the claim of the authors (lines 307-310), the determination of the mode widths is a crucial step in the analysis.

(*)

Given that the SNR is sufficient to yield statistical evidence for quadrupole modes, we are not sure how the above statement is salient to our analysis.

> 3) The absence of a clear quadrupole signal raises strong doubt about the identification of the spectrum in terms of solar=like oscillations. To assess the detection of solar-like oscillations, the authors have to justify the presence of a clear dipole signal, and the absence of a clear quadrupole signal.

(*)

The referee seems to agree with us that dipole modes are usually suppressed before quadrupole modes by the action of magnetic fields or rotation. Based on this they assert, having first rejected our identification of quadrupole modes, that the burden is on us to explain how p-modes would be observed with only quadrupole and not dipole modes suppressed.

However, they are in effect demanding that we defend a claim that we do not make, as this is the reverse of our line of reasoning. We argue instead that, since dipole modes are preferentially suppressed, a clear dipole-mode signal requires that we assign identifications as quadrupole modes to other peaks for consistency with the known properties of solar-like oscillations. However, the amount of suppression here is unclear, and in the absence of suppression, quadrupole modes are less visible and also have their power distributed across more multiplet components, and so individual peaks in such a rotating configuration will be of lower prominence.

This is in any case rendered moot by our significance testing.

> 4) Stellar inclination: because of the high noise level, the amplitudes of the l=1, m=0 components are largely unknown. This implies a large uncertainty on the inclination, which needs to be reflected in the analysis.

(*)

The actual SNR normalisation suggests that the amplitudes of m = 0 dipole modes are indeed low, correctly yielding a close to equator-on constraint on the inclination.

> 5) Differential rotation: Figure 8, as presented, show that little to nothing can be derived about differential rotation, or to rule out uniform rotation. The authors have to remove the l=2 data, to provide realistic uncertainties.

(*)

We agree that localising differential rotation from seismology is difficult, but the figure serves to explain this difficulty in the first place. Moreover, as we explain in the manuscript, other rotational constraints also suggest differential rotation, although they are not helpful in localising it.

> 6) At most, authors may have 2 pieces of information about rotation, as provided by the dipole modes and by the rotationally modulated spot signal. The paragraph ending Section 3 shows that the authors are aware of the weakness of their claims. In such case, there is no need to develop incorrect inferences over many pages.

(*)

Our rotational splittings are derived from multiple multiplets per degree, so in a strict information-theoretic sense the power spectrum permits more than 2 pieces of information about rotation to be derived from seismology alone. Our technique development for $\pi$-mode inversions directly against the power spectrum was necessary to maximise the extraction of this information from the underlying data (some of which would have been lost to peakbagging). We agree with the referee that the data has, unfortunately, not been cooperative. Nonetheless, although we conclude that the constraints on differential rotation are weak, it is still ultimately necessary to document the process by which we arrived at the conclusion that they are indeed weak in the first place.

# Discussion

> From the introduction in Section 4.1, one understands that there are multiple good reasons showing that Zvrk is not a merger remnant. Then weak reasons are provided over five pages. The Section discussion can and should be severely shortened. Interesting information is presented here, but it is only weakly connected to the paper.

The introduction to Section 4.1 explains why we believe Zvrk not to be a stellar-mass merger remnant in particular. The rest of this section concerns explaining how its unusual properties came to be in that case, and have nothing to do with whether or not it is a stellar-mass merger remnant. Given that the referee appears to have missed this, the clarity of presentation here can likely be improved. We have therefore undertaken significant renovations of this section.

While this section is indeed weakly connected to the seismic analysis (which we take that the referee assumes to be the focus of the paper given that it constitutes the majority of their feedback above), we point out that we also report several observational results aside from seismology --- in particular, a surface rotation period, and unusual lithium and carbon/nitrogen abundances. As such, seismology forms a minority of the observational evidence that we present, despite its analysis having been the most time-consuming! The astrophysical interpretation presented here connects all of this to produce a consistent explanation for how all of these features might be produced together despite Zvrk not being a stellar-mass merger remnant. In accordance with the referee's wishes, we have shortened this section. However, we would appreciate more concrete, actionable feedback from the referee regarding specific parts of our discussion that they find weak.

# Organisation and Length

> The layout of the paper is unclear and misleading, with too many wordy sections.

We have tried to make our phrasing more compact in many places. We would appreciate more specific feedback from the referee as to how, or which parts of the paper, they believe to be misleading.

> 1) Section 2.1 supposes that the star is a first ascent giant. Then, section 2.4 presenting the discussion about T Tauri stars is irrelevant and should be eliminated. If the star is an evolved RGB, it cannot be a T Tauri so the degree of relevance is too remote.

We are surprised that the referee should take this position, given that they have dismissed our detection of p-mode oscillations as spurious, since our identification of the star as a red giant is based on asteroseismology. Our discussion of the T Tauri hypothesis *directly addresses* the possibility that this p-mode detection was spurious. If we were to ignore our putative detection of p-modes, the combination of fast rotation and a high lithium abundance in this temperature range would immediately suggest identification of the object as being a T Tauri star (as indeed had accidentally been done with V501 Aur previously). This being the case, this discussion is highly relevant, as it directly connects our other, non-seismic, observational constraints on Zvrk, with the referee's own concerns that our p-mode oscillations are spurious.

> 2) From the possible measurements of the global seismic parameters, it is clear that the star cannot be in the red clump stage. A sentence and a reference are enough to making this clear, so that many paragraphs in Section 2.1 can be removed.

OK, but we note that this discussion is there specifically to rule out identifying the double ridge as radial/quadrupole modes.

> 3) It is also clear that modes, if real, can be considered as pure pressure modes. No need then to discuss what would happen if mixed modes were present.

True observationally, but standard pulsation theory will always give you mixed modes in stars, so this discussion is necessary for interpretation, and motivating subsequent technique development.

> 4) All these issues lower the quality and readability of the paper. I consider that the paper could easily fit in less than 10 pages. The actual content is limited (rapid surface rotation, if properly assessed) that it should be limited to 5-6 pages. From all the caveats that are mentioned above, the terms "identification" and "characterization" are insecure, so that the title should be changed.

(*)

No longer applicable.

The referee appears not to have identified any issues other than in our asteroseismic analysis (and even then largely owing to a misinterpretation of a single figure). However, we point out that our other non-seismic analysis (TESS+ASAS-SN photometry, spectroscopic abundance analysis, RV-scatter constraints, orbital and rotational modelling) constitutes more than half the paper, so the referee's suggestion is impossible to implement even if the asteroseismic component were to be eliminated entirely. We remind the referee that our overall work, as reflected in our title, combines all of this analysis. We might remove "asteroseismic" from the title instead, but we believe our seismic analysis also to be well-founded.

# Figures

> Figure 1: A definition of S/N should be given; furthermore, the relevance of this ratio should be made clear when the spectrum is smoothed (as it is certainly, according to the comparison with Fig 2); if not, then my first remark applies and the paper should be abandoned due to the absence of signal. Provide a broader frequency range, to more clearly show any power excess.

(*)

We now show the power spectrum in Fig. 1 over the same frequency range as in Fig. 2, and provide a description of the background-divided periodograms in the figure caption.

> Figure 2: Provide clearer figures, especially Fig 2b

Please elaborate

> Figure 3 is unneeded, since the determination of dnurot_2 is spurious.

(*)

No longer applicable.

> Figure 4: Provide a color code with more contrast; fig4c is unclear.

Please elaborate

> Figure 8 shows clearly that constraining differential rotation is impossible.

We are unsure what the referee is asking us to do here, and would appreciate clarification. As we discuss in the manuscript, combining the seismic rotation rate (interpreted as a solid-body average) with the surface rotational period already suggests rotational shear. We agree that localising this differential rotation from seismology alone is difficult given the data in hand (even at SNR of 10). However, as we also already discuss in the manuscript, the objective of figure 8 is to summarise these difficulties. Since this is basically also the referee's point, we're not sure if this comment leads to anything actionable.