We thank the referee for a detailed and critical report, to which points we respond in the order that they appear. Where necessary, excerpts appearing verbatim in our corrected manuscript are marked out in **bold face**.

Before we respond to their specific comments, there are some overarching features of the referee's notes, and our replies to them, that we must first remark upon. A substantial majority of the referee's comments appear to ultimately derive from a single misunderstanding, owing to a plotting error on our part in the construction of Fig. 1 specifically. Had it correctly depicted the data as shown, the referee judges, correctly, that our seismic detection would have been insecure. As we note below, many of the referee's subsequent comments --- including their dismissal of our mode identification as spurious, rejection of our statistical analysis as invalid, discounting of our reported uncertainties as unrealistic, and suggested elimination of our further astrophysical interpretation for being undersubstantiated --- are all coloured by this single error. Correspondingly, having rectified this error (and also, hopefully, this misunderstanding), we believe that this renders these comments substantively no longer applicable. We mark the relevant points below with an asterisk (*). Since our seismic analysis turns out not to be as invalid as the referee initially understood, we hope that the referee will also see fit to engage now with the substance of the non-seismic portions of our observations, analysis, and interpretation.

Nonetheless, we do not wish to downplay or minimise the referee's concerns, which we feel remain justifiable from a perspective of principled skepticism. Where applicable, we have supplemented our existing analysis with additional statistical tests, to more quantitatively assess and compare the strengths of the various claims that we have made. In addition, we have tried our best to address the referee's other concerns, pertaining to the overall structure and organisation of the work. It is our hope that the referee agrees that this has improved the clarity and readability of the revised work.

# Data Analysis

> 1) The Fourier spectrum in Figure 1 shows no peak with a SNR above 2.5. The application of the H0 test then indicates that noise is enough to explain all peaks. Taken in isolation, this would signal that the rest of the work is not credible.
> However, the comparison of Fig 1 with Fig 2, (a bit challenging because of imprecise captions for both figures) may indicate that the raw spectrum is not shown in Fig 1, but a smoothed spectrum. If the raw spectrum was used, then the data should be considered as consistent with pure noise. If not, then the use of the label S/N is incorrect and misleading. A proper definition of the y-axis must be given; a proper statistical analysis is needed to assess the presence of solar-like oscillations.
> [editor: showing a broader range of frequencies should help show the level of power excess as well]

There was a plotting error made in the construction of this figure. The quantity shown is actually the exact same raw Lomb-Scargle power spectrum, in the same units, as the points in the background of Fig. 2. As a further diagnostic --- the background level between peaks, which can be seen to be of a mean S/N of 1 in the Kepler panels, can be seen to take values closer to 0.2 on average in the first panel.

We agree that, were the raw SNR to have been shown, these peaks would have been consistent with being noise. While the SNR of the TESS data is significantly lower than in the Kepler data, the SNR of the most prominent peaks that we have fitted is closer to 10, so these peaks really are significant per $H_0$ test. We thank the referee for catching this error --- clearly a very important one, judging by their reaction --- and apologise for any confusion that it may have caused.

> 2) The comparison of Zvrk with two stars observed by Kepler confirms that noise is dominating. For such similar bright stars observed over the course of years, the oscillation and background oscillation power spectra should be the same, with no influence of the instrumental noise, and no influence of the observation duration (except in terms of frequency resolution, but not in terms of power spectral density). Here, regardless the exact definition of S/N plotted on the y-axis (which is not given, but is the same for the three stars), it looks like the spectrum of Zvrk is dominated by noise. A quantitative discussion about the actual detection of solar-like oscillations is needed, prior to a serious seismic analysis. The method by Viani et al (2019) seems either inappropriate (it was developed for low-noise synthetic data), or is used in an improper way.

(*)

Rotation is known to suppress oscillations, so this is not entirely surprising. The method of Viani+ 2019 is only used as an initial guess, as our final value of Δν is constructed using the fitted mode frequencies.

> 3) In the Bayesian analysis for assessing the seismic parameters is irrelevant, the authors have used such tight priors that the pseudo-blind Bayesian analysis solution is fixed to the hand-made solution. The (psuedo) fit of the quadrupole multiplets is based on peaks with a very tiny height.

(*)

This is consistent with earlier practice fitting dipole modes from red giants (e.g. DIAMONDS, FAMED, pbjam). Also, the blue shaded regions in Fig. 2 show the widths of the prior intervals, which can clearly be seen to span more than just the handmade solution.

> 4) The entire analysis is based on unrealistic uncertainties. The authors claim a precision of 10 nHz for Dnu (Table 1) and for individual frequencies (Table A1), which corresponds to the frequency resolution of the data (inverse of ~3 yr). Such a precision level is not reached for longer, higher-quality data observed by Kepler.

(*)

Yes it is! e.g. DIAMONDS, which also uses nested sampling, reports a precision of 5 nHz for cluster red giants.

> In general, the treatment of uncertainties is systematically unrealistic (e.g.: Z0 in Table 2, Gamma at line 344). The uncertainty on the inclination is also significantly underestimated (line 342)

(*)

Agreed that stastical uncertainties on these quantities are likely underestimated. For Z0 in particular, Izmir group also gets 10% statistical error, which makes them 2σ inconsistent with our adopted pipeline --- i.e. statistical error in Z0 does *not* dominate systematic error, unlike mass and radius.

However, not actually relevant to our discussion?

> 5) Lines 329-331 illustrates a problem of analysis, where an a priori is considered as a result.

Agreed that this ought not to have been done.

> 6) Basic information is often hidden in the paper and/or comes too late. The magnitude of the star should be provided very early, so that one understands that the poor signal in the Fourier spectrum is not related to a low luminosity. The actual frequency resolution should be given (either the value 0.007 µHz in the caption of Fig 2, or the description of the time series in Section 2, is incorrect).

(*)

Easy to fix; would appreciate other suggestions.

# Seismic Analysis

As far as I know, seismology alone is not able to unambiguously distinguish between RGB and AGB stars. Using epsilon_p for stating that the star is first-ascent giant is incorrect. A realistic relative uncertainty of Dnu implies a large uncertainty in epsilon_p and precludes any identification.

(*)

1) A simple sanity check strongly suggests that the identification/analysis of the quadrupole multiplets is an artifact. As shown in Fig 2, the quadrupole multiplets are artificially created by a confusion of l=2, m=-2 modes with dipole (l=1, m=+1) modes, and l=2, m=+2 with radial modes. From the figure, it is clear that the quality of the spectrum hampers the clear identification of quadrupole modes, and is by far not enough for deriving any information about them.

(*)

H0 test on quadrupole multiplets

2) The remark above indicates that, contrary to the claim of the authors (lines 307-310), the determination of the mode widths is a crucial step in the analysis.

(*)

3) The absence of a clear quadrupole signal raises strong doubt about the identification of the spectrum in terms of solar=like oscillations. To assess the detection of solar-like oscillations, the authors have to justify the presence of a clear dipole signal, and the absence of a clear quadrupole signal.

(*)

Backwards: since p-mode suppression is known to suppress dipole modes first, our identification of dipole modes requires us also to identify peak in the PS as quadrupole modes.

4) Stellar inclination: because of the high noise level, the amplitudes of the l=1, m=0 components are largely unknown. This implies a large uncertainty on the inclination, which needs to be reflected in the analysis.

(*)

The amplitudes of m = 0 dipole modes is just very low, so inclination is close to equator-on.

5) Differential rotation: Figure 8, as presented, show that little to nothing can be derived about differential rotation, or to rule out uniform rotation. The authors have to remove the l=2 data, to provide realistic uncertainties.

(*)

Mode ID for quadrupole modes is OK

6) At most, authors may have 2 pieces of information about rotation, as provided by the dipole modes and by the rotationally modulated spot signal. The paragraph ending Section 3 shows that the authors are aware of the weakness of their claims. In such case, there is no need to develop incorrect inferences over many pages.

(*)

Also: even without having constrained quadrupole modes, technique development for π-mode inversions against power spectrum still necessary.

# Discussion

> From the introduction in Section 4.1, one understands that there are multiple good reasons showing that Zvrk is not a merger remnant. Then weak reasons are provided over five pages. The Section discussion can and should be severely shortened. Interesting information is presented here, but it is only weakly connected to the paper.

The introduction to Section 4.1 explains why we believe Zvrk not to be a stellar-mass merger remnant in particular. However, its rapid rotation is not compatible with there being a companion, and has to be explained somehow. The rest of this section concerns finding such explanations. Given that the referee appears to have missed this, it is clear that the clarity of presentation here can be improved. We have therefore undertaken significant renovations of this section.

While this section is indeed weakly connected to the seismology per se (which we take that the referee assumes to be the focus of the paper given that it constitutes the majority of their feedback above), we note that we also report several observational results aside from seismology --- in particular, a surface rotation period, and unusual lithium and carbon/nitrogen abundances. The astrophysical interpretation presented here, to produce a consistent explanation for all of these features, is no less important than the various observational statements made preceding it. In accordance with the referee's wishes, we have shortened this section. However, we would appreciate more concrete feedback from the referee regarding specific parts of our discussion that they find weak.

# Organisation and Length

> The layout of the paper is unclear and misleading, with too many wordy sections.

The referee's comments are well taken: we have reorganised the paper in order to improve clarity and reduce verbosity. We would however appreciate more specific feedback from the referee as to how they find it, or which parts they find, misleading or deceptive.

> 1) Section 2.1 supposes that the star is a first ascent giant. Then, section 2.4 presenting the discussion about T Tauri stars is irrelevant and should be eliminated. If the star is an evolved RGB, it cannot be a T Tauri so the degree of relevance is too remote.

Our discussion of T Tauri hypothesis *directly addresses* possibility that p-mode detection was spurious! Our intention with this section was to consider an alternative hypothesis where this star were rather some other kind of object, ignoring observations of solar-like oscillations. We hope that our general reorganisation makes this clearer.

> 2) From the possible measurements of the global seismic parameters, it is clear that the star cannot be in the red clump stage. A sentence and a reference are enough to making this clear, so that many paragraphs in Section 2.1 can be removed.

OK, but we note that this discussion is there specifically to rule out identifying the double ridge as radial/quadrupole modes.

> 3) It is also clear that modes, if real, can be considered as pure pressure modes. No need then to discuss what would happen if mixed modes were present.

True observationally, but standard pulsation theory will always give you mixed modes in stars, so this discussion is necessary for interpretation, and motivating subsequent technique development.

> 4) All these issues lower the quality and readability of the paper. I consider that the paper could easily fit in less than 10 pages. The actual content is limited (rapid surface rotation, if properly assessed) that it should be limited to 5-6 pages. From all the caveats that are mentioned above, the terms "identification" and "characterization" are insecure, so that the title should be changed.

(*)

# Figures

> Figure 1: A definition of S/N should be given; furthermore, the relevance of this ratio should be made clear when the spectrum is smoothed (as it is certainly, according to the comparison with Fig 2); if not, then my first remark applies and the paper should be abandoned due to the absence of signal. Provide a broader frequency range, to more clearly show any power excess.

(*)

> Figure 2: Provide clearer figures, especially Fig 2b

Please elaborate

> Figure 3 is unneeded, since the determination of dnurot_2 is spurious.

(*)

> Figure 4: Provide a color code with more contrast; fig4c is unclear.

Please elaborate

> Figure 8 shows clearly that constraining differential rotation is impossible.

As we already state in the manuscript, we agree that inference of differential rotation from seismology alone is difficult given the data in hand (even at SNR of 10). However, if we use the solid-body value from seismology, it would suggest rotational shear when combined with surface rotational period.