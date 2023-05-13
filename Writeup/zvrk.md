---
header-includes:
    - \input{macros.tex}
    - \usepackage{mathptmx,txfonts,tikz,bm}
date: \today
documentclass: aastex631
classoption: astrosymb, twocolumn, tighten, linenumbers
mathspec: false
colorlinks: true
citecolor: xlinkcolor # don't override AASTeX default
urlcolor: xlinkcolor
bibliography: biblio.bib
biblio-style: aasjournal
---

\shorttitle{Gasing Pangkah I: Zvrk}
\title{Gasing Pangkah I: Asteroseismology and Preliminary Characterisation \\ of a Rapidly-Rotating Red Giant in the TESS SCVZ}
\input{preamble}
\begin{abstract}
We report the discovery of ``Zvrk'', a standalone rapidly-rotating red giant ($P_\text{rot} \sim 98\ \mathrm{d}$) observed with TESS in its Southern Continuous Viewing Zone. Zvrk's rotation rate is independently verified by the use of p-mode asteroseismology, strong perodicity in TESS and ASAS-SN photometry, and multiple measurements of spectroscopic rotational broadening. A two-component fit to APOGEE spectra indicates a spot coverage fraction consistent with the amplitude of the photometric rotational signal; modulations in this amplitude over time suggest the rapid evolution of this spot morphology, and therefore enhanced magnetic activity. We further develop and deploy new asteroseismic techniques to characterise radial differential rotation, and find strong evidence for rotational shear within Zvrk's convective envelope. This feature, in combination with such a high surface rotation rate, is categorically incompatible with even the most physically permissive models of angular-momentum transport in single-star evolution. Spectroscopic abundance estimates also indicate an unusually high surface lithium abundance, among other anomalies. Taken together, all of these suggest an ingestion scenario for the formation of this rotational configuration, various models of which we examine in detail. Such ingestion events represent an alternative mechanism by which the envelopes of post-main-sequence stars may spin up, as seen in the \textit{Kepler} sample, in conjunction with various existing hypotheses for outward angular momentum transport.
\keywords{Asteroseismology (73), Red giant stars (1372), Stellar oscillations (1617)}
\end{abstract}

# Introduction

Single-star evolution predicts the development of radial differential rotation during a star's first ascent up the red giant branch, in the sense of there being a faster-rotating radiative core than convective envelope. While seismic rotational measurements with evolved stars indicate that subgiant convective envelopes rotate faster than strict angular momentum conservation would suggest, envelope rotation rates nonetheless are predicted to further decrease dramatically as red giants expand, making them challenging to measure for more evolved red giants at higher luminosity. Indeed, many existing techniques for seismic rotation measurements in these evolved stars, which operationalise the phenomenology of the gravitoacoustic mixed modes exhibited by red giants, assume a priori that the rotation of the stellar envelope may be ignored. In the regime of (relatively) strong coupling between the interior g-mode cavity to the exterior p-mode cavity, this simplifying assumption has enabled the measurement of core rotation rates in red giants of intermediate luminosity in the Kepler sample en masse, at the expense of sacrificing information about the surface rotation of these stars. Since the surfaces of red giants are not expected to rotate in any significant fashion, such a loss of sensitivity is ordinarily considered acceptable.

However, contrary to this expectation, a small fraction of the Kepler red giant sample does exhibit nontrivial photometric variability, attributable to surface features rotating into and out of view. Such high rotation rates are not generally compatible with standard descriptions of angular momentum transport constructed to describe less evolved stars, and indicate that these rapidly-rotating red giants (RRRGs) may have passed through evolutionary scenarios quite unlike to the standard picture of single-star evolution. While some RRRGs have been found to be in binary systems (including with compact objects like black holes and neutron stars), the vast majority (85\%) do not possess detectable orbital companions. The above-described methodological shortcomings have then so far rendered these stars inaccessible to further asteroseismic characterisation in the regime of strongly mixed gravitoacoustic modes. The coupling between the interior g-mode and exterior p-mode cavities of first-ascent red giants decreases with evolution [e.g. @farnir_eggmimosa_2021;@jiang_evolution_2022;@ong_rotation_2], so a priori, p-mode or p-dominated mixed mode asteroseismology may permit a better understanding of the rotational properties of higher-luminosity RRRGs, which are in any case mostly convective by spatial extent.

Traditional numerical methods for mixed-mode asteroseismology, and in particular those which relate mode frequencies to interior structure, have been computationally very expensive to apply to these most highly luminous red giants. On the other hand, recent theoretical developments now permit the analysis of red-giant p-modes to be performed in a decoupled fashion from the interior g-mode cavity, thereby circumventing these computational difficulties. Their application to such highly-evolved red giants is also easiest to reconcile with our observational access there to only the most p-dominated mixed modes, in the weak-coupling regime. Moreover, single-star evolution predicts the slowest surface rotation rates for the most evolved red giants; thus, the most evolved standalone RRRGs potentially are also the most demonstrative (if not necessarily representative) of how RRRGs in general come to be.

We thus report the contemporaneous detection and characterisation of rotational and asteroseismic signals in TIC\ 350842552 (which we will refer to as "Zvrk" in the remainder of this work, for brevity), a highly evolved and most likely standalone RRRG. Asteroseismology (from the NASA TESS mission), direct photometry (both from TESS, and separately with the ground-based ASAS-SN network), and spectroscopy (various ground-based spectrographs) all independently confirm the presence of a 100-day photometric rotational modulation. Zvrk's spectroscopic chemical abundances exhibit peculiarities that are difficult to explain by way of single-star evolutionary modelling. In addition, we bring to bear new analytic developments in the interpretation of p-mode oscillations in this red giant, which we find to be amenable to very similar analysis to that prosecuted for the Sun. Rather than characterising a two-zone core-vs-envelope rotational contrast as is typically examined in less evolved red giants, we find instead strong evidence here for radial differential rotation situated entirely within Zvrk's convective envelope. Zvrk's high rotation rate, rotational shear, and anomalous chemical abundances are highly suggestive of an ingestion scenario for its formation, on which we place preliminary constraints. We conclude with some discussion about the astrophysical significance of this finding, and methodological implications for future work.

# Observational Characterisation

Zvrk was initially flagged for asteroseismic analysis as part of an ongoing large-scale search for unusual oscillation signatures corresponding to chemical peculiarities, in particular high lithium abundance. For this purpose, an overlapping sample of stars with both GALAH lithium abundance estimates, and TESS coverage in the Southern continuous viewing zone (CVZ), was constructed (Montet et al., in prep). Targets in this list were subjected to preliminary asteroseismic analysis, to constrain the global p-mode asteroseismic parameters $\Dnu$ and $\numax$. This was done using the 2D autocorrelation function procedure of @keaton?, applied to the SPOC light curves. For Zvrk in particular, this procedure yielded $\Dnu = (1.23 \pm 0.01)\ \mathrm{\mu Hz}; \numax = (7.5 \pm 0.3)\ \mathrm{\mu Hz}$.

Among this sample, Zvrk stood out both because (unusually for a red giant) 2-minute short cadence light curves were available, allowing our analysis to be performed with respect to the publicly available presearch data conditioning simple aperture photometry (PDCSAP) lightcurves instead; and also because its power spectrum exhibited unusual features, which defeated preliminary attempts at p-mode identification. To illustrate the latter, we show in \cref{fig:echelle-bare} the echelle power diagram obtained using the 2DACF value of $\Dnu =  1.23\ \mu$Hz. Of particular note is a double-ridged feature marked out with the white bounding box, whose significance we explain below.

![Échelle power diagram of Zvrk from PDCSAP lightcurves, using a nominal value of $\Dnu = 1.23\ \mu$Hz. The white dashed rectangle shows a double-ridge feature which, naively applying the asymptotic relation \cref{eq:asymptotic}, one would expect to identify as being radial and quadrupole modes. Under this identification, the remaining peaks must be identified as dipole mixed modes, as they do not appear to obey the same asymptotic relation. \label{fig:echelle-bare}](echelle_bare.png)

## Detailed Asteroseismology

To proceed with further asteroseismic analysis, identifications for the integer indices $n_p, \ell, m$ must be assigned to such mode frequencies as are derived from the observed power spectrum. Morphologically, these double ridges on its frequency echelle diagram, \cref{fig:echelle-bare}, are evocative of those of other p-mode oscillators, as previously observed en masse with *Kepler* and TESS, and suggest identification as being modes of low, even degree ($\ell = 0, 2$), which are known in p-mode oscillators to form double ridges of this kind as a result of the p-mode asymptotic eigenvalue equation
\begin{equation}
\nu_{n_p\ell} \sim \Dnu\left(n_p + {\ell \over 2} + \epsilon_p\right) + {\ell(\ell+1)}{\delta\nu_{02}\over 6} + \mathcal{O}(1/\nu),\label{eq:asymptotic}
\end{equation} in the absence of rotation. Here $\Delta\nu$ and $\delta\nu_{02}$ are the large and small frequency separations from the standard phenomenological description of p-mode asteroseismology, and $\epsilon_p$ is a slowly-varying phase function. The remainder of the oscillation power would, under this putative identification, be attributed to oscillations of dipole modes, which are known in other red giants to be disrupted by mode mixing with an interior g-mode cavity, producing a complex forest of gravitoacoustic mixed modes. Visually, this morphology strongly resembles that seen in core-helium-burning stars, where the coupling between the interior g-mode and exterior p-mode cavities is known to be strong. Indeed, this is the mode identification also arrived at by the use of existing "data-driven" automated methods, trained on the Kepler sample of oscillating red giants [e.g. PBJam: @pbjam].

However, such an identification would be in significant tension with other known properties of stochastically-excited red giant oscillations. In particular:

- Red clump stars are known to yield much higher values of both \Dnu\ and \numax\ than we have measured from Zvrk. Conversely, its \Dnu\ and \numax\ are too high, and thus it is not evolved enough, to be an asymptotic giant branch star. The measured values of these quantities instead strongly favour classification of Zvrk as a first-ascent red giant.
- This mode identification would imply a p-mode phase offset of $\epsilon_p \sim 0.2$. However, the value of $\epsilon_p$ in first-ascent red giants follows an extremely tight and robust relation with \Dnu, both in theoretical studies of stellar models [e.g. @white_calculating_2011;@ong_structural_2019], as well as in existing large-scale characterisation of the Kepler sample [e.g. @mosser_universal_2011;@yu_luminous_2020]. Thus, such a value of $\epsilon_p$ would be in very significant tension with the value of $0.7$ implied by the Kepler sample for red giants close to our nominal value of $\Dnu = 1.23\ \mathrm{\mu Hz}$.
- Both single-star stellar modelling [e.g. @deheuvels_seismic_2022] and observational measurements [e.g. @mosser_mixed_2014] of gravitoacoustic mixed modes in the Kepler sample also indicate that $\Delta\Pi_1$, the dipole-mode period spacing associated with the interior g-mode cavity, lies in a narrow band of allowed values with \Dnu\ for first-ascent red giants. At this value of \Dnu, $\Delta\Pi_1$ would also be far too small, and the mixed-mode coupling strength be too weak, to cause significant departures in the frequencies of the observed dipole modes from those of simple p-modes.

We show instead in \cref{fig:asteroseismology}a our proposed alternative mode identification, which takes into account all of the above constraints from the Kepler field. By necessity, our radial modes are anchored by $\epsilon_p$ measurements from the Kepler field.  To illustrate this, we show the average value of this quantity for all stars within $0.5\ \mu$Hz of $\Dnu$ to Zvrk in the catalogue of @yu_luminous_2020, and their standard deviation, using the white dashed line and red shaded interval, respectively.

In the presence of rotation, each mode at degree $\ell$ and radial order $n_p$ splits into a $(2\ell+1)$-tuplet of peaks in the power spectrum, with the distribution of observed power between peaks being determined by the inclination of the stellar rotational axis [@pesnell_observable_1985;@gizon_inclination_2003]. Rather than being modes of even degree, we instead identify the double ridge as being rotationally-split doublets of dipole ($\ell = 1$)  modes, viewed close to equator-on. In such an equator-on configuration, a mode of degree $\ell$ yields an $(\ell+1)$-tuplet; our identification of the quadrupole modes must then be constrained by having to explain the remaining peaks using rotationally-split triplets. While our putative identification of the quadrupole modes is unfortunate from the perspective of fitting the rotational splittings (as the quadrupole-mode triplets straddle both the dipole and radial modes), this identification of the quadrupole modes is required for general consistency with the values of $r_{02}$ typically returned from stellar models with comparable $\Dnu$ [e.g. @white_calculating_2011]. Moreover, in keeping with the star's evolved state, all of these modes are proposed to be essentially decoupled from the interior g-mode cavity.

Since g-mode mixing is not an observational concern, we may directly repurpose existing techniques for the derivation of p-mode frequencies from power spectra --- peakbagging --- to this star. For this purpose, we fit an ansatz model in the form
\begin{equation}
    f(\nu) = \sum_i \sum_{m=-{\ell_i}}^{\ell_i}{H_i r(m, \ell_i, i_\star) \over 1 + (\nu_i + m \delta\nu_i - \nu)^2/\Gamma_i^2} + \mathrm{BG}(\nu)\label{eq:model}
\end{equation}
directly to the power spectrum. The asteroseismic component of this power spectral model can be seen to be a sum of Lorentzians parameterised by the nonrotating frequencies $\nu_i$, the mode heights $H_i$, the inverse mode lifetimes $\Gamma_i$, the rotational multiplet splittings $\delta\nu_i$, as well as the inclination $i_\star$ of the rotational axis (through the visibility ratios $r$ of each multiplet component). In addition to this, we choose to describe our background model with a combination of a single Harvey profile and a white-noise term. 

We seek to infer posterior distributions of all of these parameters in a Bayesian sense, which necessitates the imposition of prior distributions. We place flat priors on each $\nu_i$ in windows $0.2\ \mu$Hz wide, centered on each of the manually identified values shown in \cref{fig:asteroseismology}a; a flat prior on $\mu = \cos i_\star$ for isotropy; flat priors on the logarithms of the mode lifetimes, heights, and all parameters of our noise model; and flat priors on the widths of the rotational splittings. For our main analysis in this section we moreover pool the rotational splittings of the dipole and quadrupole modes separately, assigning either $\delta\nu_{\text{rot}, \ell=1}$ or $\delta\nu_{\text{rot}, \ell=2}$ to each nonradial mode depending on its degree $\ell_i$. Additionally, we performed this exercise with and without pooling of the linewidths $\Gamma_i$. However, we found that, when linewidths were fitted on a per-mode basis, the posterior distributions of the linewidths for modes with low signal-to-noise ratios were very poorly constrained, and therefore permitted to take on unphysically large values at low amplitudes given our uninformative prior. In any case, the linewidths, which describe the damping rates of the modes, are not physically relevant for our following discussion, and their dependence on the mode frequency is weak at the low frequencies that we consider here. As such, we restrict our attention to results derived with a single linewidth $\Gamma$ being fitted against all modes, to simplify our analysis.

Using this parameterisation, these priors, and the standard $\chi^2$-2-degree-of-freedom likelihood function, we use the nested-sampling Markov-Chain Monte-Carlo (MCMC) algorithm, as implemented in the `dynesty` Python package, to infer the posterior distribution implied by the TESS data. In \cref{fig:asteroseismology}b, we show 100 draws from the posterior distribution overplotted on the power spectrum, with different contributions to \cref{eq:model} indicated with different colours of curves. The averages of these samples are shown with an RGB echelle power diagram in \cref{fig:asteroseismology}c, for easier graphical comparison with our mode identification by hand.

\begin{figure*}
\centering
\annotate{\includegraphics[width=.475\textwidth]{echelle_id.png}}{\node[white] at (.15, .9){\textbf{(a)}};}
\annotate{\includegraphics[width=.475\textwidth]{model.png}}{\node[white] at (.15, .9){\textbf{(c)}};}
\annotate{\includegraphics[width=.95\textwidth]{samples.png}}{\node at (.95, .9){\textbf{(b)}};}
\caption{Asteroseismic characterisation of Zvrk. \textbf{(a)}: \'Echelle power diagram showing putative mode identification, manually constructed from naive peakfinding. The red shaded region indicates the allowable region for radial modes as implied by $\epsilon_p$ measurements from the Kepler field. \textbf{(b)}: Samples from the joint posterior distribution for all parameters describing our model of the power spectrum (semitransparent curves, \cref{eq:model}; different colors show portions of the power spectrum attributed to modes of different degrees), overplotted against the raw power spectrum (filled circles joined with lines), and compared with a smoothed power spectrum (black line, corresponding to a Gaussian kernel of equal width to the resolution of the power spectrum, $xx\ \mu$Hz). \textbf{(c)}: RGB colour-channel \'echelle power diagram showing same samples from posterior distribution. The different colour channels show power attributed to modes of different degree: red for $\ell = 0$, green for $\ell = 1$, and blue for $\ell = 2$. Our original manual (not fitted) mode identification is overplotted for comparison.\label{fig:asteroseismology}}
\end{figure*}

\begin{figure*}[tb]
    \centering
    \includegraphics[width=\textwidth]{shared.png}
    \caption{Posterior distributions for pooled mode properties returned from peakbagging procedure.}
    \label{fig:shared}
\end{figure*}

The fitted model power spectra shown in \cref{fig:asteroseismology} can be seen to exhibit quite large rotational splitting, consistent with our initial hand-picked mode identification. We examine this more quantitatively in \cref{fig:shared}, where we show the joint posterior distributions for all pooled seismic quantities, as well as the parameters of our combined red- and white-noise background model. For rotation in particular, we find that the averaged $\ell = 1$ and $\ell = 2$ rotational splittings, at $\delta\nu_{\text{rot}, 1} = 0.091 \pm 0.007\ \mu\text{Hz}$ and $\delta\nu_{\text{rot}, 2} = 0.104\pm0.007\ \mu\text{Hz}$, are both much larger than the pooled linewidth of $\Gamma = 0.07\pm0.01\ \mu\text{Hz}$. Thus, our axial inclination constraint of $i_\star = {84 ^{+4}_{-5}} ^\circ$ is likely not susceptible to the systematic biases described in @kamiaka_reliability_2018: Zvrk is indeed viewed close to equator-on, in keeping with our mode identification by hand. Finally, we note that the dipole and quadrupole rotational splittings are slightly different from each other; we will return to this in our more detailed analysis using numerical stellar modelling.

\begin{figure*}
\centering
\annotate{\includegraphics[width=.95\textwidth]{tess-lc.pdf}}{\node at (.95, .9){\textbf{(a)}};}
\annotate{\includegraphics[width=.95\textwidth]{ls-all.pdf}}{\node at (.95, .9){\textbf{(b)}};}
\annotate{\includegraphics[width=.95\textwidth]{wavelet.pdf}}{
\node at (.95, 1){\textbf{(c)}};
\draw[blue, <->, inner sep=1](.0615, 1) -- (.525, 1) node[midway, label=below:{\tiny ASAS-SN V}]{};
\draw[red, <->, inner sep=1](.507, 1.05) -- (.613, 1.05) node[midway, label=below:{\tiny TESS Cycle 1}]{};
\draw[red, <->, inner sep=1](.71, 1.05) -- (.816, 1.05) node[midway, label=below:{\tiny TESS Cycle 3}]{};
\draw[ForestGreen, <->, inner sep=1](.42, 1.1) -- (.988, 1.1) node[midway, label=below:{\tiny ASAS-SN g}]{};
}
\caption{Photometric characterisation of Zvrk. \textbf{(a)} Stitched TESS aperture photometry (blue), shown over data from ASAS-SN in V-band (orange) and g-band (gray). The date of the APOGEE visit is shown with the vertical dashed line. \textbf{(b)} Lomb-Scargle power spectral densities from different TESS CVZ cycles and ASAS-SN bandpasses, normalised to give unity at their maximum values. A combined power spectral density incorporating all data sets (having applied offsets shown in panel a) is also shown with the red curve. From the most prominent peak of this combined power spectrum we infer a characteristic period of 98 days. \textbf{(c)} Color-separated frequency-time power diagram, with the intensity at each pixel showing the Lomb-Scargle-wavelet-transform power associated with a given time and oscillation period. Different colour channels correspond to different instruments, with TESS shown in red, ASAS-SN g in green, and ASAS-SN V in blue. A nominal oscillation period of 98 days, from our global Lomb-Scargle analysis, is marked out with the horizontal dashed line. \label{fig:photometry}}
\end{figure*}

## TESS and ASAS-SN Photometry

\label{sec:photometry}

In addition to asteroseismology, we should in principle also be able to derive estimates of an overall surface rotation rate from photometry, assuming that Zvrk exhibits spot-modulation rotational variability of a similar kind to that previously observed in rapidly-rotating Kepler red giants. However, the short-cadence PDC-SAP lightcurves from which our asteroseismic analysis above is derived have been aggressively detrended to eliminate long-term temporal variability. As a result of this detrending, rotational periods spanning multiple 27-day sectors may not be reliably determined from this data product. The rotational frequencies obtained from our seismic analysis correspond to rotational periods of up to 126 days (from our slower dipole-mode rotational frequency), thereby necessitating the use of alternative techniques.

Therefore, we instead construct our own custom aperture-photometry light curves, correcting for TESS instrumental systematics by applying the Pixel-Level Decorrelation technique [PLD: @deming_spitzer_2015], as implemented in the `lightkurve` Python package, to the TESS target pixel files. While this suffices to recover slow variability within each TESS sector, the long rotational period under consideration also requires us to stitch TESS sectors together in a principled fashion. Again, since our asteroseismic analysis suggests rotational periods which are far longer than the duration of a single sector, we opt to perform a naive linear extrapolation fit. Specifically, we fit a linear function to the last 5 days of each sector, and extrapolate the fitted function, sampling it on the first 5 days of the next sector. We then normalise the next sector such that the median flux within its first 5 days is scaled to match the median of the extrapolated function at the sampled times. This procedure is repeated for each consecutive sector. We show the results of this procedure with the blue points in \cref{fig:photometry}. Note that this procedure relies on there not being gaps between sectors, and so the missing sector 36 in Cycle 3 results in sectors 37-39 not being stitched to the preceding sector 35. Nonetheless, a clear periodic variation can be seen to emerge in Cycle 1, where all 13 sectors are successfully treated with this procedure, with a period of roughly 100 days. Likewise, similar periodicity can be seen in the first 9 consecutive sectors of Cycle 2, albeit at reduced amplitude.

To confirm that these variations are not caused by other latent, unaccounted TESS instrumental systematics, or artificially and inadvertently introduced by these choices of detrending or stitching procedures, we also examine Zvrk through independent ground-based photometry. We do so by way of the ASAS-SN network of ground-based telescopes (cite here). Since Zvrk is fairly bright ($M_V = 10.24$), this photometry had to be performed with custom apertures to account for detector saturation \remarkJO{need a good description of Ben's custom aperture stuff}. While less precise (and more noisy), any periodic instrumental systematic perturbations which affect the ASAS-SN detectors operate on very different timescales from those afflicting TESS, and thus over longer timescales we may rely on ASAS-SN to anchor our absolute photometric normalisation. We compare in \cref{fig:photometry}a the normalised ASAS-SN V-band and g-band photometry with our TESS lightcurves, choosing to display each consecutive run of TESS sectors in such a way as to match the median normalisation of the contemporaneous ASAS-SN g-band data points, which as a whole are themselves shown median-normalised. The ASAS-SN V-band data have also been scaled to yield equivalent median normalisation over the overlap period where contemporaneous observations exist with g-band. 

We can see immediately that the independent ASAS-SN data exhibit periodicity contemporaneous with that shown in TESS Cycle 1, with visually indistinguishable phase and period, indicating that the periodicity that we claim to have observed with TESS is genuine, rather than a result of TESS-specific systematics. Furthermore, this periodicity can be seen to continue in the ASAS-SN data for at least two more rotational periods after the end of TESS Cycle 1 (i.e. after a BTJD of about 1690), suggesting relatively slow evolution of any putative spot morphology causing the rotational signal. Since this procedure appears to work well on Cycle 1, and since the properties of the instrument and 2-minute-cadence data are not appreciably modified in succeeding sectors, we feel confident in assessing the apparent periodicity in Cycle 2 to be also genuine (albeit lying below ASAS-SN's noise floor for robust characterisation). In the small period of overlap between TESS Cycle 1 and ASAS-SN V-band coverage, the temporal variations in the ASAS-SN data can also be seen to be apparently in phase with the TESS modulation signal.

Many techniques exist for the derivation of the period of this kind of quasiperiodic variability. For our analysis, we will rely on the standard Lomb-Scargle technique, since it allows us to combine multiple data sets with different temporal sampling characteristics in a manner that nonetheless permits direct comparison with our asteroseismic results, obtained from Fourier power spectra. We show in \cref{fig:photometry}b the normalised Lomb-Scargle periodograms for different subsets of the data shown in panel (a), and additionally a periodogram generated from the combined data set, shown with the thick red line. We can see that all of the data sets exhibit a large peak at a period of between 90 to 100 days. This peak is located in our combined periodogram at a period of 98 days, while the scatter between various subsets of the data is of order 2 days, which we adopt as an estimate of systematic uncertainty.

The larger amplitude of this modulation as observed by ASAS-SN compared to TESS also reinforces an interpretation of this signal as originating from such a rotational spot modulation. A fixed temperature contrast between a starspot and the surrounding stellar photosphere will yield a greater reduction in intensity compared to the unspotted disk when observed in a bluer bandpass than in a redder one, and ASAS-SN's g-band observations ($\lambda \sim 477\ \mathrm{nm}$) are indeed bluer than TESS ($\lambda \sim 787\ \mathrm{nm}$). Supposing that these spots arise from near-surface magnetism like that seen in active main-sequence dwarfs, we should also expect their shapes and positions to change over many rotational periods. For these dwarfs, the spot evolution timescale is of order ~5 rotational periods; over longer observing windows, changes to both the morphology of any darkened surface features, as well as to their position on the visible disk, will affect the properties of this photometric modulation. The shapes and sizes of such spots will affect its amplitude, and their positions will, in the presence of latitudinal differential rotation, modify its apparent period.

The prominence of this photometric rotational signal, as well as to some extent its period, indeed appears to vary over time. We investigate this qualitatively in \cref{fig:photometry}c, which shows a Lomb-Scargle frequency-time power diagram generated using the ASAS-SN Sky Patrol utilities [@hart_asassn_2023], over which we mark out our nominal surface rotational period of 98 days with a horizontal line. The peaks of 100 days in our periodograms in panel (b) correspond to a horizontal band of power on this frequency-time power diagram, sampled at different times at different bandpasses, shown with different colours. Heuristically, the Lomb-Scargle periodograms (up to convolution against a kernel determined by the temporal resolution of this diagram) may be recovered by simply integrating over time. However, such an integral can be seen to be largely dominated by an episode of large photometric amplitude at around BTJD 1500, which, fortuitously, was covered contemporaneously by ASAS-SN g-band, TESS, and (at least at its beginning) ASAS-SN V-band. Prior to this episode, power can be seen to be exhibited at longer rotational periods (in ASAS-SN V-band, shown in blue), while after it, power can be seen to be exhibited at shorter rotational periods (from TESS, shown in red).

## Spectroscopy

Our initial selection of Zvrk was on the basis of a measurement of A(Li) = 3.36 dex --- almost meteoritic! \remarkJO{MARC: PLEASE PROVIDE MORE DESCRIPTION OF LITHIUM MEASUREMENTS HERE}. This lithium enrichment may be the result of highly rotational mixing, of the kind thought to be potentially induced by tidal interactions with a close binary companion\remarkJO{MELINDA: FEEL FREE TO ELABORATE AS NECESSARY}. Unfortunately, only one APOGEE visit to Zvrk was performed (MJD 58886 = BTJD 1886), so we are unable to constrain the presence or absence of an unresolved companion from the time evolution of its radial velocities. However, Zvrk has a Gaia DR3 Renormalised Unit Weight Error (RUWE) of 1.056, close to unity, which indicates that such an unresolved companion, or at least one sufficiently massive as to result in photocentre jitter, is unlikely to be present. Zvrk is also not present in the RRRG catalogue of @patton_spectroscopic_2023, which identifies rapid rotators based primarily on their fitted temperatures being anomalously cool relative to the APOGEE DR16 catalogue. This indicates that, even if Zvrk's present rotational configuration should have been produced as a result of some kind of impulsive event causing it to puff up, such a structural perturbation should have thermally relaxed by now. Thus, at least Zvrk's structure, although probably not its evolutionary history, should be amenable to description by numerical stellar models in hydrostatic equilibrium.

In addition to the asteroseismic measurements that we have presented above, spectroscopic measurements of e.g. the stellar metallicity and effective temperature are also necessary as inputs into stellar modelling for more precise constraints on stellar properties. Such measurements for Zvrk already exist in the APOGEE DR17 catalogue (under 2MASS ID 05592585-5911542), with an effective temperature of $\teff = 4320 \pm 80\ \mathrm{K}$, and a metallicity of $\mathrm{[M/H]} = -0.21 \pm 0.08$ dex. Using this value of the effective temperature, in conjunction with the solar-calibrated scaling relation $\numax \sim g / \teff$, gives us $\log g_\text{seis} = 1.76 \pm 0.03$, which is significantly lower than the APOGEE DR17 value of $2.11$.  In principle, as is often done in other works combining asteroseismology with spectroscopy, we may employ our asteroseismic measurement of $\log g$ as a direct constraint on the surface gravity, and iterate between fitting for the remaining quantities with it held fixed, and refining asteroseismic $\log g$, until the fitted values converge. However, this iterative procedure is typically performed with far more precise constraints on $\numax$ than we have available here. Conversely, using our asteroseismic surface gravity here does not significantly modify the fitted temperature and metallicity. We therefore adopt the nominal APOGEE values for $\teff$ and metallicity, and their uncertainties, in our subsequent analysis. With these values of $\log g$, $\teff$, and the metallicity, we may then estimate a spectroscopic rotational broadening $V \sin i$ through the differential-broadening technique also used in @tayar_spinning_2022; we obtain that $V \sin i = 10.1 \pm 0.7 \mathrm{km s^{-1}}$.

Given the global asteroseismic properties $\Dnu$ and $\numax$, we may apply the so-called "direct method", inverting their usual scaling relations [and including a structural correction factor $f_{\Dnu} = 0.97$ via the prescription of @sharma_stellar_2016] to obtain that $M = (1.181 \pm 0.15) M_\odot$, $R = (23.7 \pm 1.1) R_\odot$. This seismic mass estimate permits us to compare the chemical composition of Zvrk's convective envelope, and in particular its carbon-to-nitrogen enrichment $[\mathrm{C/N}]= -0.08 \pm 0.017 \text{(stat)} \pm 0.057 \text{(sys)}$ from the APOGEE DR17 catalogue, with the mass-enrichment sequence of the APOKASC3 subsample of solar-like oscillators (Pinnsonneault et al., in prep.) in the same catalogue. We show this comparison in \cref{fig:cn}, restricting this comparison to only first-ascent red giants, and to stars with with metallicities within 0.1 dex of Zvrk's. Since the sample is instrumentally and methodologically homogenous, it suffices for us to compare Zvrk against this reference population on only the basis of its statistical uncertainty. Moreover, for an apples-to-apples comparison, we use the seismic stellar mass estimated from the direct method, rather than from more detailed modelling against individual mode frequencies (which we describe below), despite the latter being more precise.

![APOGEE DR17 $[\mathrm{C/N}]$ of Zvrk relative to APOKASC3 observational sample, with a metallicity cut of 0.1 dex around the nominal value for Zvrk, and restricting attention to only first-ascent red giant stars as identified by Hon et al \remarkJO{(CITATION NEEDED)}. A well-defined mass-enrichment sequence can be seen. Zvrk can be seen to lie above this enrichment sequence, given its seismic mass and relative abundances. Note that while the scaling-relation mass is used here for an apples-to-apples comparison, the seismic mass from individual mode frequencies is constrained far more precisely than depicted here.\remarkJO{Perhaps I should also overplot the detailed-modelling seismic mass?}\label{fig:cn}](CN.pdf)

In the background of \cref{fig:cn}, a clear mass-enrichment sequence can be seen, as marked out by the dark region of histogram bins. This sequence arises as a result of the strong sensitivity of the CNO cycle to the reaction temperature, combined with the tight dependence of the stellar central temperature, and of convective core sizes in stars possessing them, on the stellar mass. As a result of the former, a larger proportion of the main-sequence nuclear processing of hydrogen is performed through the CNO cycle in more massive stars, while as a result of the latter, a larger reservoir of hydrogen is available to predominantly CNO-cycle burning in more massive stars. At core hydrogen exhaustion, more massive stars will therefore have larger residual quantities of unreacted $^{14}\mathrm{N}$ (from the rate-limiting proton-capture reaction in the CNO cycle) present in their cores, which is then distributed into the envelope during first dredge-up. Correspondingly, we would expect --- within a limited range of metallicities, as we consider here --- that the carbon-to-nitrogen enrichment [C/N] should be tightly, and negatively, correlated with the stellar mass.

Zvrk can be seen on \cref{fig:cn} to lie above this mass-enrichment sequence. While the uncertainties on the scaling-relation seismic mass are large enough to appear to permit consistency with it, we note that our seismic mass constrained by individual mode frequencies, which we describe in \autoref{sec:opt}, are significantly more precisly constrained, and strongly disfavour a stellar mass low enough to permit Zvrk to lie on this sequence. Given the above discussion, this suggests that the material in Zvrk's convective envelope is less nuclear-processed than would be typical for an ordinary red giant of comparable mass and radius.

\remarkJO{$\left[\mathrm{^{12}C/^{13}C}\right]$ = $9.6 \pm 0.88$ from \cite{hayes_bacchus_2022} --- JAMIE'S VALUE NOT IN TABLE}

<!--\begin{figure}
\centering
\annotate{\includegraphics[width=.475\textwidth]{CN.png}}{\node[fill=white,fill opacity=.5, text opacity=1] at (.25, .9){\textbf{(a)}}; \node[red] at (.5, .5) {\Huge PLACEHOLDER};}
\annotate{\includegraphics[width=.475\textwidth]{C13.png}}{\node[fill=white,fill opacity=.5, text opacity=1] at (.25, .9){\textbf{(b)}};\node[red] at (.5, .5) {\Huge PLACEHOLDER};}
\caption{Spectroscopic abundance characterisation of Zvrk. \textbf{(a)} $[\mathrm{C/N}]$ relative to observational sample, with a metallicity cut of 0.1 dex around the nominal value for Zvrk, and restricting attention to only first-ascent red giant stars as identified by Hon et al (CITATION NEEDED). A well-defined observational sequence of red giants can be seen. Zvrk can be seen to lie above this red-giant sequence, given its seismic mass and relative abundances. \textbf{(b)} $[\mathrm{^{12}C/^{13}C}]$ for the sample of \cite{hayes_bacchus_2022}.}
\end{figure}-->

# Analysis and Interpretation

## Stellar Properties and Structure

\label{sec:opt}

The uncertainties on the quantities derived from the direct method are large, both owing to the fact that $\numax$ is difficult to measure for evolved stars with comparatively few visible modes, as well as owing to the fact that the scaling relations themselves may not be reliable at advanced stages of evolution. However, these are still helpful in bounding the parameter space for further analysis.

Constraint | Value | Reference
----:+:-----:+:---:
$\teff$ / K | $4320 \pm 80$ | APOGEE
$\mathrm{[M/H]}$ | $-0.21 \pm 0.08$ | APOGEE
$L / L_\odot$ | $174 \pm 10$ | Gaia DR2
$\Dnu/\mu\mathrm{Hz}$ | $1.21 \pm 0.01$ | This work
$\numax/\mu\mathrm{Hz}$ | $7.5 \pm 0.3$ | This work
Table: Global properties adopted in stellar modelling.\label{tab:t1}

Quantity | Inferred Value | Remarks
---:+:---:+:---:
$M/M_\odot$ | $1.14^{+0.05}_{-0.03}$ | —
$R/R_\odot$ | $23.5^{+0.4}_{-0.2}$ | —
$\amlt$ | $1.89^{+0.12}_{-0.08}$ | $\alpha_\odot = 1.824$
$Y_0$ | $0.273^{+0.03}_{-0.02}$ | $Y_\odot = 0.268$
$Z_0$ | $0.010 \pm 0.001$ | $Z_\odot = 0.018$
Age/Gyr | $5.4^{+1.0}_{-0.8}$ | —
$\mathrm{Ro}_\mathrm{CZ}$ | $0.28$ (point estimate) | $\mathrm{Ro}_\odot = 1.38$
Table: Global properties returned from stellar modelling. \label{tab:t2}

From optimisation, best-fitting mass of 1.18 $M_\odot$. Slightly overluminous.

## Radial Differential Rotation

$\pi$-mode isolation prescription of @ong_semianalytic_2020. 

```{=latex}
\begin{figure}[htbp]
    \centering
    \includegraphics[width=.435\textwidth]{r_n6_l1.pdf}
    \includegraphics[width=.435\textwidth]{t_n6_l1.pdf}
    \includegraphics[width=.435\textwidth]{t_n6_l2.pdf}
    \caption{Caption here}
    \label{fig:kernels}
\end{figure}
```

\cref{fig:shear}

\begin{figure*}[htbp]
    \centering
    \annotate{\includegraphics[width=3in]{shear.pdf}}{\node at (.9,.2){\textbf{(a)}};}
    \annotate{\includegraphics[width=3in]{raw_kernel.pdf}}{\node at (.9,.15){\textbf{(b)}};}
    \caption{Raw asteroseismic constraints on sensitivity to rotational shear. \textbf{(a)} Posterior distributions for differences between the pooled quadrupole and dipole rotational splitting. The upper panel shows the posterior distribution of the differences in estimated rotation rates, which are in turn the rotational splittings divided by effective sensitivity constants $\left<\beta_\ell\right> \sim 1$. The lower panel shows the posterior distribution of the differences in acoustic radii of the centres of sensitivity associated with the underlying averaged rotational kernels of each degree, considered separately. The two quantities are statistically independent. \textbf{(b)} Differential sensitivity kernels associated with distributions in (a). Associated with each pooled rotational splitting $\delta\nu_\ell$ is an effective rotational kernel $K_\ell$, which is a weighted average of the rotational kernels associated with each mode of that degree. Faint curves show samples from the posterior distribution, while the solid coloured curves show the posterior median effective kernel for each degree. The inner turning points at $\numax$ of the best fitting stellar model from our optimisation procedure are shown with the vertical dashed lines, while the centres of sensitivity for the median kernels are indicated with the vertical dotted lines. The ordering of their positions can be seen to reversed compared to the theoretical inner turning points.}
    \label{fig:shear}
\end{figure*}

For each degree, every draw from the posterior distribution relates the pooled rotational splittings to the underlying per-mode rotational splittings as a weighted average. Such a weighted average is in turn related to the radial dependence of the rotational frequency $\Omega$ through an effective averaging kernel, which is constructed out of the per-mode averaging kernels through a weighted average with the same coefficients. Thus, every draw from the posterior distribution yields a draw from a distribution over averaging kernels for dipole and quadrupole modes considered separately.

\begin{figure*}[htbp]
    \centering
    \annotate{\includegraphics[width=.85\textwidth]{RLS_corner.pdf}}{
    \node[below,left] at (.85, .93){\includegraphics[width=.36\textwidth]{RLS.pdf}};
    \node[below,left] at (1.06, .595){\includegraphics[width=.36\textwidth]{RLS_surf.pdf}};
    \node at (.1, .97){\textbf{(a)}};
    \node at (.8, 1.03){\textbf{(b)}};
    \node at (1, .695){\textbf{(c)}};}
    \caption{Regularised constraints on rotational shear.}
    \label{fig:rls}
\end{figure*}

## Magnetic activity

Magnetic activity is known to be generated by rotational shear, which can be characterised by the Rossby number $\mathrm{Ro} = P_\text{rot} / \tau_\mathrm{CZ}$. Customarily, the convective turnover timescale $\tau_\mathrm{CZ}$ is evaluated in main-sequence stellar models locally --- e.g. at one pressure scale height $H_p$ above the base of the convection zone, near the tachocline. However, since red giants are almost entirely convective by spatial extent, such a local definition is not necessarily representative of the star as a whole. Thus we evaluate the Rossby number instead with respect to a global convective turnover timescale: $$\tau_\mathrm{CZ} = \int_{r_1}^{r_2} {\mathrm d r \over V_\text{conv}},$$ where $r_1 = r_\text{base} + H_p$ and $r_2 = r_\text{top} - H_p$, and $V_\text{conv}$ is the MLT convective velocity. Evaluating this quantity with respect to models in our optimisation trajectory, in conjunction with our nominal rotational period, gives us $\mathrm{Ro}_\star \sim 0.26$. By contrast, computing this quantity from a solar-calibrated MESA model with the same physics yields $\mathrm{Ro}_\odot = 1.38$.

Roughly speaking, the solar Rossby number is known to fluctuate between 1/2 and 2 over the course of the solar activity cycle. Thus, this value indicates that Zvrk is possibly magnetically active, which is consistent with our interpretation of the photometric variability presented in \autoref{sec:photometry} as being that of surface magnetic features, potentially spots, rotating into and out of view on the visible disk. To further test this interpretation, we subject the APOGEE spectra to the LEOPARD two-component fitting procedure described in @cao_starspots_2022, which returns a disk area fraction of $f_\text{spot} = 0.02$, and a temperature contrast of $x_\text{spot} = T_\text{spot}/T_\text{surf} = 0.8$. Assuming that secondary component corresponds to a single large, morphologically concentrated, surface feature, the spot coverage fraction and intensity contrast are consistent with the observed $2\%$ photometric variability amplitude. However, the presence of only a single APOGEE visit prevents us from assessing properties of the evolution of these putative surface features.

\cref{fig:flare}

![Candidate flare identified in short-cadence PDCSAP data. \label{fig:flare}](marginal.pdf)

GALEX photon events reported from this pointing, but only NUV

# Discussion

We identify three different classes of explanations for how Zvrk's rotational and chemical configuration came to be, which would roughly yield both a high rotation rate, and the observed enhanced lithium abundance:

(I)  Mass transfer from a hitherto undetected stellar companion would yield enhanced lithium at Zvrk's surface. In this scenario, its high rate of rotation could be attributed to either direct deposition of angular momentum from the accreted material, or tidal spin-up as a result of binary interactions.
(II)  Alternatively, both the enhanced lithium abundance and high rate of rotation could be attributed to Zvrk having engulfed at least one formerly orbiting companion [e.g. @stephan_eating_2020;@oconnor_giant_2023], with both matter and angular momentum being directly deposited into its envelope, and being redistributed over the course of several mixing timescales.
(III)  Finally, for the sake of argument, Zvrk could represent significant departures from the existing theory of single-star evolution. To match the chemical enhancements and fast rotation rate, its evolution would necessitate an efficiency of chemical mixing and angular momentum transport far in excess of that currently assumed of red giants.

No WISE infrared excess.

## Mass constraints

Given present angular momentum content, lower limit of 11 MJup for eccentric orbit and 16 MJup for circular orbit. Absolute upper limit of 80 MJup if accreted at MS radius. Even this highly massive engulfment mass would yield an enhancement in lithium of at most 2.24 dex, which is still more than an order of magnitude less than that observed here.

## Rotational Age Constraints

While rapid for a red giant, Zvrk's surface rotational period remains far slower than its breakup rotational period, which we estimate to be around $T_\text{crit} = 12.5\ \mathrm{d}$ for a star of its mass and radius. However, both its rotational period and this critical rotation rate would have changed significantly over the course of its evolution up the red giant branch. In particular, if we are to interpret the apparent photometric variability as spot modulations, the same magnetic field that drives this spot activity ought also to be responsible for magnetic rotational braking. In the absence of further deviations from single-star evolution, we therefore would expect Zvrk to have been rotating more quickly earlier on the red giant branch. While the breakup rotation rate, which scales as $R^{-3/2}$, should also have been faster at earlier times,

We show in \cref{fig:braking}a different predictions and retrodictions for Zvrk's rotational evolution, based on different possible scenarios for such magnetic braking along the evolutionary track producing our best-fitting model in the optimisation exercise of \autoref{sec:opt}. Solutions are tuned to yield a rotational period of 100 d at the best-fitting stellar model, indicated with the vertical dotted line; we plot the surface rotation rate $\Omega$ as a fraction of the breakup rotation rate $\Omega_\text{crit}$. We show with the dashed curve an unphysical, counterfactual scenario with no magnetic braking. Under this scenario, Zvrk must have been rotating at more than half the breakup rotation rate on the main sequence. Such a high rotation rate would, in an actual star, almost surely induce magnetic braking. In turn, the action of any magnetic braking whatsoever would require Zvrk to initially have been rotating still faster, in order to match its present observed rotational period on the red giant branch. 

In \cref{fig:braking}a, we also consider retrodictions generated using the magnetic braking prescription of @matt_magnetic_2012, where the rate of angular momentum loss, $\dot{J}$, scales with either $\Omega$ or $\Omega^3$, depending on whether or not the inverse Rossby number exceeds a saturation threshhold (with saturation achieved when $\Omega \tau_\text{cz} > \omega_\text{crit} \tau_{\text{cz},\odot}$). For this exercise, we adopt a standard value of $\omega_\text{crit} = \Omega_\odot/10$, and a normalisation constant for the angular momentum loss rate of $f_k = 9.37$, calibrated to produce the solar equatorial rotation rate of 25.4 d for our solar-calibrated MESA model with a disk rotation period of 8 d and a disk-locking timescale of 1 Myr. This setup is generally in keeping with previous uses of this angular-momentum-loss prescription.

\begin{figure}[htbp]
    \annotate{\includegraphics[width=.45\textwidth]{braking.pdf}}{\node at (.22, .22){\textbf{(a)}};}
    \annotate{\includegraphics[width=.4\textwidth]{timescales.pdf}}{\node at (.9, .9){\textbf{(b)}};}
    \caption{Rotational evolution of Zvrk under different extremal scenarios. \textbf{(a)} Evolution of the surface rotational frequency $\Omega_\text{surf}$ from the main sequence up the RGB, using the frequency of maximum power $\nu_\text{max}$ as an evolutionary proxy, and shown as a fraction of the breakup rotational frequency $\Omega_\text{crit}$. The solid line shows unsaturated magnetic braking according to the prescription of \citet{matt_magnetic_2012}, using solar-calibrated braking parameters $f_K$ and $\omega_\text{crit}$, where line segments are coloured by the lookback time $\Delta t$ from the present (i.e. negative values represent integration into the future). The dotted curve shows rotational evolution under braking parameters chosen to saturate magnetic braking at the present day, shown with the vertical dotted line, while the dashed curve shows the most permissive scenario of no magnetic braking (which is not significantly distinguishable from saturated braking). \textbf{(b)} Comparison of several characteristic timescales. We show in particular the spindown timescale $|P_\text{rot} / \dot{P}_\text{rot}|$ associated with structural expansion in the absence of magnetic braking; the braking timescale $|J_\text{tot} / \dot{J}_\text{tot}|$ associated with angular momentum loss by the prescription of \cite{matt_magnetic_2012}, with the dotted curve showing the same saturated trajectory as in panel (a); and the Kelvin-Helmholtz timescale $t_\text{KH} = GM^2/2RL$, as a proxy for the thermal relaxation timescale.}
    \label{fig:braking}
\end{figure}

# Conclusion

\begin{acknowledgements}

We thank S. Basu, T. Bedding, and S. Hekker for constructive feedback on preliminary versions of this work, and C. Hayes and J. Hinkle for productive discussions.

% magic incantation

This paper includes data collected by the TESS mission. Funding for the TESS mission is provided by the NASA's Science Mission Directorate.

This work has made use of data from the European Space Agency (ESA) mission {\it Gaia} (\url{https://www.cosmos.esa.int/gaia}), processed by the {\it Gaia} Data Processing and Analysis Consortium (DPAC, \url{https://www.cosmos.esa.int/web/gaia/dpac/consortium}). Funding for the DPAC has been provided by national institutions, in particular the institutions participating in the {\it Gaia} Multilateral Agreement.

JMJO, MTYH, and MS-F acknowledge support from NASA through the NASA Hubble Fellowship grants HST-HF2-51517.001, HST-HF2-51459.001, and HST-HF2-51493.001-A, respectively, awarded by STScI. STScI is operated by the Association of Universities for Research in Astronomy, Incorporated, under NASA contract NAS5-26555.
APS acknowledges partial support by the Thomas Jefferson Chair Endowment for Discovery and Space Exploration, and partial support through the Ohio Eminent Scholar Endowment.

\software{NumPy \citep{numpy}, SciPy stack \citep{scipy}, AstroPy \citep{astropy:2013,astropy:2018}, \texttt{dynesty} \citep{dynesty}, Pandas \citep{pandas}, \mesa\ \citep{mesa_paper_1,mesa_paper_2,mesa_paper_3,mesa_paper_4,mesa_paper_5}, \gyre\ \citep{townsend_gyre_2013}.}

\facilities{TESS, ASAS-SN, APOGEE, GALAH, WISE, GALEX}
\end{acknowledgements}

\appendix

# Peakbagged Mode Frequencies

For the purposes of stellar modelling, we assume that all rotational splittings are symmetric (i.e. no perturbations to the mode frequencies arising from latitudinal differential rotation or magnetic fields), and use the notional $m = 0$ mode frequencies associated with our parametric model, \cref{eq:model}, as constraints on the $m = 0$ mode frequencies returned from MESA stellar models. We provide the values of these mode frequencies in \cref{tab:peakbag}.

$\ell$ | $\nu/\mu\text{Hz}$ | $e_\nu/\mu\text{Hz}$ | $n_p$
---:+:---:+:---:+:---
0 | 5.81 | 0.01 | 4
0 | 6.91 | 0.02 | 5
0 | 8.20 | 0.02 | 6
0 | 9.41 | 0.05 | 7
0 | 10.66 | 0.03 | 8
1 | 5.34 | 0.03 | 3
1 | 6.40 | 0.01 | 4
1 | 7.64 | 0.01 | 5
1 | 8.78 | 0.02 | 6
1 | 10.09 | 0.03 | 7
2 | 5.63 | 0.05 | 3
2 | 6.70 | 0.02 | 4
2 | 7.90 | 0.03 | 5
2 | 9.17 | 0.02 | 6
2 | 10.43 | 0.04 | 7
Table: Notional $m=0$ p-mode frequencies from peakbagging procedure with multiplet model\label{tab:peakbag}

<!--\bibliography{biblio.bib}-->
