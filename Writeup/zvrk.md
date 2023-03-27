---
header-includes:
    - \input{macros.tex}
    - \usepackage{mathptmx,txfonts,tikz,bm}
date: \today
documentclass: aastex631
classoption: astrosymb, twocolumn, tighten, twocolappendix, linenumbers
mathspec: false
colorlinks: true
citecolor: xlinkcolor # don't override AASTeX default
urlcolor: xlinkcolor
bibliography: biblio.bib
biblio-style: aasjournal
---

\shorttitle{Gasing Pangkah}
\title{A Fast-Rotating Red Giant Observed by TESS}
\input{preamble}
\begin{abstract}
We report the discovery of a rapidly-rotating red giant ($P_\text{rot} \sim 98\ \mathrm{d}$) observed with TESS in its Southern Continuous Viewing Zone. The rotation rate of this red giant is independently verified by the use of p-mode asteroseismology, strong perodicity in TESS and ASAS-SN photometry, and multiple measurements of spectroscopic rotational broadening. A two-component fit to APOGEE spectra indicates a spot coverage fraction consistent with the amplitude of the photometric rotational signal; modulations in this amplitude over time suggest the rapid evolution of this spot morphology, and therefore enhanced magnetic activity. We further develop and deploy new asteroseismic techniques to characterise radial differential rotation in its convective envelope. In particular we find that the interior portions of the convective envelope rotate more slowly than the near-surface layers. This feature, in combination with such a high surface rotation rate, is categorically incompatible with even the most physically permissive models of angular-momentum transport in single-star evolution. Moreover, spectroscopic abundance estimates also indicate an unusually high surface lithium abundance. Taken together, all of these suggest an ingestion scenario for the formation of this rotational configuration, various models of which we examine in detail. Such ingestion events represent an alternative mechanism by which the envelopes of post-main-sequence stars may spin up, as seen in the \textit{Kepler} sample, in conjunction with various existing hypotheses for outward angular momentum transport.
\keywords{Asteroseismology (73), Red giant stars (1372), Stellar oscillations (1617)}
\end{abstract}

# Introduction

Single-star evolution predicts the development of radial differential rotation during a star's first ascent up the red giant branch, in the sense of there being a faster-rotating radiative core than convective envelope. While seismic rotational measurements with evolved stars indicate that subgiant convective envelopes rotate faster than strict angular momentum conservation would suggest, envelope rotation rates nonetheless are predicted to decrease as red giants expand, making them challenging to measure for more evolved red giants at higher luminosity. Indeed, many existing techniques for seismic rotation measurements in these evolved stars, which operationalise the phenomenology of the gravitoacoustic mixed modes exhibited by red giants, assume a priori that the rotation of the stellar envelope may be ignored. In the regime of (relatively) strong coupling between the interior g-mode cavity to the exterior p-mode cavity, this simplifying assumption has enabled the measurement of core rotation rates in red giants of intermediate luminosity in the Kepler sample en masse, at the expense of sacrificing information about the surface rotation of these stars.

However, seismology aside, a small fraction of the Kepler red giant sample exhibit nontrivial photometric variability, attributable to surface features rotating into and out of view. Such high rotation rates are not generally compatible with standard descriptions of angular momentum transport constructed to describe less evolved stars, and indicate that these rapidly-rotating red giants (RRRGs) may have passed through evolutionary scenarios quite unlike to the standard picture of single-star evolution. While some RRRGs have been found to be in binary systems (including with compact objects like black holes and neutron stars), the vast majority (85\%) do not possess detectable orbital companions. The above-described methodological shortcomings have so far rendered these stars inaccessible to further asteroseismic characterisation in the regime of strongly mixed gravitoacoustic modes. However, the coupling between the interior g-mode and exterior p-mode cavities of first-ascent red giants decreases with evolution [e.g. @farnir_eggmimosa_2021;@jiang_evolution_2022;@ong_rotation_2], so a priori, p-mode or p-dominated mixed mode asteroseismology may permit a better understanding of the rotational properties of higher-luminosity RRRGs, which are in any case mostly convective by spatial extent. On the other hand, traditional numerical methods for mixed-mode asteroseismology, and in particular those which relate mode frequencies to interior structure, are computationally very expensive to apply to highly luminous red giants.

We report the concurrent detection of rotational signals in TIC\ 350842552 (which we will refer to as "Zvrk" in the remainder of this work, for brevity), a RRRG, from asteroseismology (from the NASA TESS mission), direct photometry (both from TESS, and independently with the ground-based ASAS-SN network), and with spectroscopy (various ground-based spectrographs). In particular, we bring to bear new analytic developments in the interpretation of p-mode oscillations in this red giant, which we find to be amenable to very similar analysis to that prosecuted for the Sun.

# Observational Characterisation

Zvrk was initially flagged for asteroseismic analysis as part of an ongoing search for unusual oscillation signatures corresponding to chemical peculiarities, in particular high lithium abundance. For this purpose, an overlapping sample of stars with both GALAH abundances, and TESS coverage in the Southern continuous viewing zone (CVZ), was constructed, and targets in this list were subjected to preliminary asteroseismic analysis, to constrain the global p-mode asteroseismic parameters $\Dnu$, $\numax$, and $\epsilon_p$. This was done using the 2D autocorrelation function procedure of @keaton?, applied to the publicly available presearch data conditioning simple aperture photometry (PDCSAP) lightcurves.

Morphologically, the double ridges on this frequency echelle diagram are evocative of those of other p-mode oscillators, as previously observed en masse with *Kepler* and TESS, and suggest identification as being modes of low, even degree, $\ell = 0, 2$, which are known to form double ridges of this kind. The remainder of the oscillation power would, under this putative identification, be attributed to oscillations of dipole modes, which are known in other red giants to be disrupted by mode mixing with an interior g-mode cavity to produce a complex forest of gravitoacoustic mixed modes.

However, such an identification would be in significant tension with known properties of these red giants. In particular:

- This would imply a p-mode phase offset of $\epsilon_p \sim 0.2$. This is in very significant tension with the value of $0.7$ implied by the Kepler sample [e.g. @mosser_universal_2011;@yu_luminous_2020].
- At $\Dnu = 1.22\ \mathrm{\mu Hz}$, the period spacing associated with the interior g-mode cavity would be far too small to cause significant departures in the observed dipole modes from the simple p-mode asymptotic relation. 

## Detailed Asteroseismology

We show in \cref{fig:asteroseismology}

\begin{figure*}
\centering
\annotate{\includegraphics[width=.475\textwidth]{echelle_id_2.pdf}}{\node[white] at (.15, .9){\textbf{(a)}};}
\annotate{\includegraphics[width=.475\textwidth]{model.png}}{\node[white] at (.15, .9){\textbf{(b)}};}
\annotate{\includegraphics[width=.95\textwidth]{samples.png}}{\node at (.95, .9){\textbf{(c)}};}
\caption{Asteroseismic characterisation of Zvrk.\label{fig:asteroseismology}}
\end{figure*}

## ASAS-SN Photometry

ASAS-SN V and g --- need a good description of Ben's custom aperture stuff

## TESS Aperture Photometry

The PDC-SAP lightcurves from which the 

## Spectroscopy

GALAH:
APOGEE: $\teff = 4318.047 \pm 80\ \mathrm{K}$, $\mathrm{[M/H]} = -0.21 \pm 0.08$.

Unfortunately, only one APOGEE visit, so unable to constrain companion from RVs

$[\mathrm{C/N}]= -0.083321 \pm 0.06$

$\left[\mathrm{^{12}C/^{13}C}\right]$ = $9.6 \pm 0.9$

\begin{figure}
\centering
\annotate{\includegraphics[width=.475\textwidth]{CN.png}}{\node[fill=white,fill opacity=.5, text opacity=1] at (.25, .9){\textbf{(a)}}; \node[red] at (.5, .5) {\Huge PLACEHOLDER};}
\annotate{\includegraphics[width=.475\textwidth]{C13.png}}{\node[fill=white,fill opacity=.5, text opacity=1] at (.25, .9){\textbf{(b)}};\node[red] at (.5, .5) {\Huge PLACEHOLDER};}
\caption{Spectroscopic abundance characterisation of Zvrk. \textbf{(a)} $[\mathrm{C/N}]$ relative to observational sample with a metallicity cut. A well-defined observational sequence of red giants can be seen. At the notional value of $\log g$ implied by $\numax$ and $\teff$, Zvrk lies above this red-giant sequence. \textbf{(b)} $[\mathrm{^{12}C/^{13}C}]$ for the sample of \cite{hayes_bacchus_2022}.}
\end{figure}

Gaia DR3 Renormalised Unit Weight Error (RUWE) of 1.056, close to unity, indicates that an unresolved companion, sufficient to result in photocentre jitter, is unlikely to be present.

# Analysis and Interpretation

## Stellar Properties and Structure

Direct method: $M = (1.28 \pm 0.28) M_\odot$, $R = (25.0 \pm 1.8) R_\odot$. The large uncertainties on this primarily arise from the poor constraints on $\numax$. However, still helpful in bounding the parameter space for further analysis.

Constraint | Value | Reference
----:+:-----:+:---:
$\teff$ / K | $4318 \pm 80$ | APOGEE
$\mathrm{[M/H]}$ | $-0.21 \pm 0.08$ | APOGEE
$L / L_\odot$ | $174 \pm 10$ | Gaia DR2
$\Dnu/\mu\mathrm{Hz}$ | $1.21 \pm 0.01$ | This work
$\numax/\mu\mathrm{Hz}$ | $7.3 \pm 0.5$ | This work
Table: Global properties adopted in stellar modelling.\label{tab:t1}

Property | Inferred Value | Remarks
----:+:-----:+:---:
$M/M_\odot$ | $1.23 \pm 0.05$ | —
$R/R_\odot$ | $23.8 \pm 0.04$ | —
$\amlt$ | $1.8 \pm 0.05$ | —
$Y_0$ | $0.25 \pm 0.05$ | —
$Z_0$ | $0.010 \pm 0.005$ | —
Age/Gyr | $3.2 \pm 0.5$ | —
$\mathrm{Ro}_\mathrm{CZ}$ | $0.26 \pm 0.03$ | $\mathrm{Ro}_\odot = 1.38$
Table: Global properties returned from stellar modelling.\label{tab:t2}

From optimisation, best-fitting mass of 1.25 $M_\odot$. Slightly overluminous if modelled without the luminosity constraint.

## Rotational Inversions

\begin{figure}[htbp]
    \centering
    \includegraphics[width=.475\textwidth]{shear.pdf}
    \caption{Approximate characterisation of rotational shear in the JWKB approximation.}
    \label{fig:jwkb}
\end{figure}

## Magnetic activity

Magnetic activity is known to be generated by rotational shear, which can be characterised by the Rossby number $\mathrm{Ro} = P_\text{rot} / \tau_\mathrm{CZ}$. Customarily, the convective turnover timescale $\tau_\mathrm{CZ}$ is evaluated in main-sequence stellar models locally --- e.g. at one pressure scale height $H_p$ above the base of the convection zone, near the tachocline. However, since red giants are almost entirely convective by spatial extent, such a local definition is not necessarily representative of the star as a whole. Thus we evaluate the Rossby number instead with respect to a global convective turnover timescale: $$\tau_\mathrm{CZ} = \int_{r_1}^{r_2} {\mathrm d r \over V_\text{conv}},$$ where $r_1 = r_\text{base} + H_p$ and $r_2 = r_\text{top} - H_p$, and $V_\text{conv}$ is the MLT convective velocity. Evaluating this quantity with respect to models in our optimisation trajectory, in conjunction with our nominal rotational period, gives us $\mathrm{Ro}_\star \sim 0.26$. By contrast, computing this quantity from a solar-calibrated MESA model with the same physics yields $\mathrm{Ro}_\odot = 1.38$. Roughly speaking, the solar Rossby number is known to fluctuate between 1/2 and 2 over the course of the solar activity cycle.

LEOPARD: $f_\text{spot} = 0.02$, $X_\text{spot} = 0.8$ This yields a spot coverage fraction consistent with the observed $2\%$ photometric variability amplitude.

# Discussion: Possible Formation Histories

We identify three different classes of explanations for how Zvrk's rotational and chemical configuration came to be, which would roughly yield both a high rotation rate, and the observed enhanced lithium abundance:

(I)  Mass transfer from a hitherto undetected stellar companion would yield enhanced lithium at Zvrk's surface. In this scenario, its high rate of rotation could be attributed to either direct deposition of angular momentum from the accreted material, or tidal spin-up as a result of binary interactions.
(II)  Alternatively, both the enhanced lithium abundance and high rate of rotation could be attributed to Zvrk having engulfed a formerly orbiting companion, with both matter and angular momentum being directly deposited into its envelope, and being redistributed over the course of several mixing timescales.
(III)  Finally, for the sake of argument, Zvrk could represent significant departures from the existing theory of single-star evolution. To match the chemical enhancements and fast rotation rate, its evolution would necessitate an efficiency of chemical mixing and angular momentum transport far in excess of that currently assumed of red giants.

# Conclusion

\begin{acknowledgements}

% magic incantation

This paper includes data collected by the TESS mission. Funding for the TESS mission is provided by the NASA's Science Mission Directorate.

This work has made use of data from the European Space Agency (ESA) mission {\it Gaia} (\url{https://www.cosmos.esa.int/gaia}), processed by the {\it Gaia} Data Processing and Analysis Consortium (DPAC, \url{https://www.cosmos.esa.int/web/gaia/dpac/consortium}). Funding for the DPAC has been provided by national institutions, in particular the institutions participating in the {\it Gaia} Multilateral Agreement.

JMJO, MTYH, and MS-F acknowledge support from NASA through the NASA Hubble Fellowship grants HST-HF2-51517.001, HST-HF2-51459.001, and HST-HF2-51493.001-A, awarded by STScI. STScI is operated by the Association of Universities for Research in Astronomy, Incorporated, under NASA contract NAS5-26555.
APS acknowledges partial support by the Thomas Jefferson Chair Endowment for Discovery and Space Exploration, and partial support through the Ohio Eminent Scholar Endowment.

\software{NumPy \citep{numpy}, SciPy stack \citep{scipy}, AstroPy \citep{astropy:2013,astropy:2018}, \texttt{dynesty} \citep{dynesty}, Pandas \citep{pandas}, \mesa\ \citep{mesa_paper_1,mesa_paper_2,mesa_paper_3,mesa_paper_4,mesa_paper_5}, \gyre\ \citep{townsend_gyre_2013}.}
\end{acknowledgements}

<!--\bibliography{biblio.bib}-->
