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
We report the discovery of a rapidly-rotating red giant ($P_\text{rot} \sim 98\ \mathrm{d}$) observed with TESS in its Southern Continuous Viewing Zone. The rotation rate of this red giant is independently verified by the use of p-mode asteroseismology, strong perodicity in TESS and ASAS-SN photometry, and multiple measurements of spectroscopic rotational broadening. Modulations in the amplitude of the photometric rotational signal indicate the rapid evolution of spot morphology, suggesting enhanced magnetic activity, and therefore radial differential rotation. We further develop and deploy new asteroseismic techniques to characterise this rotational shear in its convective envelope. Such a high rotation rate is also categorically incompatible with even the most physically permissive models of angular-momentum transport in single-star evolution. Moreover, spectroscopy also indicates an unsually high surface lithium abundance. Taken together, all of these suggest an ingestion scenario for the formation of this rotational configuration, various models of which we examine in detail. Such ingestion events represent an alternative mechanism by which the envelopes of post-main-sequence stars may spin up, as seen in the \textit{Kepler} sample, in conjunction with various existing hypotheses for outward angular momentum transport.
\keywords{Asteroseismology (73), Red giant stars (1372), Stellar oscillations (1617)}
\end{abstract}

# Introduction

Over the course of their first ascent up the red giant branch (RGB), the surface rotation of red giant stars is thought to slow down significantly, as result of angular momentum conservation. That a non-negligible fraction of the Kepler sample of red giants exhibit nontrivial photometric variability, attributable to surface features rotating into and out of view, indicates instead that these so-called rapidly-rotating red giants (RRRGs) must have passed through evolutionary scenarios quite unlike to the standard picture of single-star evolution. While some RRRGs have been found to be in binary systems (including with compact objects like black holes and neutron stars), the vast majority (85\%) do not possess detectable orbital companions.

We report the concurrent detection of rotational signals in TIC 350842552 (which we will refer to as "Zvrk" in the remainder of this work, for brevity), a RRRG, using asteroseismology (from the NASA TESS mission), photometry (from TESS and the ground-based ASAS-SN network), and with spectroscopy (various ground-based spectrographs).

# Observational Characterisation

Zvrk was initially flagged for asteroseismic analysis as part of an ongoing search for unusual oscillation signatures corresponding to chemical peculiarities, in particular high lithium abundance. For this purpose, an overlapping sample of stars with both GALAH abundances, and TESS coverage in the Southern continuous viewing zone (CVZ), was constructed, and targets in this list were subjected to preliminary asteroseismic analysis, to constrain the global p-mode asteroseismic parameters $\Dnu$, $\numax$, and $\epsilon_p$. This was done using the 2D autocorrelation function procedure of @keaton?, applied to the publicly available presearch data conditioning simple aperture photometry (PDCSAP) lightcurves.

Morphologically, the double ridges on this frequency echelle diagram are evocative of those of other p-mode oscillators, as previously observed en masse with *Kepler* and TESS, and suggest identification as being the modes of even degree $\ell$, which are known to form double ridges of this form. The remainder of the oscillation power would, under this putative identification, be attributed to oscillations of dipole modes, which are known in other red giants to be disrupted by mode mixing with an interior g-mode cavity to produce gravitoacoustic mixed modes.

However, such an identification would be in significant tension with known properties of these red giants. In particular:

- This would imply a p-mode phase offset of $\epsilon_p \sim 0.2$. This is in very significant tension with the value of $0.7$ implied by the Kepler sample [e.g. @mosser_universal_2011, @yu_luminous_2020].
- At $\Dnu = 1.22\ \mathrm{\mu Hz}$, the period spacing associated with the interior g-mode cavity would be far too small to cause significant departures in the observed dipole modes from the simple p-mode asymptotic relation. 

## Detailed Asteroseismology


\begin{figure*}
\centering
\annotate{\includegraphics[width=.475\textwidth]{figures/echelle_id_2.pdf}}{\node[white] at (.15, .9){\textbf{(a)}};}
\annotate{\includegraphics[width=.475\textwidth]{figures/model.png}}{\node[white] at (.15, .9){\textbf{(b)}};}
\annotate{\includegraphics[width=.95\textwidth]{figures/samples.png}}{\node at (.95, .9){\textbf{(c)}};}
\caption{Asteroseismic characterisation of Zvrk.}
\end{figure*}

## TESS Photometry

The PDC-SAP lightcurves from which the 

## ASAS-SN Photometry

## Archival Spectroscopy

GALAH:
APOGEE: $\teff = 4318.047 \pm 80\ \mathrm{K}$, $\mathrm{[M/H]} = -0.21 \pm 0.08$.

Unfortunately, only one APOGEE visit, so unable to constrain companion from RVs

# Stellar Modelling

## Stellar Properties and Structure

From MESA models, $\mathrm{Ro} \sim 20$?? $\tau_\text{conv} \sim 5\ \mathrm{d}$. Order of magnitude higher than the Sun, maybe?

## Rotational Inversions

## Formation History

# Conclusion

\begin{acknowledgements}

This paper includes data collected by the TESS mission. Funding for the TESS mission is provided by the NASA's Science Mission Directorate.

This work has made use of data from the European Space Agency (ESA) mission {\it Gaia} (\url{https://www.cosmos.esa.int/gaia}), processed by the {\it Gaia} Data Processing and Analysis Consortium (DPAC, \url{https://www.cosmos.esa.int/web/gaia/dpac/consortium}). Funding for the DPAC has been provided by national institutions, in particular the institutions participating in the {\it Gaia} Multilateral Agreement.

JO, MH, and MS-F acknowledge support from NASA through the NASA Hubble Fellowship grant HST-HF2-51517.001 awarded by STScI. STScI is operated by the Association of Universities for Research in Astronomy, Incorporated, under NASA contract NAS5-26555. APS acknowledges partial support by the Thomas Jefferson
Chair Endowment for Discovery and Space Exploration and
partial support through the Ohio Eminent Scholar Endowment.

\software{NumPy \citep{numpy}, SciPy stack \citep{scipy}, AstroPy \citep{astropy:2013,astropy:2018}, \texttt{dynesty} \citep{dynesty}, Pandas \citep{pandas}, \mesa\ \citep{mesa_paper_1,mesa_paper_2,mesa_paper_3,mesa_paper_4,mesa_paper_5}, \gyre\ \citep{townsend_gyre_2013}.}
\end{acknowledgements}

<!--\bibliography{biblio.bib}-->
