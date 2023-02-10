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
We report the discovery of an extremely fast-rotating red giant ($P_\text{rot} = $) observed with TESS in its Southern Continuous Viewing Zone. The rotation rate of this red giant is independently verified by the use of p-mode asteroseismology, strong perodicity in ASAS-SN photometry, and multiple measurements of spectroscopic rotational broadening. Modulations in the amplitude of the photometric rotational signal indicate the rapid evolution of spot morpohology, suggesting enhanced magnetic activity, and therefore radial differential rotation. We further develop and deploy new asteroseismic techniques to characterise this rotational shear in its convective envelope. Such a high rotation rate is also categorically incompatible with even the most physically permissive models of angular-momentum transport in single-star evolution. Moreover, spectroscopy also indicates an unsually high surface lithium abundance. Taken together, all of these suggest an ingestion scenario for the formation of this rotational configuration, various models of which we examine in detail.
\keywords{Asteroseismology (73), Red giant stars (1372), Stellar oscillations (1617)}
\end{abstract}

# Introduction

\remark{insert small talk here}

# Observational Characterisation

## TESS Asteroseismology

- $\epsilon_p$: @mosser_universal_2011, @yu_luminous_2020

## ASAS-SN Photometry

## Archival Spectroscopy

# Stellar Modelling

## Stellar Properties and Structure

## Rotational Inversions

## Formation History

# Conclusion

\begin{acknowledgements}

% recite now the magic spell, lest thou bring the wrath of the grant gods upon thy head

This paper includes data collected by the TESS mission. Funding for the TESS mission is provided by the NASA's Science Mission Directorate.

This work has made use of data from the European Space Agency (ESA) mission {\it Gaia} (\url{https://www.cosmos.esa.int/gaia}), processed by the {\it Gaia} Data Processing and Analysis Consortium (DPAC, \url{https://www.cosmos.esa.int/web/gaia/dpac/consortium}). Funding for the DPAC has been provided by national institutions, in particular the institutions participating in the {\it Gaia} Multilateral Agreement.

JO, MH, and MS-F acknowledge support from NASA through the NASA Hubble Fellowship grant HST-HF2-51517.001 awarded by STScI. STScI is operated by the Association of Universities for Research in Astronomy, Incorporated, under NASA contract NAS5-26555. APS acknowledges partial support by the Thomas Jefferson
Chair Endowment for Discovery and Space Exploration and
partial support through the Ohio Eminent Scholar Endowment.

\software{NumPy \citep{numpy}, SciPy stack \citep{scipy}, AstroPy \citep{astropy:2013,astropy:2018}, \texttt{dynesty} \citep{dynesty}, Pandas \citep{pandas}, \mesa\ \citep{mesa_paper_1,mesa_paper_2,mesa_paper_3,mesa_paper_4,mesa_paper_5}, \gyre\ \citep{townsend_gyre_2013}.}
\end{acknowledgements}

<!--\bibliography{biblio.bib}-->
