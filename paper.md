---
title: 'EDA Toolkit: A Lightweight Python Library for Exploratory Data Analysis and Reporting'
tags:
  - Python
  - exploratory data analysis
  - data visualization
  - open source
  - data science
authors:
  - name: Leonid Shpaner
    orcid: 0009-0007-5311-8095
    equal-contrib: true
    affiliation: 1
  - name: Oscar Gil.
    orcid: 0009-0001-9438-7589 
    equal-contrib: true 
    affiliation: 2
affiliations:
 - name: Data Scientist, UCLA Health, United States
   index: 1
   ror: 00hx57361
 - name: Data Scientist, UC Riverside, United States
   index: 2
date: 30 July 2025
bibliography: paper.bib
---

# Summary

EDA Toolkit is a lightweight Python package for conducting and visualizing 
exploratory data analysis (EDA). It provides flexible plotting tools, profiling 
summaries, and exportable Excel reports tailored for both academic and industry 
workflows. Designed to be highly customizable and easily integrated into notebooks 
or pipelines, EDA Toolkit helps users rapidly understand and communicate data 
characteristics.

# Statement of need

Exploratory Data Analysis (EDA) is a critical step in any data science project. 
It helps researchers understand the structure of a dataset, detect data quality 
issues, identify meaningful patterns, and shape the direction of analysis. While 
tools such as `pandas_profiling` and `sweetviz` provide automated reports, they 
often lack the flexibility, transparency, and formatting needed in professional 
or academic settings.

EDA Toolkit addresses these limitations by offering a modular and customizable 
set of tools designed for clarity, reproducibility, and high-quality presentation. 
It supports both academic research and applied data science use cases, with an 
emphasis on producing outputs that are publication-ready and easy to interpret.

# Key Features

## Table 1 Generation

The `generate_table1()` function allows users to produce clean, formatted 
descriptive tables often used in clinical and academic research. The output 
includes summaries by group and supports filtering by data type, making it 
easier to communicate sample characteristics without relying on external tools 
like Excel.



# Mathematics

Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$

You can also use plain \LaTeX for equations
\begin{equation}\label{eq:fourier}
\hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx
\end{equation}
and refer to \autoref{eq:fourier} from text.

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project.
