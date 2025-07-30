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

| Variable | Type | Mode | Count | Proportion (%) | <=50K (n = 37155) | >50K (n = 11687) |
| --- | --- | --- | --- | --- | --- | --- |
| age_group | Categorical | 18-29 | 48,842 | 100.00 | 37155 | 11687 |
| age_group = 18-29 | Categorical | 18-29 | 13,920 | 28.50 | 13174 (35.46%) | 746 (6.38%) |
| age_group = 30-39 | Categorical | 18-29 | 12,929 | 26.47 | 9468 (25.48%) | 3461 (29.61%) |
| age_group = 40-49 | Categorical | 18-29 | 10,724 | 21.96 | 6738 (18.13%) | 3986 (34.11%) |
| age_group = 50-59 | Categorical | 18-29 | 6,619 | 13.55 | 4110 (11.06%) | 2509 (21.47%) |
| age_group = 60-69 | Categorical | 18-29 | 3,054 | 6.25 | 2245 (6.04%) | 809 (6.92%) |
| age_group = 70-79 | Categorical | 18-29 | 815 | 1.67 | 668 (1.80%) | 147 (1.26%) |
| age_group = < 18 | Categorical | 18-29 | 595 | 1.22 | 595 (1.60%) | 0 (0.00%) |
| age_group = 80-89 | Categorical | 18-29 | 131 | 0.27 | 115 (0.31%) | 16 (0.14%) |
| age_group = 90-99 | Categorical | 18-29 | 55 | 0.11 | 42 (0.11%) | 13 (0.11%) |
| age_group = 100 + | Categorical | 18-29 | 0 | 0.00 | 0 (0.00%) | 0 (0.00%) |
| marital-status | Categorical | Married-civ-spouse | 48,842 | 100.00 | 37155 | 11687 |
| marital-status = Married-civ-spouse | Categorical | Married-civ-spouse | 22,379 | 45.82 | 12395 (33.36%) | 9984 (85.43%) |
| marital-status = Never-married | Categorical | Married-civ-spouse | 16,117 | 33.00 | 15384 (41.40%) | 733 (6.27%) |
| marital-status = Divorced | Categorical | Married-civ-spouse | 6,633 | 13.58 | 5962 (16.05%) | 671 (5.74%) |
| marital-status = Separated | Categorical | Married-civ-spouse | 1,530 | 3.13 | 1431 (3.85%) | 99 (0.85%) |
| marital-status = Widowed | Categorical | Married-civ-spouse | 1,518 | 3.11 | 1390 (3.74%) | 128 (1.10%) |
| marital-status = Married-spouse-absent | Categorical | Married-civ-spouse | 628 | 1.29 | 570 (1.53%) | 58 (0.50%) |
| marital-status = Married-AF-spouse | Categorical | Married-civ-spouse | 37 | 0.08 | 23 (0.06%) | 14 (0.12%) |

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
