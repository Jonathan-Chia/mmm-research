# Abstract

Marketing Mix Modeling (MMM) has evolved from its roots in econometrics and time-series analysis into a sophisticated discipline that blends marketing theory with advanced statistical modeling [1](https://www.worldscientific.com/doi/epdf/10.1142/9789811272233_0005). Early MMMs relied on simple linear regressions to estimate channel effects, but progress in Bayesian inference and computation has driven a transition toward Bayesian models capable of capturing uncertainty, heterogeneity, and nonlinear effects such as adstock and saturation. Recent developments have focused on addressing critical issues in causality, unified measurement across marketing channels, and feature selection, all of which have profound implications for interpreting marketing effectiveness.

This paper investigates one of the most persistent and overlooked challenges in MMMs—causal mis-specification. Using simulated data, we demonstrate how failing to correctly account for causal dependencies (e.g., treating intermediate variables such as branded search as independent drivers) can lead to significant bias in parameter estimates. We compare results from a naïve MMM implementation, similar to open-source frameworks like PyMC Marketing and Google Meridian, with a more causally accurate model and explore whether incorporating incrementality-based priors can partially correct these errors.

Our results show that ignoring the underlying causal structure can substantially distort marketing effect estimates, leading to misguided optimization decisions. However, informed priors can mitigate some of these distortions. We conclude by discussing emerging approaches—such as causal graphical models, time-varying Bayesian structures, and hybrid frameworks like CausalMMM—that represent promising directions for building more accurate and interpretable marketing measurement systems.

# Introduction

The rest of this paper is organized as follows. In the [Background](#background) section, we discuss the history, progression of MMMs, and current issues. In the [Methodology](#methodology) section, we explain our data generating process where the simulated data has upper funnel channel impacts on lower funnel channels. From there, we show how we set up different MMMs to retrieve our parameters. In the [Results](#results) section, we see that better priors can help mitigate causal issues. In the [Conclusion](#conclusion) section, we discuss future developments. 

# Background

MMM research started primarily in the 1960s as scholars and practicioners sought to understand how product, price, promotion, and distribution interact and influence performance. Some foundational research include ____. Early models used aggregate data in regression-based frameworks. Further research examined the interaction between differerent marketing variables, highlighting the need to coordinate advertising, pricing, and distribution. [Handbooks in Operations Research and Management Science - Marketing Mix Models Literature Review](https://www.sciencedirect.com/science/article/pii/S0927050705800386).

Bayesian approaches to MMM started in ___.

Common issues in MMM include ____. We examine funnel effects. 