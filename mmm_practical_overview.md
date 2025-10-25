# Abstract

Marketing Mix Models (MMMs) are statistical models used to estimate marketing effectiveness and guide budget planning. In this paper, we walk through the evolution of MMMs, providing example code in PyMC for practitioners to quickly understand MMMs. Then, we demonstrate how ignoring causality can be detrimental to model estimates.

# Introduction

Give an overview of each section using a few sentences for each.

* History of MMM: MMM has come from combining econometrics, time-series analyses, and marketing theory. 
* Background - Progression from simple MMMs to Advanced: from linear models to bayesian to hierarchical bayesian MMMs to state of the art new advances.
* Current Issues: causality, unified measurement, feature selection.
* What this paper researches: demonstration of issues when causal dag is ignored / misunderstood. 

The rest of this paper is organized as follows. In the [Background](#background) section, we discuss the history, progression of MMMs, and current issues. In the [Methodology](#methodology) section, we demonstrate some of those key problems using simulated data. In the [Results](#results) section, we test out some of the proposed solutions from state of the art research. In the [Conclusion](#conclusion) section, we discuss future developments. 

# Background

## History

Give an overview of marketing science history in relation to MMMs. Research started in 1960s, became more widely known later. Bayesian MMM started to gain traction as computation and MCMC improved. 

References:
* [The History of Marketing Science](https://www.worldscientific.com/doi/epdf/10.1142/9789811272233_0005) - textbook that goes in depth on history - main source
* [Handbooks in Operations Research and Management Science - Marketing Mix Models Literature Review](https://www.sciencedirect.com/science/article/pii/S0927050705800386) - looks at all historical literature
* [BTVC at Uber](https://arxiv.org/html/2106.03322v4) - gives a brief summary of history
* [Hierarchical Marketing Mix Models with Sign Constraints](https://pmc.ncbi.nlm.nih.gov/articles/PMC9041956/) - also has two paragraphs about the history.

## 1. Bayesian Linear Regression

Regular linear regression finds coefficients, while Bayesian linear regression gives you a probability distribution for each coefficient, incorporating prior beliefs. 

Math formula:

$$ E(y_i|\beta, X) = \beta_1 x_{i1} + ... + \beta_k x_{ik} $$

$$ y_i \sim N(X_i^T \beta, \sigma^2) $$

Code Example: [Bayesian Linear Regression](https://colab.research.google.com/drive/1ZppqL9fUdPzqyL2FXa2EqFcKrzCgd28D?usp=sharing)

Difference from Linear Regression
* Set prior on Coefficients: $\beta$
* Set prior on Variance: $\sigma^2$
* Set prior on Intercept: $\beta_0$. 

Strengths:
* Core bayesian strengths such as using posterior distributions and talking about 95% probability instead of 95% confidence intervals.
* Include prior knowledge to help stabilize estimates and prevent overfitting (especially with small samples or multicollinearity).

Weaknesses:
* Prior sensitivity - need to find a good balance for the prior distributions - can't be too strong but can't be too weak in accurately estimating the data.
* Computational cost - when conjugate priors aren't used, have to use MCMC which is much more computationally intensive compared to regular regression which uses closed-form OLS solution.
* Scaling - bayesian and frequentist regression often agree (priors wash out) when you have large datasets.

References:
* [Bayesian Data Analysis](https://sites.stat.columbia.edu/gelman/book/BDA3.pdf)

## 2. MMM

MMM just adds some marketing specific adjustments to Bayesian Linear regression models.

Math Formula:

$$ y_t = \sum_{i=1}^m \beta_i s_i(c_i(x_{t-l+1,i},...,x_{t,i}; \alpha_i); k_i, \lambda_i) + \sum_{j=1}^n \gamma_j z_{t,j} + \epsilon_t $$

Code Example: [Bayesian MMM](https://colab.research.google.com/drive/1N38pqZnt7Vt1ppbxoqL_Hs_qA5ir4rHp?usp=sharing)

Adjustments:
* Adstock - ads can have a long lasting but decaying effect.
* Saturation - audience targeted by ads is finite. You can saturate your audience with ads and have diminishing returns. 
* Priors - assumption that ads won't negatively hurt sales. This assumption can be integrated in to the beta coefficient priors. 

References: 
* [Hierarchical Marketing Mix Models with Sign Constraints](https://pmc.ncbi.nlm.nih.gov/articles/PMC9041956/) - explains adstock, saturation, and priors briefly.
* [Bayesian Methods for Media Mix Modeling with Carryover and Shape Effects](https://research.google.com/pubs/archive/46001.pdf) - this is the seminal paper about adstock and saturation. 

## 3. Hierarchical Bayesian Linear Models

Hierarchical Bayesian Linear Models add a nested structure to the model. For example, if we are using a regular regression model to predict sales in the US using marketing spend as the predictor, there would be a single coefficient for marketing spend. With a hierarchical regression, we recognize that each state may have different marketing performance. Thus, we estimate a global marketing spend coefficient, representing the average marketing impact across all states, and local state-specific coefficients that capture deviations for each individual state.

Linear Regression formula:

$$ y_i \sim N(X_i^T \beta, \sigma^2) $$

Hierarchical Regression formula:

$$ y_{ij} \sim N(X_{ij}^T \beta_j, \sigma^2) $$

where:
* $i$ indexes observations within group $j$.
* each group has its own regression coefficient $\beta_j$.

Each local $\beta_j$ comes from a global distribution:

$$ \beta_j \sim N(\mu, \tau^2 I) $$

where: 
* hyperparameters $\mu$, $\tau^2$ get priors.

So, instead of a single $\beta$, we now have a distribution of $\beta_j$s that center around a population mean $\mu$. 

Code Example:

How is this different than just adding state as a variable?:
* explain the concept of pooling and why this is good in our example

References:
* [Bayesian Data Analysis](https://sites.stat.columbia.edu/gelman/book/BDA3.pdf)

## 4. Hierarchical MMMs

Hierarchical MMMs extend regular MMMs by introducing a hierarchy. For example, a hierarchical MMM might include national-level parameters for coefficients, while also estimating state-level parameters that capture local variations in the coefficients.

Math formula:

$$ y_{t,g} = \tau_g + \sum_{m=1}^M \beta_{m,g} Adstock(Hill(x_{t,m,g}; K_m, S_m), \alpha_m, L) + \sum_{c=1}^C \gamma_{c,g} * z_{t,c,g} + \epsilon_{t,g} $$

$$ \beta_{m,g} \sim \mathcal{N}(\beta_m, \eta_m^2) $$

$$ \gamma_{c,g} \sim \mathcal{N}(\gamma_c, \xi_c^2) $$

$$ \tau_g \sim \mathcal{N}(\tau, \kappa^2) $$

Where:
* $y_{t,g}$ is the outcome for geo $g$ at time $t$.
* Each geo $g$ has its own intercept $\tau_g$, coefficient $\beta_{m,g}$, and controls $\gamma_{c,g}$
* The geo-level coefficients, controls, and intercepts are hierarchically linked to global means $\beta_m$, $\gamma_c$, and $\tau$.
* This creates partial pooling: geos with little data borrow strength from the global distribution, while large geos can deviate more. 

Code Example:

Strengths:
* More data and variation = better estimates - more observations to feed the model and more variation in spend at a geo-level.
* Reduced bias under some conditions - generally at a national level ad spend and base demand are very correlated (think holidays) - but at geo level this correlation can be weaker.
* Better fitting of adstock and saturation - more variability in spends in geos lets the model explore different parts of the curves.
* Partial pooling helps smaller geos.
* Overall should improve the model and the decisions.

Weaknesses:
* Hard to get geo-level data for everything.
* National level data can be imputed to geo-level but it isn't the best.
* Assumption that adstock and saturation are same across geos may not be correct.
* Sensitivity to control variables. Imperfect controls still hurt estimates.
* Computational complexity.
* Pooling isn't perfect. Really sparse geos can have weird estimates. 

References:
* [Geo-level Bayesian Hierarchical MMM](https://research.google/pubs/geo-level-bayesian-hierarchical-media-mix-modeling/)

## 5. State of the Art

Most state of the art MMMs are built by vendors or companies. Thus, we may not know the full extent of their advances, but we can gauge general principles and new features. 

New developments in MMMs from commercial vendors:
* Recast: [advanced causal dag](https://docs.getrecast.com/docs/recast-model-technical-documentation) as framework for their model.
* Ipsos MMA: [unified measurement](https://mma.com/blog/the-current-state-of-marketing-mix-modeling/) using attribution, experiments, geo tests, and MMM all together.
* Sellforte: [advanced causal dag](https://sellforte.com/blog/what-is-causal-marketing-mix-modeling-mmm) as framework for their model.
* Analytic Partners: [smarter incrementality tests to calibrate MMMs](https://analyticpartners.com/knowledge-hub/resources/calibrating-with-chaos-analytic-partners/) and [commercial variables](https://analyticpartners.com/knowledge-hub/newsroom/analytic-partners-shapes-marketing-mix-modeling-mmm-for-25-years-and-innovates-with-commercial-analytics/).

New developments in MMMs in academia:
* [CausalMMM](https://arxiv.org/pdf/2406.16728): learns the causal structure automatically.
* [NNN](https://arxiv.org/pdf/2504.06212): uses neural networks to add qualitative features.
* [Bayesian Time-Varying Coefficient Models](https://arxiv.org/html/2106.03322v4/): allowing coefficients to drift over time.
* [Addressing Channel Influence Bias and Cross-Channel Effects](https://arxiv.org/pdf/2311.05587): physics inspired additions to Bayesian Hierarchical MMM.
* [Identification of Nonlinear and Time-varying Effects in Marketing Mix Models](https://arxiv.org/pdf/2408.07678): nonlinearities and time-varying effects are often conflated together.

Open problems:
* Experimenting with MMM is expensive
* Data sparsity: new channels don't have enough data

References:
* [Challenges and Opportunities in Media Mix Modeling](https://research.google/pubs/challenges-and-opportunities-in-media-mix-modeling/)

# Methodology

Many open source MMM tools ignore causality. Generally, the tutorials focus on just inputting every channel as parallel independent variables into the model, when in reality, marketing channels affect each other. 

For example, branded search is more of an intermediate channel:
![alt text](assets/img/image.png)

We will investigate how bad ignoring this causality can be for MMMs using simulated data that reflects this real world causality. Because the data is simulated, we will know the true parameters. 

First, we will build a naive MMM (replicating base model frameworks that we see in [Google Meridian](https://developers.google.com/meridian), and [PyMC Marketing](https://www.pymc-marketing.io/en/latest/)) and see how close it can get to the true parameters.

Here's an example of PyMC's base model framework:

![alt text](assets/img/pymc_example_dag.png)

From there, we will build a more causally accurate MMM and see how close it can get to the true parameters. Google Meridian's framework might be able to succeed off the shelf because it has added features to account for branded search being an intermediate channel. 

Lastly, we will examine if adding an incrementality estimate as a prior into the naive model can help push it in the right direction.

# Results

We expect to see that ignoring causality will have really bad results. It'll be interesting to see if a good incrementality prior could overcome this error.

# Conclusion

* Summarize background, current issues, current developments.
* Summarize results.
* Talk about future research opportunities, especially talk about how [CausalMMM](https://arxiv.org/pdf/2406.16728) might be the newest key development.


## Hierarchical Bayesian MMM

Hierarchical Bayesian MMMs extend Bayesian MMMs by introducing a hierarchy (Sun, Wang, Jin, Chan, & Koehler 2016–2017). For example, a hierarchical MMM might include national-level parameters for coefficients, while also estimating state-level parameters that capture local variations in the coefficients. This is particularly useful in practice because different regions/stores can have different outcomes. For example, if we are using a regular regression model to predict sales in the US using marketing spend as the predictor, there would be a single coefficient for marketing spend $\beta_m$. With a hierarchical regression, we would assign a local coefficient for marketing spend $\beta_{m,g}$ to each state. We would expect each state's marketing efficiency vary around the global efficiency $\beta_m$. Thus, we would have the following:

$$ \beta_{m,g} \sim \mathcal{N}(\beta_m, \eta_m^2) $$

Each of our state's coefficients come from a normal distribution centered around the global coefficient $\beta_m$ with a variance of $\eta_m^2$. 

Control variables can also be modeled in this way:

$$ \gamma_{c,g} \sim \mathcal{N}(\gamma_c, \xi_c^2) $$

The intercepts can also be modeled in this way:

$$ \tau_g \sim \mathcal{N}(\tau, \kappa^2) $$

Finally, we piece it all together into this formula:

$$ y_{t,g} = \tau_g + \sum_{m=1}^M \beta_{m,g} Adstock(Hill(x_{t,m,g}; K_m, S_m), \alpha_m, L) + \sum_{c=1}^C \gamma_{c,g} * z_{t,c,g} + \epsilon_{t,g} $$

There are many benefits of this approach as mentioned in Sun, Wang, Jin, Chan, & Koehler (2016–2017). First, because the data is now on a more granular level, there's more data and more variation between in spend to feed the model, resulting in more accurate estimates of our global parameters. This includes more accurate estimates of the adstock and saturation parameters as well because the variability in spends lets the model explore more parts of the curves. Second, the granularity can also help reduce bias from strong correlations - generally at a national level ad spend and base demand are very correlated, but at geo level this correlation can be weaker. For example, summer vacation starts at different times in different states. Third, geos with sparse data can still be estimated well thanks to partial pooling from the global parameters. Overall, the additional nuance and variability in the data can produce a much more robust model compared to a model built using national aggregated data.

Sun, Wang, Jin, Chan, & Koehler (2016–2017) also note the following limitations. First, it can be hard to gather this marketing data on a more granular level. National data can be imputed to geo levels, but with varying effectiveness. Second, the model might be incorrectly assuming that adstock and saturation are the same across geos. Third, imperfect controls still hurt estimates. Fourth, computation is much more complex and reaching stable estimates can take longer. Fifth, pooling isn't perfect. Really sparse geos can have weird estimates. 

This paper focuses on exploring national funnel effects. Future research will explore these effects in hierarchical models. 
