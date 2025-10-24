# Abstract

Marketing Mix Modeling (MMM) has evolved from its roots in econometrics and time-series analysis into a sophisticated discipline that blends marketing theory with advanced statistical modeling [1](https://www.worldscientific.com/doi/epdf/10.1142/9789811272233_0005). Early MMMs relied on simple linear regressions to estimate channel effects, but progress in Bayesian inference and computation has driven a transition toward Bayesian models capable of capturing uncertainty, heterogeneity, and nonlinear effects such as adstock and saturation. Recent developments have focused on addressing critical issues in causality, unified measurement across marketing channels, and feature selection, all of which have profound implications for interpreting marketing effectiveness.

This paper investigates one of the most persistent and overlooked challenges in MMMs—causal mis-specification. Using simulated data, we demonstrate how failing to correctly account for causal dependencies (e.g., treating intermediate variables such as branded search as independent drivers) can lead to significant bias in parameter estimates. We compare results from a naïve MMM implementation, similar to open-source frameworks like PyMC Marketing and Google Meridian, with a more causally accurate model and explore whether incorporating incrementality-based priors can partially correct these errors.

Our results show that ignoring the underlying causal structure can substantially distort marketing effect estimates, leading to misguided optimization decisions. However, informed priors can mitigate some of these distortions. We conclude by discussing emerging approaches—such as causal graphical models, time-varying Bayesian structures, and hybrid frameworks like CausalMMM—that represent promising directions for building more accurate and interpretable marketing measurement systems.

# Introduction

The rest of this paper is organized as follows. In the [Background](#background) section, we discuss the history and fundamentals of Bayesian MMMs. In the [Methodology](#methodology) section, we explain our data generating process where the simulated data has upper funnel channel impacts on lower funnel channels. From there, we show how we set up different MMMs to retrieve our parameters. In the [Results](#results) section, we see that better priors can help mitigate causal issues. In the [Conclusion](#conclusion) section, we discuss future developments. 

# Background

MMM research started primarily in the 1960s as scholars and practicioners sought to understand how product, price, promotion, and distribution interact and influence performance (Borden, 1964; McCarthy, 1978). Early models used aggregate data in regression-based frameworks. Further research examined the interaction between differerent marketing variables, highlighting the need to coordinate advertising, pricing, and distribution. [Handbooks in Operations Research and Management Science - Marketing Mix Models Literature Review](https://www.sciencedirect.com/science/article/pii/S0927050705800386).

Bayesian approaches to MMM gained momentum in the late 2010s thanks to advances in sampling computation and newer research including key research from Google on carryover and shape effects (Jin, Wang, Sun, Chan, Koehler 2017) as well as hierarchical modeling (Sun, Wang, Jin, Chan, Koehler 2016). Bayesian MMMs help mitigate issues with multicollinearity, small data but high parameters, and uncertainty propogation. 

One issue that Bayesian MMMs might also be able to mitigate is the issue of funnel effects (Chan and Perry 2017). Funnel effects are biases that occur when marketing channels influence one another across stages of the customer journey—where upper-funnel activities affect both lower-funnel channels and final outcomes. For instance, a linear TV ad may increase paid search volume as well as directly drive sales. In this case, TV ads impact sales both directly and indirectly, but naive MMMs that assume independence between TV and search fail to capture this causal structure. 

The next sub-sections demonstrates the development of Bayesian MMMs, and then the research explores methods to mitigate these funnel effects.

## Bayesian Linear Regression

Bayesian linear regression extends the traditional linear regression algorithm by modeling uncertainty in the estimated coefficients through probability distributions. The formula for the algorithm is the same where the expected value of the response variable $y_i$ is a linear combination of predictors:

$$ E(y_i|\beta, X) = \beta_1 x_{i1} + ... + \beta_k x_{ik} $$

The response $y_i$ is assumed to follow a normal distribution centered around the linear predictor, with variance $\sigma^2$. 

$$ y_i \sim N(X_i^T \beta, \sigma^2) $$

Here's an example implementation in PyMC code modeling marketing spend's effect on sales:

```python
with pm.Model() as model:

    #Priors
    beta_0 = pm.Normal('beta_0',0, sigma=10000) # intercept
    beta_1 = pm.HalfNormal('beta_1', sigma=10) # slope
    sigma = pm.HalfNormal('sigma', sigma=4000) # standard deviation - expect sales to vary by 4000

    # Deterministics
    mu = beta_0 + beta_1*spend # intercept + Beta*spend

    # Likelihood
    Ylikelihood = pm.Normal('Ylikelihood', mu, sigma, observed=sales)
```

The difference between linear regression and Bayesian linear regression is how uncertainty is modeled. In the above code example, instead of producing a point estimate for $\beta_1$ (spend's effect on sales), the Bayesian linear regression model treats the $\beta_1$ as a half normal random variable and specifies a prior belief about the $\beta_1$ (uninformative prior with wide variance of 10 that is expected to be positive and closer to 0). Priors are also placed on the intercept $\beta_0$ and the variance $\sigma$. 

Bayesian linear regression inherits the core advantages of Bayesian inference, such as the ability to express results in probabilistic terms (e.g., a 95% probability that a coefficient lies within a certain range) rather than in terms of confidence intervals (if I run the regression many times on new random samples, 95% of the output confidence intervals will contain the true value). Additionally, priors allow the model to incorporate prior knowledge or domain expertise, which can stabilize coefficient estimates and reduce overfitting. This is particularly valuable in cases with limited data, high multicollinearity, or sparse predictors.

Sadly, Bayesian linear regression also comes with challenges. One key issue is prior sensitivity: the choice of priors can heavily influence the posterior results, especially when data are limited. Priors that are too strong may bias results toward preconceived beliefs, while those that are too weak may fail to regularize effectively. Computational cost is another challenge. When conjugate priors are not available, inference requires sampling methods such as Markov Chain Monte Carlo (MCMC), which is significantly slower than the closed-form ordinary least squares (OLS) solution used in traditional regression. Finally, in large datasets, both methods' estimates often converge. Because the influence of priors diminishes with abundant data, a linear regression model is just as accurate, yet faster than Bayesian linear regression in this situation.

# References

- Borden, N. H. (1964). The concept of the marketing mix. Journal of advertising research, 4 (2),
2–7.
- Jin, Y., Wang, Y., Sun, Y., Chan, D. & Koehler, J. (2017). Bayesian methods for media mix
modeling with carryover and shape effects. research.google.com.
- McCarthy, J. E. (1978). Basic marketing: a managerial approach (6th ed.). Homewood, Il: R.D.
Irwin.
- Sun, Y., Wang, Y., Jin, Y., Chan, D., & Koehler, J. (2016–2017). Geo-level Bayesian Hierarchical Media Mix Modeling. Google Research. (Hierarchical pooling across geographies; informative priors.) Google Services
