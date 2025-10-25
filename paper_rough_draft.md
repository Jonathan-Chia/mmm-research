# Abstract

Marketing Mix Modeling (MMM) has evolved from its roots in econometrics and time-series analysis into a sophisticated discipline that blends marketing theory with advanced statistical modeling [1](https://www.worldscientific.com/doi/epdf/10.1142/9789811272233_0005). Early MMMs relied on simple linear regressions to estimate channel effects, but progress in Bayesian inference and computation has driven a transition toward Bayesian models capable of capturing uncertainty, heterogeneity, and nonlinear effects such as adstock and saturation. Recent developments have focused on addressing critical issues in causality, unified measurement across marketing channels, and feature selection, all of which have profound implications for interpreting marketing effectiveness.

This paper investigates one of the most persistent and overlooked challenges in MMMs—causal mis-specification. Using simulated data, we demonstrate how failing to correctly account for causal dependencies (e.g., treating intermediate variables such as branded search as independent drivers) can lead to significant bias in parameter estimates. We compare results from a naïve MMM implementation, similar to open-source frameworks like PyMC Marketing and Google Meridian, with a more causally accurate model and explore whether incorporating incrementality-based priors can partially correct these errors.

Our results show that ignoring the underlying causal structure can substantially distort marketing effect estimates, leading to misguided optimization decisions. However, informed priors can mitigate some of these distortions. We conclude by discussing emerging approaches—such as causal graphical models, time-varying Bayesian structures, and hybrid frameworks like CausalMMM—that represent promising directions for building more accurate and interpretable marketing measurement systems.

# Introduction

The rest of this paper is organized as follows. In the [Background](#background) section, we discuss the history of Bayesian MMMs, how to code a Bayesian MMM, and then the core issue of causal mis-specification of funnel effects. In the [Methodology](#methodology) section, we explain our data generating process where the simulated data has upper funnel channel impacts on lower funnel channels. From there, we show how we set up different MMMs to retrieve our parameters. In the [Results](#results) section, we see that better priors can help mitigate causal issues. In the [Conclusion](#conclusion) section, we discuss future developments. 

# Background

MMM research started primarily in the 1960s-1970s as scholars and practicioners sought to understand how product, price, promotion, and distribution interact and influence performance (Borden, 1964; McCarthy, 1978). Early models used aggregate data in regression-based frameworks. Bayesian approaches to MMM gained momentum in the late 2010s thanks to advances in sampling computation and newer research including key research from Google on carryover and shape effects (Jin, Wang, Sun, Chan, Koehler 2017) as well as hierarchical modeling (Sun, Wang, Jin, Chan, Koehler 2016). Bayesian MMMs help mitigate issues with multicollinearity, small data but high parameters, and uncertainty propogation. 

The next sub-sections demonstrates the development from Bayesian Linear Regression to Bayesian MMMs:

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

## Bayesian MMM

Bayesian MMM adds some marketing specific adjustments to Bayesian Linear regression models: 

$$ y_t = \sum_{i=1}^m \beta_i s_i(c_i(x_{t-l+1,i},...,x_{t,i}; \alpha_i); k_i, \lambda_i) + \sum_{j=1}^n \gamma_j z_{t,j} + \epsilon_t $$

The $s_i$ and $c_i$ are the shape and carryover transformations (Jin, Wang, Sun, Chan, & Koehler 2017), and the $\gamma_j$ are control variables. The shape transformation models the diminishing returns of advertising: as an audience continues to see ads, eventually there's no more people to convert. Meanwhile, the carryover tranformation models the delayed effect of advertising. Lastly, the control variables help account for known confounders.

Here's the implementation:

```python
with pm.Model() as model:

    # ----------------------
    # Priors
    # ----------------------
    beta_x0 = pm.Normal("beta_x0", 0.5, sigma=0.2)          # intercept
    beta_t = pm.HalfNormal("beta_t", sigma=0.02)        # trend slope

    # Media spend coefficients
    beta_x1 = pm.HalfNormal("beta_x1", sigma=0.5)
    beta_x2 = pm.HalfNormal("beta_x2", sigma=0.5)

    # Seasonality - Fourier terms 
    beta_fourier = pm.Laplace("beta_fourier", mu=0.0, b=0.2, shape=fourier_terms.shape[1])

    # Control events
    gamma_control1 = pm.Normal("gamma_control1", 0, 0.05)
    gamma_control2 = pm.Normal("gamma_control2", 0, 0.05)

    # Noise
    sigma = pm.HalfNormal("sigma", sigma=2)

    # Adstock and saturation priors
    adstock_alpha1 = pm.Beta("adstock_alpha1", alpha=2, beta=2)
    saturation_lam1 = pm.Gamma("saturation_lam1", alpha=3, beta=1)
    saturation_beta1 = pm.HalfNormal("saturation_beta1", sigma=2)

    adstock_alpha2 = pm.Beta("adstock_alpha2", alpha=2, beta=2)
    saturation_lam2 = pm.Gamma("saturation_lam2", alpha=3, beta=1)
    saturation_beta2 = pm.HalfNormal("saturation_beta2", sigma=2)

    # ----------------------
    # Forward pass (applies the transformations)
    # ----------------------
    x1_transformed = pm.Deterministic(
        "x1_transformed",
        forward_pass(scaled_data['x1'].values, adstock_alpha1, saturation_lam1, saturation_beta1)
    )

    x2_transformed = pm.Deterministic(
        "x2_transformed",
        forward_pass(scaled_data['x2'].values, adstock_alpha2, saturation_lam2, saturation_beta2)
    )
    
    # ----------------------
    # Deterministic Variables (can pull these from model later for reporting)
    # ----------------------
    channel1_contribution = pm.Deterministic("channel1_contribution", beta_x1 * x1_transformed)
    channel2_contribution = pm.Deterministic("channel2_contribution", beta_x2 * x2_transformed)
    fourier_contribution = pm.Deterministic("fourier_contribution", pm.math.dot(fourier_terms, beta_fourier))
    trend_contribution = pm.Deterministic("trend_contribution", beta_t * scaled_data['t'].values)
    control1_contribution = pm.Deterministic("control1_contribution", gamma_control1 * scaled_data['event_1'].values)
    control2_contribution = pm.Deterministic("control2_contribution", gamma_control2 * scaled_data['event_2'].values)

    # ----------------------
    # Expected value of outcome
    # ----------------------
    mu = pm.Deterministic(
        "mu",
        beta_x0
        + channel1_contribution
        + channel2_contribution
        + fourier_contribution
        + trend_contribution
        + control1_contribution
        + control2_contribution
    )

    # Likelihood
    Ylikelihood = pm.Normal("Ylikelihood", mu, sigma, observed=scaled_data['y'].values)
```

Unlike in regular MMMs, the shape and carryover parameters are random variables with probability distributions, and priors can be set for these parameters. Prior knowledge or domain expertise can be used to create very useful carryover and shape priors that improve model accuracy. For example, time to conversion estimates pulled from marketing attribution data can be used to estimate carryover priors. Similarly, ROAS estimates at different level of spends can be used to estimate shape priors. Or, domain experts can help provide simple priors. 

## Funnel Effects

One issue in regular MMMs that Bayesian MMMs might be able to mitigate is the issue of funnel effects (Chan and Perry 2017). Funnel effects are biases that occur when marketing channels influence one another across stages of the customer journey—where upper-funnel activities affect both lower-funnel channels and final outcomes. For instance, a linear TV ad may increase paid search volume as well as directly drive sales. In this case, TV ads impact sales both directly and indirectly, but naive MMMs that assume independence between TV and search fail to capture this causal structure. 

# Methodology

With smarter priors or with causally accurate modeling, the Bayesian MMM should be able to better handle these funnel effects.

We start with a simple premise - paid search is an intermediary lower funnel channel:
![alt text](assets/img/causal_dag.png)

We will investigate how bad ignoring this causality can be for MMMs using simulated data that reflects this real world causality. Because the data is simulated, we will know the true parameters. A good MMM should be able to recover these true parameters. 

## Data Generating Process

In our generated data, 70% of display ads impressions flows directly to sales, while 30% flow to paid search. Thus, there are three impacts:

* Display ads direct impact on sales
* Display and Search ads combined impact sales
* Search ads direct impact on sales

The data generating process is as follows:

1. Generate 105 weeks (about 2 years).
2. Add in the media spends for display and search. Display ads will have more spends since most companies would spend more in upper funnel marketing channels. 

![alt text](assets/img/media_spends.png)

3. Apply a carryover transformation to the spends.

- Display $\alpha$ = 0.4
- Search $\alpha$ = 0.2
- Display has a stronger carryover effect. 

4. Apply a saturation transformation to the spends.

- Display $\lambda$ = 4
- Search $\lambda$ = 2
- Display saturates faster because it is a demand creation channel that builds awareness but quickly reaches the same audience with repeated impressions. Meanwhile, paid search is a demand capture channel. Because users are actively searching for related terms, each incremental dollar can often reach new high-intent users - up to a point. 

![alt text](/assets/img/adstock_saturation.png)

5. Add seasonality and trend

![alt text](/assets/img/seasonality_trend.png)

6. Add two big promotional events on May 13, 2018 and September 14, 2019.

7. Combine all of these to generate simulated sales.



When we model, we will assume we don't know this 70/30 split and use the model to recover it; the data fed to the model will only have display impressions and search impressions. 

## Naive Model 

First, we will build a naive MMM (replicating base model frameworks that we see in [PyMC Marketing](https://www.pymc-marketing.io/en/latest/)) and see how close it can get to the true parameters.

Here's an example of PyMC's base model framework:

![alt text](assets/img/pymc_example_dag.png)

In this above framework, marketing channels are treated independently.

SHOW THE NAIVE MODEL CODE

From there, we will build a more causally accurate MMM and see how close it can get to the true parameters. 

## Causal Model

SHOW THE CAUSAL MODEL CODE

## Naive Model with Better Priors

Lastly, we will examine if adding an incrementality estimate as a prior into the naive model can help push it in the right direction.

TODO

# Results

# Conclusion



Solutions for these causal issues are already being implemented in industry and academia. Sellforte and Recast, MMM vendors, sell a causal MMM solution. ____ (2024) propose a novel MMM that estimates the causal structure automatically. 

For companies that are considering buying vs. building their own MMM, we recommend buying from these top vendors, or at least partnering with them to help build an in-house MMM, because most open-source MMM tools don't provide strong guidance to mitigate these causal issues. 

# References

- Borden, N. H. (1964). The concept of the marketing mix. Journal of advertising research, 4 (2),
2–7.
- Jin, Y., Wang, Y., Sun, Y., Chan, D. & Koehler, J. (2017). Bayesian methods for media mix
modeling with carryover and shape effects. research.google.com.
- McCarthy, J. E. (1978). Basic marketing: a managerial approach (6th ed.). Homewood, Il: R.D.
Irwin.
- Sun, Y., Wang, Y., Jin, Y., Chan, D., & Koehler, J. (2016–2017). Geo-level Bayesian Hierarchical Media Mix Modeling. Google Research. (Hierarchical pooling across geographies; informative priors.) Google Services
