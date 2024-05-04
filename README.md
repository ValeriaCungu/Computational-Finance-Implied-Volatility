# Computational-Finance-Implied-Volatility
This project is a culmination of efforts for the Computational Finance course. The objective was to explore three models, detailing their respective grids and deriving implied volatilities for call options.
The project delves into three models:

[Model 1]: Constant Elasticity of Variance Model
[Model 2]: Displaced Diffusion Model
[Model 3]: Heston Model

For each model, the grid (tn, kj) was constructed as follows:
- tn values: 1M, 2M, 3M, 6M, 12M, 18M
- kj values: 0.8, 0.9, 0.95, 0.975, 1, 1.025, 1.05, 1.1, 1.2

For each pair (tn, kj), the project computed the price of the associated call option and its implied volatility. The set of volatilities σ(tn, kj) represents the volatility surface implied by each model. 
Analytical solutions, including quasi-analytical ones, were available in literature for all models. The project meticulously compared Monte Carlo (MC) results with these analytical solutions to ascertain accuracy and efficacy.

