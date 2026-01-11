# Hierarchical Loss Weighting for Physics-Informed Neural Networks in Small-Data Regimes
## Abstract
Proxy models are widely employed in modern engineering to address problems with prohibitively high computational cost. 
Representative examples include uncertainty quantification tasks in petroleum engineering and CO$_2$ sequestration, 
where the direct approach requires a large number of numerical simulations of fluid flow in porous media. 
Such simulations are often infeasible in practice, particularly for high-resolution three-dimensional reservoir models.

A common strategy for reducing computational cost is to perform a limited number of fluid-flow simulations 
to generate training data for a proxy model. Owing to the expense of these simulations, 
the resulting datasets are typically small, which makes polynomial models, decision trees, 
and gradient-boosting methods the predominant choices for proxy modeling in this setting. 
In this work, we demonstrate that Physics-Informed Neural Networks (PINNs) are capable of changing this paradigm.

We conduct a series of numerical experiments showing that PINNs can be successfully trained on
relatively small datasets (on the order of 500 samples). 
Furthermore, in scenarios where the target function exhibits strong nonlinearities, 
PINNs consistently outperform conventional proxy models, 
highlighting their potential as an effective and scalable alternative for data-limited, 
physics-driven engineering applications.

## Results
Boosting and PCE can approximate systems with a low degree of nonlinearity quite well (1). However, if we
increase the system’s nonlinearity (as shown in the second test case by adding a nonlinear permeability field),
these algorithms can only achieve good accuracy with a large amount of data. For example, with 500 training
points, neither Boosting nor PCE were able to approximate the fluid propagation velocity in the reservoir, whereas
PINNs managed to do so with much higher accuracy.