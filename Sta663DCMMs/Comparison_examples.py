import statsmodels.api as sm
from statsmodels.formula.api import glm
import numpy as np
import matplotlib.pyplot as plt
from .Data import load_sim_data, load_james_three
from pybats.analysis import analysis

## fit a static Poisson GLM to Lebron's data
def glm_example(data):
    """fit a static Poisson GLM to Lebron's 3-pointer data using statsmodels glm and
    show confidence interval of fitted response"""

    ## extract df info
    s = sorted(set([d[:4] for d in data.date]))
    n = data.shape[0]

    # fit glm
    model = glm('three_made~home+minutes', data = data, family = sm.families.Poisson()).fit()
    coef = model.params.values
    covariates = np.c_[np.ones(n), data.loc[:,['home', 'minutes']].values]
    pred = np.exp(covariates@coef.reshape(3,1))
    coef_int = model.conf_int(alpha=0.1).values
    pred_int = np.exp(covariates@coef_int)
    upper = pred_int[:,1]
    lower = pred_int[:,0]

    # plot the fitted model, confidence interval and data
    fig, ax = plt.subplots(figsize = (12,4))
    ax.plot(data.loc[:,'three_made'], '.k', label = 'Observed')
    ax.plot(pred, '.r', label = 'Fitted')
    ax.plot(upper, '-b', label = '90% Confidence Interval')
    ax.plot(lower, '-b', label = '90% Confidence Interval')
    ax.set_title("Poisson GLM on Lebron James' Three-pointer Made (Season" + s[0] + "-" + s[1] + ")")
    ax.set_ylabel("Three-pointer Made Per Game")
    plt.legend()
    fig.savefig("Examples_plots/" + "PoissonGLM-James3PM-(Season" + s[0] + "-" + s[1] + ").png")
    plt.show()



## fit a Poisson DGLM to Lebron's 3-pointer data
def dglm_example(data, rho):
    """fit a Poisson DGLM to Lebron's 3-pointer data, using analysis from pybats"""
    Y = data.loc[:, 'three_made'].values
    X = data.loc[:, ['home', 'minutes']].values
    prior_length = 4
    nsamps = 500
    forecast_start = 40
    forecast_end = len(Y) - 1

    ## extract df info
    s = sorted(set([d[:4] for d in data.date]))

    samples, mod, coef = analysis(Y, X, prior_length=prior_length, nsamps=nsamps, family = 'poisson', k=1,
                                       forecast_start=forecast_start, forecast_end=forecast_end,
                                       mean_only=False, rho=rho, ret=['forecast', 'model', 'model_coef'])
    ## obtain the mean, median and bounds of 90% credible interval
    avg = analysis(Y, X, prior_length=prior_length, nsamps=nsamps, family='poisson', k=1,
                        forecast_start=forecast_start, forecast_end=forecast_end,
                        mean_only=True, rho=rho, ret=['forecast'])[0]
    samples = samples[:,:,0]
    avg = avg[:,0]

    med = np.median(samples, axis=0)

    upper = np.quantile(samples, 0.95, axis=0)
    lower = np.quantile(samples, 0.05, axis=0)

    ## calculate coverage
    forecast_period = np.linspace(forecast_start, forecast_end, forecast_end - forecast_start + 1)
    coverage = np.logical_and(Y[40:] <= upper, Y[40:] >= lower).sum() / len(forecast_period)

    ## make the plot
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(forecast_period, avg, '.y', label='Mean')
    ax.plot(forecast_period, med, '.r', label='Median')
    ax.plot(forecast_period, upper, '-b', label='90% Credible Interval')
    ax.plot(forecast_period, lower, '-b', label='90% Credible Interval')
    ax.plot(Y, '.k', label="Observed")
    ax.set_title("Poisson DGLM on Lebron James' Three-pointer Made (Season" + s[0] + "-" + s[1] + ")")
    ax.set_ylabel("Three-pointer Made Per Game")
    ax.annotate("Coverage rate: " + str('%1.3f' % coverage), xy=(0, 0), xytext=(50, max(Y)))
    plt.legend()
    fig.savefig("Examples_plots/" + "PoissonDGLM-James3PM-(Season" + s[0] + "-" + s[1] + ").png")
    plt.show()
