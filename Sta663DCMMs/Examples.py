import numpy as np
from numpy.random import binomial, poisson, normal
import pandas as pd
from .Data import load_sim_data, load_james_three
from .DCMMs import to_dcmm, dcmm_analysis
import matplotlib.pyplot as plt

# function to run simulated example
def sim_example():
    """run simulated example"""
    ## load simulated sales data
    sim_sales = load_sim_data()

    ## premodeling process
    Y = sim_sales.loc[:,'sales'].values
    X = [sim_sales.loc[:,['promotion', 'price']].values]
    prior_length = 8
    nsamps = 500
    forecast_start = 60
    forecast_end = 111
    rho = 0.7

    ## fitting the model
    samples, mod, coef = dcmm_analysis(Y, X, prior_length = prior_length, nsamps = nsamps,
                                       forecast_start=forecast_start, forecast_end=forecast_end,
                                       mean_only=False, rho = rho, ret = ['forecast', 'model', 'model_coef'])
    ## obtain the mean, median and bounds of 90% credible interval
    avg = dcmm_analysis(Y, X, prior_length = prior_length, nsamps = nsamps,
                                       forecast_start=forecast_start, forecast_end=forecast_end,
                                       mean_only=True, rho = rho, ret = ['forecast'])[0]
    med = np.median(samples, axis=0)

    upper = np.quantile(samples, 0.95, axis=0)
    lower = np.quantile(samples, 0.05, axis=0)

    ## calculate coverage
    coverage = np.logical_and(Y[60:] <= upper, Y[60:] >= lower).sum()/52

    ## make the plot
    forecast_period = np.linspace(forecast_start, forecast_end, 52)
    fig, ax = plt.subplots(figsize = (12,4))
    ax.plot(forecast_period, avg, '.y', label='Mean')
    ax.plot(forecast_period, med, '.r', label = 'Median')
    ax.plot(forecast_period, upper, '-b', label='90% Credible Interval')
    ax.plot(forecast_period, lower, '-b', label='90% Credible Interval')
    ax.plot(Y, '.k', label="Observed")
    ax.set_title("DCMM on Simulated Data")
    ax.set_ylabel("Sales")
    ax.annotate("Coverage rate: "+str('%1.3f' % coverage), xy=(0,0), xytext = (80, 8))
    plt.legend()
    #fig.savefig("Examples_plots/sim.png")
    plt.show()




# function to run the real example
def real_example(data, rho):
    """run real example"""
    ## premodeling process
    Y = data.loc[:, 'three_made'].values
    X = [data.loc[:, ['home', 'minutes']].values]
    prior_length = 4
    nsamps = 500
    forecast_start = 40
    forecast_end = len(Y)-1

    ## extract df info
    s = sorted(set([d[:4] for d in data.date]))

    ## fitting the model
    samples, mod, coef = dcmm_analysis(Y, X, prior_length=prior_length, nsamps=nsamps,
                                       forecast_start=forecast_start, forecast_end=forecast_end,
                                       mean_only=False, rho=rho, ret=['forecast', 'model', 'model_coef'])
    ## obtain the mean, median and bounds of 90% credible interval
    avg = dcmm_analysis(Y, X, prior_length=prior_length, nsamps=nsamps,
                        forecast_start=forecast_start, forecast_end=forecast_end,
                        mean_only=True, rho=rho, ret=['forecast'])[0]
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
    ax.set_title("DCMM on Lebron James' Three-pointer Made (Season" + s[0] + "-" + s[1] + ")")
    ax.set_ylabel("Three-pointer Made Per Game")
    ax.annotate("Coverage rate: " + str('%1.3f' % coverage), xy=(0, 0), xytext=(50, max(Y)))
    plt.legend()
    #fig.savefig("Examples_plots/"+"James3PM-(Season" + s[0] + "-" + s[1] + ").png")
    plt.show()



## load real three-pointer shooting data
james_three = load_james_three()
rhos = [0.9]*3

## run the real examples
for season, rho in zip(james_three, rhos):
    real_example(season, rho)

