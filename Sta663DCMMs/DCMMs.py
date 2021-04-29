import numpy as np
from pybats.analysis import analysis

## function to convert counts series to dcmm data
def to_dcmm(Y):
    """
    :param Y: 1D or 2D array of daily sales, across all items selected (size: T * #items)
    :return: List of 2 arrays of 0/1 indicator and counts
    """
    dcmm_data = []
    ber = Y>0
    dcmm_data.append(ber)
    poi = Y - 1
    poi[poi == -1] = int(np.mean(poi[poi > -1]))
    dcmm_data.append(poi)
    return dcmm_data



def dcmm_analysis(Y, X, forecast_start, forecast_end, prior_length,
             nsamps=500, model_prior = None, ntrend=1, mean_only = False,
          holidays=[], seasPeriods=[], seasHarmComponents=[],
          ret=['forecast'], rho = 1, delseas=[1] * 2, deltrend=[1] * 2, delregn=[1] * 2, **kwargs):

    """
    # Run updating + forecasting using a dcmm. Latent Factor option available
    :param Y: Array of daily sales or counts, across all series selected
    :param X: List of covariate arrays for the bernoulli DGLM and Poisson DGLM, one for each DGLM
    :param prior_length: number of datapoints to use for prior specification
    :param model_prior: a pre-specified model.
    :param forecast_start: day to start forecasting (beginning with 0)
    :param forecast_end:  day to end forecasting
    :param ntrend: Number of trend components in the model. 1 = local intercept , 2 = local intercept + local level
    :param holidays: List of Holidays or special events to be given a special indicator (from pandas.tseries.holiday)
    :param seasPeriods: A list of periods for seasonal effects (e.g. [7] for a weekly effect, where Y is daily data)
    :param seasHarmComponents: A list of lists of harmonic components for a seasonal period (e.g. [[1,2,3]] if seasPeriods=[7])
    :param ret: A list of values to return. Options include: ['model', 'model_trace', 'forecast']
    :param nsamps: Number of forecast samples to draw
    :param mean_only: True/False - return the mean only when forecasting, instead of samples. One array for each layer
    :param delseas: list of discount factors for seasonal effects
    :param deltrend: list of discount factors for trend
    :param delregn: list of discount factors for regressor
    :param rho: discount factor random effect
    :param **kwargs: key word arguments
    :return: List of arrays of forecasting samples. Each element of the list is one layer, in which there are nsamps of arrays for the period chosen to forecast.
    """
    np.random.seed(8)
    
    # check and potentially broadcast X, works only when one set of covariate is provided
    if len(X) == 1:
        X = X * 2

    ## convert and assign data
    Y_ber, Y_poi = to_dcmm(Y)

    if mean_only == False:
        ## forecast
        ber_samples, ber_mod, ber_coef = analysis(Y_ber, X[0], family='bernoulli', k=1, forecast_start=forecast_start, model_prior=model_prior,
                               forecast_end = forecast_end, prior_length = prior_length, nsamps = nsamps, ntrend=ntrend, mean_only = False,
                               holidays = holidays, seasPeriods = seasPeriods, seasHarmComponents = seasHarmComponents, ret = ['forecast', 'model', 'model_coef'],
                                    delseas = delseas[0], deltrend = deltrend[0], delregn = delregn[0], **kwargs)

        poi_samples, poi_mod, poi_coef = analysis(Y_poi, X[1], family='poisson', k=1, forecast_start = forecast_start, forecast_end = forecast_end, model_prior=model_prior,
                                                  prior_length = prior_length, nsamps = nsamps, ntrend=ntrend,
                               holidays = holidays, seasPeriods = seasPeriods, seasHarmComponents = seasHarmComponents, ret = ['forecast', 'model', 'model_coef'], rho = rho, mean_only = False,
                                    delseas = delseas[1], deltrend = deltrend[1], delregn = delregn[1], **kwargs)

        dcmm_samples = ber_samples * (poi_samples + 1)
        dcmm_samples = dcmm_samples[:, :, 0]
    else:
        ## mean
        ber_mean, ber_mod, ber_coef = analysis(Y_ber, X[0], family='bernoulli', k=1, forecast_start=forecast_start, model_prior=model_prior,
                               forecast_end=forecast_end, prior_length=prior_length, nsamps=nsamps, ntrend=ntrend,
                               holidays=holidays, seasPeriods=seasPeriods, seasHarmComponents=seasHarmComponents,
                               ret=['forecast', 'model', 'model_coef'], mean_only = True,
                               delseas=delseas[0], deltrend=deltrend[0], delregn=delregn[0], **kwargs)

        poi_mean, poi_mod, poi_coef = analysis(Y_poi, X[1], family='poisson', k=1, forecast_start=forecast_start, model_prior=model_prior,
                               forecast_end=forecast_end, prior_length=prior_length, nsamps=nsamps, ntrend=ntrend,
                               holidays=holidays, seasPeriods=seasPeriods, seasHarmComponents=seasHarmComponents,
                               ret=['forecast', 'model', 'model_coef'], rho=rho, mean_only = True,
                               delseas=delseas[1], deltrend=deltrend[1], delregn=delregn[1], **kwargs)

        dcmm_samples = ber_mean * (poi_mean + 1)
        dcmm_samples = dcmm_samples[0,:]


    dcmm_model = [ber_mod, poi_mod]
    dcmm_coef = [ber_coef, poi_coef]
    ## return results
    out = []
    for obj in ret:
        if obj == 'forecast':
            out.append(dcmm_samples)
        if obj == 'model':
            out.append(dcmm_model)
        if obj == 'model_coef':
            out.append(dcmm_coef)

    return out








