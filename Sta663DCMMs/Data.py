import numpy as np
from numpy.random import binomial, poisson, normal
import pandas as pd
import os



## simulate sales data
np.random.seed(8)
promotion = binomial(1, 0.4, 112)
sales = (poisson(3, 112)+1)*promotion
price = np.round(normal(5,0.2,112) - normal(2.5, 0.5, 112)*promotion,2)

sim_sales = pd.DataFrame({'sales':sales, 'promotion':promotion, 'price':price})
data_dir = os.path.dirname(os.path.abspath(__file__)) + '/Examples_data/'
sim_sales.to_csv(data_dir+"sim_data.csv")

## function to load simulated data
def load_sim_data():
    """load simulated data"""
    data_dir = os.path.dirname(os.path.abspath(__file__)) + '/Examples_data/'
    return pd.read_csv(data_dir+"sim_data.csv")

## function to load real 3 pointer data of Lebron James
def load_james_three():
    """load Lebron James 3PM data"""
    seasons = ["12-13", "16-17", "17-18"]
    data_dir = os.path.dirname(os.path.abspath(__file__)) + '/Examples_data/'
    return [pd.read_csv(data_dir+"James" + season + ".csv") for season in seasons]
