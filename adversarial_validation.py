import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn import model_selection
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


melb = pd.read_csv("listings_Melbourne.csv")
sydney = pd.read_csv("listings_Sydney.csv")


def pre_processer(df:pd.DataFrame, max_price:float, min_city_counts:int, columns:list):
    
    # Convert price into float
    df.price = df.price.str.slice(start=1, stop=-3).str.replace(",","").astype("float")
    
    # Remove outliers in price
    df = df.loc[df.price < 2000,]
    
    # Remove rare examples of city column
    city_counts = pd.DataFrame(df.city.value_counts())
    df = df.loc[df.city.isin(city_counts.loc[city_counts.city>200,].index.to_list()),]
    
    # Remove missing values in bedrooms and turn into integer
    df = df.loc[df.bedrooms.notna(),]
    df.bedrooms = df.bedrooms.astype("int").copy()
    
    # Select the required columns
    df = df[columns].reset_index(drop=True)
    
    return df