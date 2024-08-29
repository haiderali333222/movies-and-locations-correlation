import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# Load data from CSV files
location_data = pd.read_csv('data/location_popularity_final.csv')
movie_data = pd.read_csv('data/movie_popularity_final.csv')


print(location_data.head())
print(movie_data.head())
