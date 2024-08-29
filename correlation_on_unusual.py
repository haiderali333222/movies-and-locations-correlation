import pandas as pd
import numpy as np


# Load data from CSV files
locations_df = pd.read_csv('data/location_popularity_final.csv')
movies_df = pd.read_csv('data/movie_popularity_final.csv')

# Step 2: Data Cleaning and Preparation

# Reshape the movie data
movies_long_df = pd.melt(
    movies_df, 
    id_vars=['film_id', 'city', 'country', 'date_range', 'name'], 
    var_name='date', 
    value_name='movie_popularity'
)


print(movies_long_df)
# Convert the 'date' column in movies_long_df to datetime
movies_long_df['date'] = pd.to_datetime(movies_long_df['date'], format='%d/%m/%Y')

# Separate the first column from date columns
location_name_column = locations_df.columns[0]  # This should be 'name'
date_columns = locations_df.columns[1:]  # These should be the date columns

# Ensure the first column is named 'name' or rename it if needed
if location_name_column != 'name':
    locations_df.rename(columns={location_name_column: 'name'}, inplace=True)

# Convert the date columns in locations_df to datetime, ignoring the first column
locations_df.columns = ['name'] + [pd.to_datetime(col, errors='coerce', format='%Y-%m-%d') for col in date_columns]

# Check the columns again after renaming
print("Columns in locations_df after processing:", locations_df.columns)

# Reshape locations_df to long format
locations_long_df = pd.melt(
    locations_df, 
    id_vars=['name'], 
    var_name='date', 
    value_name='location_popularity'
)

# Convert 'date' column to datetime in locations_long_df
locations_long_df['date'] = pd.to_datetime(locations_long_df['date'])

# Drop rows in locations_long_df where 'date' is NaT (Not a Time)
locations_long_df = locations_long_df.dropna(subset=['date'])

# Convert 'date' columns in both dataframes to datetime (ensuring type consistency)
movies_long_df['date'] = pd.to_datetime(movies_long_df['date'])
locations_long_df['date'] = pd.to_datetime(locations_long_df['date'])

# Step 3: Merge the movies and location data on city/location name and date
merged_df = pd.merge(
    movies_long_df, 
    locations_long_df, 
    left_on=['city', 'date'], 
    right_on=['name', 'date'],
    how='inner'
)


# Step 4: Calculate the correlation
correlation = merged_df[['movie_popularity', 'location_popularity']].corr().iloc[0, 1]

print(f"Correlation between movie popularity and location popularity: {correlation}")

# Step 5: Analyze impact of movies on location popularity
# Calculate the difference in location popularity before and after each movie's release
impact_df = merged_df.copy()
impact_df['year'] = impact_df['date'].dt.year

# Create a function to calculate the impact of each movie
def calculate_impact(group):
    # Assuming movie's popularity period starts 2 years after the first year of data for simplicity
    pre_movie_pop = group.loc[group['year'] < group['year'].min() + 2, 'location_popularity'].mean()
    post_movie_pop = group.loc[group['year'] >= group['year'].min() + 2, 'location_popularity'].mean()
    impact = post_movie_pop - pre_movie_pop
    return pd.Series({'impact': impact, 'pre_movie_pop': pre_movie_pop, 'post_movie_pop': post_movie_pop})

# Correct grouping to use 'name_y' which represents location names
movie_impact_df = impact_df.groupby(['film_id', 'name_y']).apply(calculate_impact).reset_index()

# Sort movies by impact and get the top 300 movies
top_300_movies = movie_impact_df.sort_values(by='impact', ascending=False).head(300)

# Calculate the correlation for top 300 movies
correlation = top_300_movies['impact'].corr(top_300_movies['pre_movie_pop'])
print(f"Correlation between impact and pre-movie popularity for top 300 movies: {correlation}")

# Output the top 300 movies with impact on locations
top_300_movies[['film_id', 'name_y', 'impact', 'pre_movie_pop', 'post_movie_pop']].to_csv('top_300_movies_impact.csv', index=False)