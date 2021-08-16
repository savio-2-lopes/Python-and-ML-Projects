# importando bibliotecas
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# Obtendo os dados da biblioteca

movies_df=pd.read_csv('movies.csv', usecols=['movieId', 'title'], dtype={'movieId': 'int32', 'title': 'str'})
rating_df=pd.read_csv('ratings.csv', usecols=['userId', 'movieId', 'rating'], dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'})

print('\nExibindo dados dos filmes\n', movies_df.head())
print('\nExibindo dados das classificações\n', movies_df.head())

df=pd.merge(rating_df, movies_df, on="movieId")
print(df.head())

combine_movie_rating = df.dropna(axis=0, subset=['title'])
movie_ratingCount = (combine_movie_rating.groupby(by=['title'])['rating'].count().reset_index().rename(columns={'rating': 'totalRatingCount'})[['title', 'totalRatingCount']])
print('\n', movie_ratingCount.head())

rating_with_totalRatingCount=combine_movie_rating.merge(movie_ratingCount, left_on='title', right_on='title', how='left')
print('\n', rating_with_totalRatingCount.head())

pd.set_option('display.float_format', lambda x: '%.3f' % x)
print(movie_ratingCount['totalRatingCount'].describe())

populaty_threshold=50
rating_popular_movie=rating_with_totalRatingCount.query('totalRatingCount >= @populaty_threshold')
rating_popular_movie.head()

rating_popular_movie.shape

movie_features_df=rating_popular_movie.pivot_table(index='title', columns='userId', values='rating').fillna(0)
print('\n', movie_features_df.head())

movie_features_df_matrix=csr_matrix(movie_features_df.values)
model_knn=NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(movie_features_df_matrix)

print(movie_features_df.shape)

query_index=np.random.choice(movie_features_df.shape[0])
print(query_index)

distances, indices = model_knn.kneighbors(movie_features_df.iloc[query_index,:].values.reshape(1, -1), n_neighbors=6)

print('\n', movie_features_df.head())

for i in range(0, len(distances.flatten())):
  if i == 0:
    print('Recomendação para {0}: \n'.format(movie_features_df.index[query_index]))

  else:
    print('{0}:{1}, de distância de {2}'.format(i, movie_features_df.index[indices.flatten()[i]], distances.flatten()[i]))