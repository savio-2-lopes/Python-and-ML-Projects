import os
import time

# Importando bibliotecas
import math
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# Importando utilitários
from fuzzywuzzy import fuzz

# Carregando dataset
df_movies = pd.read_csv(
    'movies.csv',
    usecols=['movieId', 'title'],
    dtype={'movieId': 'int32', 'title': 'str'})

df_ratings = pd.read_csv(
    'ratings.csv',
    usecols=['userId', 'movieId', 'rating'],
    dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'})

num_users = len(df_ratings.userId.unique())
num_items = len(df_ratings.movieId.unique())
print('Existem {} usuários exclusivos e {} filmes exclusivos neste conjunto de dados '.format(num_users, num_items))

# Obtendo contagem
df_ratings_cnt_tmp = pd.DataFrame(df_ratings.groupby('rating').size(), columns=['count'])
df_ratings_cnt_tmp

# há muito mais contagens na classificação de zero 
total_cnt = num_users * num_items
rating_zero_cnt = total_cnt - df_ratings.shape[0]

# anexar contagens de classificação zero a df_ratings_cnt 
df_ratings_cnt = df_ratings_cnt_tmp.append(
    pd.DataFrame({'count': rating_zero_cnt}, index=[0.0]),
    verify_integrity=True,
).sort_index()
print(df_ratings_cnt)

# adicionar contagem de log 
df_ratings_cnt['log_count'] = np.log(df_ratings_cnt['count'])
print(df_ratings_cnt)

print(df_ratings.head())

df_movies_cnt = pd.DataFrame(df_ratings.groupby('movieId').size(), columns=['count'])
print(df_movies_cnt.head())

df_movies_cnt['count'].quantile(np.arange(1, 0.6, -0.05))
# Filtrando dados
popularity_thres = 50
popular_movies = list(set(df_movies_cnt.query('count >= @popularity_thres').index))
df_ratings_drop_movies = df_ratings[df_ratings.movieId.isin(popular_movies)]
print('forma dos dados de avaliações originais : ', df_ratings.shape)
print('forma dos dados de classificação após o lançamento de filmes impopulares: ', df_ratings_drop_movies.shape)

# Obter o número de avaliações dadas por cada usuário 

df_users_cnt = pd.DataFrame(df_ratings_drop_movies.groupby('userId').size(), columns=['count'])
df_users_cnt.head()
df_users_cnt['count'].quantile(np.arange(1, 0.5, -0.05))

# Filtrando dados

ratings_thres = 50
active_users = list(set(df_users_cnt.query('count >= @ratings_thres').index))
df_ratings_drop_users = df_ratings_drop_movies[df_ratings_drop_movies.userId.isin(active_users)]
print('dados de classificação de forma de origem: ', df_ratings.shape)
print('forma de dados de classificação após descartar filmes impopulares e usuários inativos: ', df_ratings_drop_users.shape)

# Dinamizar e criar matriz de usuário de filme 

movie_user_mat = df_ratings_drop_users.pivot(index='movieId', columns='userId', values='rating').fillna(0)

# Criar mapeador do título do filme para o índice 

movie_to_idx = {
    movie: i for i, movie in 
    enumerate(list(df_movies.set_index('movieId').loc[movie_user_mat.index].title))
}

# Transforma a matriz em matriz esparsa scipy 

movie_user_mat_sparse = csr_matrix(movie_user_mat.values)

# Definir modelo

model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
model_knn.fit(movie_user_mat_sparse)

"""
Retorna a correspondência mais próxima por meio de proporção difusa. 
Se nenhuma correspondência for encontrada, retorna None 
"""

def fuzzy_matching(mapper, fav_movie, verbose=True):
    match_tuple = []
    # Obter itens correspondentes 
    for title, idx in mapper.items():
        ratio = fuzz.ratio(title.lower(), fav_movie.lower())
        if ratio >= 60:
            match_tuple.append((title, idx, ratio))
    
    # Ordenar dados
    match_tuple = sorted(match_tuple, key=lambda x: x[2])[::-1]
    if not match_tuple:
        print('Ops! Nenhuma correspondência encontrada ')
        return

    if verbose:
        print('Possíveis correspondências encontradas em nosso banco de dados: {0}\n'.format([x[0] for x in match_tuple]))
    return match_tuple[0][1]

"""
Retornar as principais recomendações de filmes semelhantes 
com base na entrada do filme do usuário 
"""

def make_recommendation(model_knn, data, mapper, fav_movie, n_recommendations):
    model_knn.fit(data)

    # Obter índice de entrada do filme 
    print('Você introduziu o filme:', fav_movie)
    idx = fuzzy_matching(mapper, fav_movie, verbose=True)
    print('Sistema de recomendação começa a fazer inferências ')
    print('......\n')
    distances, indices = model_knn.kneighbors(data[idx], n_neighbors=n_recommendations+1)

    # obter lista de idx bruta de recomendações 
    raw_recommends = \
        sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())), key=lambda x: x[1])[:0:-1]

    # obter mapeador reverso 
    reverse_mapper = {v: k for k, v in mapper.items()}

    # Exibindo recomendações
    print('Recomendado para {}:'.format(fav_movie))
    for i, (idx, dist) in enumerate(raw_recommends):
        print('{0}: {1}, por uma distância de {2}'.format(i+1, reverse_mapper[idx], dist))

my_favorite = 'Ant-Man and the Wasp'

make_recommendation(
    model_knn=model_knn,
    data=movie_user_mat_sparse,
    fav_movie=my_favorite,
    mapper=movie_to_idx,
    n_recommendations=10)

# calcular o número total de entradas na matriz filme-usuário 
num_entries = movie_user_mat.shape[0] * movie_user_mat.shape[1]

# calcular o número total de entradas com valores zero 
num_zeros = (movie_user_mat==0).sum(axis=1).sum()

# calcular a proporção do número de zeros para o número de entradas 
ratio_zeros = num_zeros / num_entries
print('Há cerca de {:.2%} das avaliações em nossos dados em falta '.format(ratio_zeros))

