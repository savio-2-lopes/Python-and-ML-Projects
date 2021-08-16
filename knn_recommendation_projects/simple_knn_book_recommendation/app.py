import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

books = pd.read_csv('BX-Books.csv', sep=';', error_bad_lines=False, encoding="latin-1")
books.columns = ['ISBN', 'bookTitle', 'bookAuthor', 'yearOfPublication', 'publisher', 'imageUrlS', 'imageUrlM', 'imageUrlL']

users = pd.read_csv('BX-Users.csv', sep=';', error_bad_lines=False, encoding="latin-1")
users.columns = ['userID', 'Location', 'Age']

ratings = pd.read_csv('BX-Book-Ratings.csv', sep=';', error_bad_lines=False, encoding="latin-1")
ratings.columns = ['userID', 'ISBN', 'bookRating']

""" Dados de classificação """

print(ratings.shape)
print(list(ratings.columns))

""" Dados de livros """

print(books.shape)
print(list(books.columns))

""" Dados  dos usuário """

# Este conjunto de dados fornece informações
# demográficas ao usuário. Inclui 3 campos: 
# Identificação do usuário, localização e idade

print(users.shape)
print(list(users.columns))

""" Recomendações baseadas em contagem de classificação """

rating_count = pd.DataFrame(ratings.groupby('ISBN')['bookRating'].count())
rating_count.sort_values('bookRating', ascending=False).head()
print(rating_count.sort_values('bookRating', ascending=False).head())

# O livro com o ISBN recebeu a maioria das contagens de classificação

most_rated_books = pd.DataFrame(['0330299891', '0375404120', '0586045007', '9022906116', '9032803328'], index=np.arange(5), columns = ['ISBN'])
most_rated_books_summary=pd.merge(most_rated_books, books, on="ISBN")
print('\n', most_rated_books_summary)

""" Filtragem colaborativa usando KNN """

# KNN é um algoritmo de aprendizado de máquinas para 
# encontrar clusters de usuários semelhantes com base em classificações
# comuns de livros, e fazer previsões usando a classificando média dos 
# vizinhos mais próximos top-k.
# Por exemplo, apresentamos primeiramente classificações 
# em uma matriz com a matriz tendo uma linha para cada item (livro)
# e uma coluna para cada usuário.
# Em seguida, encontramos o item k que tem os vetores de enganjamento 
# do usuário mais semelhantes. 
# neste caso, vizinhos mais próximos do item id 5=[7, 4, 8, ...].
# Agora, vamos implementar KNN em nosso sistema de recomendação de livros
# A partiri do conjunto de dados original, estaremos apenas olhando para os 
# livros populares. Para descobrir quais livros são populares, combinamos dados de
# livros com dados de classificação.

combine_book_rating = pd.merge(ratings, books, on='ISBN')
columns=['yearOfPublication', 'publisher', 'bookAuthor', 'imageUrlS', 'imageUrlM', 'imageUrlL']
combine_book_rating = combine_book_rating.drop(columns, axis=1)
print(combine_book_rating.head())

# Em seguida, agrupamos por titulo de livros e criamos
# uma nova coluna para contagem total de classificação.

combine_book_rating=combine_book_rating.dropna(axis=0, subset=['bookTitle'])
book_ratingCount=(combine_book_rating.groupby(by=['bookTitle'])['bookRating'].count().reset_index().rename(columns={'bookRating': 'totalRatingCount'})[['bookTitle', 'totalRatingCount']])
print(book_ratingCount.head())

# Combinamos os dados de classificação 
# com os dados totais da contagem de classificação,
# isso nos dá exatamente o que precisamos para descobrir quais
# livros são populares e filtrar livros menos conhecidos.

rating_with_totalRatingCount = combine_book_rating.merge(book_ratingCount, left_on = 'bookTitle', right_on = 'bookTitle', how = 'left')
print(rating_with_totalRatingCount.head())

# Vejamos as estatísticas da contagem total de classificações:

pd.set_option('display.float_format', lambda x: '%.3f' % x)
print(book_ratingCount['totalRatingCount'].describe())

# O livro mediano foi classificado apenas uma vez. 
# Vamos olhar para o topo da distribuição

print(book_ratingCount['totalRatingCount'].quantile(np.arange(.9, 1, .01)))

# Cerca de 1% dos livros receberam 50 ou mais avaliações .
# Como temos tantos livros em nossos dados, vamos limitá-lo ao top 1%,
# e isso nos dará 2713 livros exclusivos.

popularity_threshold = 50
rating_popular_book = rating_with_totalRatingCount.query('totalRatingCount >= @popularity_threshold')
print(rating_popular_book.head())

""" Filtrar apenas para usuários nos EUA e Canadá """

# A fim de melhorar a velocidade da computação, e não esbarrar no problema
# "MemoryError", limitarei nossos dados de usuários àqueles nos EUA e canadá.
# E em seguida, combinar dados do usuários com os dados de classificação e dados totais de contagem de classificação.

combined = rating_popular_book.merge(users, left_on = 'userID', right_on = 'userID', how = 'left')
us_canada_user_rating = combined[combined['Location'].str.contains("usa|canada")]
us_canada_user_rating=us_canada_user_rating.drop('Age', axis=1)
print(us_canada_user_rating.head())

""" Implementação de KNN """

# Comvertemos nossa tabela em uma matriz 2D e preechemos os valores faltantes com zeros (já que calcularemos distâncias entre vetores de classificação).
# Em seguida, trasformamos os valores (classificações) do quadro de dados matricialem uma matriz esparsa para cálculos mais eficientes

""" Encontrand os vizinhos mais próximos """

# Usamos algoritmos não supervisionados com sklearn.neighbors.
# O algoritmo que usamos para calclar os vizinhos mais próximos é "bruto",
# e especificamos "metric=cosine" para que o algoritmo calcule a semelhança cossine entre
# vetores de classificação. 
# Finalmente, encaixamos o modelo:

us_canada_user_rating = us_canada_user_rating.drop_duplicates(['userID', 'bookTitle'])
us_canada_user_rating_pivot = us_canada_user_rating.pivot(index = 'bookTitle', columns = 'userID', values = 'bookRating').fillna(0)
us_canada_user_rating_matrix = csr_matrix(us_canada_user_rating_pivot.values)

model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
model_knn.fit(us_canada_user_rating_matrix)
print(model_knn.fit(us_canada_user_rating_matrix))

""" Teste nosso modelo e faça algumas recomendações """

# Nesta etapa, o algoritmo KNN mede a distância para determinar a 
# "proximidade" das instâncias.
# Em seguida, classifica uma instância ao encontrar seus vizinhos mais
# próximos, e escolher a classe mais popular entre os vizinhos

query_index = np.random.choice(us_canada_user_rating_pivot.shape[0])
distances, indices = model_knn.kneighbors(us_canada_user_rating_pivot.iloc[query_index, :].reshape(1, -1), n_neighbors = 6)

for i in range(0, len(distances.flatten())):
  if i == 0:
    print('Recomendação para {}: \n'.format(us_canada_user_pivot.index[query_index]))

  else:
    print('{0}: {1}, por uma distância de {2}:'.format(i, us_canada_user_rating_pibot.index[indices.flatten()[i]], distances.flatten()[i]))