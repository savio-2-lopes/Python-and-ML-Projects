# Importando bibliotecas

import pandas as pd
import numpy as np

from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# Dados do livro
books = pd.read_csv('BX-Books.csv', sep=';', error_bad_lines=False, encoding="latin-1")
books.columns = ['ISBN', 'bookTitle', 'bookAuthor', 'yearOfPublication', 'publisher', 'imageUrlS', 'imageUrlM', 'imageUrlL']

# Dados do usuário
users = pd.read_csv('BX-Users.csv', sep=';', error_bad_lines=False, encoding="latin-1")
users.columns = ['userID', 'Location', 'Age']

# Dados do usuário
ratings = pd.read_csv('BX-Book-Ratings.csv', sep=';', error_bad_lines=False, encoding="latin-1")
ratings.columns = ['userID', 'ISBN', 'bookRating']

""" Filtragem colaborativa usando KNN """

"""
kNN é um algoritmo de aprendizado de máquina para encontrar clusters de
usuários semelhantes com base em classificações comuns de livros, e fazer
previsões usando a classificação média dos vizinhos mais próximos top-k.
Por exemplo, apresentamos primeiramente classificações em uma matriz com a 
matriz tendo uma linha para cada item (livro) e uma coluna para cada usuário.
Em seguida, encontramos o item k que tem os vetores de engajamento do usuário
mais semelhantes. Neste caso, vizinhos mais próximos do item id 5=[7, 4, 8, ...].
Agora, vamos implementar kNN em nosso sistema de recomendações de livros.
A partir do conjunto de dados original, estaremos apenas olhando para os livros
populares. Para descobrir quais livros são populares, combinamos dados de livros com
dados de classificação.
"""

combine_book_rating = pd.merge(ratings, books, on='ISBN')
columns = ['yearOfPublication', 'publisher', 'bookAuthor', 'imageUrlS', 'imageUrlM', 'imageUrlL']

combine_book_rating = combine_book_rating.drop(columns, axis=1)
print(combine_book_rating.head())

"""
Em seguida, agrupamos por títulos de livros e 
criamos uma nova coluna para contagem total de classificação
"""

combine_book_rating = combine_book_rating.dropna(axis = 0, subset = ['bookTitle'])
book_ratingCount = (combine_book_rating.
  groupby(by = ['bookTitle'])['bookRating'].
  count().
  reset_index().
  rename(columns = {'bookRating': 'totalRatingCount'})
  [['bookTitle', 'totalRatingCount']]
)

print('\nContagem total de classificação\n', book_ratingCount.head())

"""
Combinamos os dados de classificação com os dados 
totais da contagem e classificação, isso nos dá exatamente o que
precisamos para descobrir quais livros são populares e filtrar
livros menos conhecidos.
"""

rating_with_totalRatingCount = combine_book_rating.merge(book_ratingCount, left_on='bookTitle', right_on='bookTitle', how='left')
print(rating_with_totalRatingCount.head())

""" Vejamos as estatísticas da contagem total de classificações: """

pd.set_option('display.float_format', lambda x: '%.3f' % x)
print('\nEstátisticas da contagem total de classificações\n', book_ratingCount['totalRatingCount'].describe())

""" 
O livro mediano foi classificado apenas uma vez.
Vamos olhar para o topo da distribuição.
"""

print(book_ratingCount['totalRatingCount'].quantile(np.arange(.9, 1, .01)))

"""
Cerca de 1% dos livros receberam 50 ou mais avaliações.
Como temos tantos livros em nossos dados, vamos limitá-lo ao top 1%, 
e isso nos dará 2713 livros exclusivos.
"""

popularity_threshold = 50
rating_popular_book = rating_with_totalRatingCount.query('totalRatingCount >= @popularity_threshold')
print('\nAvaliações limitadas para o top 1%\n', rating_popular_book.head())

""" Filtrar apenas para usuários nos EUA e Canadá """

"""
A fim de melhorar a velocidade da computação, e não esbarrar no problema
"MemoryError", limitarei nossos dados de usuário àqueles nos EUA e Canadá.
E, em seguida, combinar dados do usuário com os dados de classificação e dados
totais de contagem de classificação.
"""

combined = rating_popular_book.merge(users, left_on = 'userID', right_on = 'userID', how = 'left')
us_canada_user_rating = combined[combined['Location'].str.contains("usa|canada")]
us_canada_user_rating=us_canada_user_rating.drop('Age', axis=1)
print('\nDados filtrados para os usuários dos EUA e Canadá\n', us_canada_user_rating.head())

""" Implementação de KNN """

"""
Convertemos nossa tabela em uma matriz 2D e preechemos os valores faltantes
com zeros (já que calcularesmo distâncias entre vetores de classificação).
Em seguida, transformamos os valores (classificações) do quadro de dados matricial
em uma matriz esparsa para cálculos mais eficientes.
"""

""" Encontrando os vizinhos mais próximos """

"""
Usamos algoritmos não supervisionados com sklearn.neighbors. 
O algoritmo que usamos para calcular os vizinhos mais próximos é "bruto", 
e especificamos "metric=cosine" para que o algoritmo calcule a semelhança 
cossina entre vetores de classificação. Finalmente, encaixamos no modelo.
"""

us_canada_user_rating = us_canada_user_rating.drop_duplicates(['userID', 'bookTitle'])
us_canada_user_rating_pivot = us_canada_user_rating.pivot(index = 'bookTitle', columns = 'userID', values = 'bookRating').fillna(0)
us_canada_user_rating_matrix = csr_matrix(us_canada_user_rating_pivot.values)

model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
model_knn.fit(us_canada_user_rating_matrix)

""" Teste nosso modelo e faça algumas recomendações """

"""
Nesta etapa, o algoritmo kNN mede a distância para determinar a "proximidade"
das instâncias. Em seguida, classifica uma instância ao encontrar seus vizinhos 
mais próximos, e escolhe a classe mais popular entre os vizinhos.
"""

query_index = np.random.choice(us_canada_user_rating_pivot.shape[0])
distances, indices = model_knn.kneighbors(us_canada_user_rating_pivot.iloc[query_index, :].values.reshape(1, -1), n_neighbors=6)

for i in range(0, len(distances.flatten())):
  if i == 0:
    print('Sua recomendação é {0}: \n'.format(us_canada_user_rating_pivot.index[query_index]))

  else:
    print('{0}: {1}, por uma distância de {2}: '.format(i, us_canada_user_rating_pivot.index[indices.flatten()[i]], distances.flatten()[i]))