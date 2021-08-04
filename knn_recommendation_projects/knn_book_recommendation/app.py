# Instalando depedências
import pandas as pd
from flask import Flask, jsonify, request
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import fuzz

class Recommender:
  def train(self):
        books = pd.read_csv("dataset/BX_Books.csv", sep=";", encoding="latin-1")
        ratings = pd.read_csv("dataset/BX-Book-Ratings.csv", sep=";", encoding="latin-1")

        user_rating_count = pd.DataFrame(ratings.groupby('User-ID').size(), columns=['count'])
        book_rating_count = pd.DataFrame(ratings.groupby('ISBN').size(), columns=['count'])
        
        popularity_thres = 8
        
        popular_books = list(set(book_rating_count.query('count > @popularity_thres').index))
        
        tmp = ratings[(ratings.ISBN.isin(popular_books))]
        
        ratings_popular = tmp[(tmp.ISBN.isin(books.ISBN.values))]
        ratings_thres = 10
        
        active_users = list(set(user_rating_count.query('count > @ratings_thres').index))
        relevant_ratings = ratings_popular[ratings_popular['User-ID'].isin(active_users)]
        book_user_mat = relevant_ratings.pivot(index='ISBN', columns='User-ID', values='Book-Rating').fillna(0)
        
        self.book_user_mat_sparse = csr_matrix(book_user_mat.values)
        self.model_knn = NearestNeighbors(n_neighbors=11, n_jobs=-1)
        self.model_knn.fit(self.book_user_mat_sparse)
        self.mapper = {book: i for i, book in enumerate(book_user_mat.index)}
        self.relevant_books = books[books['ISBN'].isin(relevant_ratings['ISBN'])]

      def predict(self, fav_book, n_recommendations):
        print('Você tem um livro de entrada', fav_book)

        all_books_cpy = self.relevant_books.copy()
        all_books_cpy['fuzz_ratio'] = all_book_cpy.apply(
          lambda row: fuzz.ratio(row['Book-Title'].lower(), fav_book.lower()), axis=1
        )
        best_match = all_book_cpy.loc[all_books_cpy['fuzz_ratio'].idxmax()]

        print('Recomendar livros para {}'.format(best_match['Book-Title']))
        best_match_ix = self.mapper[best_match.ISBN]
        print('Sistema de recomendação começa a fazer inferências')
        print('......\n')
        distances, indices = self.model_knn.kneighbors(self.book_user_mat_sparse [best_match_ix], n_neighbors=n_recommendations + 1)

        raw_recommends = \
            sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())), key=lambda x: x[1])[1:]

        reverse_mapper = {v: k for k, v in self.mapper.items()}

        print('Recomendações para {}:'.format(fav_book))
        recommendation_list = []

        for i, (idx, dist) in enumerate(raw_recommends):
            title = all_books_cpy[all_books_cpy['ISBN'] == reverse_mapper[idx]].iloc[0]['Book-Title']
            print('{0}: {1}, com uma distância de {2}'.format(i + 1, title, dist))
            recommendation_list.append(str(title))

        return {
            "book_title": str(best_match['Book-Title']),
            "match_with_query": int(best_match['fuzz_ratio']),
            "recommendations": recommendation_list
        }

server = Flask(__name__)
recommender = Recommender()
recommender.train()

@server.route("/")
def hello():
    return "O servidor está instalado e funcionando"

@server.route("/recomendacao", methods=['POST'])
def recomendacao():
    if 'book' in request.form:
        b = request.form['book']
        rec = recommender.predict(b, 10)
        return jsonify(rec)
    return "Nenhum livro fornecido"

if __name__ == "__main__":
   server.run(host='0.0.0.0') 