import heapq
import os
from math import sqrt
import numpy as np
import pandas as pd
import warnings
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from tabulate import tabulate

K = 10

_PRECISION_K_ = 'Precision@k'
_ARHR_ = "ARHR"
_RMSE_ = "RMSE"

COSINE = "cosine"
EUCLIDEAN = "euclidean"
JACCARD = "jaccard"

SIMILARITY_METRICS = [COSINE, EUCLIDEAN, JACCARD]


class RecommendationSystem:

    def __init__(self):

        # Using absolute paths for reaching the data files.
        self.books = pd.read_csv(os.path.abspath("books.csv"), low_memory=False, encoding="ISO-8859-1")
        self.ratings = pd.read_csv(os.path.abspath("ratings.csv"), low_memory=False)
        self.users = pd.read_csv(os.path.abspath("users.csv"), low_memory=False)
        self.test = pd.read_csv(os.path.abspath("test.csv"), low_memory=False)

    def get_simply_recommendation(self, k, DEBUG=False):

        # Use book_id as the index of the DataFrame and filter all the irrelevant columns.
        books = self.books[["book_id", "title"]].set_index('book_id')
        # Sort the ratings DataFrame by book_id and filter all the irrelevant columns.
        ratings = self.ratings.sort_values(by=['book_id'], ascending=True)[["book_id", "rating"]]

        books['vote_count'] = ratings.pivot_table(index=['book_id'], aggfunc='size')
        books['vote_average'] = ratings.pivot_table(index=['book_id'], aggfunc='mean')

        # Use the Weighted Average Ratings similarity measure to get the top10 books.
        q_books = self.weighted_average_ratings_measure(books)
        ids, titles, scores = list(q_books['book_id'][:k]), list(q_books['title'][:k]), list(q_books['score'][:k])

        if DEBUG:  # Print the top10 books.
            print(tabulate(list(zip(range(1, k + 1), ids, titles, scores)), headers=['rank', 'book_id', 'title', 'score'],
                           tablefmt='grid', colalign=("center", "center", "center", "center")))

        # Return the top10 books list with a pattern of (book_id, title, score).
        return list(zip(ids, titles, scores))

    def weighted_average_ratings_measure(self, books):

        # Calculate the mean of vote average column.
        C = books['vote_average'].mean()
        # Calculate the minimum number of votes required to be in the chart, m.
        m = books['vote_count'].quantile(0.90)

        # Filter all the books that don't have the minimum number of votes.
        q_books = books.copy().loc[books['vote_count'] >= m]

        # Function that computes the weighted rating of each book.
        def weighted_rating(x, m, C):
            v = x['vote_count']
            R = x['vote_average']
            return (v / (v + m) * R) + (m / (m + v) * C)

        # Define a new feature - 'score' - and calculate its value with `weighted_rating()`.
        q_books['score'] = q_books.apply(weighted_rating, args=(m, C), axis=1)
        # Sort the books based on the score calculated above.
        q_books = q_books.sort_values('score', ascending=False).reset_index()

        return q_books

    def get_simply_place_recommendation(self, place, k, DEBUG=False):

        # Use book_id as the index of the books DataFrame and filter all the irrelevant columns.
        books = self.books[["book_id", "title"]].set_index('book_id')
        users = self.users.set_index('user_id')  # Use user_id as the index of the users DataFrame.
        ratings = self.ratings.set_index('user_id')  # Use user_id as the index of the ratings DataFrame.

        # Define a location feature in ratings and update it with the values under this feature in the users DataFrame.
        ratings['location'] = np.nan
        ratings.update(users)

        # Filter all other locations.
        ratings = ratings[ratings['location'] == place]

        # Calculate the vote_count and vote_average for each book.
        ratings = ratings.reset_index().filter(items=['book_id', 'user_id', 'rating'])
        pivot_table = ratings.pivot_table(index=['book_id'], columns=['user_id'])
        books['vote_count'] = pivot_table.count(axis=1)
        books['vote_average'] = pivot_table.mean(axis=1)

        # Use the Weighted Average Ratings similarity measure to get the top10 books in the relevant location.
        q_books = self.weighted_average_ratings_measure(books)
        ids, titles, scores = list(q_books['book_id'][:k]), list(q_books['title'][:k]), list(q_books['score'][:k])

        if DEBUG:  # Print the top10 books.
            print(tabulate(list(zip(range(1, k + 1), ids, titles, scores)), headers=['rank', 'book_id', 'title', 'score'],
                           tablefmt='grid', colalign=("center", "center", "center", "center")))

        # Return the top10 books list with a pattern of (book_id, title, score).
        return list(zip(ids, titles, scores))

    def get_simply_age_recommendation(self, age, k, DEBUG=False):

        # Use book_id as the index of the books DataFrame and filter all the irrelevant columns.
        books = self.books[["book_id", "title"]].set_index('book_id')
        users = self.users.set_index('user_id')  # Use user_id as the index of the users DataFrame.
        ratings = self.ratings.set_index('user_id')  # Use user_id as the index of the ratings DataFrame.

        # Define an age feature in ratings and update it with the values under this feature in the users DataFrame.
        ratings['age'] = np.nan
        ratings.update(users)

        # Filter the ratings DataFrame around to the age range (x1-y0).
        min_age = (age // 10) * 10 + 1
        max_age = ((age // 10) + 1) * 10
        ratings = ratings[(ratings['age'] >= min_age) & (ratings['age'] <= max_age)]

        # Calculate the vote_count and vote_average for each book.
        ratings = ratings.reset_index().filter(items=['book_id', 'user_id', 'rating'])
        pivot_table = ratings.pivot_table(index=['book_id'], columns=['user_id'])
        books['vote_count'] = pivot_table.count(axis=1)
        books['vote_average'] = pivot_table.mean(axis=1)

        # Use the Weighted Average Ratings similarity measure to get the top10 books for the relevant age range.
        q_books = self.weighted_average_ratings_measure(books)
        ids, titles, scores = list(q_books['book_id'][:k]), list(q_books['title'][:k]), list(q_books['score'][:k])

        if DEBUG:  # Print the top10 books.
            print(tabulate(list(zip(range(1, k + 1), ids, titles, scores)), headers=['rank', 'book_id', 'title', 'score'],
                           tablefmt='grid', colalign=("center", "center", "center", "center")))

        # Return the top10 books list with a pattern of (book_id, title, score).
        return list(zip(ids, titles, scores))

    def get_data_matrix(self):

        # Calculate the number of unique users and books.
        n_users = self.ratings.user_id.unique().shape[0]
        n_items = self.ratings.book_id.unique().shape[0]

        # Get the unique books and sort them.
        book_ids = self.ratings.book_id.unique()
        book_ids.sort()

        # Create a mapping from book id to a new index and vice versa (for a continuous range of book ids).
        self.idx2id = {i: real_i for i, real_i in enumerate(book_ids)}
        id2idx = {real_i: i for i, real_i in enumerate(book_ids)}

        # Build the data matrix using the user_ids and the continuous range of book ids.
        data_matrix = np.empty((n_users, n_items))
        data_matrix[:] = np.nan
        for line in self.ratings.itertuples():
            user = line[1] - 1
            movie = id2idx[line[2]]
            rating = line[3]
            data_matrix[user, movie] = rating

        return data_matrix

    def build_CF_prediction_matrix(self, sim):

        def keep_top_k(arr, k):
            smallest = heapq.nlargest(k, arr)[-1]
            arr[arr < smallest] = 0  # replace anything lower than the cut off with 0
            return arr

        # Build the data_matrix.
        self.data_matrix = self.get_data_matrix()

        # Calculate the mean.
        mean_user_rating = np.nanmean(self.data_matrix, axis=1).reshape(-1, 1)
        ratings_diff = (self.data_matrix - mean_user_rating)

        # Replace nan -> 0.
        ratings_diff[np.isnan(ratings_diff)] = 0

        # Calculate user x user similarity matrix.
        warnings.filterwarnings("ignore")  # Ignore Jaccard metric warning.
        user_similarity = 1 - pairwise_distances(ratings_diff, metric=sim)

        # For each user (i.e., for each row) keep only k most similar users, set the rest to 0.
        user_similarity = np.array([keep_top_k(np.array(arr), K) for arr in user_similarity])

        # Since n-k users have similarity=0, for each user only k most similar users contribute to the predicted ratings
        self.pred = mean_user_rating + user_similarity.dot(ratings_diff) / np.array(
            [np.abs(user_similarity).sum(axis=1)]).T

    def get_CF_recommendation(self, user_id, k, DEBUG=False):

        user_id = user_id - 1
        predicted_ratings_row = self.pred[user_id]
        data_matrix_row = self.data_matrix[user_id]

        # Get the top10 books for this user.
        ids, titles, ratings = self.get_recommendations(predicted_ratings_row, data_matrix_row, k)

        if DEBUG:  # Print the top10 books.
            print(tabulate(list(zip(range(1, k + 1), ids, titles, ratings)),
                           headers=['rank', 'book_id', 'title', 'rating'], tablefmt='grid',
                           colalign=("center", "center", "center", "center")))

        # Return the top10 books list with a pattern of (book_id, title).
        return list(zip(ids, titles))

    def get_recommendations(self, predicted_ratings_row, data_matrix_row, k):

        # Use book_id as the index of the books DataFrame and filter all the irrelevant columns.
        books = self.books[["book_id", "title"]].set_index('book_id')

        # Replace nan -> 0 and get the top rated books for the user.
        predicted_ratings_row[~np.isnan(data_matrix_row)] = 0
        idx = np.argsort(-predicted_ratings_row)[:k]

        # Get the ids, the titles and the ratings of the top10 books.
        ids = [self.idx2id[i] for i in idx]
        titles = list(books.loc[np.array(ids)]['title'])
        ratings = np.sort(predicted_ratings_row)[::-1][:k]

        return ids, titles, ratings

    def get_feature_matrix(self):

        # Use these features for the similarity matrix.
        X = self.books[["title", "authors", "original_publication_year", "language_code"]]

        # Keeping only the first author out of a list of several authors of one book.
        def filter_authors(x):
            authors = x.split(',')
            return x if len(authors) <= 1 else authors[0]

        # Apply filter_authors function to the 'authors' feature.
        X['authors'] = X['authors'].apply(filter_authors)

        def clean_data(x):
            if isinstance(x, list):
                return [str.lower(i.replace(" ", "")) for i in x]
            else:
                if isinstance(x, str):  # Check if director exists. If not, return empty string
                    return str.lower(x.replace(" ", "_"))
                else:
                    return ''

        # Apply clean_data function to the 'title' and 'authors' features.
        for feature in ["title", "authors"]:
            X[feature] = X[feature].apply(clean_data)

        # Encode the 'title' and 'authors' features which are categorical features.
        X = pd.concat([X, pd.get_dummies(X['title'], prefix='title')], axis=1)
        X = X.drop(columns=['title'])  # Remove the original features.
        X = pd.concat([X, pd.get_dummies(X['authors'], prefix='author')], axis=1)
        X = X.drop(columns=['authors'])  # Remove the original features.

        #  I use the pandas cut function to bin the publication year values into discrete intervals.
        # The values were taken from the X['original_publication_year'].describe() as follows.
        bins = [-1750, 1987, 2003, 2010, 2016]  # (min, 25%, 50%, 75%, max)
        labels = range(1, 5)
        X['publication_year_binned'] = pd.cut(X['original_publication_year'], bins=bins, labels=labels)
        X = X.drop(columns=['original_publication_year'])  # Remove the original feature.

        # Now let's encode 'publication_year_binned' as one-hot representation.
        X = pd.concat([X, pd.get_dummies(X['publication_year_binned'], prefix='pub_year_binned')], axis=1)
        # Remove the original feature.
        X = X.drop(columns=['publication_year_binned'])

        # Encode the 'language_code' feature, which is also a categorical feature and remove the original feature.
        X = pd.concat([X, pd.get_dummies(X['language_code'], prefix='lang_code')], axis=1)
        X = X.drop(columns=['language_code'])

        # Return the Encoded feature matrix.
        return X

    def build_contact_sim_metrix(self):

        # Encode the feature matrix.
        feature_matrix = self.get_feature_matrix().values

        # Compute the Cosine Similarity matrix,
        self.cosine_sim = cosine_similarity(feature_matrix, feature_matrix)

        # Reset index of books DataFrame and construct reverse mapping.
        books = self.books.reset_index()
        self.idx_2_book_id = pd.Series(books['book_id'], index=books.index)
        self.idx2title = pd.Series(books['title'], index=books.index)

        return self.cosine_sim

    def get_contact_recommendation(self, book_name, k, DEBUG=False):

        # Get the book_id of the book that matches the title.
        book_id = self.books.set_index(keys=['title'], drop=True).loc[book_name]['book_id']
        # Get the index of the book_id.
        row_index = self.books.reset_index().set_index(keys=['book_id']).loc[book_id]['index']

        # Get the pairwise similarity scores of all books with that book.
        book_row = self.cosine_sim[row_index]
        idx = np.argsort(book_row)[::-1][1:k + 1]  # Get the top10 books.

        # Get the ids and titles of the top10 books.
        ids = [self.idx_2_book_id[i] for i in idx]
        titles = [self.idx2title[i] for i in idx]

        if DEBUG:  # Print the top10 books.
            print(tabulate(list(zip(range(1, k + 1), ids, titles)), headers=['rank', 'book_id', 'title'], tablefmt='grid',
                           colalign=("center", "center", "center"), ))

        # Return the top10 books list with a pattern of (book_id, title).
        return list(zip(ids, titles))

    def filter_test_data(self, k):

        # Get the test DataFrame.
        test = self.test.copy()

        # Filter the books that their rating is 3 and down.
        def filter_by_rating(rating):
            return True if rating >= 4 else False

        # Define a new feature - 'is_high_rated' and calculate its value with `filter_by_rating()`.
        test["is_high_rated"] = test["rating"].apply(filter_by_rating)

        # Keep only the users (and their ratings) that have rated more that k books as high rated books.
        users_passed_the_cut = {user_id: list(group[group['is_high_rated']]["book_id"]) for user_id, group in
                                test.groupby(by="user_id") if len(group[group['is_high_rated']]) >= k}

        return users_passed_the_cut

    def precision_k(self, k, DEBUG=False):

        # Get the users that passed the cut.
        users_passed_the_cut = self.filter_test_data(k)
        result = []

        # For each similarity metric.
        for sim_metric in SIMILARITY_METRICS:

            # Build the prediction matrix with the current similarity metric.
            self.build_CF_prediction_matrix(sim_metric)
            sum_precision_k = 0.0

            for user_id, high_rated_books in users_passed_the_cut.items():

                # Get the recommended books for the user.
                recommendations = self.get_CF_recommendation(user_id, k)

                # Sum the number of hits.
                hits = sum([1 for (book_id, _) in recommendations if book_id in high_rated_books])
                sum_precision_k += hits / k  # Sum the precision_k .

            # Normalize by the number of users passed the cut.
            precision__k = round(sum_precision_k / len(users_passed_the_cut), 3)
            result.append(precision__k)

            if DEBUG:  # Print the accuracy.
                print(f"Accuracy with the {sim_metric} similarity metric is {precision__k}")

        # Return the accuracies list - the format is [cosine, euclidean, jaccard]
        return result

    def ARHR(self, k, DEBUG=False):

        # Get the users that passed the cut.
        users_passed_cut = self.filter_test_data(k)
        result = []

        for sim_metric in SIMILARITY_METRICS:

            # Build the prediction matrix with the current similarity metric.
            self.build_CF_prediction_matrix(sim_metric)

            sum_arhr = 0.0

            for user_id, high_rated_books in users_passed_cut.items():

                # Get the recommended books for the user.
                recommendations = self.get_CF_recommendation(user_id, k)

                # Sum the number of hits.
                hits = sum([1 / i for i, (book_id, _) in enumerate(recommendations, 1) if book_id in high_rated_books])
                sum_arhr += hits  # Sum the ARHR .

            # Normalize by the number of users passed the cut.
            arhr = round(sum_arhr / len(users_passed_cut), 3)
            result.append(arhr)

            if DEBUG:  # Print the accuracy.
                print(f"Accuracy with the {sim_metric} similarity metric is {arhr}")

        # Return the accuracies list - the format is [cosine, euclidean, jaccard]
        return result

    def get_all_CF_recommendations(self, user_id):

        user_id = user_id - 1
        predicted_ratings_row = self.pred[user_id]
        data_matrix_row = self.data_matrix[user_id]

        # Replace nan -> 0 and get the top rated books for the user.
        predicted_ratings_row[~np.isnan(data_matrix_row)] = 0
        idx = np.argsort(-predicted_ratings_row)

        # Get the rating of the top rated books.
        ratings = np.sort(predicted_ratings_row)[::-1]

        # Return a dictionary mapping each book_id to it's rating.
        return {self.idx_2_book_id[i]: rating for i, rating in zip(idx, ratings)}

    def RMSE(self, DEBUG=False):

        # Get the test DataFrame.
        test = self.test.copy()
        result = []

        for sim_metric in SIMILARITY_METRICS:

            # Build the prediction matrix with the current similarity metric.
            self.build_CF_prediction_matrix(sim_metric)

            sum = 0.0
            N = 0

            for user_id, group in test.groupby(by="user_id"):

                # Get all the recommended books for the user.
                predicted_recommendations = self.get_all_CF_recommendations(user_id)

                # For each book recommendation of the user in the test data.
                for user_recommendation in group.itertuples(index=False):
                    _, book_id, rating = tuple(user_recommendation)
                    predicted_rating = predicted_recommendations[book_id]  # Get the predicted rating.
                    sum += pow((predicted_rating - rating), 2)
                    N += 1

            # Normalize by N.
            rmse = round(sqrt(sum / N), 3)
            result.append(rmse)

            if DEBUG:  # Print the accuracy.
                print(f"Accuracy with the {sim_metric} similarity metric is {rmse}")

        # Return the accuracies list - the format is [cosine, euclidean, jaccard]
        return result


def run_all(DEBUG=True):
    print('2: Simply recommendation')
    get_simply_recommendation(K, DEBUG)
    print('\n3: Simply place recommendation')
    get_simply_place_recommendation('Ohio', K, DEBUG)
    print('\n4: Simply age recommendation')
    get_simply_age_recommendation(28, K, DEBUG)
    print(f"\n6: CF recommendation using the '{COSINE}' similarity metric")
    build_CF_prediction_matrix(COSINE)
    get_CF_recommendation(1, K, DEBUG)
    print(f"\n7: CF recommendation using the '{EUCLIDEAN}' similarity metric")
    build_CF_prediction_matrix(EUCLIDEAN)
    get_CF_recommendation(1, K, DEBUG)
    print(f"\n7: CF recommendation using the '{JACCARD}' similarity metric")
    build_CF_prediction_matrix(JACCARD)
    get_CF_recommendation(1, K, DEBUG)
    print('\n10: Content-based recommendation')
    build_contact_sim_metrix()
    get_contact_recommendation("Twilight (Twilight, #1)", K, DEBUG)
    print(f'\n11: {_PRECISION_K_}')
    precision_k(K, DEBUG)
    print(f'\n11: {_ARHR_}')
    ARHR(K, DEBUG)
    print(f'\n11: {_RMSE_}')
    RMSE(DEBUG)


RC = RecommendationSystem()
get_simply_recommendation = RC.get_simply_recommendation
get_simply_place_recommendation = RC.get_simply_place_recommendation
get_simply_age_recommendation = RC.get_simply_age_recommendation
build_CF_prediction_matrix = RC.build_CF_prediction_matrix
get_CF_recommendation = RC.get_CF_recommendation
build_contact_sim_metrix = RC.build_contact_sim_metrix
get_contact_recommendation = RC.get_contact_recommendation
precision_k = RC.precision_k
ARHR = RC.ARHR
RMSE = RC.RMSE

# run_all()
