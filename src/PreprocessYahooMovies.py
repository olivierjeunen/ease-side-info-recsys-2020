import argparse
import numpy as np
import os
import pandas as pd
import pickle
import util
from datetime import datetime
from scipy.sparse import save_npz, vstack
from sklearn.preprocessing import LabelEncoder

if __name__ == '__main__':
    # Commandline arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', type = str, help = 'Directory containing the data')
    args = parser.parse_args()

    # Fix seed for reproducibility
    np.random.seed(42)

    # Load rating data
    print(datetime.now(), 'Loading in ratings...')
    ratings = pd.read_csv(args.dir + 'ydata-ymovies-user-movie-ratings-train-v1_0.txt',
                          sep = '\t',
                          header = None)
    ratings.columns = ['user', 'item', 'weird_rating','rating']
    ratings = ratings.loc[ratings.rating > 3.0]

    # Only keep users who have rated at least 5 movies
    user_counts = ratings['user'].value_counts().reset_index().rename(columns = {'index': 'user', 'user': 'count'})
    user_counts = user_counts.loc[user_counts['count'] >= 5]
    ratings = ratings.merge(user_counts, on = 'user', how = 'right').drop('count', axis = 1)
    print('\t{0:8} ratings'.format(ratings.shape[0]))
    print('\t{0:8} unique users, {1:8} unique items'.format(ratings['user'].nunique(), ratings['item'].nunique()))

    # Load side info
    print(datetime.now(), 'Loading in side-info...')
    side_columns = [
        'item',
        'title',
        'synposis',
        'runtime',
        'MPAA',
        'MPAA_reason',
        'release_date',
        'distributor',
        'dummy_1', # HUH?
        'poster',
        'genre',
        'directors',
        'director_ids',
        'crew_members',
        'crew_ids',
        'crew_types',
        'actors',
        'actor_ids',
        'avg_rating',
        'n_rating',
        'n_awards',
        'n_nominated',
        'list_won',
        'list_nominated',
        'rating_moviemom',
        'review_moviemom',
        'list_review_summaries',
        'list_reviewers',
        'list_captions',
        'preview',
        'DVD_review',
        'GNPP',
        'avg_train',
        'num_train'
    ]
    side = pd.read_csv(args.dir + 'ydata-ymovies-movie-content-descr-v1_0.txt',
                       sep = '\t',
                       encoding = 'latin',
                       names = side_columns)#[['item','genre']]

    # Extract genres properly
    genres = pd.DataFrame(side.genre.str.split('|').tolist(), index = side.item).stack().reset_index([0, 'item'])
    genres.columns = ['item', 'genre']
    genres = genres.loc[genres.genre != '\\N']
    genres = genres.merge(ratings[['item']].drop_duplicates(), on = 'item', how = 'right').dropna()

    # Extract directors properly
    directors = pd.DataFrame(side.director_ids.str.split('|').tolist(), index = side.item).stack().reset_index([0, 'item'])
    directors.columns = ['item', 'director']
    directors = directors.loc[directors.director != '\\N']
    directors = directors.merge(ratings[['item']].drop_duplicates(), on = 'item', how = 'inner').dropna()

    # Extract actors properly
    actors = pd.DataFrame(side.actor_ids.str.split('|').tolist(), index = side.item).stack().reset_index([0, 'item'])
    actors.columns = ['item', 'actor']
    actors = actors.loc[actors.actor != '\\N']
    actors = actors.merge(ratings[['item']].drop_duplicates(), on = 'item', how = 'inner').dropna()

    # Drop those that appear less than twice (wouldn't affect Gram-matrix)
    dir2count = directors.groupby('director')['item'].apply(lambda x: len(set(x))).reset_index().rename(columns = {'item': 'count'})
    dir2count = dir2count[dir2count['count'] >= 2]
    directors = directors.merge(dir2count[['director']], on = 'director', how = 'right')
    act2count = actors.groupby('actor')['item'].apply(lambda x: len(set(x))).reset_index().rename(columns = {'item': 'count'})
    act2count = act2count[act2count['count'] >= 2]
    actors = actors.merge(act2count[['actor']], on = 'actor', how = 'right')

    # Ensure proper integer identifiers
    user_enc = LabelEncoder()
    item_enc = LabelEncoder()
    genre_enc = LabelEncoder()
    direc_enc = LabelEncoder()
    actor_enc = LabelEncoder()
    ratings['user'] = user_enc.fit_transform(ratings['user'])
    ratings['item'] = item_enc.fit_transform(ratings['item'])
    genres['item'] = item_enc.transform(genres['item'])
    genres['genre'] = genre_enc.fit_transform(genres['genre'].astype(str))
    directors['item'] = item_enc.transform(directors['item'])
    directors['director'] = direc_enc.fit_transform(directors['director'])
    actors['item'] = item_enc.transform(actors['item'])
    actors['actor'] = actor_enc.fit_transform(actors['actor'])

    # Generate Metadata-to-item mapping
    X_genres = util.generate_csr_matrix(genres, 'genre', ratings['item'].max() + 1)
    X_directors = util.generate_csr_matrix(directors, 'director', ratings['item'].max() + 1)
    X_actors = util.generate_csr_matrix(actors, 'actor', ratings['item'].max() + 1)
    X_meta = vstack((X_genres,X_directors,X_actors))
    
    # Check whether output directory already exists - make it if necessary
    if not os.path.exists(args.dir + 'preprocessed/'):
        os.makedirs(args.dir + 'preprocessed/')

    # Write out metadata-item matrix
    print(datetime.now(), 'Writing out metadata-item matrix...')
    save_npz(args.dir + 'preprocessed/X_meta.npz', X_meta)

    print(datetime.now(), 'Train-validation-test split...')
    X_train, _, val_dict, _, test_dict = util.train_val_test_split_Karypis(ratings) 

    # Write out user-item matrix and held-out dictionaries
    print(datetime.now(), 'Writing out training, validation and test data...')
    save_npz(args.dir + 'preprocessed/X_train.npz', X_train)
    with open(args.dir + 'preprocessed/val_dict.pkl', 'wb') as handle:
        pickle.dump(val_dict, handle)
    with open(args.dir + 'preprocessed/test_dict.pkl', 'wb') as handle:
        pickle.dump(test_dict, handle)
    print(datetime.now(), 'Finished!')
