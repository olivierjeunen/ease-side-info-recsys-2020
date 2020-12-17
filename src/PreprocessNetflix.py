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
    parser.add_argument('--test_users', type = int, default = 40000)
    args = parser.parse_args()

    # Fix seed for reproducibility
    np.random.seed(42)
    
    # Load rating data
    print(datetime.now(), 'Loading in ratings...')
    ratings = pd.read_csv(args.dir + 'netflix_ratings.csv')
    ratings.columns = ['user', 'item', 'rating', 'time']
    
    # Preprocessing as in Liang et al. @ WWW 2018
    # Only keep ratings of 4 or higher
    ratings = ratings.loc[ratings.rating >= 4]
    # Only keep users who have rated at least 5 movies
    user_counts = ratings['user'].value_counts().reset_index().rename(columns = {'index': 'user', 'user': 'count'})
    user_counts = user_counts.loc[user_counts['count'] >= 5]
    ratings = ratings.merge(user_counts, on = 'user', how = 'right').drop('count', axis = 1)
    print('\t{0:8} ratings'.format(ratings.shape[0]))
    print('\t{0:8} unique users, {1:8} unique items'.format(ratings['user'].nunique(), ratings['item'].nunique()))

    # Load side info
    print(datetime.now(), 'Loading in side-info...')
    ####################
    # SERIES AND YEARS #
    ####################
    # Load in data
    series = pd.read_csv(args.dir + 'netflixid2series.csv')
    # Drop movies that don't appear in preference data
    #series = series.merge(ratings[['item']].drop_duplicates(), on = 'item', how = 'right')

    # Load in data
    years = pd.read_csv(args.dir + 'netflixid2year.csv')
    # Drop movies that don't appear in preference data
    years = years.merge(ratings[['item']].drop_duplicates(), on = 'item', how = 'right')

    ########
    # CREW #
    ########
    # Load IMDB data links with movielens
    links = pd.read_csv(args.dir + 'netflixid2imdbid.csv')

    # Side info - genres
    side = pd.read_csv(args.dir + 'ml-imdb_sideinfo.csv')[['imdb_title_id','genre']]
    side.columns = ['imdb_id', 'genre']
    side = side.merge(links, on = 'imdb_id', how = 'right')

    # Extract genres
    genres = pd.DataFrame(side.genre.str.split(',').tolist(), index = side.item).stack().reset_index([0, 'item'])
    genres.columns = ['item', 'genre']
    genres = genres.loc[genres.genre != '\\N']
    
    # Load IMDB crew data and link it properly
    crew = pd.read_csv(args.dir + 'imdb_crew_info.csv')
    crew.columns = ['imdb_id', 'directors', 'writers']
    crew = crew.merge(links, on = 'imdb_id', how = 'right')
    
    # We don't care about movies without ratings
    crew = crew.merge(ratings[['item']].drop_duplicates(), on = 'item', how = 'right')[['item','directors','writers']]
    crew['directors'] = crew['directors'].apply(lambda s: str(s))
    crew['writers'] = crew['writers'].apply(lambda s: str(s))
 
    # Extract directors
    directors = pd.DataFrame(crew.directors.str.split(',').tolist(), index = crew.item).stack().reset_index([0, 'item'])
    directors.columns = ['item', 'director']
    directors = directors.loc[directors.director != '\\N']
    
    # Drop directors that appear less than once (wouldn't affect Gram-matrix)
    dir2count = directors.groupby('director')['item'].apply(lambda x: len(set(x))).reset_index().rename(columns = {'item': 'count'})
    dir2count = dir2count[dir2count['count'] >= 2]
    directors = directors.merge(dir2count[['director']], on = 'director', how = 'right')

    # Extract writers
    writers = pd.DataFrame(crew.writers.str.split(',').tolist(), index = crew.item).stack().reset_index([0, 'item'])
    writers.columns = ['item', 'writer']
    writers = writers.loc[writers.writer != '\\N']
    
    # Drop writers that appear less than once (wouldn't affect Gram-matrix)
    writer2count = writers.groupby('writer')['item'].apply(lambda x: len(set(x))).reset_index().rename(columns = {'item': 'count'})
    writer2count = writer2count[writer2count['count'] >= 2]
    writers = writers.merge(writer2count[['writer']], on = 'writer', how = 'right')

    # Ensure proper integer identifiers
    user_enc = LabelEncoder()
    item_enc = LabelEncoder()
    year_enc = LabelEncoder()
    genre_enc = LabelEncoder()
    direc_enc = LabelEncoder()
    write_enc = LabelEncoder()
    ratings['user'] = user_enc.fit_transform(ratings['user'])
    ratings['item'] = item_enc.fit_transform(ratings['item'])
    years['item'] = item_enc.transform(years['item'])
    years['year'] = year_enc.fit_transform(years['year'])
    series['item'] = item_enc.transform(series['item'])
    genres['item'] = item_enc.transform(genres['item'])
    genres['genre'] = genre_enc.fit_transform(genres['genre'])
    directors['item'] = item_enc.transform(directors['item'])
    directors['director'] = direc_enc.fit_transform(directors['director'])
    writers['item'] = item_enc.transform(writers['item'])
    writers['writer'] = write_enc.fit_transform(writers['writer'])

    # Generate Metadata-to-item mapping
    X_years = util.generate_csr_matrix(years, 'year', ratings['item'].max() + 1)
    X_series = util.generate_csr_matrix(series, 'title_id', ratings['item'].max() + 1)
    X_genres = util.generate_csr_matrix(genres, 'genre', ratings['item'].max() + 1)
    X_directors = util.generate_csr_matrix(directors, 'director', ratings['item'].max() + 1)
    X_writers = util.generate_csr_matrix(writers, 'writer', ratings['item'].max() + 1)
    X_meta = vstack((X_years, X_series, X_genres, X_directors, X_writers))
    
    # Check whether output directory already exists - make it if necessary
    if not os.path.exists(args.dir + 'preprocessed/'):
        os.makedirs(args.dir + 'preprocessed/')

    # Write out metadata-item matrix
    print(datetime.now(), 'Writing out metadata-item matrix...')
    save_npz(args.dir + 'preprocessed/X_meta.npz', X_meta)

    # Train - validation - test split
    print(datetime.now(), 'Train-validation-test split...')
    X_train, X_val, val_dict, X_test, test_dict = util.train_val_test_split_Jebara(ratings, n_test_users = args.test_users) 

    # Write out validation and test data
    print(datetime.now(), 'Writing out validation and test data...')
    save_npz(args.dir + 'preprocessed/X_val.npz', X_val)
    with open(args.dir + 'preprocessed/val_dict.pkl', 'wb') as handle:
        pickle.dump(val_dict, handle)
    save_npz(args.dir + 'preprocessed/X_test.npz', X_test)
    with open(args.dir + 'preprocessed/test_dict.pkl', 'wb') as handle:
        pickle.dump(test_dict, handle)

    # Write out full user-item training matrix
    print(datetime.now(), 'Writing out train data...')
    save_npz(args.dir + 'preprocessed/X_train.npz', X_train)

    # Subsample training data on a user-level
    print(datetime.now(), 'Subsampling training users...')
    train_users = np.unique(X_train.nonzero()[0])
    np.random.shuffle(train_users)
    for frac_train_users in [0.01, .05, .1, .25, .5]:
        train_users[:int(frac_train_users * len(train_users))]
        pd.DataFrame(train_users[:int(frac_train_users * len(train_users))], columns = ['user']).to_csv(args.dir + 'preprocessed/train_users_{}.csv'.format(frac_train_users), index = False)
    print(datetime.now(), 'Finished!')
