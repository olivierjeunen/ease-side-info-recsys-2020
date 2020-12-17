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
    parser.add_argument('--test_users', type = int, default = 10000)
    args = parser.parse_args()

    # Fix seed for reproducibility
    np.random.seed(42)

    # Load rating data
    print(datetime.now(), 'Loading in ratings...')
    ratings = pd.read_csv(args.dir + 'preprocessed_pref.csv')
    ratings.columns = ['user', 'item']
    print('\t{0:8} ratings'.format(ratings.shape[0]))
    print('\t{0:8} unique users, {1:8} unique items'.format(ratings['user'].nunique(), ratings['item'].nunique()))

    # Load side info
    print(datetime.now(), 'Loading in side-info...')
    ###########################
    # ARTISTS - GENRES - TAGS #
    ###########################
    # Load in data
    artists = pd.read_csv(args.dir + 'preprocessed_artists.csv')
    artists.columns = ['item', 'artist']
    genres = pd.read_csv(args.dir + 'preprocessed_genres.csv')
    genres.columns = ['item', 'genre']
    tags = pd.read_csv(args.dir + 'preprocessed_tags.csv')
    tags.columns = ['item', 'tag']

    # Drop those not appearing in preference data
    artists = artists.merge(ratings[['item']].drop_duplicates(), how = 'right').dropna()
    genres = genres.merge(ratings[['item']].drop_duplicates(), how = 'right').dropna()
    tags = tags.merge(ratings[['item']].drop_duplicates(), how = 'right').dropna()

    # Ensure proper integer identifiers
    user_enc = LabelEncoder()
    item_enc = LabelEncoder()
    ratings['user'] = user_enc.fit_transform(ratings['user'])
    ratings['item'] = item_enc.fit_transform(ratings['item'])
    artists['item'] = item_enc.transform(artists['item'])
    genres['item'] = item_enc.transform(genres['item'])
    tags['item'] = item_enc.transform(tags['item'])

    # Generate Metadata-to-item mapping
    X_artists = util.generate_csr_matrix(artists, 'artist', ratings['item'].max() + 1)
    X_genres = util.generate_csr_matrix(genres, 'genre', ratings['item'].max() + 1)
    X_tags = util.generate_csr_matrix(tags, 'tag', ratings['item'].max() + 1)
    X_meta = vstack((X_artists, X_genres, X_tags))
    
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
