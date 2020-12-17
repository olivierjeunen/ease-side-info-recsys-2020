import argparse
import numpy as np
import os
import pandas as pd
import pickle
import string
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
    ratings = pd.read_csv(args.dir + 'reviews_Sports_and_Outdoors_5.csv')[['reviewerID','asin','overall']]
    ratings.columns = ['user', 'item', 'rating']
    ratings = ratings.loc[ratings.rating > 3.0]

    # Only keep users who have rated at least 5 movies
    user_counts = ratings['user'].value_counts().reset_index().rename(columns = {'index': 'user', 'user': 'count'})
    user_counts = user_counts.loc[user_counts['count'] >= 5]
    ratings = ratings.merge(user_counts, on = 'user', how = 'right').drop('count', axis = 1)
    print('\t{0:8} ratings'.format(ratings.shape[0]))
    print('\t{0:8} unique users, {1:8} unique items'.format(ratings['user'].nunique(), ratings['item'].nunique()))

    # Load in metadata
    meta = pd.read_csv(args.dir + 'meta_Sports_and_Outdoors.csv')[['asin','description','categories','title','brand']]
    meta.columns = ['item','desc','cat','title','brand']
    
    # We only want metadata for items we have ratings for
    meta = meta.merge(ratings[['item']].drop_duplicates(), how = 'right', on = 'item')
    minsup = 3
    maxsup = ratings['item'].nunique() // 4
   
    # Clean up categorical strings
    meta['cat'] = meta['cat'].apply(lambda s: s.replace('[','').replace(']','').replace('\'','').strip())
    cat = pd.DataFrame(meta.cat.str.split(',').tolist(), index = meta.item).stack().reset_index([0, 'item'])
    cat.columns = ['item', 'cat']
    cat['cat'] = cat['cat'].apply(lambda s: s.strip())
    cat.drop_duplicates(inplace=True)
    cat = cat.loc[cat.cat != 'Sports & Outdoors'] # Appears too often
    cat = cat.loc[cat.cat != ' ']
    cat2count = cat.groupby('cat')['item'].apply(lambda x: len(set(x))).reset_index().rename(columns = {'item': 'count'})
    cat2count = cat2count[cat2count['count'] >= 2]
    cat = cat.merge(cat2count[['cat']], on = 'cat', how = 'right')
    print(cat['cat'].value_counts())

    # Clean up description strings
    meta['desc'] = meta['desc'].apply(lambda s: str(s).lower().translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation))))
    desc = pd.DataFrame(meta.desc.str.split(' ').tolist(), index = meta.item).stack().reset_index([0, 'item'])
    desc.columns = ['item', 'desc']
    desc.drop_duplicates(inplace=True)
    desc.dropna(inplace = True)
    desc = desc.loc[desc.desc != ' ']
    word2count = desc.groupby('desc')['item'].apply(lambda x: len(set(x))).reset_index().rename(columns = {'item': 'count'})
    word2count = word2count[word2count['count'] >= minsup]
    word2count = word2count[word2count['count'] <= maxsup]
    desc = desc.merge(word2count[['desc']], on = 'desc', how = 'right')
    print(desc['desc'].value_counts())

    # Clean up Title strings
    meta['title'] = meta['title'].apply(lambda s: str(s).lower().translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation))))
    title = pd.DataFrame(meta.title.str.split(' ').tolist(), index = meta.item).stack().reset_index([0, 'item'])
    title.columns = ['item', 'title']
    title.drop_duplicates(inplace=True)
    title.dropna(inplace = True)
    title = title.loc[title.title != ' ']
    word2count = title.groupby('title')['item'].apply(lambda x: len(set(x))).reset_index().rename(columns = {'item': 'count'})
    word2count = word2count[word2count['count'] >= minsup]
    word2count = word2count[word2count['count'] <= maxsup]
    title = title.merge(word2count[['title']], on = 'title', how = 'right')
    print(title['title'].value_counts())

    # Clean up description strings
    meta['brand'] = meta['brand'].apply(lambda s: str(s).lower().translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation))))
    brand = pd.DataFrame(meta.brand.str.split(' ').tolist(), index = meta.item).stack().reset_index([0, 'item'])
    brand.columns = ['item', 'brand']
    brand.drop_duplicates(inplace=True)
    brand.dropna(inplace = True)
    brand = brand.loc[brand.brand != ' ']
    word2count = brand.groupby('brand')['item'].apply(lambda x: len(set(x))).reset_index().rename(columns = {'item': 'count'})
    word2count = word2count[word2count['count'] >= minsup]
    word2count = word2count[word2count['count'] <= maxsup]
    brand = brand.merge(word2count[['brand']], on = 'brand', how = 'right')
    print(brand['brand'].value_counts())
    
    # Ensure proper integer identifiers
    user_enc = LabelEncoder()
    item_enc = LabelEncoder()
    cat_enc = LabelEncoder()
    desc_enc = LabelEncoder()
    title_enc = LabelEncoder()
    brand_enc = LabelEncoder()
    ratings['user'] = user_enc.fit_transform(ratings['user'])
    ratings['item'] = item_enc.fit_transform(ratings['item'])
    cat['item'] = item_enc.transform(cat['item'])
    cat['cat'] = cat_enc.fit_transform(cat['cat'].astype(str))
    desc['item'] = item_enc.transform(desc['item'])
    desc['desc'] = desc_enc.fit_transform(desc['desc'])
    title['item'] = item_enc.transform(title['item'])
    title['title'] = title_enc.fit_transform(title['title'])
    brand['item'] = item_enc.transform(brand['item'])
    brand['brand'] = brand_enc.fit_transform(brand['brand'])

    # Generate Metadata-to-item mapping
    X_cat = util.generate_csr_matrix(cat, 'cat', ratings['item'].max() + 1)
    X_desc = util.generate_csr_matrix(desc, 'desc', ratings['item'].max() + 1)
    X_title = util.generate_csr_matrix(title, 'title', ratings['item'].max() + 1)
    X_brand = util.generate_csr_matrix(brand, 'brand', ratings['item'].max() + 1)
    X_meta = vstack((X_cat,X_desc,X_title,X_brand))
    
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
