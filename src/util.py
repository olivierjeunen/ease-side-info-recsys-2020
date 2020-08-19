import pandas as pd 
import numpy as np
import sys
from collections import defaultdict
from datetime import datetime
from joblib import Parallel, delayed
from scipy.sparse import csr_matrix, coo_matrix, vstack
from tqdm import tqdm

# From https://stackoverflow.com/questions/24455615/python-how-to-display-size-of-all-variables
def sizeof_fmt(num, suffix='B'):
    ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)

def pretty_print_results(results):
    ''' Pretty print results in a defaultdict. '''
    print('\tRecall@K')
    for K in results[0].keys():
        print('\t', K,'\t',results[0][K])
    print('\tNDCG@K')
    for K in results[1].keys():
        print('\t', K,'\t',results[1][K])

def generate_csr_matrix(meta_df, colname, ncols, alpha = 1.):
    ''' Generate Metadata-to-item mapping in the form of a CSR matrix. '''
    data = np.ones(meta_df.shape[0]) * alpha
    rows, cols = meta_df[colname].values, meta_df['item'].values
    nrows = meta_df[colname].max() + 1
    return csr_matrix((data, (rows, cols)), shape = (int(nrows), int(ncols)))

def normalize_idf(X):
    ''' Normalize matrix X according to column-wise IDF. '''
    # Log-normalised Smoothed Inverse Document Frequency
    row_counts = X.sum(axis = 1)
    row_counts -= (row_counts.min() - 2.0) # Start from 0 for more expressive log-scale
    idf = (1.0 / np.log(row_counts)).A1.ravel()
    return csr_matrix(np.diag(idf)) @ X

def compute_sparsity(A):
    ''' Compute the sparsity level (% of non-zeros) of matrix A. '''
    return 1.0 - np.count_nonzero(A) / (A.shape[0] * A.shape[1])

def sparsify(B, rho = .95):
    ''' Get B to the required sparsity level by dropping out the rho % lower absolute values. '''
    min_val = np.quantile(np.abs(B), rho)
    B[np.abs(B) < min_val] = .0
    return B

def compute_EASE(X, l2 = 5e2):
    ''' Compute a closed-form OLS SLIM-like item-based model. (H. Steck @ WWW 2019) '''
    G = X.T @ X + l2 * np.identity((X.shape[1]))
    B = np.linalg.inv(G)
    B /= -np.diag(B)
    B[np.diag_indices(B.shape[0])] = .0
    return B

def compute_cosine(X):
    ''' Compute a cosine similarity item-based model. '''
    # Base similarity matrix (all dot products)
    similarity = X.T.dot(X).toarray()
    # Squared magnitude of preference vectors (number of occurrences)
    square_mag = np.diag(similarity)
    # Inverse squared magnitude
    inv_square_mag = 1 / square_mag
    # If it doesn't occur, set it's inverse magnitude to zero (instead of inf)
    inv_square_mag[np.isinf(inv_square_mag)] = 0
    # inverse of the magnitude
    inv_mag = np.sqrt(inv_square_mag)
    # cosine similarity (elementwise multiply by inverse magnitudes)
    cosine = similarity * inv_mag
    cosine = cosine.T * inv_mag
    cosine[np.diag_indices(X.shape[1])] = .0
    return cosine

def generate_eval_format(ratings, nrows, ncols, hist_frac = .8):
    ''' Split 'ratings' into a historical and held-out fraction '''
    # Split ratings into 'history' and 'held-out' set
    test_ratings = ratings.groupby('user').apply(lambda df: df.sample(frac = 1. - hist_frac)).reset_index(drop = True)
    hist_ratings = pd.concat([test_ratings, ratings]).drop_duplicates(keep = False)

    # Generate user-item matrix for history and dictionary for hold-out
    data = np.ones(hist_ratings.shape[0])
    rows, cols = hist_ratings['user'], hist_ratings['item']
    X_hist = csr_matrix((data, (rows, cols)), shape = (nrows, ncols))
    
    # Generate dictionary for hold-out (fast lookup)
    test_dict = defaultdict(set)
    for row in test_ratings.itertuples():
        test_dict[row.user].add(row.item)
    
    return X_hist, test_dict

def train_val_test_split_strong(ratings, n_test_users = 10000, hist_frac = .8, n_train_users = 0):
    ''' Split into train/validation/test ratings for strong generalisation.
        i.e. unseen users during training time '''
    # Sample validation and testing users without replacement
    val_test_users = np.random.choice(ratings['user'].max() + 1, size = n_test_users * 2, replace = False)
    val_users = val_test_users[:n_test_users]
    test_users = val_test_users[n_test_users:]
    
    # Extract ratings for these users from the full set
    val_ratings = ratings.merge(pd.DataFrame(val_users, columns = ['user']), how = 'right')
    test_ratings = ratings.merge(pd.DataFrame(test_users, columns = ['user']), how = 'right')
    train_ratings = pd.concat([test_ratings, val_ratings, ratings]).drop_duplicates(keep = False)

    # Split into historical and held-out sets
    nrows, ncols = ratings['user'].max() + 1, ratings['item'].max() + 1
    X_val, val_dict = generate_eval_format(val_ratings, nrows, ncols, hist_frac = hist_frac)
    X_test, test_dict = generate_eval_format(test_ratings, nrows, ncols, hist_frac = hist_frac)

    # Subsample training data if specified
    if n_train_users:
        # Randomly sample training users - only keep their ratings
        train_users = train_ratings[['user']].sample(n = n_train_users)
        train_ratings = train_ratings.merge(train_users, on = 'user', how = 'right')

    # Generate historical matrix for training ratings
    X_train, _ = generate_eval_format(train_ratings, nrows, ncols, hist_frac = 1.)

    return X_train, X_val, val_dict, X_test, test_dict

def train_val_test_split_loocb(ratings, n_train_users = 0):
    ''' Split into train/validation/test ratings via leave-one-out. '''
    # For every user - randomly sample a single item for test and validation
    val_ratings = ratings.groupby('user').apply(lambda df: df.sample(1)).reset_index(drop = True)
    rest_ratings = pd.concat([val_ratings, ratings]).drop_duplicates(keep = False)

    test_ratings = rest_ratings.groupby('user').apply(lambda df: df.sample(1)).reset_index(drop = True)
    train_ratings = pd.concat([test_ratings, rest_ratings]).drop_duplicates(keep = False)
    
    # Generate historical matrix for training ratings
    nrows, ncols = ratings['user'].max() + 1, ratings['item'].max() + 1
    X_hist, _ = generate_eval_format(train_ratings, nrows, ncols, hist_frac = 1.)
    _, val_dict = generate_eval_format(val_ratings, nrows, ncols, hist_frac = 0.)
    _, test_dict = generate_eval_format(test_ratings, nrows, ncols, hist_frac = 0.)

    # Subsample training data if specified
    if n_train_users:
        # Randomly sample training users - only keep their ratings
        train_users = train_ratings[['user']].sample(n = n_train_users)
        train_ratings = train_ratings.merge(train_users, on = 'user', how = 'right')

    # Generate historical matrix for training ratings
    X_train, _ = generate_eval_format(train_ratings, nrows, ncols, hist_frac = 1.)

    return X_train, X_hist, val_dict, X_hist, test_dict

def evaluate(X, scores, test, k_values = [1, 5, 10, 20, 50, 100], compute_item_counts = True):
    ''' Evaluate an approximation X with historical user-item matrix 'X' and user to held-out item dictionary 'test'. '''
    # Placeholder for results
    recall = defaultdict(float)
    NDCG = defaultdict(float)
    item2count = csr_matrix((1,scores.shape[0]))

    # Function per user to parallellise
    def evaluate_user(scores, items, k_values = k_values):
        # Placeholder for results per user
        item2count = None
        recall = []
        NDCG = []
        # Top-K for multiple K's
        for K in k_values:
            ##########
            # RECALL #
            ##########
            # Extract top-K highest scores into a set
            topK_list = np.argpartition(scores, -K)[-K:]
            topK_set = set(topK_list)
            # Compute recall
            recall.append(len(topK_set.intersection(items)) / min(K, len(items)))
            ########
            # NDCG #
            ########
            # Extract top-K highest scores into a sorted list
            topK_list = topK_list[np.argsort(scores[topK_list])][::-1]
            # Compute NDCG discount template
            discount_template = 1. / np.log2(np.arange(2, K + 2))
            # Compute ideal DCG
            IDCG = discount_template[:min(K, len(items))].sum()
            # Compute DCG
            DCG = sum((discount_template[rank] * (item in items)) for rank, item in enumerate(topK_list))
            # Normalise and store
            NDCG.append(DCG / IDCG)
            #############
            # LONG TAIL # 
            #############
            if K == 100:
                item2count = coo_matrix(([1] * K,([0] * K,topK_list)), shape = (1, scores.shape[0]))
        # Stack batches
        return recall + NDCG, item2count 

    # Parallellise every batch
    val = Parallel(n_jobs=-1)(delayed(evaluate_user)(scores[new_row,:].A1, items, k_values) for new_row, (user, items) in tqdm(enumerate(test.items()), total = len(test)))
    if compute_item_counts:
        # Properly extract evaluation metrics and item counts for analysis
        item2counts = [v[1] for v in val]
        item2count = vstack(item2counts).sum(axis=0).A1
    # Merge evaluation-metrics per user
    val = [v[0] for v in val]
    val = np.vstack(val)
    for idx, K in enumerate(k_values):
        recall[K] = np.mean(val[:,idx])
        NDCG[K] = np.mean(val[:,idx+len(k_values)])
    return recall, NDCG, item2count
