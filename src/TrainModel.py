import argparse
import gc
import itertools
import numpy as np
import os
import pandas as pd
import pickle
import time
from copy import deepcopy
from datetime import datetime
from scipy.sparse import load_npz, vstack
from sklearn.preprocessing import LabelEncoder, normalize
import sys
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import torch
from tqdm import trange

import models
import util

# Only needed for tensorflow when 1st GPU is already in use
#import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if __name__ == '__main__':
    # Commandline arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', type = str, help = 'Directory containing the data')
    parser.add_argument('-a', '--algorithms', nargs='+', help='Algorithms to train and evaluate')
    parser.add_argument('-frac', '--frac_train_users', nargs='+', help='Fractions of training users to go over', default = [0.01, .05, .1, .25, .5, 1.])
    parser.add_argument('-l2', '--l2_values', nargs='+', help='Values to test for EASE\'s l2 regularisation strength', default = [50, 100, 200, 500, 1000])
    parser.add_argument('-alpha', '--alpha_values', nargs='+', help='Values to test for EASE\'s alpha side-info weight', default = np.linspace(.0,1.,21))
    parser.add_argument('-eval_style', help='Evaluation style - either strong generalisation or LOOCV', default = 'strong')
    args = parser.parse_args()

    # Fix seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    tf.set_random_seed(42)

    # Check whether output directory already exists - make it if necessary
    if not os.path.exists(args.dir + 'preprocessed/'):
        print('Directory {0} not found.\nPlease run the accompanying preprocessing script first.'.format(args.dir + 'preprocessed/'))
        exit(1)
    
    print('Directory with data:', args.dir)
    print('Models to evaluate:', args.algorithms)
    print('Evaluation style:', args.eval_style )

    # Load everything you need
    print(datetime.now(), 'Loading in data...')
    X_meta = load_npz(args.dir + 'preprocessed/X_meta.npz').astype(np.int32)
    X_train = load_npz(args.dir + 'preprocessed/X_train.npz').astype(np.int32)
    if args.eval_style == 'strong':
        X_val = load_npz(args.dir + 'preprocessed/X_val.npz').astype(np.int32)
        X_test = load_npz(args.dir + 'preprocessed/X_test.npz').astype(np.int32)
    elif args.eval_style == 'LOOCV':
        X_val = X_train
        X_test = X_train
    else:
        print('Unknown evaluation style, aborting...')
        exit(1)

    with open(args.dir + 'preprocessed/val_dict.pkl', 'rb') as handle:
        val_dict = pickle.load(handle)
    with open(args.dir + 'preprocessed/test_dict.pkl', 'rb') as handle:
        test_dict = pickle.load(handle)

    # Check whether output directory already exists - make it if necessary
    if not os.path.exists(args.dir + 'results/'):
        os.makedirs(args.dir + 'results/')
        
    # For every sampled subset of training users
    subsampling_fractions = args.frac_train_users if args.eval_style == 'strong' else [1.]
    for frac_train_users in subsampling_fractions:
        frac_train_users = float(frac_train_users)
        print(datetime.now(), '---- Frac of train users:\t{} ----'.format(frac_train_users))
    
        # Placeholder for results
        results = []
    
        # Normally, we train on everything
        train_users = np.unique(X_train.nonzero()[0])
        # But subsample when necessary
        if frac_train_users < 1.:
            # Read training users
            train_users = pd.read_csv(args.dir + 'preprocessed/train_users_{}.csv'.format(frac_train_users))['user'].values.astype(np.int32)
        # Only keep these rows in X_train
        X_train_subset = deepcopy(X_train[train_users,:])

        print('\tTraining data # users:', X_train_subset.shape[0])
        print('\tTraining data # prefs:', X_train_subset.count_nonzero())
        print('\tSide-information # tags:', X_meta.shape[0])
        print('\tSide-information # pairs:', X_meta.count_nonzero())
        print('\tTraining and evaluating models...')  
        
        for algo in args.algorithms:
            print('\t\t---- {0} ----'.format(algo))
            if algo == 'cosine':
                ###########################################
                # 1. Item-kNN (cosine) (Sarwar, WWW 2001) #
                ###########################################
                # Get dictionary with results - 0 is recall, 1 is NDCG, 2 is item counts for analysis
                recall, ndcg, item2count = models.run_itemknn(X_train_subset, X_test, test_dict)
                util.pretty_print_results((recall, ndcg))
                results.append({
                    'Recall@20': recall[20],
                    'Recall@50': recall[50],
                    'NDCG@100': ndcg[100],
                    'frac_U': frac_train_users,
                    'Alg': 'Item-kNN'
                })
                np.savez(args.dir + 'results/item_counts_{0}_{1}.npz'.format(algo, frac_train_users), item2count)

            elif algo == 'cvae':
                ####################################
                # 2. cVAE (Chen, DLRS@RecSys 2018) #
                ####################################
                # Get dictionary with results - 0 is recall, 1 is NDCG, 2 is item counts for analysis
                recall, ndcg, _ = models.run_cVAE(X_train_subset, X_meta, X_val, X_test, val_dict, test_dict)
                util.pretty_print_results((recall, ndcg))
                results.append({
                    'Recall@20': recall[20],
                    'Recall@50': recall[50],
                    'NDCG@100': ndcg[100],
                    'frac_U': frac_train_users,
                    'Alg': 'cVAE'
                })

            elif algo == 'vlm':
                ###################################################################
                # 3.a VARIATONAL LOW-RANK MULTINOMIALS (Elahi, RecSys 2019) #
                ###################################################################
                # Original implementation in Tensorflow
                # Get dictionary with results - 0 is recall, 1 is NDCG, 2 is item counts for analysis
                print('------------------ WITH SIDE INFO ---------------')
                recall, ndcg, _ = models.run_VLM(X_train_subset, train_users, normalize(X_meta, norm = 'l1', axis = 0), X_val, X_test, val_dict, test_dict, side_info = True)
                util.pretty_print_results((recall, ndcg))
                results.append({
                    'Recall@20': recall[20],
                    'Recall@50': recall[50],
                    'NDCG@100': ndcg[100],
                    'frac_U': frac_train_users,
                    'Alg': 'VLM-Side'
                })
                print('------------------ WITHOUT SIDE INFO ---------------')
                recall, ndcg, _ = models.run_VLM(X_train_subset, train_users, normalize(X_meta, norm = 'l1', axis = 0), X_val, X_test, val_dict, test_dict, side_info = False)
                util.pretty_print_results((recall, ndcg))
                results.append({
                    'Recall@20': recall[20],
                    'Recall@50': recall[50],
                    'NDCG@100': ndcg[100],
                    'frac_U': frac_train_users,
                    'Alg': 'VLM-NoSide'
                })

            elif algo == 'vlm_pytorch':
                ###################################################################
                # 3.b VARIATONAL LOW-RANK MULTINOMIALS (Elahi, RecSys 2019) #
                ###################################################################
                # Our implementation in PyTorch, generally quite a bit faster
                # Get dictionary with results - 0 is recall, 1 is NDCG, 2 is item counts for analysis
                print('------------------ WITH SIDE INFO ---------------')
                recall, ndcg, _ = models.run_VLM_PyTorch(X_train_subset, train_users, normalize(X_meta, norm = 'l1', axis = 0), X_val, X_test, val_dict, test_dict, side_info = True, eval_style = args.eval_style)
                util.pretty_print_results((recall, ndcg))
                results.append({
                    'Recall@20': recall[20],
                    'Recall@50': recall[50],
                    'NDCG@100': ndcg[100],
                    'frac_U': frac_train_users,
                    'Alg': 'VLM-PyTorch-Side'
                })
                print('------------------ WITHOUT SIDE INFO ---------------')
                recall, ndcg, _ = models.run_VLM_PyTorch(X_train_subset, train_users, normalize(X_meta, norm = 'l1', axis = 0), X_val, X_test, val_dict, test_dict, side_info = False, eval_style = args.eval_style)
                util.pretty_print_results((recall, ndcg))
                results.append({
                    'Recall@20': recall[20],
                    'Recall@50': recall[50],
                    'NDCG@100': ndcg[100],
                    'frac_U': frac_train_users,
                    'Alg': 'VLM-PyTorch-NoSide'
                })
            
            elif algo == 'slim':
                #########################################
                # 4. (c)SLIM (Ning and Karypis, ICDM 2011) #
                #########################################
                # Get dictionary with results - 0 is recall, 1 is NDCG, 2 is item counts for analysis
                print('------------------ WITHOUT SIDE INFO ---------------')
                recall, ndcg, _ = models.run_SLIM(X_train_subset, train_users, X_meta, X_val, X_test, val_dict, test_dict, side_info = False, eval_style = args.eval_style)
                util.pretty_print_results((recall, ndcg))
                results.append({
                    'Recall@20': recall[20],
                    'Recall@50': recall[50],
                    'NDCG@100': ndcg[100],
                    'frac_U': frac_train_users,
                    'Alg': 'SLIM'
                })
                print('------------------ WITH SIDE INFO ---------------')
                recall, ndcg, _ = models.run_SLIM(X_train_subset, train_users, X_meta, X_val, X_test, val_dict, test_dict, side_info = True, eval_style = args.eval_style)
                util.pretty_print_results((recall, ndcg))
                results.append({
                    'Recall@20': recall[20],
                    'Recall@50': recall[50],
                    'NDCG@100': ndcg[100],
                    'frac_U': frac_train_users,
                    'Alg': 'cSLIM'
                })

            elif algo == 'ease':
                #############################
                ## 4. EASE (Steck, WWW 2019) #
                #############################
                print(datetime.now(), '---- EASE ----')
                # Find optimal l2 on validation set via grid-search for optimal NDCG@100
                NDCG_values = []
                optimal_model_EASE = None
                optimal_l2_value = None
                val_users = list(val_dict.keys())
                # For every parameter combination
                for l2 in args.l2_values:
                    # Compute the model
                    start = time.perf_counter()
                    model = util.compute_EASE(X_train_subset, l2 = int(l2))
                    end = time.perf_counter()
                    print('\t... took {0} seconds!'.format(end - start))
                    # Evaluate the model
                    val_scores = X_val[val_users,:] @ model - 987654321 * X_val[val_users,:]
                    NDCG = util.evaluate(X_val, val_scores, val_dict)[1][100]
                    NDCG_values.append(NDCG)
                    print('\tL2:', l2, 'NDCG@100:', NDCG)
                    if np.max(NDCG_values) == NDCG:
                        optimal_model_EASE = model
                        optimal_l2_value = int(l2)
                # Compute prediction scores for all test users - subtract already seen items
                test_users = list(test_dict.keys())
                test_scores = X_test[test_users,:] @ optimal_model_EASE - 987654321 * X_test[test_users,:]
                recall, ndcg, item2count = util.evaluate(X_test, test_scores, test_dict)
                util.pretty_print_results((recall, ndcg))
                results.append({
                    'Recall@20': recall[20],
                    'Recall@50': recall[50],
                    'NDCG@100': ndcg[100],
                    'frac_U': frac_train_users,
                    'Alg': 'EASE'
                })
                #np.savez(args.dir + 'results/item_counts_{0}_{1}.npz'.format('ease', frac_train_users), item2count)
                ############################
                # 5. Add-EASE (contrib. 1) #
                ############################
                print(datetime.now(), '---- Add-EASE ----')
                NDCG_values = []
                optimal_model_ADDEASE = None
                # Compute EASE model on tag-item matrix
                side_model = util.compute_EASE(X_meta, l2 = optimal_l2_value)
                # For every parameter combination
                for alpha in args.alpha_values:
                    # Blend
                    model = (1. - float(alpha)) * optimal_model_EASE + float(alpha) * side_model
                    # Evaluate the model
                    val_scores = X_val[val_users,:] @ model - 987654321 * X_val[val_users,:]
                    NDCG = util.evaluate(X_val, val_scores, val_dict)[1][100]
                    NDCG_values.append(NDCG)
                    print('\tAlpha:', alpha, 'L2:', optimal_l2_value, 'NDCG@100:', NDCG)
                    if np.max(NDCG_values) == NDCG:
                        optimal_model_ADDEASE = model
                # Compute prediction scores for all test users - subtract already seen items
                test_scores = X_test[test_users,:] @ optimal_model_ADDEASE - 987654321 * X_test[test_users,:]
                recall, ndcg, item2count = util.evaluate(X_test, test_scores, test_dict)
                util.pretty_print_results((recall, ndcg))
                results.append({
                    'Recall@20': recall[20],
                    'Recall@50': recall[50],
                    'NDCG@100': ndcg[100],
                    'frac_U': frac_train_users,
                    'Alg': 'Add-EASE'
                })
                #np.savez(args.dir + 'results/item_counts_{0}_{1}.npz'.format('addease', frac_train_users), item2count)
                del optimal_model_EASE, side_model, optimal_model_ADDEASE

                #########################
                # 6. cEASE (contrib. 2) #
                #########################
                print(datetime.now(), '---- cEASE ----')
                NDCG_values = []
                optimal_model_CEASE = None
                # For every parameter combination
                for alpha in args.alpha_values:
                    # Stack matrix 
                    X_full = vstack((X_train_subset, X_meta * float(alpha)))
                    # Compute the model
                    model = util.compute_EASE(X_full, l2 = optimal_l2_value)
                    # Evaluate the model
                    val_scores = X_val[val_users,:] @ model - 987654321 * X_val[val_users,:]
                    NDCG = util.evaluate(X_val, val_scores, val_dict)[1][100]
                    NDCG_values.append(NDCG)
                    print('\tAlpha:', alpha, 'L2:', optimal_l2_value, 'NDCG@100:', NDCG)
                    if np.max(NDCG_values) == NDCG:
                        optimal_model_CEASE = model
                # Compute prediction scores for all test users - subtract already seen items
                test_scores = X_test[test_users,:] @ optimal_model_CEASE - 987654321 * X_test[test_users,:]
                recall, ndcg, item2count = util.evaluate(X_test, test_scores, test_dict)
                util.pretty_print_results((recall, ndcg))
                results.append({
                    'Recall@20': recall[20],
                    'Recall@50': recall[50],
                    'NDCG@100': ndcg[100],
                    'frac_U': frac_train_users,
                    'Alg': 'cEASE'
                })
                #np.savez(args.dir + 'results/item_counts_{0}_{1}.npz'.format('cease', frac_train_users), item2count)
                del optimal_model_CEASE
                gc.collect()

            else:
                print('\t\t\tUnknown algorithm, skipping...')

        # Write out results
        pd.DataFrame(results).to_csv(args.dir + 'results/{0}_{1}.csv'.format('_'.join(args.algorithms),str(frac_train_users)), index = False)
