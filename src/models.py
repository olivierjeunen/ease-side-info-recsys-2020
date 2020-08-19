from copy import deepcopy
import itertools
import numpy as np
import pandas as pd
import util
from scipy.sparse import lil_matrix, coo_matrix, csr_matrix, vstack
from sklearn.preprocessing import normalize
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import torch
import torch.utils.data
from tqdm import trange

from baselines.cvae.vae import VAE
from baselines.vlm.vlm import VLM
from baselines.vlm.vlm_pytorch import VLM_PyTorch

from SLIM import SLIM, SLIMatrix

def run_itemknn(X_train, X_test, test_dict):
    # Compute item-item matrix with cosine similarities
    S_cosine = util.compute_cosine(X_train)

    # Compute prediction scores for all test users - subtract already seen items
    test_users = list(test_dict.keys())
    test_scores = X_test[test_users,:] @ S_cosine - 987654321 * X_test[test_users,:]
    
    # Evaluate and pretty print
    results_cosine = util.evaluate(X_test, test_scores, test_dict)
    return results_cosine

def run_cVAE(X_train, X_meta, X_val, X_test, val_dict, test_dict):
    # Parameters for cVAE
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    params = {
        'layers': [400, 100],
        'n_items': X_meta.shape[1],
        'device': device 
    }
   
    batch_size = 1024
    tol = 1e-7
    patience = 200
    alpha = 1.
    beta = 1.
    # Instantiate model
    model = VAE(params).to(device)
    
    # Multi-GPU 
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        batch_size = int(batch_size * 2)

    # Set up optimizer
    optimizer = torch.optim.Adam(model.parameters(),lr=5e-4)

    # Loss function is MSE and annealed ELBO
    def loss_function(recon_x, x, mu, logvar, anneal=1.0):
        MSE = torch.sum((x - torch.sigmoid(recon_x)) ** 2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return MSE + anneal * KLD

    # Pre-train on meta-data
    train_tensor = torch.from_numpy(X_meta.A.astype('float32')).to(device)
    train_loader = torch.utils.data.DataLoader(train_tensor, batch_size, shuffle=True)
    t = trange(2000, desc = 'Meta')
    best_loss, best_epoch = np.inf, -1
    for epoch in t:
        # Put the model into training mode
        model.train()
        loss_value = 0
        # Every batch
        for batch_idx, data in enumerate(train_loader):
            # Clear gradients
            optimizer.zero_grad()
            # Get predictions
            recon_batch, mu, logvar = model(data)
            # Compute loss
            loss = loss_function(recon_batch, data, mu, logvar, anneal = 0.2)
            # Back-propagate 
            loss.backward()
            loss_value += loss.item()
            optimizer.step()
        loss = loss_value / len(train_loader.dataset)
        t.set_postfix(loss = loss)
        # Early stopping - are we improving by at least 'tol'?
        if (best_loss - loss) > tol:
            # If yes - keep going
            best_loss = loss
            best_epoch = epoch
        # If we're not improving, have we improved at all in the past 'patience' epochs?
        if (epoch - best_epoch) > patience:
            print('Converged after {0} epochs, stopping...'.format(epoch))
            break

    del train_tensor, train_loader
    torch.cuda.empty_cache()
    
    # Loss function is cross-entropy and annealed ELBO
    def loss_function(recon_x, x, mu, logvar, anneal=1., alpha = 1.):
        BCE = -torch.sum(alpha * torch.log(torch.sigmoid(recon_x) + 1e-8) * x + torch.log(1 - torch.sigmoid(recon_x) + 1e-8) * (1 - x))
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + anneal * KLD
    
    # Refine on ratings
    # Get rating data into torch format
    train_tensor = torch.from_numpy(X_train.A.astype('float32')).to(device)
    train_loader = torch.utils.data.DataLoader(train_tensor, batch_size, shuffle=True)
    t = trange(2000, desc = 'Pref')
    best_loss, best_epoch = np.inf, -1
    for epoch in t:
        # Put the model into training mode
        model.train()
        loss_value = 0
        # Every batch
        for batch_idx, data in enumerate(train_loader):
            # Clear gradients
            optimizer.zero_grad()
            # Get predictions
            recon_batch, mu, logvar = model(data)
            # Compute loss
            loss = loss_function(recon_batch, data, mu, logvar, anneal = beta, alpha = alpha)
            # Back-propagate 
            loss.backward()
            loss_value += loss.item()
            optimizer.step()
        loss = loss_value / len(train_loader.dataset)
        t.set_postfix(loss = loss)
        # Early stopping - are we improving by at least 'tol'?
        if (best_loss - loss) > tol:
            # If yes - keep going
            best_loss = loss
            best_epoch = epoch
        # If we're not improving, have we improved at all in the past 'patience' epochs?
        if (epoch - best_epoch) > patience:
            print('Converged after {0} epochs, stopping...'.format(epoch))
            break
        
    del train_tensor, train_loader
    torch.cuda.empty_cache()
    
    # Scores for test set
    test_users = list(test_dict.keys())
    with torch.no_grad():
        test_tensor = torch.from_numpy(X_test[test_users,:].A.astype('float32')).to(device)
        test_loader = torch.utils.data.DataLoader(test_tensor, batch_size, shuffle=False)
        pred_test = []
        for batch_idx, data in enumerate(test_loader):
            scores, _, _ = model(data)
            pred_test.append(scores.detach().cpu().numpy())
        pred_test = np.vstack(pred_test)
    # Subtract previously seen items from predicted scores
    test_scores = pred_test - 987654321 * X_test[test_users,:]
    results_cVAE = util.evaluate(X_test, test_scores, test_dict)
    return results_cVAE

def run_VLM(X_train_subset, train_users, X_meta, X_val, X_test, val_dict, test_dict, side_info = True):
    # Parameters for VLM
    var_prior = 1.0
    lr = 5e-3
    reg = 1e-9
    num_factors = 100
    batch_size = 512
    num_epochs = 1500

    if not side_info:
        X_meta = csr_matrix((1,X_train_subset.shape[1]))

    # Specific input format for VLM
    val_users = list(val_dict.keys())
    test_users = list(test_dict.keys()) 

    # X_train_subset only has relevant non-zero rows at the moment
    # We want a matrix with zeroes everywhere but these training vectors in the right spot
    X_train_full = lil_matrix(X_val.shape).astype(np.int32)
    X_train_full[train_users,:] = X_train_subset
    X_train_full = X_train_full.tocsr()
    X_all = (X_train_full+X_val+X_test).toarray().astype(np.float32)
    video_metadata_array = X_meta.T.todense().astype(np.float32)

    ##############################
    # TRAINING PROCEDURE FOR VLM #
    ##############################
    tf.reset_default_graph()
    
    # Instantiate TensorFlow Execution DAG
    with tf.Graph().as_default():
        # Generate model
        model = VLM(X_test.shape[0], # Num users
                    X_test.shape[1], # Num items
                    X_meta.shape[0], # Num tags
                    num_factors,
                    var_prior,
                    reg,
                    video_metadata_array) 

        # Innitialise model
        batch_logits, batch_logits_validation, log_softmax, avg_loss, batch_conditional_log_likelihood,\
        batch_kl_div, num_items_per_document = model.construct_graph()

        # Optimisation procedure for training users
        train_op = tf.train.AdamOptimizer(learning_rate=lr)\
        .minimize(avg_loss, global_step=tf.Variable(0, name='global_step_1', trainable=False))

        # Optimisation procedure for validation and test users (keep items/tags fixed)
        train_op_validation = tf.train.AdamOptimizer(learning_rate=lr)\
        .minimize(avg_loss,
                  var_list = [model.Mu_Zu, model.lsdev_Zu],
                  global_step=tf.Variable(0, name='global_step_1_validation', trainable=False))

        ####Summary####
        avg_loss_summary_ph = tf.placeholder(dtype = tf.float32)
        tf.summary.scalar('avg_loss', avg_loss_summary_ph)

        ndcg_summary_ph = tf.placeholder(dtype=tf.float32)
        tf.summary.scalar('ndcg_100', ndcg_summary_ph)
        summary = tf.summary.merge_all()

        ####Start####
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # Initialise session
        with tf.Session(config=config) as sess:
            
            sess.run(init)
            ndcgs_vad = []
            best_ndcg_sofar = -1000
            # For every epoch
            progress_bar = trange(num_epochs)
            for epoch_ind in progress_bar:
                ####################################
                ## COMPUTATIONS FOR TEST SET #
                ####################################
                ###Optimize parameters for test users ####    
                for batch_ind, st_index in enumerate(range(0, len(test_users), batch_size)):
                    # Put batch into the right format
                    end_index = min(st_index + batch_size, len(test_users))
                    user_indices = test_users[st_index:end_index]
                    # Optimise user factors for validation data
                    _, loss_val = sess.run([train_op_validation, avg_loss], feed_dict = {model.users_ph : user_indices, model.played_videos_ph : X_all[user_indices]})

                ##################################
                ## COMPUTATIONS FOR TRAINING SET #
                ##################################
                avg_loss_dataset = 0
                num_batches = 0
                np.random.shuffle(train_users)
                ## For every training batch
                for batch_ind, st_index in enumerate(range(0, len(train_users), batch_size)):
                    # Put batch into the right format
                    end_index = min(st_index + batch_size, len(train_users))
                    user_indices = train_users[st_index:end_index]
                    # Optimise user factors for training data
                    _, loss_val = sess.run([train_op, avg_loss], feed_dict = {model.users_ph : user_indices, model.played_videos_ph : X_all[user_indices,:]})
                    avg_loss_dataset += loss_val
                    num_batches += 1
                # Average out loss
                avg_loss_dataset = avg_loss_dataset / max(num_batches, 1)
                ####Summary####
                progress_bar.set_postfix(loss = avg_loss_dataset)
            
            # Compute NDCG on test set
            predictions_test = sess.run(batch_logits_validation, feed_dict = {model.users_ph: test_users})
    # Clear model, memory and such
    tf.reset_default_graph()

    # Subtract previously seen items from predicted scores
    test_scores = predictions_test - 987654321 * X_test[test_users,:]
    VLM_results = util.evaluate(X_test, test_scores, test_dict)
    return VLM_results

def run_VLM_PyTorch(X_train, train_users, X_meta, X_val, X_test, val_dict, test_dict, side_info, eval_style = 'strong'):
    # Parameters for VLM
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    params = {
        'num_users': X_val.shape[0],
        'num_items': X_val.shape[1],
        'num_tags': X_meta.shape[0],
        'num_factors': 100,
        'var_prior': 1.0,
        'reg': 1e-9,
        'device': device,
        'item_tag_mat': X_meta.T.astype(np.float32),
        'side_info': side_info
    }
    lr = 3e-3
    num_epochs = 5000
    batch_size = 1024
    tol = 1e-8
    patience = 50

    # Heldout data from dictionary to csr matrix 
    vals = np.ones(len(val_dict))
    rows = np.asarray(list(val_dict.keys()))
    cols = np.asarray([list(v)[0] for v in val_dict.values()])
    val_csr = csr_matrix((vals,(rows,cols)), shape = X_val.shape)
   
    # Instantiate model
    model = VLM_PyTorch(params).to(device)
    
    # Multi-GPU if possible
    multi_gpu = False
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        multi_gpu = True
        batch_size = int(batch_size * 2)

    def compute_kl_div(lsdev_Zu_batch, Mu_Zu_batch, num_factors, var_prior):
        sdev_Zu_batch = torch.exp(lsdev_Zu_batch)
        comp1 = num_factors * (0.5 * np.log(var_prior) - lsdev_Zu_batch)
        comp2 = (num_factors / (2 * var_prior)) * sdev_Zu_batch.pow(2)
        comp3 = (1.0 / (2 * var_prior)) * torch.sum(Mu_Zu_batch.pow(2), dim = 1)
        comp4 = (num_factors / 2.0)
        return comp1 + comp2 + comp3 - comp4
    
    def loss_function(x, scores, Mu_Zu, lsdev_Zu, num_factors = params['num_factors'], var_prior = params['var_prior'], reg = params['reg']):
        scores = scores.masked_fill(~x,.0)
        batch_conditional_log_likelihood = torch.sum(scores, dim = 1)
        batch_kl_div = compute_kl_div(lsdev_Zu, Mu_Zu, num_factors, var_prior)
        items_per_user = torch.sum(x, dim = 1, dtype = torch.float)
        batch_elbo = (1.0 / items_per_user) * (batch_conditional_log_likelihood - batch_kl_div)
        if multi_gpu and side_info:
            return -1 * torch.mean(batch_elbo) + reg * (torch.norm(model.module.Mu_Zv.weight, 2) + torch.norm(model.module.Mu_Zt.weight, 2))
        elif (not multi_gpu) and side_info:
            return -1 * torch.mean(batch_elbo) + reg * (torch.norm(model.Mu_Zv.weight, 2) + torch.norm(model.Mu_Zt.weight, 2))
        elif multi_gpu and (not side_info):
            return -1 * torch.mean(batch_elbo) + reg * (torch.norm(model.module.Mu_Zv.weight, 2))
        elif (not multi_gpu) and (not side_info):
            return -1 * torch.mean(batch_elbo) + reg * (torch.norm(model.Mu_Zv.weight, 2))

    # Set up data for training
    train_tensor = torch.from_numpy(X_train.A.astype(bool)).to(device)
    train_users = torch.from_numpy(train_users.astype(np.int64)).to(device)
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_tensor, train_users), batch_size, shuffle=True)

    # Set up data for validation 
    val_batch_size = batch_size #1024
    val_users = np.asarray(list(val_dict.keys()))
    val_tensor = torch.from_numpy(X_val[val_users,:].A.astype(np.bool)).to(device)
    heldout_val_tensor = torch.from_numpy(val_csr[val_users,:].A.astype(np.float32)).to(device)
    val_users = torch.from_numpy(val_users.astype(np.int64)).to(device)
    val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(val_tensor, heldout_val_tensor, val_users), val_batch_size, shuffle=False)

    # Optimise everything for training data
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)

    # For every epoch
    t = trange(num_epochs, desc = 'Train')
    best_loss, best_epoch = np.inf, -1
    for epoch in t:
        # Put the model into training mode
        model.train()
        loss_value = 0
        # Optimise for every batch of training data
        for batch_idx, (data, users) in enumerate(train_loader):
            # Clear gradients
            optimizer.zero_grad()
            # Get predictions
            scores, Mu_Zu, lsdev_Zu = model(users)
            # Compute loss
            loss = loss_function(data, scores, Mu_Zu, lsdev_Zu)
            # Back-propagate 
            loss.backward()
            loss_value += loss.item()
            optimizer.step()
        loss = loss_value / len(train_loader.dataset)
        t.set_postfix(loss = loss)

        # Early stopping - are we improving by at least 'tol'?
        if (best_loss - loss) > tol:
            # If yes - keep going
            best_loss = loss
            best_epoch = epoch
        # If we're not improving, have we improved at all in the past 'patience' epochs?
        if (epoch - best_epoch) > patience:
            print('Converged after {0} epochs, stopping...'.format(epoch))
            break

    del train_tensor, train_users, train_loader
    torch.cuda.empty_cache()

    # Scores for test set
    # Only optimise user vectors for test data
    test_users = np.asarray(list(test_dict.keys()))
    test_tensor = torch.from_numpy(X_test[test_users,:].A.astype(bool)).to(device)
    # If we have distinct train/test users - we should learn vectors for test users
    if eval_style == 'strong':
        if multi_gpu:
            optimizer = torch.optim.Adam([model.module.Mu_Zu.weight, model.module.lsdev_Zu.weight],lr=lr)
        else:
            optimizer = torch.optim.Adam([model.Mu_Zu.weight, model.lsdev_Zu.weight],lr=lr)
        test_users = torch.from_numpy(test_users.astype(np.int64)).to(device)
        test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_tensor, test_users), batch_size, shuffle=True)
        t = trange(num_epochs, desc = 'Test ')
        best_loss, best_epoch = np.inf, -1
        for epoch in t:
            # Put the model into training mode
            model.train()
            loss_value = 0
            # Every batch
            for batch_idx, (data, users) in enumerate(test_loader):
                # Clear gradients
                optimizer.zero_grad()
                # Get predictions
                scores, Mu_Zu, lsdev_Zu = model(users)
                # Compute loss
                loss = loss_function(data, scores, Mu_Zu, lsdev_Zu)
                # Back-propagate 
                loss.backward()
                loss_value += loss.item()
                optimizer.step()
            loss = loss_value / len(test_loader.dataset)
            t.set_postfix(loss = loss)
            # Early stopping - are we improving by at least 'tol'?
            if (best_loss - loss) > tol:
                # If yes - keep going
                best_loss = loss
                best_epoch = epoch
            # If we're not improving, have we improved at all in the past 'patience' epochs?
            if (epoch - best_epoch) > patience:
                print('Converged after {0} epochs, stopping...'.format(epoch))
                break

    # Scores for test set - only for optimal model
    torch.cuda.empty_cache()
    with torch.no_grad():
        test_users = np.asarray(list(test_dict.keys()))
        test_users = torch.from_numpy(test_users.astype(np.int64)).to(device)
        test_loader = torch.utils.data.DataLoader(test_users, batch_size, shuffle=False)
        pred_test = []
        for batch_idx, users in enumerate(test_loader):
            scores, _, _ = model(users, add_noise = False)
            pred_test.append(scores.detach().cpu().numpy())
        pred_test = np.vstack(pred_test)

    # Subtract previously seen items from predicted scores
    test_users = np.asarray(list(test_dict.keys()))
    test_scores = pred_test - 987654321 * X_test[test_users,:]
    results_VLM = util.evaluate(X_test, test_scores, test_dict)
    return results_VLM

def run_SLIM(X_train, train_users, X_meta, X_val, X_test, val_dict, test_dict, side_info, eval_style = 'strong'):
    # Values for grid-search
    NDCG_values = []
    optimal_model_SLIM = None
    best_values = None
    l1_values = [0, 2.5, 5.0, 10.0, 20] 
    l2_values = [0, 5.0, 10.0, 20, 50, 100]
    al_values = [.5, 1.0, 2.5, 5.0, 10.0] if side_info else [1.0]
    for l1r, l2r, alpha in itertools.product(l1_values, l2_values, al_values): 
        print('L1: {0}\tL2: {1}\tAlpha: {2}'.format(l1r,l2r, alpha))
        # Set up parameters
        params = {'algo':'cd', 'nthreads':16, 'l1r':l1r, 'l2r':l2r}

        # Build training matrix
        trainmat = X_train
        if side_info:
            trainmat = vstack((trainmat, alpha * X_meta))
        trainmat = SLIMatrix(trainmat)

        # Train model
        model = SLIM()
        model.train(params, trainmat)
        print('Converting out of SLIM format...')
        # To CSR works, but densifying it crashes sometimes? Very strange
        # S_SLIM = model.to_csr().todense()
        # Work-around by writing to disk and reading in
        model.save_model(modelfname='slim_model.csr', mapfname='slim_map.csr')
        def read_csr(filename):
            f = open(filename, 'r')
            all_rows = []
            all_cols = []
            all_vals = []
            for i, line in enumerate(f.readlines()):
                strs = line.split(' ')
                cols = [int(s) for s in strs[1::2]]
                vals = [float(s) for s in strs[2::2]]
                all_cols.extend(cols)
                all_vals.extend(vals)
                all_rows.extend([i for _ in cols])
            all_rows = np.array(all_rows, dtype=np.int64)
            all_cols = np.array(all_cols, dtype=np.int64)
            all_vals = np.array(all_vals, dtype=np.float32)
            mat = coo_matrix((all_vals, (all_rows, all_cols)), shape = (X_train.shape[1],X_train.shape[1]))
            return mat
        S_SLIM = read_csr('slim_model.csr')
        print('... done!')
        S_SLIM = S_SLIM.todense()

        # Evaluate on validation data
        print('Evaluating...')
        val_users = list(val_dict.keys())
        val_scores = X_val[val_users,:] @ S_SLIM - 987654321 * X_val[val_users,:]
        
        # Evaluate and pretty print
        NDCG = util.evaluate(X_val, val_scores, val_dict)[1][100]
        NDCG_values.append(NDCG)

        print('\tNDCG@100:\t{0}'.format(NDCG))
        if np.max(NDCG_values) == NDCG:
            optimal_model_SLIM = S_SLIM
            best_values = (l1r, l2r, alpha)

    print('Best grid-search values:', best_values)

    # Compute prediction scores for all test users - subtract already seen items
    test_users = list(test_dict.keys())
    test_scores = X_test[test_users,:] @ optimal_model_SLIM - 987654321 * X_test[test_users,:]
    
    # Evaluate and pretty print
    results_SLIM = util.evaluate(X_test, test_scores, test_dict)
    return results_SLIM
