import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 

# Code from Ehtsham Elahi
# https://github.com/ehtsham/recsys19vlm/blob/master/RecSys2019-VLMPaper.ipynb

class VLM(object):
    def __init__(self, num_users, num_items, num_tags, num_factors, var_prior, reg, video_metadata_array):
        self.num_users = num_users
        self.num_items = num_items
        self.num_tags = num_tags
        self.num_factors = num_factors
        self.var_prior = var_prior
        self.reg = reg
        self.video_metadata_array_const = tf.constant(video_metadata_array, dtype = tf.float32)
        self.construct_placeholders()
        
    def construct_placeholders(self):
        # Placeholders for training samples
        self.users_ph = tf.placeholder(dtype=tf.int32, shape=[None])
        self.played_videos_ph = tf.placeholder(dtype=tf.float32, shape=[None, self.num_items])
    
    def construct_model_variables(self):
        # Mean for user latent factors
        self.Mu_Zu = tf.Variable(dtype=tf.float32,
                            initial_value=tf.random_normal(shape=[self.num_users, self.num_factors]), 
                            name = 'mean_latent_factors_zu')
        # Log(std-deviation) for user latent factors
        self.lsdev_Zu = tf.Variable(dtype=tf.float32,
                               initial_value=tf.random_normal(shape=[self.num_users, 1]), name='lsdev_Zu')
        # Mean for item latent factors
        self.Mu_Zv = tf.Variable(dtype=tf.float32,
                                 initial_value=tf.random_normal(shape=[self.num_items, self.num_factors]),
                                 name = 'mean_latent_factors_zv')
        # Mean for tag latent factors
        self.Mu_Zt = tf.Variable(dtype=tf.float32,
                            initial_value=tf.random_normal(shape=[self.num_tags, self.num_factors]),
                            name = 'mean_latent_factors_zt')
        
    def compute_kl_div(self, lsdev_Zu_batch, Mu_Zu_batch):
        # KL Divergence needed for ELBO
        sdev_Zu_batch = tf.exp(lsdev_Zu_batch)
        comp1 = self.num_factors * (0.5 * tf.math.log(self.var_prior) - lsdev_Zu_batch)
        comp2 = (self.num_factors / (2 * self.var_prior)) * (tf.pow(sdev_Zu_batch, 2))
        comp3 = (1.0 / (2 * self.var_prior)) * tf.reduce_sum(tf.pow(Mu_Zu_batch, 2), axis=1, keep_dims = True)
        comp4 = (self.num_factors / 2.0)

        return comp1 + comp2 + comp3 - comp4
        
    def construct_graph(self):
        # Boilerplate Tensorflow
        self.construct_model_variables()
        
        # Mean, log(std-deviation) and Gaussian noise for user latent factors
        Mu_Zu_batch = tf.gather(self.Mu_Zu, self.users_ph)
        lsdev_Zu_batch = tf.gather(self.lsdev_Zu, self.users_ph)
        Eps_u_ph = tf.random_normal(shape = [tf.size(self.users_ph), self.num_factors],
                                    mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name="eps")
        Zu_batch = Mu_Zu_batch + Eps_u_ph * tf.exp(lsdev_Zu_batch)
        
        # Tag factors mapped to items
        Mu_Zv_hat = tf.matmul(self.video_metadata_array_const, self.Mu_Zt)
        batch_logits = tf.matmul(Zu_batch, self.Mu_Zv + Mu_Zv_hat, transpose_b=True)
        batch_logits_validation = tf.matmul(Mu_Zu_batch, self.Mu_Zv + Mu_Zv_hat, transpose_b=True)
    
        log_softmax = tf.nn.log_softmax(batch_logits)
        
        num_items_per_document = tf.reduce_sum(self.played_videos_ph, axis=1, keep_dims=True)
        
        batch_conditional_log_likelihood = tf.reduce_sum(self.played_videos_ph * log_softmax, axis = 1, keep_dims=True)
        batch_kl_div = self.compute_kl_div(lsdev_Zu_batch, Mu_Zu_batch)
        
        batch_elbo = (1.0 / num_items_per_document) * (batch_conditional_log_likelihood - batch_kl_div)
        
        avg_loss = -1 * tf.reduce_mean(batch_elbo) + self.reg * (tf.nn.l2_loss(self.Mu_Zv) +
                                                                 tf.nn.l2_loss(self.Mu_Zt))
        
        return batch_logits, batch_logits_validation, log_softmax, avg_loss, batch_conditional_log_likelihood, batch_kl_div, num_items_per_document
