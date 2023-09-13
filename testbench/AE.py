import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import silhouette_score


class AE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(AE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.beta = 1.0
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.cluster_loss_tracker = keras.metrics.Mean(
            name="cluster_loss"
        )

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
        ]
    def train_step(self, data):
        with tf.GradientTape() as tape:
            z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.mse(data, reconstruction), axis=(1, 2)
                )
            )
            total_loss = reconstruction_loss
            
        
        grads = tape.gradient(total_loss, self.trainable_weights)
        
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        # self.cluster_loss_tracker.update_state(self.custom_score())
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
        }
    def test_step(self, data_in):
        data, _ = data_in
        with tf.GradientTape() as tape:
            z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.mse(data, reconstruction), axis=(1, 2)
                )
            )
            total_loss = reconstruction_loss 
        grads = tape.gradient(total_loss, self.trainable_weights)
        

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.cluster_loss_tracker.update_state(self.custom_score())
        return {
            "test_loss": self.total_loss_tracker.result(),
            "test_reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "clustering_loss": self.cluster_loss_tracker.result(),
        }
    
#     def _intra_cluster_distance_slow(self, X, labels, metric, i):
#         indices = tf.where(labels == labels[i])[0]
#         if indices.shape == 0:
#             return 0.
#         a = tf.math.reduce_mean([metric(X[i]-X[j]) for j in indices if not i == j])
#         return a
    
#     def _nearest_cluster_distance_slow(self, X, labels, metric, i):
#         label = labels[i]
#         b = tf.math.minimum(
#                 [tf.math.reduce_mean(
#                     [metric(X[i]-X[j]) for j in tf.where(labels == cur_label)[0]]
#                 ) for cur_label in set(labels) if not cur_label == label])
#         return b

#     def silhouette_score(self, X, labels):
#         metric = tf.norm
#         n = labels.shape[0]
#         print(n)
#         A = tf.convert_to_tensor([self._intra_cluster_distance_slow(X, labels, metric, i)
#                       for i in range(n)])
#         B = tf.convert_to_tensor([self._nearest_cluster_distance_slow(X, labels, metric, i)
#                       for i in range(n)])
#         sil_samples = (B - A) / tf.math.maximum(A, B)
#         return np.nan_to_num(sil_samples)

    def silhouette_score(self, X):
        center_spread = []
        for i in range(30):
            center_spread.append(tf.math.reduce_std(X[i*100:(i+1)*100]))
        intra_cluster = tf.stack(center_spread, axis=0)
        
        relative_distances = []
        for k in range(30):
            for j in range(30):
                if j != k:
                    center_1 = tf.math.reduce_mean(X[j*100:(j+1)*100], axis = 0)
                    center_2 = tf.math.reduce_mean(X[k*100:(k+1)*100], axis = 0)
                    relative_distances.append(tf.norm(center_1-center_2))
        extra_cluster = tf.stack(relative_distances, axis=0)
        
        return tf.reduce_mean(intra_cluster)/ (tf.reduce_mean(extra_cluster)+0.01)
            

    def custom_score(self):
        data = tf.convert_to_tensor(np.load("../../../../../datax/scratch/pma/reverse_search/test/clustering_tests/clustering_hyperparam_test.npy")[:3_000])
        # data = np.load("../../../../../datax/scratch/pma/reverse_search/test/clustering_hyperparam_test.npy")[:30_000]
        labels = np.load("../../../../../datax/scratch/pma/reverse_search/test/clustering_tests/clustering_hyperparam_test_labels.npy")[:3_000]
        feautres = self.encoder(data)
        
        score = self.silhouette_score(X = feautres)
        return score

        
    def call(self, inputs):
        return self.decoder.predict(self.encoder.predict(inputs))
    
