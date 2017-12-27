from keras.layers import Input, Embedding, Dense, Lambda, merge, concatenate, Activation
from keras.models import Model, load_model
import keras.backend as K
import tensorflow as tf
import pickle


class SiameseCBOW():
    """Implementation of SiameseCBOW using keras.
    
    Args:
        input_dim (int): the size of the vocabulary
        output_dim (int): the length of the distributed representation of the word vectors
        input_length (int): the length of the sentences
        n_positive (int): the number of the positive samples
        n_negative (int): the number of the negatice samples
    """
    def __init__(input_dim, output_dim, input_length=100, n_positive=2, n_negative=5):
        def antirectifier(x):
            sums = K.sum(x, axis=1, keepdims=False)
            normalisers = tf.count_nonzero(
                tf.count_nonzero(x, axis=2, keep_dims=False, dtype=tf.float32),
                axis=1, keep_dims=True, dtype=tf.float32)
            return sums / normalisers
        def antirectifier_output_shape(input_shape):
            shape = list(input_shape)
            assert len(shape) == 3  # only valid for 3D tensors
            return (shape[0], shape[-1],)
        def cossim(x):
            dot_products = K.batch_dot(x[0], x[1], axes=[1,1])
            norm0 = tf.norm(x[0], ord=2, axis=1, keep_dims=True)
            norm1 = tf.norm(x[1], ord=2, axis=1, keep_dims=True)
            return dot_products / norm0 / norm1
        main_input = Input(shape=(input_length,), dtype='int32', name='main_input')
        pos_inputs = [Input(shape=(input_length,), dtype='int32', name='positive_input_{}'.format(i)) for i in range(n_positive)]
        neg_inputs = [Input(shape=(input_length,), dtype='int32', name='negative_input_{}'.format(i)) for i in range(n_negative)]
        embed = Embedding(output_dim=output_dim, input_dim=input_dim, input_length=input_length, name='embedding')
        s = embed(main_input)
        s_p = [embed(i) for i in pos_inputs]
        s_n = [embed(i) for i in neg_inputs]
        ave = Lambda(antirectifier, output_shape=antirectifier_output_shape, name='average')
        ave_s = ave(s)
        ave_s_p = [ave(i) for i in s_p]
        ave_s_n = [ave(i) for i in s_n]
        cos_p = [merge([ave_s, l], mode=lambda x: cossim(x), output_shape=(1,), name='p_cos_sim_{}'.format(i)) for i, l in enumerate(ave_s_p)]
        cos_n = [merge([ave_s, l], mode=lambda x: cossim(x), output_shape=(1,), name='n_cos_sim_{}'.format(i)) for i, l in enumerate(ave_s_n)]
        z = concatenate(cos_p + cos_n, axis=1, name='concatenate')
        pred = Activation('softmax')(z)
        model = Model(inputs=[main_input] + pos_inputs + neg_inputs, outputs=pred)
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        self.model = model
    
    def fit(self, x, y, epochs=50):
        """Train the model
        
        Args:
            x: input
                x must be a list of (1 + n_positive + n_negative) numpy.ndarray elements
                and the shape of each element should be (batch_size, input_length).
                The order of the list must be in a order like [a target sentence, positive samples, negative samples]
            y: output
                y must be a numpy.ndarray whose shape is (batch_size, (n_positice + n_negative)).
        """
        self.model.fit(x, y, epochs=epochs)
    
    def predict(self, x):
        return self.model.predict(x)
    
    def get_embedding_vectors(self):
        return self.model.get_weights()[0]
    
    def save_embedding_vectors(self, path):
        with open(path, mode='wb') as f:
            pickle.dump(self.get_embed_vectors, f)
    
    def save(self, path):
        self.model.save(path)
    
    def load(self, path):
        self.model = load_model(path)
    