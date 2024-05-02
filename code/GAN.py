import tensorflow as tf
import numpy as np
import IPython.display



"""
Compute the GAN loss.

Inputs:
- logits_real: Tensor, shape [batch_size, 1], output of discriminator for each real image
- logits_fake: Tensor, shape[batch_size, 1], output of discriminator for each fake image
"""

bce_func = tf.keras.backend.binary_crossentropy
acc_func = tf.keras.metrics.binary_accuracy

# TODO: fill in loss functions!
def d_loss(d_fake:tf.Tensor, d_real:tf.Tensor)  -> tf.Tensor:
    real_loss = bce_func(tf.ones_like(d_real), d_real)  # Real images should be classified as 1
    fake_loss = bce_func(tf.zeros_like(d_fake), d_fake)  # Fake images should be classified as 0
    return tf.reduce_mean(real_loss + fake_loss)

def g_loss(d_fake:tf.Tensor, d_real:tf.Tensor) -> tf.Tensor: return tf.reduce_mean(bce_func(tf.ones_like(d_fake), d_fake))    

def d_acc_fake(d_fake:tf.Tensor, d_real:tf.Tensor) -> tf.Tensor: return tf.reduce_mean(acc_func(tf.zeros_like(d_fake), d_fake))

def d_acc_real(d_fake:tf.Tensor, d_real:tf.Tensor) -> tf.Tensor: return tf.reduce_mean(acc_func(tf.ones_like(d_real), d_real))

def g_acc(d_fake:tf.Tensor, d_real:tf.Tensor) -> tf.Tensor: return tf.reduce_mean(acc_func(tf.ones_like(d_fake), d_fake))


def get_dis_model(name="dis_model"):
    return tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256),
        tf.keras.layers.LeakyReLU(alpha=0.01),
        tf.keras.layers.Dense(256),
        tf.keras.layers.LeakyReLU(alpha=0.01),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ], name=name)

def get_gen_model(name="gen_model"):
    return tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(96,)),
        tf.keras.layers.Dense(1000),
        tf.keras.layers.LeakyReLU(alpha=0.01),
        tf.keras.layers.Dense(1000),
        tf.keras.layers.LeakyReLU(alpha=0.01),
        tf.keras.layers.Dense(784, activation='tanh'),
        tf.keras.layers.Reshape((28, 28, 1))
    ], name=name)

################################################################################################################################################
# GAN CLASS
################################################################################################################################################
class GAN_MODEL(tf.keras.Model):
    '''
    self.gen_model = generator model;           z_like -> x_like
    self.dis_model = discriminator model;       x_like -> probability
    self.z_sampler = sampling strategy for z;   z_dims -> z
    self.z_dims    = dimensionality of generator input
    self.sample_z(n)      = sample n sampled z realizations
    self.generate(z)      = generate x-likes from provided z
    self.discriminate(x)  = predict whether x-like is real or generated
    '''
    def __init__(self, dis_model, gen_model, z_dims, z_sampler=tf.random.normal, **kwargs):
        '''
        self.gen_model = generator model;           z_like -> x_like
        self.dis_model = discriminator model;       x_like -> probability
        self.z_sampler = sampling strategy for z;   z_dims -> z
        self.z_dims    = dimensionality of generator input
        '''
        super().__init__(**kwargs)
        self.z_dims = z_dims
        self.z_sampler = z_sampler
        self.gen_model = gen_model
        self.dis_model = dis_model

    def call(self, inputs, **kwargs):
        b_size = tf.shape(inputs)[0]
        z_samp = self.sample_z(b_size)          #tf.constant([0.])   ## Generate a z sample
        g_samp = self.generate(z_samp)          #tf.constant([0.])   ## Generate an x-like image
        d_samp = self.discriminate(g_samp)      #tf.constant([0.])   ## Predict whether x-like is real
        return d_samp


    def sample_z(self, num_samples, **kwargs): return self.z_sampler([num_samples, *self.z_dims[1:]]) #'''generates an z based on the z sampler'''
    def discriminate(self, inputs, **kwargs): return self.dis_model(inputs, **kwargs) #'''predict whether input input is a real entry from the true dataset'''
    def generate(self, z, **kwargs): return self.gen_model(z, **kwargs) #'''generates an output based on a specific z realization'''
        

    def compile(self, optimizers, losses, accuracies, **kwargs):
        super().compile(
            loss        = losses.values(),
            optimizer   = optimizers.values(),
            metrics     = accuracies.values(),
            **kwargs
        )
        self.loss_funcs = losses
        self.optimizers = optimizers
        self.acc_funcs  = accuracies


    def fit(self, *args, dis_steps=1, gen_steps=1, **kwargs):
        self.gen_steps = gen_steps
        self.dis_steps = dis_steps
        super().fit(*args, **kwargs)


    def test_step(self, data):
        x_real, l_real = data                                  ## - x_real: Real Images from dataset
        batch_size = tf.shape(x_real)[0]

        z_samp = self.z_sampler((batch_size, *self.z_dims[1:]))
        x_fake = self.gen_model(z_samp, training=False)        ## - x_fake: Images generated by generator
        d_real = self.dis_model(x_real, training=False)        ## - d_real: The discriminator's prediction of the reals
        d_fake = self.dis_model(x_fake, training=False)        ## - d_fake: The discriminator's prediction of the fakes

        ########################################################################

        metrics = dict()
        all_funcs = {**self.loss_funcs, **self.acc_funcs}
        for key, func in all_funcs.items():
            metrics[key] = func(d_fake, d_real)

        return metrics


    def train_step(self, data):
        x_real, l_real = data
        batch_size = tf.shape(x_real)[0]

        z_samp = tf.constant([0.])

        for i in range(self.dis_steps):
            with tf.GradientTape() as tape:
                z_samp = self.z_sampler((batch_size, *self.z_dims[1:]))
                x_fake = self.gen_model(z_samp, training=True)
                d_real = self.dis_model(x_real, training=True)
                d_fake = self.dis_model(x_fake, training=True)
                d_loss = self.loss_funcs['d_loss'](d_fake, d_real)
            grads = tape.gradient(d_loss, self.dis_model.trainable_variables)
            self.optimizers['d_opt'].apply_gradients(zip(grads, self.dis_model.trainable_variables))

        for j in range(self.gen_steps):
            with tf.GradientTape() as tape:
                z_samp = self.z_sampler((batch_size, *self.z_dims[1:]))
                x_fake = self.gen_model(z_samp, training=True)
                d_fake = self.dis_model(x_fake, training=True)
                # If g_loss requires both d_fake and d_real:
                g_loss = self.loss_funcs['g_loss'](d_fake, d_real)
            grads = tape.gradient(g_loss, self.gen_model.trainable_variables)
            self.optimizers['g_opt'].apply_gradients(zip(grads, self.gen_model.trainable_variables))
        ########################################################################

        metrics = dict()
        all_funcs = {**self.loss_funcs, **self.acc_funcs}
        for key, func in all_funcs.items():
            metrics[key] = func(d_fake, d_real)

        return metrics


