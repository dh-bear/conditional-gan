import tensorflow as tf
from GAN import GAN_MODEL
from GAN import d_loss, g_loss, d_acc_fake, d_acc_real, g_acc, get_dis_model, get_gen_model
import os
import tensorflow as tf


# ensures that we run only on cpu
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

##PREPROCESSING################################################################################################################################
if "X0" not in globals():
    (X0, L0), (_, _) = tf.keras.datasets.mnist.load_data()
    X0 = tf.cast(X0, tf.float32) / 255.0
    X0 = tf.expand_dims(X0, -1)

##VISUALIZATION#################################################################################################################################
##########################################################################################################################################

gan_model = GAN_MODEL(
    dis_model = get_dis_model(),
    gen_model = get_gen_model(),
    z_dims    = (None, 96),
    name      = "gan"
)

gan_model.compile(
    optimizers = {
        'd_opt' : tf.keras.optimizers.legacy.Adam(1e-3, beta_1=0.5),
        'g_opt' : tf.keras.optimizers.legacy.Adam(1e-3, beta_1=0.5),
    },
    losses = {
        'd_loss' : d_loss,
        'g_loss' : g_loss,
    },
    accuracies = {
        'd_acc_real' : d_acc_real,
        'd_acc_fake' : d_acc_fake,
        'g_acc'      : g_acc,
    }
)

train_num = 10000       ## Feel free to bump this up to 50000 when your architecture is done
true_sample = X0[train_num-2:train_num+2]                       ## 4 real images
fake_sample = gan_model.z_sampler((4, *gan_model.z_dims[1:]))   ## 4 fake images
# viz_callback = EpochVisualizer(gan_model, [true_sample, fake_sample])

gan_model.fit(
    X0[:train_num], L0[:train_num],
    dis_steps  = 5,
    gen_steps  = 5,
    epochs     = 10, ## Feel free to bump this up to 20 when your architecture is done
    batch_size = 50,
    # callbacks  = [viz_callback]
)

# viz_callback.save_gif('generation')
# IPython.display.Image(open('generation.gif','rb').read())
