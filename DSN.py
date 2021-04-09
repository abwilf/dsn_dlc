from utils import *
import tensorflow_addons as tfa
import tensorflow as tf
from tensorflow.keras.layers import TimeDistributed, Flatten, Dense, Input, Activation, Dropout, Bidirectional, LSTM, Conv1D, Conv1DTranspose, Masking, MaxPool1D, Flatten, Layer, Reshape, UpSampling1D, ZeroPadding1D, ReLU, GlobalMaxPool1D, Lambda
import tensorflow.keras.backend as K
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping, Callback
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from scipy.stats import mode

class CLF_Layer(Layer):
    def __init__(self, k):
        super().__init__()
        self.DENSE1 = Dense(128, activation='relu')
        self.DENSE2 = Dense(128, activation='relu')
        self.DENSE3 = Dense(k, activation='softmax')

    def call(self, mfbs):
        mfbs = self.DENSE1(mfbs)
        mfbs = self.DENSE2(mfbs)
        mfbs = self.DENSE3(mfbs)
        return mfbs

class Conv_In(Layer):
    def __init__(self, filters, kernel_size, strides, activation, dilation_rate):
        super().__init__(name=self.__class__.__name__)
        self.conv = Conv1D(filters=filters, kernel_size=kernel_size, strides=strides, activation=activation, dilation_rate=dilation_rate, padding='same', data_format='channels_last', dtype='float32')
        self.IN = tfa.layers.InstanceNormalization(axis=-1, center=True, scale=True, beta_initializer="random_uniform", gamma_initializer="random_uniform")
    
    def call(self, x, training):
        x = self.conv(x)
        x = self.IN(x)
        return x

class Deconv_In(Layer):
    def __init__(self, filters, kernel_size, strides, activation):
        super().__init__(name=self.__class__.__name__)
        self.conv = Conv1DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, activation=activation, padding='same', data_format='channels_last', dtype='float32')
        self.IN = tfa.layers.InstanceNormalization(axis=-1, center=True, scale=True, beta_initializer="random_uniform", gamma_initializer="random_uniform")
    
    def call(self, x, training):
        x = self.conv(x)
        x = self.IN(x)
        return x

class Downsampler(Layer):
    def __init__(self, k, train_batch, activation):
        super().__init__(name=self.__class__.__name__)
        self.layer1 = Conv_In(filters=128, kernel_size=16, strides=1, activation=activation, dilation_rate=2)
        self.layer2 = Conv_In(filters=100, kernel_size=16, strides=1, activation=activation, dilation_rate=1)
        self.layer3 = Conv_In(filters=128, kernel_size=16, strides=1, activation=activation, dilation_rate=2)
        self.pool1 = MaxPool1D(pool_size=4, strides=4, padding='same')

    def call(self, x, training):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool1(x)
        return x

class Upsampler(Layer):
    def __init__(self, feat_size, activation):
        super().__init__(name=self.__class__.__name__)
        self.deconv_in1 = Deconv_In(filters=64, kernel_size=3, strides=2, activation=activation)
        self.deconv_in2 = Deconv_In(filters=32, kernel_size=3, strides=2, activation=activation)
        self.conv = Conv1D(filters=feat_size, kernel_size=3, strides=1, padding='same')
    
    def call(self, x, training):
        x = self.deconv_in1(x)
        x = self.deconv_in2(x)
        x = self.conv(x)
        return x

class Enc_Comb(Layer):
    def __init__(self, args):
        super().__init__(name=self.__class__.__name__)
        self.downsamplers = [Downsampler(k=args['num_noises'],activation=args['activation'], train_batch=args['train_data_noisy'][:10]) for _ in range(args['num_noises'])]
        self.shared = Downsampler(k=args['num_noises'], activation=args['activation'], train_batch=args['train_data_noisy'][:10])
        
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.noise = None

    def call(self, x, noise):
        enc_priv = self.downsamplers[noise](x)
        enc_shared = self.shared(x)
        enc_comb = self.batch_norm(enc_priv + enc_shared)
        return enc_priv, enc_shared, enc_comb

def mean_sum_loss():
    def loss_f(y_true, y_pred, sample_weight):
        '''y_true is dummy array of dataset idxs, y_pred is softmax distribution over those idxs'''
        idx_x = tf.cast(tf.squeeze(tf.where(y_true[:,0]==y_true[:,0])), tf.int32)
        idx_y = tf.cast(K.argmax(y_true, axis=-1), tf.int32)
        idxs = tf.stack([idx_x, idx_y], axis=1)
        return tf.reduce_mean(tf.gather_nd(y_pred, idxs)*sample_weight)
    return loss_f

@tf.custom_gradient
def GradientReversalOperator(x):
	def grad(dy):
		return -1 * dy
	return x, grad

class GradientReversalLayer(tf.keras.layers.Layer):
	def __init__(self):
		super(GradientReversalLayer, self).__init__()
	def call(self, inputs):
		return GradientReversalOperator(inputs)

def Noop_Layer():
    return Lambda(lambda x: x)

class Noop_Callback(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()

class Generator(keras.Model):
    def __init__(self, args, name=None):
        super().__init__(name=self.__class__.__name__)
        self.upsampler = Upsampler(feat_size=args['train_data_noisy'].shape[-1], activation=args['activation'])
        self.task_clf = CLF_Layer(k=args['train_label'].shape[-1])
        self.num_noises = args['num_noises']
        self.enc_comb = Enc_Comb(args)
        self.sim = tf.keras.models.Sequential([
            Noop_Layer() if args['iterative'] else GradientReversalLayer(),
            Flatten(),
            CLF_Layer(k=args['num_noises']),
        ])

    def compile(self, recon_optimizer, l_recon_fn, diff_optimizer, sim_fe_optimizer, sim_clf_optimizer, task_loss_fn, l_diff_fn, l_sim):
        super().compile()
        self.recon_optimizer = recon_optimizer
        self.l_recon_fn = l_recon_fn
        self.sim_fe_optimizer = sim_fe_optimizer
        self.sim_clf_optimizer = sim_clf_optimizer
        
        self.diff_optimizer = diff_optimizer
        self.task_loss_fn = task_loss_fn
        self.l_diff = l_diff_fn
        self.l_sim = l_sim

    def train_step(self, data):
        (noisy, sample_weights, ids), (clean, label) = data

        # get gradients for similarity loss
        ids_onehot = K.one_hot(ids, num_classes=self.num_noises)
        with tf.GradientTape(persistent=True) as tape:
            enc_shared = self.enc_comb.shared(noisy)
            sim_out = self.sim(enc_shared)
            l_sim = self.l_sim(ids_onehot, sim_out, sample_weight=args['l_sim_const']*tf.ones(tf.shape(enc_shared)[0]))
        sim_fe_grad = tape.gradient(l_sim, self.enc_comb.shared.trainable_variables)
        if not args['iterative']:
            sim_clf_grad = tape.gradient(l_sim, self.sim.trainable_variables)
        del tape

        output_created = False
        tot_l_recon, tot_l_diff = 0, 0
        for noise in range(self.num_noises):
            idxs = tf.where(ids == noise)
            noisy_in = tf.gather_nd(noisy, idxs)
            sample_weights_in = tf.gather_nd(sample_weights, idxs)
            clean_in = tf.gather_nd(clean, idxs)
            label_in = tf.gather_nd(label, idxs)

            noise_label = tf.ones((tf.shape(noisy_in, out_type=tf.dtypes.int64)[0],))*noise
            ids_onehot = K.one_hot(tf.cast(noise_label, 'int32'), num_classes=self.num_noises)
            self.enc_comb.noise = noise
            with tf.GradientTape(persistent=True) as tape:
                enc_priv, enc_shared, enc_comb = self.enc_comb(noisy_in, noise)
                recon = self.upsampler(enc_comb)
                l_recon = self.l_recon_fn(clean_in, recon, sample_weight=args['l_recon_const']*tf.ones(tf.shape(enc_shared)[0]))
                tot_l_recon += l_recon

                task = self.task_clf(Flatten()(enc_comb))
                task_loss = self.task_loss_fn(label_in, task, sample_weight=sample_weights_in)

                l_diff = tf.cond(tf.shape(idxs)[0]==0, lambda: tf.constant(0, dtype='float32'), lambda: self.l_diff(enc_shared, enc_priv, sample_weight=args['l_diff_const']))
                tot_l_diff += l_diff

            trainable_variables = self.trainable_variables
            recon_grad = tape.gradient(l_recon, trainable_variables)
            
            def apply_diff_grad():
                diff_grad = tape.gradient(l_diff, self.enc_comb.trainable_variables)
                self.diff_optimizer.apply_gradients(zip(diff_grad, self.enc_comb.trainable_variables))
            
            tf.cond(tf.shape(idxs)[0]==0, lambda: None, apply_diff_grad)

            del tape
            self.recon_optimizer.apply_gradients(zip(recon_grad, trainable_variables))

            if not output_created:
                output = tf.scatter_nd(indices=idxs, updates=recon, shape=tf.shape(noisy, out_type=tf.dtypes.int64))
                output_created = True
            
            else:
                output = output + tf.scatter_nd(indices=idxs, updates=recon, shape=tf.shape(noisy, out_type=tf.dtypes.int64))
        
        self.sim_fe_optimizer.apply_gradients(zip(sim_fe_grad, self.enc_comb.trainable_variables))
        if not args['iterative']:
            self.sim_clf_optimizer.apply_gradients(zip(sim_clf_grad, self.sim.trainable_variables))
        
        return {'loss': tot_l_recon+tot_l_diff+l_sim, 'l_recon': tot_l_recon, 'l_diff': tot_l_diff, 'l_sim': l_sim}

    def test_step(self, data):
        (noisy, sample_weights, ids), (clean, label) = data

        # get gradients for similarity loss
        ids_onehot = K.one_hot(ids, num_classes=self.num_noises)
        enc_shared = self.enc_comb.shared(noisy)
        sim_out = self.sim(enc_shared)
        l_sim = self.l_sim(ids_onehot, sim_out, sample_weight=args['l_sim_const']*tf.ones(tf.shape(enc_shared)[0]))

        output_created = False
        tot_l_recon, tot_l_diff = 0, 0
        for noise in range(self.num_noises):
            idxs = tf.where(ids == noise)
            noisy_in = tf.gather_nd(noisy, idxs)
            sample_weights_in = tf.gather_nd(sample_weights, idxs)
            clean_in = tf.gather_nd(clean, idxs)
            label_in = tf.gather_nd(label, idxs)

            noise_label = tf.ones((tf.shape(noisy_in, out_type=tf.dtypes.int64)[0],))*noise
            ids_onehot = K.one_hot(tf.cast(noise_label, 'int32'), num_classes=self.num_noises)
            self.enc_comb.noise = noise
            enc_priv, enc_shared, enc_comb = self.enc_comb(noisy_in, noise)
            recon = self.upsampler(enc_comb)
            l_recon = self.l_recon_fn(clean_in, recon, sample_weight=args['l_recon_const']*tf.ones(tf.shape(enc_shared)[0]))
            
            tot_l_recon += l_recon

            l_diff = tf.cond(tf.shape(idxs)[0]==0, lambda: tf.constant(0, dtype='float32'), lambda: self.l_diff(enc_shared, enc_priv, sample_weight=args['l_diff_const']))
            tot_l_diff += l_diff

            if not output_created:
                output = tf.scatter_nd(indices=idxs, updates=recon, shape=tf.shape(noisy, out_type=tf.dtypes.int64))
                output_created = True
            else:
                output = output + tf.scatter_nd(indices=idxs, updates=recon, shape=tf.shape(noisy, out_type=tf.dtypes.int64))
        return {'loss': tot_l_recon+tot_l_diff+l_sim, 'l_recon': tot_l_recon, 'l_diff': tot_l_diff, 'l_sim': l_sim}

    def predict_step(self, data):
        noisy, ids = data[0]
        output_created = False
        for noise in range(self.num_noises):
            idxs = tf.where(ids == noise)
            noisy_in = tf.gather_nd(noisy, idxs)

            self.enc_comb.noise = noise
            enc_priv, enc_shared, enc_comb = self.enc_comb(noisy_in, noise)

            recon = self.upsampler(enc_comb)
            task = self.task_clf(Flatten()(enc_comb))

            if not output_created:
                output = tf.scatter_nd(indices=idxs, updates=recon, shape=tf.shape(noisy, out_type=tf.dtypes.int64))
                output_created = True
            else:
                output = output + tf.scatter_nd(indices=idxs, updates=recon, shape=tf.shape(noisy, out_type=tf.dtypes.int64))

        return output, task

def l_diff_fn():
    def loss_fn(enc_shared, enc_priv, sample_weight):
        enc_shared = K.l2_normalize(enc_shared, -1)
        enc_priv = K.l2_normalize(enc_priv, -1)
        matmul = tf.linalg.matmul(a=enc_shared, b=enc_priv, transpose_a=True) ** 2
        cost = tf.reduce_mean(matmul)
        return cost*sample_weight
    return loss_fn

def dsn_main(args_in):
    global args, mapped_train_ids, mapped_val_ids, mapped_test_ids
    args = args_in

    train_idxs, val_idxs = train_test_split(np.arange(args['train_label'].shape[0]), test_size=0.2)
    val_data_noisy, val_data_clean, val_label, mapped_val_ids = [elt[val_idxs] for elt in [args['train_data_noisy'], args['train_data_clean'], args['train_label'], args['mapped_train_ids']]]
    train_data_noisy, train_data_clean, train_label, mapped_train_ids = [elt[train_idxs] for elt in [args['train_data_noisy'], args['train_data_clean'], args['train_label'], args['mapped_train_ids']]]
    test_data_noisy, mapped_test_ids = args[['test_data_noisy', 'mapped_test_ids']]
    assert mapped_train_ids.shape[0] == train_label.shape[0] and mapped_val_ids.shape[0] == val_label.shape[0] and mapped_test_ids.shape[0] == args['test_label'].shape[0]

    class_weights = args['class_weights']
    train_sample_weights = ar(lmap(lambda elt: class_weights[elt], np.argmax(train_label, axis=-1)))
    val_sample_weights = ar(lmap(lambda elt: class_weights[elt], np.argmax(val_label, axis=-1)))

    if args['verbose']:
        print('Training dsn...')
    
    model = Generator(args, name='noisy_to_clean')
    model.compile(
        recon_optimizer=Adam(learning_rate=0.001, beta_1=0.5),
        diff_optimizer=Adam(learning_rate=0.001, beta_1=0.5, clipvalue=args['clipvalue']),
        sim_fe_optimizer=Adam(learning_rate=0.001, beta_1=0.5),
        sim_clf_optimizer=Adam(learning_rate=0.001, beta_1=0.5),
        task_loss_fn=tf.keras.losses.CategoricalCrossentropy(),
        l_recon_fn=tf.keras.losses.MeanSquaredError(),
        l_diff_fn=l_diff_fn(),
        l_sim=mean_sum_loss(),
    )

    model.train_step(data=
        ((train_data_noisy[:10], train_sample_weights[:10], mapped_train_ids[:10]), (train_data_clean[:10], train_label[:10]))
    )
    model.train_step(data=
        ((train_data_noisy[10:15], train_sample_weights[10:15], mapped_train_ids[10:15]), (train_data_clean[10:15], train_label[10:15]))
    )

    model.fit(
        x=(train_data_noisy, train_sample_weights, mapped_train_ids),
        y=(train_data_clean, train_label),
        validation_data=((val_data_noisy, val_sample_weights, mapped_val_ids), (val_data_clean, val_label)),
        epochs=1 if (args['test'] or args['epoch_test']) else 500,
        batch_size=32,
        callbacks=[
            EarlyStopping(monitor='val_l_recon', patience=15, restore_best_weights=True),
        ],
        verbose=args['verbose'],
    )
    
    return model.predict(test_data_noisy, batch_size=32)[1]
