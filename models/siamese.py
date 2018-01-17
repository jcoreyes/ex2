import numpy as np
import theano
import theano.tensor as T
from theano import function
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import lasagne
from lasagne import nonlinearities
import lasagne.layers as L
from ex2.utils.distributions import log_stdnormal, log_normal2, log_bernoulli, kl_normal2_stdnormal
import numpy as np
from ex2.utils.theano_utils import compile_timer


NORM_CONSTANT = 2 * np.sqrt(np.pi * 2).item()


def make_mlp(l_in, hidden_sizes, input_dim, output_dim,
             hidden_act=nonlinearities.tanh, final_act=None):
    for i, hidden_size in enumerate(hidden_sizes):
        l_in = lasagne.layers.DenseLayer(l_in, num_units=hidden_size, nonlinearity=hidden_act)
    l_out = lasagne.layers.DenseLayer(l_in, num_units=output_dim, nonlinearity=final_act)
    return l_out


def sample_batch(data, data_size, batch_size):
    idxs = np.random.randint(data_size, size=batch_size)
    return data[idxs]

class SimpleSampleLayer(L.MergeLayer):
    """
    Simple sampling layer drawing a single Monte Carlo sample to approximate
    E_q [log( p(x,z) / q(z|x) )]. This is the approach described in [KINGMA]_.
    """
    def __init__(self, mean, log_var,
                 seed=lasagne.random.get_rng().randint(1, 2147462579),
                 **kwargs):
        super(SimpleSampleLayer, self).__init__([mean, log_var], **kwargs)

        self._srng = RandomStreams(seed)

    def seed(self, seed=lasagne.random.get_rng().randint(1, 2147462579)):
       self._srng.seed(seed)

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, input, deterministic=False):
        mu, log_var = input
        eps = self._srng.normal(mu.shape)
        z = mu + T.exp(0.5 * log_var) * eps
        if deterministic:
            z = mu
        return z

class CosLayer(L.MergeLayer):
    def __init__(self, a, b, **kwargs):
        super().__init__([a, b], **kwargs)

    def get_output_shape_for(self, input_shapes):
        assert input_shapes[0] == input_shapes[1]  # both inputs should be of the same size
        return [input_shapes[0][0]] + [1]  # output will drop the last dimension

    def get_output_for(self, inputs, **kwargs):
        a, b = inputs
        sim = ((a * b).sum(axis=-1) / a.norm(2, axis=-1) / b.norm(2, axis=-1)).reshape((-1, 1))
        return sim

class MLP:
    def __init__(self, input_layer, output_dim, hidden_sizes,
                 hidden_act=nonlinearities.tanh,
                 output_act=nonlinearities.identity,
                 params=None,
                 batch_norm=False,
                 dropout=False):
        out_layer = input_layer

        param_idx = 0
        for hidden_size in hidden_sizes:
            w_args = {}
            if params is not None:
                w_args = dict(W=params[param_idx], b=params[param_idx+1])
            out_layer = L.DenseLayer(out_layer, hidden_size, nonlinearity=hidden_act,
                                     **w_args)
            if batch_norm:
                out_layer = L.batch_norm(out_layer)
            if dropout:
                out_layer = L.dropout(out_layer)
            param_idx += 2

        w_args = {}
        if params is not None:
            w_args = dict(W=params[param_idx], b=params[param_idx + 1])
        out_layer = L.DenseLayer(out_layer, output_dim, nonlinearity=output_act,
                                 **w_args)

        self.out_layer = out_layer
        self.output = L.get_output(self.out_layer)

    def params(self):
        return L.get_all_params(self.out_layer)

    def output_layer(self):
        return self.out_layer

class ConvNet():
    def __init__(self, input_layer, 
                 filter_sizes=((4,4), (4,4)),
                 num_filters=(16,16),
                 strides=((2,2), (2,2)),
                 hidden_act=nonlinearities.rectify,
                 ):
        out_layer = input_layer

        for i, (filter_size, num_filter, stride) in enumerate(zip(filter_sizes,
                                                                  num_filters,
                                                                  strides)):
            out_layer = L.Conv2DLayer(out_layer, num_filters=num_filter, filter_size=filter_size,
                                      stride=stride, pad='full', nonlinearity=hidden_act)

        out_layer = L.FlattenLayer(out_layer)
        self.out_layer = out_layer
        self.output = L.get_output(self.out_layer)

    def params(self):
        return L.get_all_params(self.out_layer)

    def output_layer(self):
        return self.out_layer


# Copy from https://github.com/casperkaae/parmesan/blob/master/examples/vae_vanilla.py
class Siamese:
    def __init__(self, input_dim, feature_dim, hidden_sizes,
                 l2_reg=0,
                 hidden_act=nonlinearities.tanh,
                 learning_rate=1e-4,
                 kl_weight=1,
                 batch_norm=False,
                 use_cos=False,
                 dropout=False,
                 env_name='Maze'):
        self.input_dim = input_dim
        self.env_name = env_name

        self.sym_x1 = T.matrix()
        self.sym_x2 = T.matrix()
        self.sym_labels = T.matrix()

        self.lin1 = lasagne.layers.InputLayer((None, input_dim))
        self.lin2 = lasagne.layers.InputLayer((None, input_dim))
        self.labels = lasagne.layers.InputLayer((None, 1))

        self.base1 = MLP(self.lin1, hidden_sizes[0], hidden_sizes, hidden_act, hidden_act,
                         batch_norm=batch_norm)
        self.base2 = MLP(self.lin2, hidden_sizes[0], hidden_sizes, hidden_act, hidden_act, dropout=dropout,
                         batch_norm=batch_norm)


        l1_enc_h2 = self.base1.output_layer()
        l2_enc_h2 = self.base2.output_layer()

        self.mean_net1 = MLP(l1_enc_h2, feature_dim, hidden_sizes, hidden_act)
        self.mean_net2 = MLP(l2_enc_h2, feature_dim, hidden_sizes, hidden_act, dropout=dropout)

        self.logvar_net1 = MLP(l1_enc_h2, feature_dim, hidden_sizes, hidden_act)
        self.logvar_net2 = MLP(l1_enc_h2, feature_dim, hidden_sizes, hidden_act, dropout=dropout)

        l1_mu = self.mean_net1.output_layer()
        l1_log_var = self.logvar_net1.output_layer()

        l2_mu = self.mean_net2.output_layer()
        l2_log_var = self.logvar_net2.output_layer()

        # Sample latent variables
        l1_z = SimpleSampleLayer(mean=l1_mu, log_var=l1_log_var)
        l2_z = SimpleSampleLayer(mean=l2_mu, log_var=l2_log_var)

        combined_z = L.ConcatLayer([l1_z, l2_z])

        if use_cos:
            l_output = CosLayer(l1_z, l2_z)
            l_output = L.NonlinearityLayer(l_output, nonlinearity=nonlinearities.sigmoid)
        else:
        # Classify from latent
            self.class_net = MLP(combined_z, 1, hidden_sizes, hidden_act=hidden_act,
                                 output_act=nonlinearities.sigmoid)
            l_output = self.class_net.output_layer()

        combined_mu = L.ConcatLayer([l1_mu, l2_mu])
        combined_logvar = L.ConcatLayer([l1_log_var, l2_log_var])
        z_train, z_mu_train, z_log_var_train, output_train = L.get_output(
            [combined_z, combined_mu, combined_logvar, l_output],
            inputs={self.lin1: self.sym_x1, self.lin2: self.sym_x2},
            deterministic=False
        )
        output_test = L.get_output(l_output,
                                   inputs={self.lin1: self.sym_x1, self.lin2: self.sym_x2},
                                   deterministic=True)

        self.LL_train, self.class_loss, self.kl_loss = latent_gaussian_x_bernoulli(z_train, z_mu_train, z_log_var_train,
                                               output_train, self.sym_labels, True, kl_weight)
        self.LL_train *= -1

        if l2_reg != 0:
           self.LL_train += l2_reg * lasagne.regularization.regularize_network_params(l_output,
                                                                                     lasagne.regularization.l2)

        self.l_output = l_output
        self.output_test = output_test

        params = self.params()

        grads = T.grad(self.LL_train, params)

        updates = lasagne.updates.adam(grads, params, learning_rate=learning_rate)

        self.train_model = theano.function([self.sym_x1, self.sym_x2, self.sym_labels],
                                           [self.LL_train, self.class_loss, self.kl_loss],
                                           updates=updates)
        self.test_model = theano.function([self.sym_x1, self.sym_x2], self.output_test)

    def reset_weights(self):
        params = self.params()
        for v in params:
            val = v.get_value()
            if (len(val.shape) < 2):
                v.set_value(lasagne.init.Constant(0.0)(val.shape))
            else:
                v.set_value(lasagne.init.GlorotUniform()(val.shape))

    def params(self):
        return L.get_all_params([self.l_output], trainable=True)

    def train_batch(self, x1, x2, labels):
        if 'Cheetah' in self.env_name and self.input_dim == 1:
            x1 = x1[:, -3].reshape((-1, 1))
            x2 = x2[:, -3].reshape((-1, 1))
        loss, class_loss, kl_loss = self.train_model(x1, x2, labels)
        return loss, class_loss, kl_loss

    def test(self, x):
        if 'Cheetah' in self.env_name and self.input_dim == 1:
            x = x[:, -3].reshape((-1, 1))
        # preds = []
        # for i in range(500):
        #     pred = self.test_model(x, x)
        #     preds.append(pred)
        # avg = np.mean(np.concatenate(preds, 1), 1)
        avg = self.test_model(x, x)
        avg = np.clip(np.squeeze(avg), 1e-5, 1-1e-5)
        prob = (1 - avg) / (avg)
        return prob


class SiameseConv:
    def __init__(self, input_size,
                 img_width,
                 img_height,
                 channel_size=1,
                 action_size=None,
                 feature_dim=10,
                 hidden_sizes=(32,32),
                 conv_args={},
                 l2_reg=0,
                 kl_weight=1,
                 learning_rate=1e-4,
                 hidden_act=nonlinearities.tanh,
                 use_actions=False,
                 set_norm_constant=None):
        self.input_size = input_size
        self.set_norm_constant = set_norm_constant
        self.sym_x1 = T.matrix()
        self.sym_x2 = T.matrix()
        self.sym_labels = T.matrix()

        self.lin1 = lasagne.layers.InputLayer((None, input_size))
        self.lin2 = lasagne.layers.InputLayer((None, input_size))

        if use_actions:
            lin1 = L.SliceLayer(self.lin1, slice(0, -action_size))
            lin2 = L.SliceLayer(self.lin2, slice(0, -action_size))
            lact1 = L.SliceLayer(self.lin1, slice(-action_size, None))
            lact2 = L.SliceLayer(self.lin2, slice(-action_size, None))
        else:
            lin1 = self.lin1
            lin2 = self.lin2

        lin1 = L.ReshapeLayer(lin1, (-1, channel_size, img_width, img_height))
        lin2 = L.ReshapeLayer(lin2, (-1, channel_size, img_width, img_height))

        self.base1 = ConvNet(lin1, **conv_args)
        self.base2 = ConvNet(lin2, **conv_args)

        l1_enc_h2 = self.base1.output_layer()
        l2_enc_h2 = self.base2.output_layer()

        if use_actions:
            l1_enc_h2 = L.ConcatLayer([l1_enc_h2, lact1])
            l2_enc_h2 = L.ConcatLayer([l2_enc_h2, lact2])

        self.mean_net1 = MLP(l1_enc_h2, feature_dim, hidden_sizes, hidden_act)
        self.mean_net2 = MLP(l2_enc_h2, feature_dim, hidden_sizes, hidden_act)

        self.logvar_net1 = MLP(l1_enc_h2, feature_dim, hidden_sizes, hidden_act)
        self.logvar_net2 = MLP(l1_enc_h2, feature_dim, hidden_sizes, hidden_act)

        l1_mu = self.mean_net1.output_layer()
        l1_log_var = self.logvar_net1.output_layer()

        l2_mu = self.mean_net2.output_layer()
        l2_log_var = self.logvar_net2.output_layer()

        # Sample latent variables
        l1_z = SimpleSampleLayer(mean=l1_mu, log_var=l1_log_var)
        l2_z = SimpleSampleLayer(mean=l2_mu, log_var=l2_log_var)

        combined_z = L.ConcatLayer([l1_z, l2_z])
        # Classify from latent
        self.class_net = MLP(combined_z, 1, hidden_sizes, output_act=nonlinearities.sigmoid)
        l_output = self.class_net.output_layer()

        combined_mu = L.ConcatLayer([l1_mu, l2_mu])
        combined_logvar = L.ConcatLayer([l1_log_var, l2_log_var])
        z_train, z_mu_train, z_log_var_train, output_train = L.get_output(
            [combined_z, combined_mu, combined_logvar, l_output],
            inputs={self.lin1: self.sym_x1, self.lin2: self.sym_x2},
            deterministic=False
        )

        l1_z_t, l2_z_t = L.get_output(
            [l1_z, l2_z],
            inputs={self.lin1: self.sym_x1, self.lin2: self.sym_x2},
            deterministic=False
        )

        output_test = L.get_output(l_output,
                                   inputs={self.lin1: self.sym_x1, self.lin2: self.sym_x2},
                                   deterministic=True)

        self.LL_train, self.class_loss, self.kl_loss = latent_gaussian_x_bernoulli(z_train, z_mu_train, z_log_var_train,
                                               output_train, self.sym_labels, True, kl_weight)
        self.LL_train *= -1

        if l2_reg != 0:
           self.LL_train += l2_reg * lasagne.regularization.regularize_network_params(l_output,
                                                                                     lasagne.regularization.l2)

        self.l_output = l_output
        self.output_test = output_test

        params = self.params()

        grads = T.grad(self.LL_train, params)

        updates = lasagne.updates.adam(grads, params, learning_rate=learning_rate)

        with compile_timer('train_fn'):
            self.train_model = theano.function([self.sym_x1, self.sym_x2, self.sym_labels],
                                               [self.LL_train, self.class_loss, self.kl_loss],
                                                updates=updates)
        with compile_timer('test_fn'):
            self.test_model = theano.function([self.sym_x1, self.sym_x2], self.output_test)

    def reset_weights(self):
        params = self.params()
        for v in params:
            val = v.get_value()
            if (len(val.shape) < 2):
                v.set_value(lasagne.init.Constant(0.0)(val.shape))
            else:
                v.set_value(lasagne.init.GlorotUniform()(val.shape))

    def params(self):
        return L.get_all_params([self.l_output], trainable=True)

    def train_batch(self, x1, x2, labels):

        loss, class_loss, kl_loss = self.train_model(x1, x2, labels)
        return loss, class_loss, kl_loss

    def test(self, x):
        avg = self.test_model(x, x)
        avg = np.clip(np.squeeze(avg), 1e-5, 1-1e-5)
        prob = (1 - avg) / (avg)
        return prob


#Calculate the loglikelihood(x) = E_q[ log p(x|z) + log p(z) - log q(z|x)]
def latent_gaussian_x_bernoulli(z, z_mu, z_log_var, x_mu, x, analytic_kl_term, kl_weight):
    """
    Latent z       : gaussian with standard normal prior
    decoder output : bernoulli
    When the output is bernoulli then the output from the decoder
    should be sigmoid. The sizes of the inputs are
    z: (batch_size, num_latent)
    z_mu: (batch_size, num_latent)
    z_log_var: (batch_size, num_latent)
    x_mu: (batch_size, num_features)
    x: (batch_size, num_features)
    """
    if analytic_kl_term:
        kl_term = kl_normal2_stdnormal(z_mu, z_log_var).sum(axis=1)
        log_px_given_z = log_bernoulli(x, x_mu, eps=1e-6).sum(axis=1)
        LL = T.mean(kl_weight * -kl_term + log_px_given_z)
    else:
        log_qz_given_x = log_normal2(z, z_mu, z_log_var).sum(axis=1)
        log_pz = log_stdnormal(z).sum(axis=1)
        log_px_given_z = log_bernoulli(x, x_mu, eps=1e-6).sum(axis=1)
        LL = T.mean(log_pz + log_px_given_z - log_qz_given_x)
    log_px_given_z_mean = T.mean(log_px_given_z)
    kl_mean = T.mean(kl_term)
    return LL, -log_px_given_z_mean, kl_mean



def test1d(train_itrs=40000, batch_size=512):
    import matplotlib.pyplot as plt 
    mode = 'bi'
    replay_size = 100000
    if mode == 'uni':
        replay = np.random.randn(replay_size)
        # replay = theano.shared(value=np.random.randn(replay_size))
    else:
        replay = np.concatenate([np.random.randn(replay_size // 2) - 4, np.random.randn(replay_size // 2) + 4])

    replay = np.expand_dims(replay, 1).astype(np.float32)
    # replay = replay.view(replay_size, 1).type_as(torch.FloatTensor()).cuda()

    siamese = Siamese(1, 4, (32, 32), learning_rate=1e-3)

    positives_np = np.expand_dims(np.linspace(-8, 8, 200).astype(np.float32), 1)
    positives = positives_np

    labels = np.expand_dims(np.concatenate([np.ones(batch_size), np.zeros(batch_size)]), 1).astype(np.float32)

    hist, bin_edges = np.histogram(replay, density=True, bins=100)
    bin_edges += (bin_edges[1] - bin_edges[0]) / 2
    #hist /= hist.max()

    log_step = 0.01 * train_itrs
    plt.ion()
    for train_itr in range(train_itrs):

        pos = sample_batch(positives, positives.shape[0], batch_size)
        neg = sample_batch(replay, replay.shape[0], batch_size)

        x1 = np.concatenate([pos, pos])
        x2 = np.concatenate([pos, neg])
        #import pdb; pdb.set_trace()

        loss = siamese.train_batch(x1, x2, labels)

        if train_itr % log_step == 0:
            print(loss)

            pred = siamese.test(positives)

            #pred_np /= pred_np.max()
            plt.clf()

            plt.plot(bin_edges[:-1], hist)
            plt.plot(positives_np, pred)
            plt.show()
            plt.pause(0.05)

    while True:
        plt.pause(0.05)
    #pred = test(siamese, positives)
    #plt.plot(positives_np, pred.data.cpu().numpy())
    #plt.show()


def test2d(train_itrs=5000, batch_size=128):
    import matplotlib.pyplot as plt 
    def generate_2d_test_data(N):
        NNeg = N
        dX = 2
        cluster1 = np.random.randn(NNeg, dX) + np.array([1.3, 2.3])
        cluster1 = cluster1[cluster1[:, 0] > 1.3, :]
        cluster2 = np.random.randn(NNeg, dX) - np.array([1.5, 2.5])
        cluster2 = cluster2[cluster2[:, 1] < -2.5, :]
        cluster3 = np.random.random(size=(int(round(NNeg / 4)), 2)) * 2
        cluster3[:, 0] *= -1
        cluster3 += np.array([-1, 1])
        cluster4 = cluster3 + np.array([-1, 1])
        cluster5 = np.random.random(size=(int(round(NNeg / 4)), 2)) * 6
        cluster5[:, 1] *= -1

        negatives = np.concatenate([cluster1, cluster2, cluster3, cluster4, cluster5])
        return negatives
    import os
    from ex2.envs.twod_mjc_env import map_config, get_dense_gridpoints, predictions_to_heatmap, make_density_map

    grid_config = map_config(xs=(-5,5), ys=(-5,5))
    #logging.basicConfig(level=logging.DEBUG)
    np.set_printoptions(suppress=True)

    dX = 2
    plot_exemplars = get_dense_gridpoints(grid_config)
    nPlotEx = plot_exemplars.shape[0]
    positives = plot_exemplars.astype(np.float32)

    replay = generate_2d_test_data(100000).astype(np.float32)
    negative_density = make_density_map(replay, grid_config)
    # replay = replay.view(replay_size, 1).type_as(torch.FloatTensor()).cuda()

    siamese = Siamese(2, 16, (32,), hidden_act=nonlinearities.tanh, dropout=False,
                      learning_rate=1e-3, use_cos=False)

    labels = np.expand_dims(np.concatenate([np.ones(batch_size), np.zeros(batch_size)]), 1).astype(np.float32)

    log_step = 0.1 * train_itrs

    for train_itr in range(train_itrs):

        pos = sample_batch(positives, positives.shape[0], batch_size)
        neg = sample_batch(replay, replay.shape[0], batch_size)

        x1 = np.concatenate([pos, pos])
        x2 = np.concatenate([pos, neg])

        loss = siamese.train_batch(x1, x2, labels)

        if train_itr % log_step == 0 or train_itr == train_itrs - 1:
            print(loss)

            pred = siamese.test(positives)
            pred /= pred.max()

            plt.imshow(np.c_[predictions_to_heatmap(pred, grid_config), negative_density],
                       cmap='afmhot')
            plt.savefig(filename='%s/Pictures/densities/itr_%d.png' % (os.path.expanduser('~'), train_itr))


if __name__ == '__main__':
    test2d()




