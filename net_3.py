import six

import chainer
import chainer.functions as F
import chainer.links as L
import xavier


def calculate_means(mu, ln_var):
    xp = chainer.cuda.get_array_module(mu)
    mean_mu = xp.mean(mu.data)
    sigma = xp.exp(ln_var.data / 2)
    mean_sigma = xp.mean(sigma)
    return mean_mu, mean_sigma


def generate_std_params(a):
    xp = chainer.cuda.get_array_module(a)
    mu = chainer.Variable(xp.zeros(a.shape).astype(xp.float32))
    ln_var = chainer.Variable(xp.zeros(a.shape).astype(xp.float32))
    return mu, ln_var


class VAE(chainer.Chain):
    """Variational AutoEncoder"""

    def __init__(self, n_in, n_latent, n_h, activation=F.tanh):
        super(VAE, self).__init__()
        self.activation = activation
        with self.init_scope():
            # encoder
            self.le1 = L.Linear(n_in, n_h, initialW=xavier.Xavier(n_in, n_h))
            self.le2 = L.Linear(n_h, n_h, initialW=xavier.Xavier(n_h, n_h))
            self.le3_mu = L.Linear(n_h, n_latent, initialW=xavier.Xavier(n_h, n_latent))
            self.le3_ln_var = L.Linear(n_h, n_latent, initialW=xavier.Xavier(n_h, n_latent))
            # decoder
            self.ld1 = L.Linear(n_latent, n_h, initialW=xavier.Xavier(n_latent, n_h))
            self.ld2 = L.Linear(n_h, n_h, initialW=xavier.Xavier(n_h, n_h))
            self.ld3 = L.Linear(n_h, n_in, initialW=xavier.Xavier(n_h, n_in))

            # self.le1 = L.Linear(n_in, n_h, initialW=chainer.initializers.Normal(scale=0.01))
            # self.le2 = L.Linear(n_h, n_h, initialW=chainer.initializers.Normal(scale=0.01))
            # self.le3_mu = L.Linear(n_h, n_latent, initialW=chainer.initializers.Normal(scale=0.01))
            # self.le3_ln_var = L.Linear(n_h, n_latent, initialW=chainer.initializers.Normal(scale=0.01))
            # # decoder
            # self.ld1 = L.Linear(n_latent, n_h, initialW=chainer.initializers.Normal(scale=0.01))
            # self.ld2 = L.Linear(n_h, n_h, initialW=chainer.initializers.Normal(scale=0.01))
            # self.ld3 = L.Linear(n_h, n_in, initialW=chainer.initializers.Normal(scale=0.01))

    def __call__(self, x, sigmoid=True):
        """AutoEncoder"""
        return self.decode(self.encode(x)[0], sigmoid)

    def encode(self, x):
        h1 = self.activation(self.le1(x))
        h2 = self.activation(self.le2(h1))
        mu = self.le3_mu(h2)
        ln_var = self.le3_ln_var(h2)  # log(sigma**2)
        return mu, ln_var

    def decode(self, z, sigmoid=True):
        h1 = self.activation(self.ld1(z))
        h2 = self.activation(self.ld2(h1))
        h3 = self.ld3(h2)
        if sigmoid:
            return F.sigmoid(h3)
        else:
            return h3

    def get_loss_func(self, C=1.0, k=1):
        """Get loss function of VAE.

        The loss value is equal to ELBO (Evidence Lower Bound)
        multiplied by -1.

        Args:
            C (int): Usually this is 1.0. Can be changed to control the
                second term of ELBO bound, which works as regularization.
            k (int): Number of Monte Carlo samples used in encoded vector.
        """
        def lf(x):
            mu, ln_var = self.encode(x)
            mean_mu, mean_sigma = calculate_means(mu, ln_var)
            batchsize = len(mu.data)
            std_mu, std_ln_var = generate_std_params(mu)

            # reconstruction loss
            rec_loss = 0
            kl_loss = 0
            for l in six.moves.range(k):
                z = F.gaussian(mu, ln_var)
                rec_loss += F.bernoulli_nll(x, self.decode(z, sigmoid=False)) / (k * batchsize)
                kl_loss += -F.gaussian_nll(z, mu, ln_var) / (k * batchsize)
                kl_loss += F.gaussian_nll(z, std_mu, std_ln_var) / (k * batchsize)

            self.rec_loss = rec_loss
            self.kl_loss = kl_loss
            self.loss = self.rec_loss + C * self.kl_loss
            chainer.report(
                {
                    'rec_loss': rec_loss,
                    'kl': self.kl_loss,
                    'loss': self.loss,
                    'mu': mean_mu,
                    'sigma': mean_sigma,
                },
                observer=self
            )
            return self.loss
        return lf
