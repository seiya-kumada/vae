import six

import chainer
import chainer.functions as F
from chainer.functions.loss.vae import gaussian_kl_divergence
import chainer.links as L
import xavier


def calculate_means(mu, ln_var):
    xp = chainer.cuda.get_array_module(mu)
    mean_mu = xp.mean(mu.data)
    sigma = xp.exp(ln_var.data / 2)
    mean_sigma = xp.mean(sigma)
    return mean_mu, mean_sigma


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
            self.ld3_mu = L.Linear(n_h, n_in, initialW=xavier.Xavier(n_h, n_in))
            self.ld3_ln_var = L.Linear(n_h, n_in, initialW=xavier.Xavier(n_h, n_in))

    def __call__(self, x, sigmoid=True):
        """AutoEncoder"""
        return self.decode(self.encode(x)[0], sigmoid)

    def encode(self, x):
        h1 = self.activation(self.le1(x))
        h2 = self.activation(self.le2(h1))
        mu = self.le3_mu(h2)
        ln_var = self.le3_ln_var(h2)  # log(sigma**2)
        return mu, ln_var

    def decode(self, z):
        h1 = self.activation(self.ld1(z))
        h2 = self.activation(self.ld2(h1))
        mu = self.ld3_mu(h2)
        ln_var = self.ld3_ln_var(h2)
        return mu, ln_var

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
            # reconstruction loss
            rec_loss = 0
            for l in six.moves.range(k):
                z = F.gaussian(mu, ln_var)
                mu_, ln_var_ = self.decode(z)
                rec_loss += F.gaussian_nll(x, mu_, ln_var_) / (k * batchsize)
            self.rec_loss = rec_loss
            kl = gaussian_kl_divergence(mu, ln_var) / batchsize
            self.loss = self.rec_loss + C * kl
            chainer.report(
                {
                    'rec_loss': rec_loss,
                    'loss': self.loss,
                    'kl': kl,
                    'mu': mean_mu,
                    'sigma': mean_sigma,
                },
                observer=self
            )
            return self.loss
        return lf
