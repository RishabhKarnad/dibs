import jax
import jax.random as random
import jax.numpy as jnp
import numpy as np

from dibs.target import make_linear_gaussian_equivalent_model
from dibs.utils import visualize_ground_truth
from dibs.inference import MarginalDiBS
from dibs.metrics import expected_shd, threshold_metrics, neg_ave_log_marginal_likelihood

key = random.PRNGKey(123)
print(f"JAX backend: {jax.default_backend()}")


datasets = [('3var', 3), ('4var', 4), ('7var', 7)]


def svgd_callback(*, dibs, t, zs):
    print(f'Epoch {t}')


for ds in datasets:
    dataset_dir = f'../dag-gwg/datasets/{ds[0]}'

    key, subk = random.split(key)
    _, graph_model, likelihood_model = make_linear_gaussian_equivalent_model(
        key=subk, n_vars=ds[1], graph_prior_str="sf")

    data = jnp.array(np.load(f'{dataset_dir}/data.npy'))
    data_ho = jnp.array(np.load(f'{dataset_dir}/data_ho.npy'))
    g = jnp.array(np.load(f'{dataset_dir}/G.npy'))

    dibs = MarginalDiBS(x=data, interv_mask=None,
                        graph_model=graph_model, likelihood_model=likelihood_model)
    key, subk = random.split(key)
    gs = dibs.sample(key=subk, n_particles=20, steps=2000,
                     callback_every=20, callback=svgd_callback)

    np.save(f'./output/{ds[0]}/posterior.npy', gs)

    dibs_empirical = dibs.get_empirical(gs)
    dibs_mixture = dibs.get_mixture(gs)

    for descr, dist in [('DiBS ', dibs_empirical), ('DiBS+', dibs_mixture)]:

        eshd = expected_shd(dist=dist, g=g)
        auroc = threshold_metrics(dist=dist, g=g)['roc_auc']
        negll = neg_ave_log_marginal_likelihood(dist=dist, x=data_ho,
                                                eltwise_log_marginal_likelihood=dibs.eltwise_log_marginal_likelihood_observ)

        print(
            f'{descr} |  E-SHD: {eshd:4.1f}    AUROC: {auroc:5.2f}    neg. MLL {negll:5.2f}')
