from recommendation_models.pop import pop
from recommendation_models.cdae import cdae, CDAE
from recommendation_models.vae import vae_cf, VAE
from recommendation_models.ifvae import ifvae, IFVAE
from recommendation_models.autorec import autorec, AutoRec
from recommendation_models.bpr import bpr
#from recommendation_models.wrmf import als
from recommendation_models.cml import cml
from recommendation_models.puresvd import puresvd
from recommendation_models.nceplrec import nceplrec
from recommendation_models.plrec import plrec
from recommendation_models.ncesvd import ncesvd

models = {
    "POP": pop,
    "AutoRec": autorec,
    "CDAE": cdae,
    "VAE-CF": vae_cf,
    "IFVAE": ifvae,
    "BPR": bpr,
#    "WRMF": als,
    "CML": cml,
    "PureSVD": puresvd,
    "NCE-PLRec": nceplrec,
    "NCE-SVD": ncesvd,
    "PLRec": plrec,
}

autoencoders = {
    "AutoRec": AutoRec,
    "CDAE": CDAE,
    "VAE-CF": VAE,
    "IFVAE": IFVAE
}

vaes = {
    "VAE-CF": VAE,
    "IFVAE": IFVAE
}
