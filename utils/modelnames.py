from models.cdae import cdae, CDAE
from models.vae import vae_cf, VAE
from models.ifvae import ifvae, IFVAE
from models.autorec import autorec, AutoRec
from models.bpr import bpr
from models.wrmf import als
from models.cml import cml


models = {
    "AutoRec": autorec,
    "CDAE": cdae,
    "VAE-CF": vae_cf,
    "IFVAE": ifvae,
    "BPR": bpr,
    "WRMF": als,
    "CML": cml
}

autoencoders = {
    "AutoRec": AutoRec,
    "CDAE": CDAE,
    "VAE-CF": VAE,
    "IFVAE": IFVAE
}