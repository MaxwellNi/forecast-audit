"""Independent fixed-grid mixture arithmetic; no producer imports."""
import math
import numpy as np
import independent_math as ind
FRACTIONS=np.geomspace(1e-4,.99,64)
DELTA=.0001

def betting(h, lower, upper, bias, delta=DELTA):
    if bias<lower:
        return delta, np.nan
    if bias>=upper:
        return 1.,0.
    if bias==lower:
        return (delta if np.any(h>lower) else 1.),np.nan
    # Algebraically equivalent but avoids the producer's normalized-X formula.
    logs=np.log1p(FRACTIONS[:,None]*(np.asarray(h)[None,:]-bias)/(bias-lower)).sum(1)
    maximum=logs.max()
    log_e=float(maximum+np.log(np.exp(logs-maximum).mean()))
    return (1. if log_e<=0 else min(1.,delta+math.exp(-log_e))),log_e
