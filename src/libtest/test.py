import functools,numpy as np
closetol=1e-12
npclose=functools.partial(np.allclose,atol=closetol)

