from ..libelem import *
from typing import Union,Tuple

TOUTGETELM = Union[LinElas1D]

def getelem(elemname:str,ninteg,gdofn)->TOUTGETELM:
    if ( elemname == 'linelas1d'):
        return LinElas1D(ninteg=ninteg,gdofn=gdofn)
    
    if ( elemname == 'linelas2d'):
        return LinElas2D(ninteg=ninteg,gdofn=gdofn)

    if ( elemname == 'linelastrac2d'):
        return LinElasTrac2D(ninteg=ninteg,gdofn=gdofn)

    raise RuntimeError(f'element {elemname} not found')
    

