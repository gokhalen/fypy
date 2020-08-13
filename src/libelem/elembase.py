from scipy import sparse

class ElemBase():

    def __init__(self,eltype,ninteg,gdofn):
        # eltype: element type, string
        # ninteg: integration points, integer
        # gdofn : number of global degrees of freedom in the system
        self.eltype = eltype
        self.ninteg = ninteg
        self.gdofn  = gdofn


