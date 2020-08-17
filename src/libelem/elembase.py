from scipy import sparse,linalg

class ElemBase():

    def __init__(self,eltype,ninteg,gdofn):
        
        # eltype: element type, string
        # ninteg: integration points, integer
        # gdofn : number of global degrees of freedom in the system
        self.eltype   = eltype
        self.ninteg   = ninteg
        self.gdofn    = gdofn
        
        # create sparse matrix and rhs
        self.kmatrix = sparse.coo_matrix((gdofn,gdofn),dtype='float64')
        self.rhs     = sparse.coo_matrix((gdofn,1),dtype='float64')

    


