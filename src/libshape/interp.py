def interp_parent(data,shp):
    #  interpolation over the parent domain
    # data: iterable yielding data at element nodes to be interpolated.
    #       yielded data should support multiplication operator
    #       yielded data should ideally be numpy arrays
    # shp: iterable yielding namedtuple Shape. length of shp is number of points at
    #      which interpolation is to be desired.
    #      length of data and the number of shape functions in the Shape namedtuple should be same

    return [  sum( s*d  for s,d in zip(ss.shape,data) )    for ss in shp ]
    

