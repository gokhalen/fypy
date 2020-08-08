    def compare_gaussnd(self,ndime:int,npoints:int,ptn=None,wtn=None,data_supplied=False)->Union[None,AssertionError]:
        # this method returns either None or an raises an AssertionError.
        # I've type hinted it to return either, though type hinting for exceptions seems verboten
        # fgauss: either gauss1d,gauss2d,gauss3d that can be compared against gaussnd
        # nfunc: which gauss function to be called (1,2 or 3)

        msgpt, msgwt = '',''
        pt,wt        = eval(f'gauss{ndime}d(npoints)')
        
        if ( not data_supplied ):
            ptn,wtn = gaussnd(ndime,npoints)
            msg     = f'gauss{ndime}d does not match gaussnd for {npoints} integration points'

        if ( data_supplied ) :
            msg = f"gauss{ndime}d does not match Hughes' data for {npoints} integration points"

        # To make sure the test is working, set arrays to wrong value
        # if ( (ndime==2) and (npoints==3) ):
        #    pt[3][1] += 1
        #    wt[4]    += 1
        
        boolpt,boolwt  = map(npclose,(ptn,wtn),(pt,wt))

        if (not boolpt):
            # call function to determine which entry is off
            # if really wants to get fancy, we can decorate get_mismatch with make_mismatch_message
            idx,aa,bb = get_mismatch(ptn,pt,closetol=closetol)
            msgpt     = make_mismatch_message(idx,aa,bb)

        if (not boolwt):
            # call function to determine which entry is off
            idx,aa,bb  = get_mismatch(wtn,wt,closetol=closetol)
            msgwt      = make_mismatch_message(idx,aa,bb)
            
        self.assertTrue(boolpt,msg=msg+msgpt)
        self.assertTrue(boolwt,msg=msg+msgwt)

def compare_hughes(self,npoints,ftest:Callable,ptsh,wtsh):
    '''
    npoints: number of points to use for integration in each direction
    ftest:   a function which is to be tested against the data from
             Hughes' book
    ptsh:  integration points from Hughes' book
    wtsh:  weights from Hughes' boook
    '''
    msgpts = f"{ftest.__name__} {npoints=} pts does not match Hughes' linear"
    msgwts = f"{ftest.__name__} {npoints=} wts does not match Hughes' linear"

    (pts,wts)=map(np.asarray,ftest(npoints))
    (wtclose,ptclose)=map(npclose,(pts,wts),(ptsh,wtsh))

    self.assertTrue(wtclose,msg=msgwts)
    self.assertTrue(ptclose,msg=msgpts)


def compare_shape(self,fshape:Callable,pts:Iterable,exout:Iterable):
    '''
    fshape: callable to be tested: shape1d or shape2d
    pts:    points at which shape functions and their derivatives have to be tested
    exout:  expected output containing Shape named tuples 
    '''
    #  actout: actual output to be compard with 'exout' expected output
    *actout,             = map(fshape,pts)
    msgshp,msgder   = 'Mismatch in shape functions: ','Mismatch in shape function derivatives: '

    for i,(act,exp) in enumerate(zip(actout,exout)):
        boolshp,boolder = map(npclose,(act.shape,act.der),(exp.shape,exp.der))
        if ( not boolshp ):
            idx,aa,bb = get_mismatch(act.shape,exp.shape,closetol=closetol)
            msgshp    += make_mismatch_message(idx,aa,bb) + f'for {i}th entry in testing data'
            pass

        if ( not boolder ):
            print(f'{act=}',f'{exp=}')
            idx,aa,bb  = get_mismatch(act.der,exp.der,closetol=closetol)
            msgder    += make_mismatch_message(idx,aa,bb) + f'for {i}th entry in testing data'
            pass

        self.assertTrue(boolshp,msg=msgshp)
        self.assertTrue(boolder,msg=msgder)

