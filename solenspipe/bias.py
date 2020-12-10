
import numpy as np
from pixell import utils # These are needed for MPI. Relevant functions can be copied over.
import healpy as hp
from enlib import bench
from orphics import stats,io,mpi

"""
Extremely general functions for lensing power spectrum bias subtraction
======================================================================

These are quite general (independent of pixelization), FFT/SHT, region, array, etc.

One abstracts out the following:

qfunc(XY,x,y)
which for estimator XY e.g. "TT"
x,y accepts fourier transformed beam deconvolved low pass filtered inpainted purified T, E or B maps.
It should probably belong to some prepared object which knows about filters etc.

get_kmap(X,seed)
This is the tricky part.
X can be "T", "E", or "B"
The way seed is used is very important. It should have the following logic.
seed = (icov,set,i)
If i==0, it shouldn't matter what icov or set are, it always returns the corresponding "data" kmap.
Otherwise i should range from 1 to nsims.
This assumes there are icov x nset x nsims sims.
set=0 has nsims
set=1 has nsims
set=2 has nsims
set=3 has nsims
set 2 and 3 should have common phi between corresponding sims, used in MCN1

e.g. implementation:
if set==0 or set==1:
    cmb_seed = (icov,set,i)+(0,)
    kappa_seed = (icov,set,i)+(1,)
    noise_seed = (icov,set,i)+(2,)
elif set==2 or set==3:
    cmb_seed = (icov,set,i)+(0,)
    kappa_seed = (icov,2,i)+(1,)
    noise_seed = (icov,set,i)+(2,)

=================
"""
def rdn0(icov,alpha,beta,qfunc,get_kmap,comm,power,nsims,
         include_meanfield=False,gaussian_sims=False,include_main=True,rdn0_error=False,
         qxy=None,qab=None,type=None,ils=None, blens=None, bhps=None, Alpp=None, A_ps=None):
    """
    Anisotropic MC-RDN0 for alpha=XY cross beta=AB
    qfunc(XY,x,y) returns QE XY reconstruction 
    get_kmap("T",(0,0,1)
    e.g. rdn0(0,"TT","TE",qest.get_kappa,get_kmap,comm,power)
    gaussian_sims=True indicates we don't need to involve pairs
    of sims because the sims are not lensed.
    This is the usual 1xnsims RDN0. If type is unspecified, this will return the usual QE RDN0. 
    type='bh' returns the point source hardened RDN0.
    if rdn0_error=True, calculate the scatter between realizations of RDN0. return the rdn0 standard deviation as the second component.
    """
    eX,eY = alpha
    eA,eB = beta

    if type=='bh':
        qa = lambda x,y: qfunc(alpha,x,y,ils, blens, bhps, Alpp, A_ps)
        qb = lambda x,y: qfunc(beta,x,y,ils, blens, bhps, Alpp, A_ps)
    else:
        qa = lambda x,y: qfunc(alpha,x,y)
        qb = lambda x,y: qfunc(beta,x,y)

    # Data
    X = get_kmap((0,0,0))
    Y = X
    A = X
    B = X
    if include_meanfield: 
        qxy = qa(X,Y) if qxy is None else qxycd
        qab = qb(A,B) if qab is None else qab
    # Sims
    rdn0 = 0.
    if comm is not None:
        rank,size = comm.rank, comm.size
        s = stats.Stats(comm)
    else:
        rank,size = 0, 1
    with bench.show("sim"):
        for i in range(rank+1, nsims+1, size):
            print(i)
            error=0. #only used to calculate the rdn0 error
            Xs  = get_kmap((icov,0,i+1))
            Ys  = Xs
            As  = Xs
            Bs  = Xs
            if include_meanfield:
                mf= ((power(qa(Xs,Ys),qab) + power(qxy,qb(As,Bs)))) 
                rdn0+=mf
                error+=mf
            if include_main:
                print("main rdn0")
                main= power(qa(X,Ys),qb(A,Bs)) + power(qa(Xs,Y),qb(A,Bs)) \
                        + power(qa(Xs,Y),qb(As,B)) + power(qa(X,Ys),qb(As,B))
                rdn0+=main
                error+=main
            
                
                if not(gaussian_sims):
                    print("non gaussian")
                    Ysp = get_kmap((icov,1,i+1))
                    Asp = Ysp
                    Bsp = Ysp
                    sim_rdn0= (- power(qa(Xs,Ysp),qb(As,Bsp)) - power(qa(Xs,Ysp),qb(Asp,Bs)))
                    rdn0+=sim_rdn0
                    error+=sim_rdn0
       
                else:
                    gauss_sim=  (-power(qa(Xs,Ys),qb(As,Bs)))
                    rdn0+=gauss_sim
                    error+=gauss_sim
            if comm is not None:
                s.add_to_stats('rdn0',error)
    try:
        totrdn0 = utils.allreduce(rdn0,comm)
        if rdn0_error:
            s.get_stats()
            s.vectors['rdn0']
            std=np.std(s.vectors['rdn0'], axis=0)
    except AttributeError:
        totrdn0 = rdn0

    if rdn0_error: 
        return totrdn0/nsims,std
    
    return totrdn0/nsims

def mean_rdn0(icov,alpha,beta,qfunc,get_kmap,comm,power,nsims,
         include_meanfield=False,gaussian_sims=False,include_main=True,
         qxy=None,qab=None,type=None,ils=None, blens=None, bhps=None, Alpp=None, A_ps=None):
    """
    Compute the average of nsimsx1 RDN0. Each of the nsims RDN0's are calculated as follows: Taking 2*nsims, we divide this batch of sims such as we treat the first half of nsims as the data and the second half
    are the simulations. Such as the ith RDN0 have the ith simulation as the data and the ith+nsims simulation is the simulation.
    Such pairing of sims reduces fluctuations of a particular sim and this  is used in the null test routines when there is no signal and also in bias subtraction to obtain the MC bias.


    Anisotropic MC-RDN0 for alpha=XY cross beta=AB
    qfunc(XY,x,y) returns QE XY reconstruction 
    get_kmap("T",(0,0,1)

    e.g. rdn0(0,"TT","TE",qest.get_kappa,get_kmap,comm,power)


    gaussian_sims=True indicates we don't need to involve pairs
    of sims because the sims are not lensed. 
    """
    eX,eY = alpha
    eA,eB = beta
    if type=='bh':
        qa = lambda x,y: qfunc(alpha,x,y,ils, blens, bhps, Alpp, A_ps)
        qb = lambda x,y: qfunc(beta,x,y,ils, blens, bhps, Alpp, A_ps)
    else:
        qa = lambda x,y: qfunc(alpha,x,y)
        qb = lambda x,y: qfunc(beta,x,y)
    # Data #need to shuffle this so that make all sims same as data
    #for loop here as well, change the data
    if include_meanfield: 
        qxy = qa(X,Y) if qxy is None else qxy
        qab = qb(A,B) if qab is None else qab
    # Sims
    rdn0 = 0.
    for i in range(comm.rank+1, nsims+1, comm.size):
            #data
            X=get_kmap((icov,0,i))
            Y = get_kmap((icov,0,i))
            A = get_kmap((icov,0,i))
            B = get_kmap((icov,0,i))
            j=i+nsims
            Xs=get_kmap((icov,0,j))
            Ys  = get_kmap((icov,0,j))
            As  = get_kmap((icov,0,j))
            Bs  = get_kmap((icov,0,j))
            if include_meanfield:
                rdn0 += ((power(qa(Xs,Ys),qab) + power(qxy,qb(As,Bs)))) 
            if include_main:
                rdn0 += power(qa(X,Ys),qb(A,Bs)) + power(qa(Xs,Y),qb(A,Bs)) \
                        + power(qa(Xs,Y),qb(As,B)) + power(qa(X,Ys),qb(As,B))
            if not(gaussian_sims):
                Ysp=get_kmap((icov,1,j))
                Asp=get_kmap((icov,1,j))
                Bsp = get_kmap((icov,1,j))
                rdn0 += (- power(qa(Xs,Ysp),qb(As,Bsp)) - power(qa(Xs,Ysp),qb(Asp,Bs)))
            else:
                rdn0 +=  (-power(qa(Xs,Ys),qb(As,Bs)))
    totrdn0 = utils.allreduce(rdn0,comm) 

    return totrdn0/nsims

def structure(icov,alpha,beta,qfunc,get_kmap,comm,power,nsims,nmc=50,n=100,
         include_meanfield=False,gaussian_sims=False,include_main=True,
         qxy=None,qab=None):
    """
    MC-RDN0 for alpha=XY cross beta=AB
    qfunc(XY,x,y) returns QE XY reconstruction 
    get_kmap("T",(0,0,1)
    Generates nmcxn rdn0s for covariance simulation
    e.g. structure(0,"TT","TE",qest.get_kappa,get_kmap,comm,power)


    gaussian_sims=True indicates we don't need to involve pairs
    of sims because the sims are not lensed. Fluctuations reduced by dividing nsims into 2 halfs. Treat the first half as data the second half as sims.
    """

    eX,eY = alpha
    eA,eB = beta
    qa = lambda x,y: qfunc(alpha,x,y)
    qb = lambda x,y: qfunc(beta,x,y)
    
    rdn0list=[]
    step=np.int(np.ceil(1950/nmc))
    for i in range(0,1950,step):
        with bench.show("rdn0"):
            #these are the data values
            X = get_kmap((0,0,i))
            Y = X
            A = X
            B = X
            if include_meanfield: 
                qxy = qa(X,Y) if qxy is None else qxy
                qab = qb(A,B) if qab is None else qab
            # Sims
            rdn0 = 0.       
            for j in range(i+1+comm.rank,i+n,comm.size):
            #for j in range(i+1+comm.rank,i+201,comm.size):
                print(j)
                Xs  = get_kmap((icov,0,j))
                Ys  = get_kmap((icov,0,j))
                As  = get_kmap((icov,0,j))
                Bs  = get_kmap((icov,0,j))
                if include_meanfield:
                    rdn0 += ((power(qa(Xs,Ys),qab) + power(qxy,qb(As,Bs)))) 
                if include_main:
                    rdn0 += power(qa(X,Ys),qb(A,Bs)) + power(qa(Xs,Y),qb(A,Bs)) \
                            + power(qa(Xs,Y),qb(As,B)) + power(qa(X,Ys),qb(As,B))
                    if not(gaussian_sims):
                        Ysp = get_kmap((icov,1,j))
                        Asp = Ysp
                        Bsp = Ysp
                        rdn0 += (- power(qa(Xs,Ysp),qb(As,Bsp)) - power(qa(Xs,Ysp),qb(Asp,Bs)))
                    else:
                        rdn0 +=  (-power(qa(Xs,Ys),qb(As,Bs)))
                
            totrdn0 = utils.allreduce(rdn0,comm)
            totrdn0=np.array(totrdn0/n)
            rdn0list.append(totrdn0)
    return rdn0list

def rdn0_shear(icov,alpha,beta,qfunc,get_kmap,comm,power,nsims,
         include_meanfield=False,gaussian_sims=False,include_main=True,
         qxy=None,qab=None):
    """
    Anisotropic MC-RDN0  shear 
    Still need to incorporate this in the template RDN0
    for alpha=XY cross beta=AB
    qfunc(XY,x,y) returns QE XY reconstruction 
    get_kmap("T",(0,0,1)
    e.g. rdn0(0,"TT","TE",qest.get_kappa,get_kmap,comm,power)
    gaussian_sims=True indicates we don't need to involve pairs
    of sims because the sims are not lensed.
    """
    eX,eY = alpha
    eA,eB = beta
    qa = lambda x,y: qfunc(x,y)
    qb = lambda x,y: qfunc(x,y)
    # Data
    X = get_kmap((0,0,0))
    Y = X
    A = X
    B = X
    if include_meanfield: 
        qxy = qa(X[0],Y[1]) if qxy is None else qxy
        qab = qb(A[0],B[1]) if qab is None else qab
    # Sims
    rdn0 = 0.
    with bench.show("sim"):
        for i in range(comm.rank+1, nsims+1, comm.size):
            print(i)
            Xs  = get_kmap((icov,0,i))
            Ys  = Xs
            As  = Xs
            Bs  = Xs
            if include_meanfield:
                rdn0 += ((power(qa(Xs[0],Ys[1]),qab) + power(qxy,qb(As[0],Bs[1])))) 
                print(rdn0)
            if include_main:
                print("main rdn0")
                print(power(qa(X[0],Ys[1]),qb(A[0],Bs[1])))
                rdn0 += power(qa(X[0],Ys[1]),qb(A[0],Bs[1])) + power(qa(Xs[0],Y[1]),qb(A[0],Bs[1])) \
                        + power(qa(Xs[0],Y[1]),qb(As[0],B[1])) + power(qa(X[0],Ys[1]),qb(As[0],B[1]))
                if not(gaussian_sims):
                    print("non gaussian")
                    Ysp = get_kmap((icov,1,i))
                    Asp = Ysp
                    Bsp = Ysp
                    rdn0 += (- power(qa(Xs[0],Ysp[1]),qb(As[0],Bsp[1])) - power(qa(Xs[0],Ysp[1]),qb(Asp[0],Bs[1])))
                else:
                    rdn0 +=  (-power(qa(Xs[0],Ys[1]),qb(As[0],Bs[1])))
    totrdn0 = utils.allreduce(rdn0,comm) 
    return totrdn0/nsims


def mcn1_shear(icov,alpha,beta,qfunc,get_kmap,comm,power,nsims,verbose=False):
    """
    MCN1 for shear
     for alpha=XY cross beta=AB
    qfunc(x,y) returns QE reconstruction minus mean-field in fourier space
    need to incorporate this in the template mcn1
    """
    eX,eY = alpha
    eA,eB = beta
    qa = lambda x,y: qfunc(x,y)
    qb = lambda x,y: qfunc(x,y)
    n1 = 0.
    term_list=[]
    for i in range(comm.rank+1, nsims+1, comm.size):        
        if verbose: print("Rank %d doing task %d" % (comm.rank,i))
        Xsk   = get_kmap((icov,2,i))
        Yskp  = get_kmap((icov,3,i))
        Ask   = Xsk
        Bskp  = Yskp
        Askp  = Yskp
        Bsk   = Xsk
        Xs    = get_kmap((icov,0,i))
        Ysp   = get_kmap((icov,1,i))
        As    = Xs
        Bsp   = Ysp
        Asp   = Ysp
        Bs    = Xs
        term = power(qa(Xsk[0],Yskp[1]),qb(Ask[0],Bskp[1])) + power(qa(Xsk[0],Yskp[1]),qb(Askp[0],Bsk[1])) \
            - power(qa(Xs[0],Ysp[1]),qb(As[0],Bsp[1])) - power(qa(Xs[0],Ysp[1]),qb(Asp[0],Bs[1]))
        n1 = n1 + term

    return  utils.allreduce(n1,comm) /nsims




def mcn1(icov,alpha,beta,qfunc,get_kmap,comm,power,nsims,verbose=False,type=None,ils=None, blens=None, bhps=None, Alpp=None, A_ps=None):
    """
    MCN1 for alpha=XY cross beta=AB
    qfunc(x,y) returns QE reconstruction minus mean-field in fourier space
    """
    eX,eY = alpha
    eA,eB = beta
    if type=='bh':
        qa = lambda x,y: qfunc(alpha,x,y,ils, blens, bhps, Alpp, A_ps)
        qb = lambda x,y: qfunc(beta,x,y,ils, blens, bhps, Alpp, A_ps)
    else:
        qa = lambda x,y: qfunc(alpha,x,y)
        qb = lambda x,y: qfunc(beta,x,y)
    n1 = 0.
    term_list=[]
    for i in range(comm.rank+1, nsims+1, comm.size):        
        if verbose: print("Rank %d doing task %d" % (comm.rank,i))
        Xsk   = get_kmap((icov,2,i))
        Yskp  = get_kmap((icov,3,i))
        Ask   = Xsk
        Bskp  = Yskp
        Askp  = Yskp
        Bsk   = Xsk
        Xs    = get_kmap((icov,0,i))
        Ysp   = get_kmap((icov,1,i))
        As    = Xs
        Bsp   = Ysp
        Asp   = Ysp
        Bs    = Xs
        term = power(qa(Xsk,Yskp),qb(Ask,Bskp)) + power(qa(Xsk,Yskp),qb(Askp,Bsk)) \
            - power(qa(Xs,Ysp),qb(As,Bsp)) - power(qa(Xs,Ysp),qb(Asp,Bs))
        n1 = n1 + term

    return  utils.allreduce(n1,comm) /nsims


def mcmf(icov,alpha,qfunc,get_kmap,comm,nsims):
    """
    MCMF for alpha=XY
    qfunc(x,y) returns QE reconstruction minus mean-field in fourier space
    """
    eX,eY = alpha
    qe = lambda x,y: qfunc(alpha,x,y)
    mf = 0.
    ntot = 0.
    for i in range(comm.rank+1, nsims+1, comm.size):        
        for j in range(2):
            kx   = get_kmap((icov,j,i))
            ky   = get_kmap((icov,j,i))
            mf += qe(kx,ky)
            ntot += 1.
    mftot = utils.allreduce(mf,comm) 
    totnot = utils.allreduce(ntot,comm) 
    return mftot/totntot
        

def _validate_splits(xyuv,x,y,u,v):
    xn,yn,un,vn = xyuv
    nsplits = x.shape[0]
    assert nsplits>=4
    for a in [x,y,u,v]:
        assert a.ndim in [2,4]
        assert a.shape[0]==nsplits
    return nsplits

def cross_estimator(xyuv,qe_func,pow_func,xsplits,ysplits,usplits,vsplits,
                    xcoadd=None,ycoadd=None,ucoadd=None,vcoadd=None):
    """

    Returns the cross-only estimate of the raw 4-point lensing power spectrum given
    splits of the data and functions for lensing reconstrucion and power spectrum
    calculation.

    xyuv: string containing 4-letter lensing power spectrum combination. e.g. TTTT,
    TTTE, EEEB, etc. These are used in function calls with qe_func.

    qe_func: a function qfunc(xy,x,y) that calculates the un-normalized quadratic estimator
    between x and y given an estimator name xy. If the input fields are filtered,
    this function should not apply filtering, and vice versa. The function should
    not distinguish between whether the input fields are coadds or individual splits,
    and in particular, any filtering for coadds and splits should be identical.
    
    pow_func: A generic function pow_func(x,y) that takes harmonics x and y
    and converts to a power spectrum. e.g. hp.alm2cl for SHTs and
    (x*y.conj()).real for 2D FFTs.

    xsplits,ysplits,usplits,vsplits: (nsplits,nalms) or (nsplits,Ny,Nx)
    arrays containing the splits for each of X,Y,U,V data.

    xcoadd,ycoadd,ucoadd,vcoadd: coadds of above. If not provided, will
    use np.mean(splits,axis=0).


    """
    from sympy import Symbol
    X,Y,U,V = xyuv
    XY = X + Y
    UV = U + V
    m = _validate_splits(xyuv,xsplits,ysplits,usplits,vsplits)
    xs = np.asanyarray(xsplits)
    ys = np.asanyarray(ysplits)
    us = np.asanyarray(usplits)
    vs = np.asanyarray(vsplits)
    cfunc = lambda x: np.mean(x,axis=0)
    xc = cfunc(xs)
    yc = cfunc(ys)
    uc = cfunc(us)
    vc = cfunc(vs)
    feed_dict = {'xc': xc, 'yc': yc, 'uc': uc,'vc': vc}
    for i in range(int(m)):
        feed_dict[f'xs{i}'] = xs[i]
        feed_dict[f'ys{i}'] = ys[i]
        feed_dict[f'us{i}'] = us[i]
        feed_dict[f'vs{i}'] = vs[i]
    return cross_estimator_symbolic(xyuv,m,qe_func,pow_func,eval_feed_dict=feed_dict)

def cross_estimator_symbolic(xyuv,nsplits,qe_func,pow_func,
                             coadd_name_func = lambda x: f'{x}c',field_names=['x','y','u','v'],
                             split_name_func = lambda x,y: f'{x}s{y}',eval_feed_dict=None):

    """
    Returns a symbolic expression for the cross-only estimate of the raw 4-point 
    lensing power spectrum given splits of the data and functions for 
    lensing reconstrucion and power spectrum calculation.

    xyuv: string containing 4-letter lensing power spectrum combination. e.g. TTTT,
    TTTE, EEEB, etc. These are used in function calls with qe_func.

    nsplits: number of splits

    qe_func: function for lensing reconstruction. This function should either operate
    on symbols and return symbols, or alternatively, eval_feed_dict can be provided
    which will be used to evaluate the symbols prior to application of the function.

    pow_func: function for power spectrum estimation. This function should either operate
    on symbols and return symbols, or alternatively, eval_feed_dict can be provided
    which will be used to evaluate the symbols prior to application of the function.

    """
    from sympy import Symbol
    X,Y,U,V = xyuv
    XY = X + Y
    UV = U + V
    m = nsplits
    c = coadd_name_func
    f = field_names
    s = split_name_func
    e = lambda inp: eval_feed_dict[inp.name] if eval_feed_dict is not None else inp
    xc = e(Symbol(f'{c(f[0])}'))
    yc = e(Symbol(f'{c(f[1])}'))
    uc = e(Symbol(f'{c(f[2])}'))
    vc = e(Symbol(f'{c(f[3])}'))
    xs = [None]*m ; ys = [None]*m ; us = [None]*m ; vs = [None]*m
    for i in range(m):
        xs[i] = e(Symbol(f'{s(f[0],i)}'))
        ys[i] = e(Symbol(f'{s(f[1],i)}'))
        us[i] = e(Symbol(f'{s(f[2],i)}'))
        vs[i] = e(Symbol(f'{s(f[3],i)}'))

    phi_A_coadd = qe_func(XY,xc,yc,term=0)
    phi_B_coadd = qe_func(UV,uc,vc,term=1)
    sum_phiiiA = 0
    sum_phiiiB = 0
    sum_CphiixA_phiixB = 0
    sum_CphiijA_phiijB = 0
    for i in range(int(m)):
        phiiA = (qe_func(XY,xs[i],yc,term=0)+qe_func(XY,xc,ys[i],term=0))/2.
        phiiB = (qe_func(UV,us[i],vc,term=1)+qe_func(UV,uc,vs[i],term=1))/2.
        phiiiA = qe_func(XY,xs[i],ys[i],term=0) # = (qe_func(XY,xs[i],ys[i])+qe_func(XY,xs[i],ys[i]))/2.
        phiiiB = qe_func(UV,us[i],vs[i],term=1) # = (qe_func(UV,us[i],vs[i])+qe_func(UV,us[i],vs[i]))/2.
        sum_phiiiA += phiiiA
        sum_phiiiB += phiiiB
        phiixA = phiiA - phiiiA/m
        phiixB = phiiB - phiiiB/m
        sum_CphiixA_phiixB += pow_func(phiixA,phiixB)
        for j in range(i+1,int(m)):
            phiijA = (qe_func(XY,xs[i],ys[j],term=0)+qe_func(XY,xs[j],ys[i],term=0))/2.
            phiijB = (qe_func(UV,us[i],vs[j],term=1)+qe_func(UV,us[j],vs[i],term=1))/2.
            sum_CphiijA_phiijB += pow_func(phiijA,phiijB)
    phixA = phi_A_coadd - sum_phiiiA / m**2
    phixB = phi_B_coadd - sum_phiiiB / m**2
    C_phixA_phixB = pow_func(phixA,phixB)
    return ( m**4. * C_phixA_phixB- 4. * m**2. * sum_CphiixA_phiixB + \
               4. *sum_CphiijA_phiijB ) /m / (m-1.) / (m-2.) / (m-3.)


def get_feed_dict(shape,wcs,theory,noise_t,noise_p,fwhm,gtfunc=None,split_estimator=False,noise_scale=1.):
    from pixell import enmap
    import symlens
    modlmap = enmap.modlmap(shape,wcs)
    feed_dict = {}
    feed_dict['uC_T_T'] = theory.lCl('TT',modlmap) if (gtfunc is None) else gtfunc(modlmap)
    feed_dict['uC_T_E'] = theory.lCl('TE',modlmap)
    feed_dict['uC_E_E'] = theory.lCl('EE',modlmap)

    feed_dict['tC_T_T'] = theory.lCl('TT',modlmap) + (noise_t * np.pi/180./60.)**2. / symlens.gauss_beam(modlmap,fwhm)**2.
    feed_dict['tC_T_E'] = theory.lCl('TE',modlmap)
    feed_dict['tC_E_E'] = theory.lCl('EE',modlmap) + (noise_p * np.pi/180./60.)**2. / symlens.gauss_beam(modlmap,fwhm)**2.
    feed_dict['tC_B_B'] = theory.lCl('BB',modlmap) + (noise_p * np.pi/180./60.)**2. / symlens.gauss_beam(modlmap,fwhm)**2.

    if split_estimator:
        ntt = 0
        npp = 0
    else:
        ntt = noise_scale*(noise_t * np.pi/180./60.)**2. / symlens.gauss_beam(modlmap,fwhm)**2.
        npp = noise_scale*(noise_p * np.pi/180./60.)**2. / symlens.gauss_beam(modlmap,fwhm)**2.
    feed_dict['nC_T_T'] = theory.lCl('TT',modlmap) + ntt
    feed_dict['nC_T_E'] = theory.lCl('TE',modlmap)
    feed_dict['nC_E_E'] = theory.lCl('EE',modlmap) + npp
    feed_dict['nC_B_B'] = theory.lCl('BB',modlmap) + npp

    return feed_dict


def RDN0_analytic(shape,wcs,theory,fwhm,noise_t,noise_p,powdict,estimator,XY,UV,
                  xmask,ymask,kmask,AXY,AUV,split_estimator=False,gtfunc=None,noise_scale=1.):
    import symlens
    feed_dict = get_feed_dict(shape,wcs,theory,noise_t,noise_p,fwhm,gtfunc=gtfunc,
                              split_estimator=split_estimator,noise_scale=noise_scale)
    feed_dict['dC_T_T'] = powdict['TT']
    feed_dict['dC_T_E'] = powdict['TE']
    feed_dict['dC_E_E'] = powdict['EE']
    feed_dict['dC_B_B'] = powdict['BB']
    return symlens.RDN0_analytic(shape,wcs,feed_dict,estimator,XY,estimator,UV,
                                 Aalpha=AXY,Abeta=AUV,xmask=xmask,ymask=ymask,kmask=kmask,
                                 field_names_alpha=None,field_names_beta=None,skip_filter_field_names=False,
                                 split_estimator=split_estimator)
