import numpy as np

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
    
def rdn0(icov,alpha,beta,qfunc,get_kmap,comm,power,nsims):
    """
    RDN0 for alpha=XY cross beta=AB
    qfunc(XY,x,y) returns QE XY reconstruction minus mean-field in fourier space
    get_kmap("T",(0,0,1)

    e.g. rdn0(0,"TT","TE",qest.get_kappa,get_kmap,comm,power)
    """
    eX,eY = alpha
    eA,eB = beta
    qa = lambda x,y: qfunc(alpha,x,y)
    qb = lambda x,y: qfunc(beta,x,y)
    # Data
    X = get_kmap(eX,(0,0,0))
    Y = get_kmap(eY,(0,0,0))
    A = get_kmap(eA,(0,0,0))
    B = get_kmap(eB,(0,0,0))
    # Sims
    rdn0 = 0.
    for i in range(comm.rank+1, nsims+1, comm.size):        
        Xs  = get_kmap(eX,(icov,0,i))
        Ys  = get_kmap(eY,(icov,0,i))
        As  = get_kmap(eA,(icov,0,i))
        Bs  = get_kmap(eB,(icov,0,i))
        Ysp = get_kmap(eY,(icov,1,i))
        Asp = get_kmap(eA,(icov,1,i))
        Bsp = get_kmap(eB,(icov,1,i))
        rdn0 += power(qa(X,Ys),qb(A,Bs)) + power(qa(Xs,Y),qb(A,Bs)) \
            + power(qa(Xs,Y),qb(As,B)) + power(qa(X,Ys),qb(As,B)) \
            - power(qa(Xs,Ysp),qb(As,Bsp)) - power(qa(Xs,Ysp),qb(Asp,Bs))
    from pixell import utils
    totrdn0 = utils.allreduce(rdn0,comm) 
    return totrdn0/nsims

def mcn1(icov,alpha,beta,qfunc,get_kmap,comm,power,nsims,verbose=False):
    """
    MCN1 for alpha=XY cross beta=AB
    qfunc(x,y) returns QE reconstruction minus mean-field in fourier space
    """
    eX,eY = alpha
    eA,eB = beta
    qa = lambda x,y: qfunc(alpha,x,y)
    qb = lambda x,y: qfunc(beta,x,y)
    mcn1 = 0.
    for i in range(comm.rank+1, nsims+1, comm.size):
        if verbose: print("Rank %d doing task %d" % (comm.rank,i))
        Xsk   = get_kmap(eX,(icov,2,i))
        Yskp  = get_kmap(eY,(icov,3,i))
        Ask   = get_kmap(eA,(icov,2,i))
        Bskp  = get_kmap(eB,(icov,3,i))
        Askp  = get_kmap(eA,(icov,3,i))
        Bsk   = get_kmap(eB,(icov,2,i))
        Xs    = get_kmap(eX,(icov,0,i))
        Ysp   = get_kmap(eY,(icov,1,i))
        As    = get_kmap(eA,(icov,0,i))
        Bsp   = get_kmap(eB,(icov,1,i))
        Asp   = get_kmap(eA,(icov,1,i))
        Bs    = get_kmap(eB,(icov,0,i))
        mcn1 += power(qa(Xsk,Yskp),qb(Ask,Bskp)) + power(qa(Xsk,Yskp),qb(Askp,Bsk)) \
            - power(qa(Xs,Ysp),qb(As,Bsp)) - power(qa(Xs,Ysp),qb(Asp,Bs)) # FIXME: make mpi aware
    from pixell import utils
    totmcn1 = utils.allreduce(mcn1,comm) 
    return mcn1/nsims


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
            kx   = get_kmap(eX,(icov,j,i))
            ky   = get_kmap(eY,(icov,j,i))
            mf += qe(kx,ky)
            ntot += 1.
    from pixell import utils
    mftot = utils.allreduce(mf,comm) 
    totnot = utils.allreduce(ntot,comm) 
    return mftot/totntot
        
