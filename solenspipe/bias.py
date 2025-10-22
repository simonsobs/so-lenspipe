import numpy as np
from orphics import mpi
from pixell import utils # These are needed for MPI. Relevant functions can be copied over.
import healpy as hp
from enlib import bench
from pixell.mpi import FakeCommunicator
from pixell import lensing as plensing

"""
Extremely general functions for lensing power spectrum bias subtraction
======================================================================

These are quite general (independent of pixelization), FFT/SHT, region, array, etc.

One abstracts out the following:

qfunc(x,y)
which for estimator XY e.g. "TT"
x,y accepts fourier transformed beam deconvolved filtered inpainted purified [T, E, B] maps.
It should probably belong to some prepared object which knows about filters etc.

get_kmap(seed)
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
if set==0 or set==1: Corresponding to S and S'
    cmb_seed = (icov,set,i)+(0,)
    kappa_seed = (icov,set,i)+(1,)
    noise_seed = (icov,set,i)+(2,)
elif set==2 or set==3: Corresponds to S_phi and S'_phi of the lensing papers
    cmb_seed = (icov,set,i)+(0,)
    kappa_seed = (icov,2,i)+(1,)
    noise_seed = (icov,set,i)+(2,)

=================
"""

def mcrdn0(icov, get_kmap, power, nsims, qfunc1,get_kmap1=None, qfunc2=None, Xdat=None,Xdat1=None, use_mpi=True, 
         verbose=True, skip_rd=False):
         
    """
    Using Monte Carlo sims, this function calculates
    the anisotropic Gaussian N0 bias
    between two quadratic estimators. It returns both the
    data realization-dependent version (RDN0) and the pure-simulation
    version (MCN0).
    e.g. 
    >> mcrdn0(0,qfunc,get_kmap,comm,power)
    
    Parameters
    ----------
    icov: int
        The index of the realization passed to get_kmap if performing 
    a covariance calculation - otherwise, set to zero.
    get_kmap: function
        Function for getting the filtered a_lms of  data and simulation
    maps. See notes at top of module.
    power: function
        Returns C(l) from two maps x,y, as power(x,y). 
    nsims: int
        Number of sims
    qfunc1: function
        Function for reconstructing lensing from maps x and y,
    called as e.g. qfunc(x, y). See e.g. SoLensPipe.qfunc.
    The x and y arguments accept a [T_alm,E_alm,B_alm] tuple.
    The function should return an (N,...) array where N is typically
    two components for the gradient and curl. 
    qfunc2: function, optional
        Same as above, for the third and fourth legs of the 4-point
    RDN0.
    comm: object, optional
        MPI communicator
    verbose: bool, optional
        Whether to show progress
    skip_rd: bool, optional
        Whether to skip the RDN0 terms. The first returned component
    is then None.

    Returns
    -------
    rdn0: (N*(N+1)/2,...) array
        Estimate of the RDN0 bias. If N=2 for gradient and curl,
    the three components correspond to the gradient RDN0, the
    curl RDN0 and the gradient x curl RDN0. None is returned
    if skip_rd is True.

    mcn0: (N*(N+1)/2,...) array
        Estimate of the MCN0 bias. If N=2 for gradient and curl,
    the three components correspond to the gradient RDN0, the
    curl RDN0 and the gradient x curl RDN0.
    
    """
    qa = qfunc1 
    qb = qfunc2

    mcn0evals = []
    if not(skip_rd): 
        assert Xdat is not None # Data
        if Xdat1 is None:
            Xdat1=Xdat
        rdn0evals = []

    if use_mpi:
        comm,rank,my_tasks = mpi.distribute(nsims)
    else:
        comm,rank,my_tasks = FakeCommunicator(), 0, range(nsims)
        

    for i in my_tasks:
        i=i+2
        if rank==0 and verbose: print("MCRDN0: Rank %d doing task %d" % (rank,i))
        Xs  = get_kmap((icov,0,i))
        if get_kmap1 is None:
            Xs1=Xs
        else:
            Xs1=get_kmap1((icov,0,i))
        if not(skip_rd): 
            qaXXs = qa(Xdat,Xs) #for the two split case, one would need two get_kmaps?, or instead returns an array of maps, [i] for split i
            qbXXs = qb(Xdat1,Xs1) if qb is not None else qaXXs #this is split 2
            qaXsX = qa(Xs,Xdat)  #split 1
            qbXsX = qb(Xs1,Xdat1) if qb is not None else qaXsX #this is split 2
            rdn0_only_term = power(qaXXs,qbXXs) + power(qaXsX,qbXXs) \
                    + power(qaXsX,qbXsX) + power(qaXXs,qbXsX)
            
        Xsp = get_kmap((icov,0,i+1)) 
        if get_kmap1 is None:
            Xsp1=Xsp
        else:
            Xsp1=get_kmap1((icov,0,i+1))

        qaXsXsp = qa(Xs,Xsp) #split1 
        qbXsXsp = qb(Xs1,Xsp1) if qb is not None else qaXsXsp #split2

        qbXspXs = qb(Xsp1,Xs1) if qb is not None else qa(Xsp,Xs) #this is not present

        mcn0_term = (power(qaXsXsp,qbXsXsp) + power(qaXsXsp,qbXspXs))
        mcn0evals.append(mcn0_term.copy())
        if not(skip_rd):  rdn0evals.append(rdn0_only_term - mcn0_term)

    if not(skip_rd):
        avgrdn0 = utils.allgatherv(rdn0evals,comm)
    else:
        avgrdn0 = None
    avgmcn0 = utils.allgatherv(mcn0evals,comm)
    return avgrdn0, avgmcn0


def mcrdn0_reuse(icov, get_kmap, power, nsims, qfunc1,get_kmap1=None, qfunc2=None, Xdat=None,Xdat1=None, use_mpi=True, 
         verbose=True, skip_rd=False):
         
    """
    Using Monte Carlo sims, this function calculates
    the anisotropic Gaussian N0 bias
    between two quadratic estimators. It returns both the
    data realization-dependent version (RDN0) and the pure-simulation
    version (MCN0).
    e.g. 
    >> mcrdn0(0,qfunc,get_kmap,comm,power)
    
    Parameters
    ----------
    icov: int
        The index of the realization passed to get_kmap if performing 
    a covariance calculation - otherwise, set to zero.
    get_kmap: function
        Function for getting the filtered a_lms of  data and simulation
    maps. See notes at top of module.
    power: function
        Returns C(l) from two maps x,y, as power(x,y). 
    nsims: int
        Number of sims
    qfunc1: function
        Function for reconstructing lensing from maps x and y,
    called as e.g. qfunc(x, y). See e.g. SoLensPipe.qfunc.
    The x and y arguments accept a [T_alm,E_alm,B_alm] tuple.
    The function should return an (N,...) array where N is typically
    two components for the gradient and curl. 
    qfunc2: function, optional
        Same as above, for the third and fourth legs of the 4-point
    RDN0.
    comm: object, optional
        MPI communicator
    verbose: bool, optional
        Whether to show progress
    skip_rd: bool, optional
        Whether to skip the RDN0 terms. The first returned component
    is then None.

    Returns
    -------
    rdn0: (N*(N+1)/2,...) array
        Estimate of the RDN0 bias. If N=2 for gradient and curl,
    the three components correspond to the gradient RDN0, the
    curl RDN0 and the gradient x curl RDN0. None is returned
    if skip_rd is True.

    mcn0: (N*(N+1)/2,...) array
        Estimate of the MCN0 bias. If N=2 for gradient and curl,
    the three components correspond to the gradient RDN0, the
    curl RDN0 and the gradient x curl RDN0.
    
    """
    qa = qfunc1 
    qb = qfunc2

    mcn0evals = []
    if not(skip_rd): 
        assert Xdat is not None # Data
        if Xdat1 is None:
            Xdat1=Xdat
        rdn0evals = []

    if use_mpi:
        comm,rank,my_tasks = mpi.distribute(nsims)
    else:
        comm,rank,my_tasks = FakeCommunicator(), 0, range(nsims)
        

    for i in my_tasks:
        i=i+2
        if rank==0 and verbose: print("MCRDN0: Rank %d doing task %d" % (rank,i))
        Xs  = get_kmap((icov,1,i))
        if get_kmap1 is None:
            Xs1=Xs
        else:
            Xs1=get_kmap1((icov,1,i))
        if not(skip_rd): 
            qaXXs = qa(Xdat,Xs) #for the two split case, one would need two get_kmaps?, or instead returns an array of maps, [i] for split i
            qbXXs = qb(Xdat1,Xs1) if qb is not None else qaXXs #this is split 2
            qaXsX = qa(Xs,Xdat)  #split 1
            qbXsX = qb(Xs1,Xdat1) if qb is not None else qaXsX #this is split 2
            rdn0_only_term = power(qaXXs,qbXXs) + power(qaXsX,qbXXs) \
                    + power(qaXsX,qbXsX) + power(qaXXs,qbXsX)
        Xsp = get_kmap((icov,0,i)) 
        if get_kmap1 is None:
            Xsp1=Xsp
        else:
            Xsp1=get_kmap1((icov,0,i))

        qaXsXsp = qa(Xs,Xsp) #split1 
        qbXsXsp = qb(Xs1,Xsp1) if qb is not None else qaXsXsp #split2

        qbXspXs = qb(Xsp1,Xs1) if qb is not None else qa(Xsp,Xs) #this is not present

        mcn0_term = (power(qaXsXsp,qbXsXsp) + power(qaXsXsp,qbXspXs))
        mcn0evals.append(mcn0_term.copy())
        if not(skip_rd):  rdn0evals.append(rdn0_only_term - mcn0_term)

    if not(skip_rd):
        avgrdn0 = utils.allgatherv(rdn0evals,comm)
    else:
        avgrdn0 = None
    avgmcn0 = utils.allgatherv(mcn0evals,comm)
    return avgrdn0, avgmcn0

def rdn0(icov, get_kmap, power, nsims, qfunc1, qfunc2=None, Xdat=None, comm=None, 
         verbose=False):
         
    """
    Using Monte Carlo sims, this function calculates
    the anisotropic realization-dependent Gaussian RDN0 bias
    between two quadratic estimators. 
    e.g. 
    >> rdn0(0,qfunc,get_kmap,comm,power)
    
OA    Parameters
    ----------
    icov: int
        The index of the realization passed to get_kmap if performing 
    a covariance calculation - otherwise, set to zero.
    get_kmap: function
        Function for getting the filtered a_lms of  data and simulation
    maps. See notes at top of module.
    power: function
        Returns C(l) from two maps x,y, as power(x,y). 
    nsims: int
        Number of sims
    qfunc1: function
        Function for reconstructing lensing from maps x and y,
    called as e.g. qfunc(x, y). See e.g. SoLensPipe.qfunc.
    The x and y arguments accept a [T_alm,E_alm,B_alm] tuple.
    The function should return an (N,...) array where N is typically
    two components for the gradient and curl. 
    qfunc2: function, optional
        Same as above, for the third and fourth legs of the 4-point
    RDN0.
    comm: object, optional
        MPI communicator
    verbose: bool, optional
        Whether to show progress

    Returns
    -------
    rdn0: (N*(N+1)/2,...) array
        Estimate of the RDN0 bias. If N=2 for gradient and curl,
    the three components correspond to the gradient RDN0, the
    curl RDN0 and the gradient x curl RDN0. 

    """
    return mcrdn0(icov, get_kmap, power, nsims, qfunc1, qfunc2=qfunc2, Xdat=Xdat, comm=comm, 
           verbose=verbose, skip_rd=False)[0]

def mcn0(icov, get_kmap, power, nsims, qfunc1, qfunc2=None, comm=None, 
         verbose=False):
         
    """
    Using Monte Carlo sims, this function calculates
    the anisotropic Gaussian N0 bias
    between two quadratic estimators, using only simulations (MCN0).
    e.g. 
    >> mcrdn0(0,qfunc,get_kmap,comm,power)
    
    Parameters
    ----------
    icov: int
        The index of the realization passed to get_kmap if performing 
    a covariance calculation - otherwise, set to zero.
    get_kmap: function
        Function for getting the filtered a_lms of  data and simulation
    maps. See notes at top of module.
    power: function
        Returns C(l) from two maps x,y, as power(x,y). 
    nsims: int
        Number of sims
    qfunc1: function
        Function for reconstructing lensing from maps x and y,
    called as e.g. qfunc(x, y). See e.g. SoLensPipe.qfunc.
    The x and y arguments accept a [T_alm,E_alm,B_alm] tuple.
    The function should return an (N,...) array where N is typically
    two components for the gradient and curl. 
    qfunc2: function, optional
        Same as above, for the third and fourth legs of the 4-point
    RDN0.
    comm: object, optional
        MPI communicator
    verbose: bool, optional
        Whether to show progress

    Returns
    -------
    mcn0: (N*(N+1)/2,...) array
        Estimate of the MCN0 bias. If N=2 for gradient and curl,
    the three components correspond to the gradient RDN0, the
    curl RDN0 and the gradient x curl RDN0.
    
    """
    return mcrdn0(icov, get_kmap, power, nsims, qfunc1, qfunc2=qfunc2, Xdat=None, comm=comm, 
                  verbose=verbose, skip_rd=True)[1]


def mcn1(icov,get_kmap,power,nsims,qfunc1,qfunc2=None,comm=None,verbose=True,shear=False):
    """
    MCN1 for alpha=XY cross beta=AB
    qfunc(x,y) returns QE reconstruction minus mean-field in fourier space


    Parameters
    ----------
    icov: int
        The index of the realization passed to get_kmap if performing 
    a covariance calculation - otherwise, set to zero.
    get_kmap: function
        Function for getting the filtered a_lms of  data and simulation
    maps. See notes at top of module.
    power: function
        Returns C(l) from two maps x,y, as power(x,y). 
    nsims: int
        Number of sims
    qfunc1: function
        Function for reconstructing lensing from maps x and y,
    called as e.g. qfunc(x, y). See e.g. SoLensPipe.qfunc.
    The x and y arguments accept a [T_alm,E_alm,B_alm] tuple. 
    The function should return an (N,...) array where N is typically
    two components for the gradient and curl. 
    qfunc2: function, optional
        Same as above, for the third and fourth legs of the 4-point
    MCN1.
    comm: object, optional
        MPI communicator
    verbose: bool, optional
        Whether to show progress

    Returns
    -------
    mcn1: (N*(N+1)/2,...) array
        Estimate of the MCN1 bias. If N=2 for gradient and curl,
    the three components correspond to the gradient MCN1, the
    curl MCN1 and the gradient x curl MCN1.
    
        Estimate of the MCN1 bias

    """
    qa = qfunc1
    qb = qfunc2
    comm,rank,my_tasks = mpi.distribute(nsims)
    n1evals = []
    for i in my_tasks:
        i=i+2
        if rank==0 and verbose: print("MCN1: Rank %d doing task %d" % (comm.rank,i))
        Xs    = get_kmap((icov,0,i)) # S
        Ysp   = get_kmap((icov,1,i)) # S'
        Xsk   = get_kmap((icov,2,i)) # Sphi
        Yskp  = get_kmap((icov,3,i)) # Sphi'
        if shear:
            qa_Xsk_Yskp = 0.5*(qa(Xsk[0],Yskp[1])+qa(Yskp[0],Xsk[1]))
            qb_Xsk_Yskp = 0.5*(qb(Xsk[0],Yskp[1])+qb(Yskp[0],Xsk[1])) if qb is not None else qa_Xsk_Yskp
            qb_Yskp_Xsk = 0.5*(qb(Yskp[0],Xsk[1])+qb(Xsk[0],Yskp[1])) if qb is not None else 0.5*(qa(Yskp[0],Xsk[1])+qa(Xsk[0],Yskp[1]))
            qa_Xs_Ysp = 0.5*(qa(Xs[0],Ysp[1])+qa(Ysp[0],Xs[1]))
            qb_Xs_Ysp = 0.5*(qb(Xs[0],Ysp[1])+qb(Ysp[0],Xs[1])) if qb is not None else qa_Xs_Ysp
            qb_Ysp_Xs = 0.5*(qb(Ysp[0],Xs[1])+qb(Xs[0],Ysp[1])) if qb is not None else 0.5*(qa(Ysp[0],Xs[1])+qa(Xs[0],Ysp[1]))
            qb_Yskp_Xsk = 0.5*(qb(Yskp[0],Xsk[1])+qb(Xsk[0],Yskp[1])) if qb is not None else 0.5*(qa(Yskp[0],Xsk[1])+qa(Xsk[0],Yskp[1]))
        else:
            qa_Xsk_Yskp = 0.5*(qa(Xsk,Yskp)+qa(Yskp,Xsk))
            qb_Xsk_Yskp = 0.5*(qb(Xsk,Yskp)+qb(Yskp,Xsk)) if qb is not None else qa_Xsk_Yskp
            qb_Yskp_Xsk = 0.5*(qb(Yskp,Xsk)+qb(Xsk,Yskp)) if qb is not None else qa(Yskp,Xsk)
            qa_Xs_Ysp = 0.5*(qa(Xs,Ysp)+qa(Ysp,Xs))
            qb_Xs_Ysp = 0.5*(qb(Xs,Ysp)+qb(Ysp,Xs)) if qb is not None else qa_Xs_Ysp
            qb_Ysp_Xs = 0.5*(qb(Ysp,Xs)+qb(Xs,Ysp)) if qb is not None else 0.5*(qa(Ysp,Xs)+qa(Xs,Ysp))
            qb_Yskp_Xsk = 0.5*(qb(Yskp,Xsk)+qb(Xsk,Yskp)) if qb is not None else 0.5*(qa(Yskp,Xsk)+qa(Xsk,Yskp))
        term = power(qa_Xsk_Yskp,qb_Xsk_Yskp) + power(qa_Xsk_Yskp,qb_Yskp_Xsk) \
            - power(qa_Xs_Ysp,qb_Xs_Ysp) - power(qa_Xs_Ysp,qb_Ysp_Xs)
        n1evals.append(term.copy())
    n1s = utils.allgatherv(n1evals,comm)
    return n1s

def mcn1_s4(icov,get_kmap,power,phifunc,nsims,qfunc1,
            qfunc2=None,use_mpi=True,verbose=True,shear=False):
    """
    MCN1 for alpha=XY cross beta=AB (4 split noisy sims)
    qfunc(x,y) returns QE reconstruction minus mean-field in fourier space


    Parameters
    ----------
    icov: int
        The index of the realization passed to get_kmap if performing 
    a covariance calculation - otherwise, set to zero.
    get_kmap: function
        Function for getting the filtered a_lms of  data and simulation
    maps. See notes at top of module.
    power: function
        Returns C(l) from two maps x,y, as power(x,y). 
    nsims: int
        Number of sims
    qfunc1: function
        Function for reconstructing lensing from maps x and y,
    called as e.g. qfunc(x, y). See e.g. SoLensPipe.qfunc.
    The x and y arguments accept a [T_alm,E_alm,B_alm] tuple. 
    The function should return an (N,...) array where N is typically
    two components for the gradient and curl. 
    qfunc2: function, optional
        Same as above, for the third and fourth legs of the 4-point
    MCN1.
    comm: object, optional
        MPI communicator
    verbose: bool, optional
        Whether to show progress

    Returns
    -------
    mcn1: (N*(N+1)/2,...) array
        Estimate of the MCN1 bias. If N=2 for gradient and curl,
    the three components correspond to the gradient MCN1, the
    curl MCN1 and the gradient x curl MCN1.
    
        Estimate of the MCN1 bias

    """
    qf = phifunc
    qa = qfunc1
    qb = qfunc2


    if use_mpi:
        comm,rank,my_tasks = mpi.distribute(nsims)
    else:
        comm,rank,my_tasks = FakeCommunicator(), 0, range(nsims)

    n1evals = []
    for i in my_tasks:
        i=i+2 # (0 = data, 1 = noisy sim treated as data)
        if verbose: print("MCN1: Rank %d doing task %d" % (comm.rank,i))
        Xs    = [get_kmap((icov,0,i), split=j) for j in range(4)] # S
        Ysp   = [get_kmap((icov,1,i), split=j) for j in range(4)] # S'
        Xsk   = [get_kmap((icov,2,i), split=j) for j in range(4)] # Sphi
        Yskp  = [get_kmap((icov,3,i), split=j) for j in range(4)] # Sphi'

        def shear_field(field, shear_i):
            return [field[k][shear_i] for k in range(4)]
        
        if shear:
            qa_Xsk_Yskp = 0.5*(qf(*shear_field(Xsk,0), *shear_field(Yskp,1), qa) + \
                               qf(*shear_field(Yskp,0), *shear_field(Xsk,1), qa))
            qb_Xsk_Yskp = 0.5*(qf(*shear_field(Xsk,0), *shear_field(Yskp,1), qb) + \
                               qf(*shear_field(Yskp,0), *shear_field(Xsk,1), qb)) if qb is not None else qa_Xsk_Yskp
            if qb is None:
                qb_Yskp_Xsk = 0.5*(qf(*shear_field(Yskp,0), *shear_field(Xsk,1), qa) + \
                                   qf(*shear_field(Xsk,0), *shear_field(Yskp,1), qa))
            else:
                qb_Yskp_Xsk = 0.5*(qf(*shear_field(Yskp,0), *shear_field(Xsk,1), qb) + \
                                   qf(*shear_field(Xsk,0), *shear_field(Yskp,1), qb))
                
            qa_Xs_Ysp = 0.5*(qf(*shear_field(Xs,0), *shear_field(Ysp,1), qa) + \
                             qf(*shear_field(Ysp,0), *shear_field(Xs,1), qa))
            qb_Xs_Ysp = 0.5*(qf(*shear_field(Xs,0), *shear_field(Ysp,1), qb) + \
                             qf(*shear_field(Ysp,0), *shear_field(Xs,1), qb)) if qb is not None else qa_Xs_Ysp
            if qb is None:
                qb_Ysp_Xs = 0.5*(qf(*shear_field(Ysp,0), *shear_field(Xs,1), qa) + \
                                 qf(*shear_field(Xs,0), *shear_field(Ysp,1), qa))
                qb_Yskp_Xsk = 0.5*(qf(*shear_field(Yskp,0), *shear_field(Xsk,1), qa) + \
                                   qf(*shear_field(Xsk,0), *shear_field(Yskp,1), qa))
            else:
                qb_Ysp_Xs = 0.5*(qf(*shear_field(Ysp,0), *shear_field(Xs,1), qb) + \
                                 qf(*shear_field(Xs,0), *shear_field(Ysp,1), qb))
                qb_Yskp_Xsk = 0.5*(qf(*shear_field(Yskp,0), *shear_field(Xsk,1), qb) + \
                                   qf(*shear_field(Xsk,0), *shear_field(Yskp,1), qb))
        else:
            qa_Xsk_Yskp = 0.5*(qf(*Xsk,*Yskp,qa)+qf(*Yskp,*Xsk,qa))
            qb_Xsk_Yskp = 0.5*(qf(*Xsk,*Yskp,qb)+qf(*Yskp,*Xsk,qb)) if qb is not None else qa_Xsk_Yskp
            qb_Yskp_Xsk = 0.5*(qf(*Yskp,*Xsk,qb)+qf(*Xsk,*Yskp,qb)) if qb is not None else qf(*Yskp,*Xsk,qa)
            qa_Xs_Ysp = 0.5*(qf(*Xs,*Ysp,qa)+qf(*Ysp,*Xs,qa))
            qb_Xs_Ysp = 0.5*(qf(*Xs,*Ysp,qb)+qf(*Ysp,*Xs,qb)) if qb is not None else qa_Xs_Ysp
            if qb is None:
                qb_Ysp_Xs = 0.5*(qf(*Ysp,*Xs,qa)+qf(*Xs,*Ysp,qa))
                qb_Yskp_Xsk = 0.5*(qf(*Yskp,*Xsk,qa)+qf(*Xsk,*Yskp,qa))
            else:
                qb_Ysp_Xs = 0.5*(qf(*Ysp,*Xs,qb)+qf(*Xs,*Ysp,qb))
                qb_Yskp_Xsk = 0.5*(qf(*Yskp,*Xsk,qb)+qf(*Xsk,*Yskp,qb))

        term = power(qa_Xsk_Yskp,qb_Xsk_Yskp) + power(qa_Xsk_Yskp,qb_Yskp_Xsk) \
             - power(qa_Xs_Ysp,qb_Xs_Ysp) - power(qa_Xs_Ysp,qb_Ysp_Xs)
        n1evals.append(term.copy())

    n1s = utils.allgatherv(n1evals,comm)
    return n1s

def mcmf(icov,qfunc,get_kmap,comm,nsims):
    """
    MCMF for alpha=XY
    qfunc(x,y) returns QE reconstruction minus mean-field in fourier space
    """
    qe = lambda x,y: qfunc(x,y)
    mf = 0.
    ntot = 0.
    for i in range(comm.rank+1, nsims+1, comm.size):        
        for j in range(2):
            kx   = get_kmap((icov,j,i))
            ky   = get_kmap((icov,j,i))
            mf += qe(kx,ky)
            ntot += 1.
    mftot = utils.allreduce(mf,comm) 
    totntot = utils.allreduce(ntot,comm) 
    return mftot/totntot

def mcmf_pair(icov,qfunc,get_kmap,comm,nsims):
    """
    MCMF for alpha=XY
    Computes a pair of mean-fields from two independent sets of sims.
    qfunc(x,y) returns QE reconstruction minus mean-field in fourier space
    """
    qe = lambda x,y: qfunc(x,y)
    mf1, mf2 = 0., 0.
    ntot = 0.
    for i in range(comm.rank+1, nsims+1, comm.size):        
        kx   = get_kmap((icov,0,i))
        ky   = get_kmap((icov,1,i))
        mf1 += qe(kx,kx)
        mf2 += qe(ky,ky)
        ntot += 1.
    mf1tot = utils.allreduce(mf1,comm) 
    mf2tot = utils.allreduce(mf2,comm)
    totntot = utils.allreduce(ntot,comm) 
    return mf1tot/totntot, mf2tot/totntot
        

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

def mcrdn0_s4(sim_set, get_kmap, power,phifunc, nsims, qfunc1,
              #get_kmap1=None,get_kmap2=None,get_kmap3=None,
              qfunc2=None, Xdat=None,Xdat1=None,Xdat2=None,Xdat3=None, 
              use_mpi=True, verbose=True, skip_rd=False,shear=False,power_mcn0=None):
    """
    This function performs the Realization dependent N0 (RDN0) using combinations of simulations and data for the cross-correlation based estimator. Currently for 4 splits of data.

    Parameters:
    sim_set: A tuple containing the simulation set and the start index used for the MC. This specifies the simulations used to estimate the RDN0
    get_kmap: A function to retrieve a specific map based on the provided parameters.
    power: A function to calculate the power spectra
    phifunc: function that returns the different phi combinations used to produce a phi_cl without noise bias. Eq.(28)-(30) of 2011.02475
    nsims: The number of simulations used to estimate RDN0.
    qfunc1: QE wrapper used.
    get_kmap1, get_kmap2, get_kmap3: Functions to retrieve the data map splits
    qfunc2: An optional QE wrapper if the second phi used to creat clphiphi is different to the first one.
    Xdat, Xdat1, Xdat2, Xdat3: Data used in the calculations.
    use_mpi: A boolean indicating whether to use MPI for distributing tasks.
    verbose: A boolean indicating whether to print verbose messages.
    skip_rd: A boolean indicating whether to skip certain operations.
    shear: A boolean indicating whether to apply shear in the calculations.
    power_mcn0: An optional parameter related to power calculations.

    Returns:
        A tuple of shape (nsims, 2, Lmax) for the gradient and the curl  modes.

    """

    i_set,start=sim_set
    qa = phifunc 
    qf1 = qfunc1
    qf2=qfunc2

    mcn0evals = []
    if not(skip_rd): 
        assert Xdat is not None # Data
        if Xdat1 is None:
            Xdat1=Xdat
        rdn0evals = []

    if use_mpi:
        comm,rank,my_tasks = mpi.distribute(nsims)
    else:
        comm,rank,my_tasks = FakeCommunicator(), 0, range(nsims)
        

    for i in my_tasks:
        i=i+start
        if rank==0 and verbose: print("MCRDN0: Rank %d doing task %d" % (rank,i))
        Xs  = get_kmap((0,i_set,i), split=0)
        
        if power_mcn0 is not None:
            print('noiseless as input, duplicating Xs')
            Xs1 = Xs
            Xs2 = Xs
            Xs3 = Xs
            
        else:
            Xs1 = get_kmap((0,i_set,i), split=1)
            Xs2 = get_kmap((0,i_set,i), split=2)
            Xs3 = get_kmap((0,i_set,i), split=3)


        if not(skip_rd): 
            if shear:
                qaXXs = qa(Xdat[0],Xdat1[0],Xdat2[0],Xdat3[0],Xs[1],Xs1[1],Xs2[1],Xs3[1],qf1) 
                qbXXs = qa(Xdat[0],Xdat1[0],Xdat2[0],Xdat3[0],Xs[1],Xs1[1],Xs2[1],Xs3[1],qf2) if qf2 is not None else qaXXs 
                qaXsX = qa(Xs[0],Xs1[0],Xs2[0],Xs3[0],Xdat[1],Xdat1[1],Xdat2[1],Xdat3[1],qf1) 
                qbXsX = qa(Xs[0],Xs1[0],Xs2[0],Xs3[0],Xdat[1],Xdat1[1],Xdat2[1],Xdat3[1],qf2) if qf2 is not None else qaXsX 
                rdn0_only_term = power(qaXXs,qbXXs)+ power(qaXXs,qbXsX) + power(qaXsX,qbXXs) \
                        + power(qaXsX,qbXsX) 
            else:
                qaXXs = qa(Xdat,Xdat1,Xdat2,Xdat3,Xs,Xs1,Xs2,Xs3,qf1)
                qbXXs = qa(Xdat,Xdat1,Xdat2,Xdat3,Xs,Xs1,Xs2,Xs3,qf2) if qf2 is not None else qaXXs 
                qaXsX = qa(Xs,Xs1,Xs2,Xs3,Xdat,Xdat1,Xdat2,Xdat3,qf1) 
                qbXsX = qa(Xs,Xs1,Xs2,Xs3,Xdat,Xdat1,Xdat2,Xdat3,qf2) if qf2 is not None else qaXsX 
                rdn0_only_term = power(qaXXs,qbXXs)+ power(qaXXs,qbXsX) + power(qaXsX,qbXXs) \
                        + power(qaXsX,qbXsX) 

        Xsp  = get_kmap((0,i_set,i+1), split=0) 
        Xsp1 = get_kmap((0,i_set,i+1), split=1) 
        Xsp2 = get_kmap((0,i_set,i+1), split=2) 
        Xsp3 = get_kmap((0,i_set,i+1), split=3) 


        if power_mcn0 is None:

            if shear:
                qaXsXsp = plensing.phi_to_kappa(qf1(Xs[0],Xsp[1])) #split1 
                qbXsXsp = plensing.phi_to_kappa(qf2(Xs[0],Xsp[1])) if qf2 is not None else qaXsXsp #split2
                qbXspXs = plensing.phi_to_kappa(qf2(Xsp[0],Xs[1])) if qf2 is not None else plensing.phi_to_kappa(qf1(Xsp[0],Xs[1])) #this is not present
            else:
                qaXsXsp = qa(Xs,Xs1,Xs2,Xs3,Xsp,Xsp1,Xsp2,Xsp3,qf1) #split1 
                qbXsXsp = qa(Xs,Xs1,Xs2,Xs3,Xsp,Xsp1,Xsp2,Xsp3,qf2) if qf2 is not None else qaXsXsp #split2
                qbXspXs = qa(Xsp,Xsp1,Xsp2,Xsp3,Xs,Xs1,Xs2,Xs3,qf2) if qf2 is not None else qa(Xsp,Xsp1,Xsp2,Xsp3,Xs,Xs1,Xs2,Xs3,qf1) #this is not present
            mcn0_term = (power(qaXsXsp,qbXsXsp) + power(qaXsXsp,qbXspXs))
        else:

            if shear:
                qaXsXsp = plensing.phi_to_kappa(qf1(Xs[0],Xsp[1])) #split1 
                qbXsXsp = plensing.phi_to_kappa(qf2(Xs[0],Xsp[1])) if qf2 is not None else qaXsXsp #split2
                qbXspXs = plensing.phi_to_kappa(qf2(Xsp[0],Xs[1])) if qf2 is not None else plensing.phi_to_kappa(qf1(Xsp[0],Xs[1])) #this is not present
            else:
                qaXsXsp = plensing.phi_to_kappa(qf1(Xs,Xsp)) #split1 
                qbXsXsp = plensing.phi_to_kappa(qf2(Xs,Xsp)) if qf2 is not None else qaXsXsp #split2
                qbXspXs = plensing.phi_to_kappa(qf2(Xsp,Xs)) if qf2 is not None else plensing.phi_to_kappa(qf1(Xsp,Xs)) #this is not present
            mcn0_term = (power_mcn0(qaXsXsp,qbXsXsp) + power_mcn0(qaXsXsp,qbXspXs))

        mcn0evals.append(mcn0_term.copy())
        if not(skip_rd):  rdn0evals.append(rdn0_only_term - mcn0_term)

    if not(skip_rd):
        avgrdn0 = utils.allgatherv(rdn0evals,comm)
    else:
        avgrdn0 = None
    avgmcn0 = utils.allgatherv(mcn0evals,comm)
    return avgrdn0, avgmcn0


def mcrdn0_only(icov, get_kmap, power,phifunc, nsims, qfunc1,get_kmap1=None,get_kmap2=None,get_kmap3=None, qfunc2=None, Xdat=None,Xdat1=None,Xdat2=None,Xdat3=None, use_mpi=True, 
         verbose=True, skip_rd=False,shear=False):
         
    #boost another 400 mcn0 for final run i,i+2
    qa = phifunc 
    qf1 = qfunc1
    qf2=qfunc2
    

    mcn0evals = []
    if not(skip_rd): 
        assert Xdat is not None # Data
        if Xdat1 is None:
            Xdat1=Xdat
        rdn0evals = []

    if use_mpi:
        comm,rank,my_tasks = mpi.distribute(nsims)
    else:
        comm,rank,my_tasks = FakeCommunicator(), 0, range(nsims)
        

    for i in my_tasks:
        i=i+2
        if rank==0 and verbose: print("MCRDN0: Rank %d doing task %d" % (rank,i))
        Xs  = get_kmap((icov,0,i))
        Xs1= get_kmap1((icov,0,i))
        Xs2= get_kmap2((icov,0,i))
        Xs3= get_kmap3((icov,0,i))


        if not(skip_rd): 
            if shear:
                qaXXs = qa(Xdat[0],Xdat1[0],Xdat2[0],Xdat3[0],Xs[1],Xs1[1],Xs2[1],Xs3[1],qf1) #for the two split case, one would need two get_kmaps?, or instead returns an array of maps, [i] for split i
                qbXXs = qa(Xdat[0],Xdat1[0],Xdat2[0],Xdat3[0],Xs[1],Xs1[1],Xs2[1],Xs3[1],qf2) if qf2 is not None else qaXXs 
                qaXsX = qa(Xs[0],Xs1[0],Xs2[0],Xs3[0],Xdat[1],Xdat1[1],Xdat2[1],Xdat3[1],qf1) 
                qbXsX = qa(Xs[0],Xs1[0],Xs2[0],Xs3[0],Xdat[1],Xdat1[1],Xdat2[1],Xdat3[1],qf2) if qf2 is not None else qaXsX 
                rdn0_only_term = power(qaXXs,qbXXs)+ power(qaXXs,qbXsX) + power(qaXsX,qbXXs) \
                        + power(qaXsX,qbXsX) 
            else:
                qaXXs = qa(Xdat,Xdat1,Xdat2,Xdat3,Xs,Xs1,Xs2,Xs3,qf1) #for the two split case, one would need two get_kmaps?, or instead returns an array of maps, [i] for split i
                qbXXs = qa(Xdat,Xdat1,Xdat2,Xdat3,Xs,Xs1,Xs2,Xs3,qf2) if qf2 is not None else qaXXs 
                qaXsX = qa(Xs,Xs1,Xs2,Xs3,Xdat,Xdat1,Xdat2,Xdat3,qf1) 
                qbXsX = qa(Xs,Xs1,Xs2,Xs3,Xdat,Xdat1,Xdat2,Xdat3,qf2) if qf2 is not None else qaXsX 
                rdn0_only_term = power(qaXXs,qbXXs)+ power(qaXXs,qbXsX) + power(qaXsX,qbXXs) \
                        + power(qaXsX,qbXsX) 

        Xsp = get_kmap((icov,0,i+2)) 
        Xsp1 = get_kmap1((icov,0,i+2)) 
        Xsp2 = get_kmap2((icov,0,i+2)) 
        Xsp3 = get_kmap3((icov,0,i+2)) 

        if shear:
            qaXsXsp = qa(Xs[0],Xs1[0],Xs2[0],Xs3[0],Xsp[1],Xsp1[1],Xsp2[1],Xsp3[1],qf1) #split1 
            qbXsXsp = qa(Xs[0],Xs1[0],Xs2[0],Xs3[0],Xsp[1],Xsp1[1],Xsp2[1],Xsp3[1],qf2) if qf2 is not None else qaXsXsp #split2
            qbXspXs = qa(Xsp[0],Xsp1[0],Xsp2[0],Xsp3[0],Xs[1],Xs1[1],Xs2[1],Xs3[1],qf2) if qf2 is not None else qa(Xsp[0],Xsp1[0],Xsp2[0],Xsp3[0],Xs[1],Xs1[1],Xs2[1],Xs3[1],qf1) #this is not present
        else:
            qaXsXsp = qa(Xs,Xs1,Xs2,Xs3,Xsp,Xsp1,Xsp2,Xsp3,qf1) #split1 
            qbXsXsp = qa(Xs,Xs1,Xs2,Xs3,Xsp,Xsp1,Xsp2,Xsp3,qf2) if qf2 is not None else qaXsXsp #split2
            qbXspXs = qa(Xsp,Xsp1,Xsp2,Xsp3,Xs,Xs1,Xs2,Xs3,qf2) if qf2 is not None else qa(Xsp,Xsp1,Xsp2,Xsp3,Xs,Xs1,Xs2,Xs3,qf1) #this is not present

        mcn0_term = (power(qaXsXsp,qbXsXsp) + power(qaXsXsp,qbXspXs))
        mcn0evals.append(mcn0_term.copy())
        if not(skip_rd):  rdn0evals.append(rdn0_only_term - mcn0_term)

    if not(skip_rd):
        avgrdn0 = utils.allgatherv(rdn0evals,comm)
    else:
        avgrdn0 = None
    avgmcn0 = utils.allgatherv(mcn0evals,comm)
    return avgrdn0, avgmcn0


def simple_rdn0(icov,alpha,beta,qfunc,get_kmap,comm,power,nsims,Xdata,symmetric=False):
    """
    Original RDN0 function.
    RDN0 for alpha=XY cross beta=AB
    qfunc(XY,x,y) returns QE XY reconstruction minus mean-field in fourier space
    get_kmap("T",(0,0,1)

    e.g. rdn0(0,"TT","TE",qest.get_kappa,get_kmap,comm,power)
    """
    qa = lambda x,y: qfunc(alpha,x,y)
    qb = lambda x,y: qfunc(beta,x,y)
    # Sims
    rdn0 = 0.
    for i in range(comm.rank+1, nsims+1, comm.size):
        if comm.rank==0: print("RDN0 step ", i)
        Xs  = get_kmap((icov,0,i))
        Ysp = get_kmap((icov,1,i))
        qadys = qa(Xdata,Xs)
        qbdys = qb(Xdata,Xs) if alpha!=beta else qadys
        iqadys = qa(Xs,Xdata) if not(symmetric) else qadys
        iqbdys = qb(Xs,Xdata) if not(symmetric) else qbdys
        qaxsysp = qa(Xs,Ysp)
        qbxsysp = qb(Xs,Ysp) if alpha!=beta else qaxsysp
        qbyspxs = qb(Ysp,Xs) if not(symmetric) else qbxsysp
        rdn0 += power(qadys,qbdys) + power(iqadys,qbdys) \
            + power(iqadys,iqbdys) + power(qadys,iqbdys) \
            - power(qaxsysp,qbxsysp) - power(qaxsysp,qbyspxs)
        if comm.rank==0: print("RDN0 step ", i, " done")
    totrdn0 = utils.allreduce(rdn0,comm)
    print("RDN0 done")
    return totrdn0/nsims

# helper function for subset rdn0
def load_save_power(x_index, y_index, u_index, v_index,
                    qa, qf1, get_kmap, power, i_set = 0, repeat=False,
                    qf2=None, get_kmap1=None, get_kmap2=None, get_kmap3=None,
                    out_root="", format_cl="cl_cross_s1,sp1_x_s2,sp2_shear.txt",
                    format_phi="phi_cross_s1,sp1_shear.npy", shear=False,
                    noiseless_sims=True, verbose=True):
    
    new_format = out_root + format_cl.replace("s1", str(x_index)).replace("sp1", str(y_index)) \
                                     .replace("s2", str(u_index)).replace("sp2", str(v_index)) \
                                     .replace(".txt", "_noiseless.txt" if noiseless_sims else ".txt")
    new_format_phi = out_root + format_phi.replace(".npy", "_noiseless.npy" \
                                                           if noiseless_sims else ".npy")
    if not shear:
        new_format = new_format.replace("shear", "noshear")
        new_format_phi = new_format_phi.replace("shear", "noshear")

    if verbose:
        print("Format for cls: ", new_format)
        print("Format for intermediate phi: ", new_format_phi)

    # make sure {xy, yx} = {uv, vu}
    assert x_index in [u_index, v_index] and y_index in [u_index, v_index], "bad indices"

    # lazy evaluation, don't evaluate at all if we can find the saved cls in the first place
    def load_maps_split(i_set, i, j):
        As, Bs   = get_kmap((0,i_set,i)), get_kmap((0,i_set,j))
        As1, Bs1 = get_kmap1((0,i_set,i)), get_kmap1((0,i_set,j))
        As2, Bs2 = get_kmap2((0,i_set,i)), get_kmap2((0,i_set,j))
        As3, Bs3 = get_kmap3((0,i_set,i)), get_kmap3((0,i_set,j))

        return As, As1, As2, As3, Bs, Bs1, Bs2, Bs3

    def load_maps_nosplit(i_set, i, j):
        As, Bs = get_kmap((0,i_set,i)), get_kmap((0,i_set,j))

        return As, Bs

    # this parameter says xy = uv and to only compute one qe for the pair
    to_repeat = (repeat and qf2 is None)
    if qf2 is None: qf2 = qf1
    
    if shear:
        if noiseless_sims:
            def phi_xy():
                Xs, Ys = load_maps_nosplit(i_set, x_index, y_index)
                return plensing.phi_to_kappa(qf1(Xs[0], Ys[1]))
            def phi_uv():
                Us, Vs = load_maps_nosplit(i_set, u_index, v_index)
                return plensing.phi_to_kappa(qf2(Us[0], Vs[1]))
        else:
            def phi_xy():
                Xs, Xs1, Xs2, Xs3, Ys, Ys1, Ys2, Ys3 = load_maps_split(i_set, x_index, y_index)
                return qa(Xs[0], Xs1[0], Xs2[0], Xs3[0], Ys[1], Ys1[1], Ys2[1], Ys3[1], qf1)
            def phi_uv():
                Us, Us1, Us2, Us3, Vs, Vs1, Vs2, Vs3 = load_maps_split(i_set, u_index, v_index)
                return qa(Us[0], Us1[0], Us2[0], Us3[0], Vs[1], Vs1[1], Vs2[1], Vs3[1], qf2)
    else:
        if noiseless_sims:
            def phi_xy():
                Xs, Ys = load_maps_nosplit(i_set, x_index, y_index)
                return plensing.phi_to_kappa(qf1(Xs, Ys))
            def phi_uv():
                Us, Vs = load_maps_nosplit(i_set, u_index, v_index)
                return plensing.phi_to_kappa(qf2(Us, Vs))
        else:
            def phi_xy():
                Xs, Xs1, Xs2, Xs3, Ys, Ys1, Ys2, Ys3 = load_maps_split(i_set, x_index, y_index)
                return qa(Xs, Xs1, Xs2, Xs3, Ys, Ys1, Ys2, Ys3, qf1)
            def phi_uv():
                Us, Us1, Us2, Us3, Vs, Vs1, Vs2, Vs3 = load_maps_split(i_set, u_index, v_index)            
                return qa(Us, Us1, Us2, Us3, Vs, Vs1, Vs2, Vs3, qf2)

    def cl_xy_uv(p_xy_saved = None, p_uv_saved = None, to_repeat=to_repeat):
        with bench.show(f"phi xy ({x_index}, {y_index})"):
            p_xy = p_xy_saved if p_xy_saved is not None else phi_xy()
        with bench.show(f"phi uv ({u_index}, {v_index})"):
            if to_repeat:
                p_uv = p_xy
            else:
                p_uv = p_uv_saved if p_uv_saved is not None else phi_uv()
        return p_xy, p_uv, power(p_xy, p_uv)

    # check for saved cls
    try:
        print(f"Looking for cls: {new_format}")
        cl = np.loadtxt(new_format)
        print(f"Found cls: {new_format}")

    except FileNotFoundError:
        new_format_phi_xy = new_format_phi.replace("s1", str(x_index)).replace("sp1", str(y_index))
        new_format_phi_uv = new_format_phi.replace("s1", str(u_index)).replace("sp1", str(v_index))
        try:
            p_xy_loaded = np.load(new_format_phi_xy, allow_pickle=True)
            print(f"Found phi_xy: {new_format_phi_xy}")
        except FileNotFoundError:
            p_xy_loaded = None
        try:
            p_uv_loaded = np.load(new_format_phi_uv, allow_pickle=True)
            print(f"Found phi_uv: {new_format_phi_uv}!")
        except FileNotFoundError:
            p_uv_loaded = None
        
        with bench.show(f"cl_xy_uv: (({x_index},{y_index}),({u_index},{v_index}))"):
            p_xy, p_uv, cl = cl_xy_uv(p_xy_saved=p_xy_loaded,
                                      p_uv_saved=p_uv_loaded,
                                      to_repeat=to_repeat)

        if p_xy_loaded is None: np.save(new_format_phi_xy, p_xy)
        if p_uv_loaded is None: np.save(new_format_phi_uv, p_uv)
        np.savetxt(new_format, cl)

    return cl


def mcrdn0_subset_s4(sim_set, data_sim_id, get_kmap, power, phifunc, nsims,
                     qfunc1, get_kmap1=None, get_kmap2=None, get_kmap3=None,
                     qfunc2=None, use_mpi=True, verbose=True, skip_rd=False,
                     shear=False, power_mcn0=None, noiseless_mcn0=False, out_root="",
                     format_cl="cl_cross_s1,sp1_x_s2,sp2_shear.txt",
                     format_phi="phi_cross_s1,sp1_shear.npy"):
    """
    This function performs the Realization-dependent N0 (RDN0) using combinations of simulations for the cross-correlation based estimator.
    This variant treats a specific simulation index "data_sim_id" as the data, and computes an average over nsims.

    Parameters:
    sim_set: A tuple containing the simulation set and the start index used for the MC. This specifies the simulations used to estimate the RDN0
    data_sim_id: Sim index for the data.
    get_kmap: A function to retrieve a specific map based on the provided parameters.
    power: A function to calculate the power spectra
    phifunc: function that returns the different phi combinations used to produce a phi_cl without noise bias. Eq.(28)-(30) of 2011.02475
    nsims: The number of simulations used to estimate RDN0.
    qfunc1: QE wrapper used.
    get_kmap1, get_kmap2, get_kmap3: Functions to retrieve the data map splits
    qfunc2: An optional QE wrapper if the second phi used to creat clphiphi is different to the first one.
    use_mpi: A boolean indicating whether to use MPI for distributing tasks.
    verbose: A boolean indicating whether to print verbose messages.
    skip_rd: A boolean indicating whether to skip certain operations.
    shear: A boolean indicating whether to apply shear in the calculations.
    power_mcn0: An optional parameter related to power calculations.
    out_root: An output path / prefix to load/save intermediate rdn0 calculations.

    Returns:
        A tuple of shape (nsims, 2, Lmax) for the gradient and the curl  modes.

    """
    
    d            = data_sim_id
    i_set, start = sim_set
    qa           = phifunc 
    qf1, qf2     = qfunc1, qfunc2

    mcn0evals = []
    if not skip_rd: 
        rdn0evals = []

    if use_mpi:
        comm,rank,my_tasks = mpi.distribute(nsims)
    else:
        comm,rank,my_tasks = FakeCommunicator(), 0, range(nsims)
        
    for s in my_tasks:
        s = s + start
        
        def rdn0_terms(xy, uv, repeat=False):
            return load_save_power(xy[0], xy[1], uv[0], uv[1],
                                   qa, qf1, get_kmap, power, 
                                   repeat=repeat, i_set=i_set, 
                                   qf2=qf2, get_kmap1=get_kmap1,
                                   get_kmap2=get_kmap2, get_kmap3=get_kmap3,
                                   out_root=out_root, shear=shear,
                                   format_cl=format_cl, format_phi=format_phi,
                                   noiseless_sims=False, verbose=verbose)
            
        def mcn0_terms(xy, uv, repeat=False):
            return load_save_power(xy[0], xy[1], uv[0], uv[1],
                                   qa, qf1, get_kmap,
                                   power if power_mcn0 is None else power_mcn0,
                                   repeat=repeat, i_set=i_set, qf2=qf2,
                                   get_kmap1=get_kmap1,
                                   get_kmap2=get_kmap2, get_kmap3=get_kmap3,
                                   out_root=out_root, shear=shear,
                                   format_cl=format_cl, format_phi=format_phi,
                                   noiseless_sims=noiseless_mcn0, verbose=verbose)
        
        # mcn0 terms
        with bench.show(f"Rank {rank}, sim index {s}, mcn0 terms"):
            with bench.show(f"Rank {rank}, mcn0 term sspssp (s: {s}, s+1: {s+1})"):
                sspssp = mcn0_terms((s,s+1), (s,s+1), repeat=True)
            with bench.show(f"Rank {rank}, mcn0 term sspsps (s: {s}, s+1: {s+1})"):
                sspsps = mcn0_terms((s,s+1), (s+1,s))
        mcn0_only_term = sspssp + sspsps

        mcn0evals.append(mcn0_only_term.copy())

        # using notation from equation (43) of 2011.02475v2
        if not skip_rd:
            with bench.show(f"Rank {rank}, (data {data_sim_id}, sim {s}), rdn0 terms"):
                with bench.show(f"Rank {rank}, rdn0 term dsds (d: {data_sim_id}, s: {s})"):
                    dsds = rdn0_terms((d,s), (d,s), repeat=True)
                with bench.show(f"Rank {rank}, rdn0 term dssd (d: {data_sim_id}, s: {s})"):
                    dssd = rdn0_terms((d,s), (s,d))
                with bench.show(f"Rank {rank}, rdn0 term sdds (d: {data_sim_id}, s: {s})"):
                    sdds = rdn0_terms((s,d), (d,s))
                with bench.show(f"Rank {rank}, rdn0 term sdsd (d: {data_sim_id}, s: {s})"):
                    sdsd = rdn0_terms((s,d), (s,d), repeat=True)

            rdn0_only_term = dsds + dssd + sdds + sdsd

            rdn0evals.append(rdn0_only_term - mcn0_only_term)

    with bench.show(f"allgather for 1 mcrdn0 run"):
        if skip_rd:
            avgrdn0 = None
        else:
            avgrdn0 = utils.allgatherv(rdn0evals,comm) if use_mpi else rdn0evals
        avgmcn0 = utils.allgatherv(mcn0evals,comm) if use_mpi else mcn0evals

    return avgrdn0, avgmcn0