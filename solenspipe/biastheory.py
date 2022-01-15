from __future__ import print_function
from orphics import maps,io,cosmology
from orphics import stats,mpi
from pixell import utils # These are needed for MPI.
from pixell import enmap,lensing as plensing,curvedsky as cs,reproject
import numpy as np
import os,sys
import healpy as hp
from enlib import bench
from mapsims import noise,SOChannel
from falafel import qe
from solenspipe._lensing_biases import lensingbiases as lensingbiases_f
from solenspipe._lensing_biases import checkproc as checkproc_f
import pytempura

def compute_n1_py(clpp,normarray,cls,cltt,clee,clbb,clte,nells,nellsp,lmin,Lmaxout,lmax_TT,lcorr_TT,tmp_output,Lstep,Lmin_out):

    """Calculation of the theoretical N1 bias for the different polcomb combinations
    Parameters
    ----------
    clpp : 1d array of lensing field phi starting at multipole L=2
     
    normarray : Array of Als arrays (Lensing potential N0s)
                np.array([N0TT,N0EE,N0EB,N0TE,N0TB,N0BB])
    cls : Array of CMB Cls arrays used for the weights F
        np.array([l,ClTT,ClEE,ClBB,ClTE])    
    cltt: 1d ClTT array used by the filters (cltt=cls[1])
    clee: cls[2]
    clbb: cls[3]
    clte: cls[4]
    nells: 1d array of the temperature noise
    nellsp: 1d array of the polarization noise
            Size of nells and nellsp (int) determine lmax the maximum multipole used to compute N1
    lmin: int
          minimum multipole used to compute N1
    Lmaxout: int
             Maximum multipole for the output
    lmax_TT: int
             Maximum multipole for temperature
    lcorr_TT: int
            Cut-off in ell for correlated noise ( zero if not wanted)
    tmp_output: str
        Output folder, where files can be saved
    Lstep: int
           Step size specifing the L's in which the N1 will be calculated
           
    Lmin_out: Minimum multipole for the output.
    
    Output:
        return n1tt,n1ee,n1eb,n1te,n1tb as 1D arrays for phi (multiply by (ell*(ell+1))/2)**2 for kappa N1)
    """
    n1tt,n1ee,n1eb,n1te,n1tb=lensingbiases_f.compute_n1(clpp,normarray,cls,cltt,clee,clbb,clte,nells,nellsp,lmin,Lmaxout,lmax_TT,lcorr_TT,tmp_output,Lstep,Lmin_out)
    
    return n1tt,n1ee,n1eb,n1te,n1tb  
    
def compute_n1mix(clpp,normarray,cls,cltt,clee,clbb,clte,nells,nellsp,lmin,Lmaxout,lmax_TT,lcorr_TT,tmp_output,Lstep,Lmin_out):
    """Calculation of the theoretical N1 bias for the different off diagonal polcomb combinations
    Parameters
    ----------
    clpp : 1d array of lensing field phi starting at multipole L=2
     
    normarray : Array of Als arrays (Lensing potential N0s)
                np.array([N0TT,N0EE,N0EB,N0TE,N0TB,N0BB])
    cls : Array of CMB Cls arrays used for the weights F
        np.array([l,ClTT,ClEE,ClBB,ClTE])    
    cltt: 1d ClTT array used by the filters (cltt=cls[1])
    clee: cls[2]
    clbb: cls[3]
    clte: cls[4]
    nells: 1d array of the temperature noise 
    nellsp: 1d array of the polarization noise
            Size of nells and nellsp determine lmax the maximum multipole used to compute N1
    lmin: int
          minimum multipole used to compute N1
    Lmaxout: int
             Maximum multipole for the output
    lmax_TT: int
             Maximum multipole for temperature
    lcorr_TT: int
            Cut-off in ell for correlated noise ( zero if not wanted)
    Lstep: int
           Step size specifing the L's in which the N1 will be calculated
           
    Lmin_out: Minimum multipole for the output.
    Output:
        return n1ttee,n1tteb,n1ttte,n1tttb,n1eeeb,n1eete,n1eetb,n1ebte,n1ebtb,n1tetb for phi as 1D arrays (multiply by (ell*(ell+1))/2)**2 for kappa N1)
    
    """        

    n1ttee,n1tteb,n1ttte,n1tttb,n1eeeb,n1eete,n1eetb,n1ebte,n1ebtb,n1tetb=lensingbiases_f.compute_n1mix(clpp,normarray,cls,cltt,clee,clbb,clte,nells,nellsp,lmin,Lmaxout,lmax_TT,lcorr_TT,tmp_output,Lstep,Lmin_out)
    return n1ttee,n1tteb,n1ttte,n1tttb,n1eeeb,n1eete,n1eetb,n1ebte,n1ebtb,n1tetb  

def compute_n1mv(clpp,normarray,cls,cltt,clee,clbb,clte,nells,nellsp,lmin,Lmaxout,lmax_TT,lcorr_TT,tmp_output,Lstep,Lmin_out):
    
    """Calculation of the theoretical N1 bias for MV combination
    Parameters
    ----------
    clpp : 1d array of lensing field phi starting at multipole L=2
     
    normarray : Array of Als arrays (Lensing potential N0s)
                np.array([N0TT,N0EE,N0EB,N0TE,N0TB,N0BB])
    cls : Array of CMB Cls arrays used for the weights F
        np.array([l,ClTT,ClEE,ClBB,ClTE])    
    cltt: 1d ClTT array used by the filters (cltt=cls[1])
    clee: cls[2]
    clbb: cls[3]
    clte: cls[4]
    nells: 1d array of the temperature noise 
    nellsp: 1d array of the polarization noise
            Size of nells and nellsp determine lmax the maximum multipole used to compute N1
    lmin: int
          minimum multipole used to compute N1
    Lmaxout: int
             Maximum multipole for the output
    lmax_TT: int
             Maximum multipole for temperature
    lcorr_TT: int
            Cut-off in ell for correlated noise ( zero if not wanted)
    Lstep: int
           Step size specifing the L's in which the N1 will be calculated
           
    Lmin_out: Minimum multipole for the output.
    Output:
        return n1mv as 1D arrays
    
    """
    bins=np.arange(len(normarray[0][0]))
    fac= bins*(bins+1.) / 4.
    n0tt=normarray[0][0]*bins**2*fac
    n0ee=normarray[1][0]*bins**2*fac
    n0eb=normarray[2][0]*bins**2*fac
    n0te=normarray[3][0]*bins**2*fac
    n0tb=normarray[4][0]*bins**2*fac
    n1bins=np.arange(Lmin_out,Lmaxout,Lstep)
    n1tt,n1ee,n1eb,n1te,n1tb=lensingbiases_f.compute_n1(clpp,normarray,cls,cltt,clee,clbb,clte,nells,nellsp,lmin,Lmaxout,lmax_TT,lcorr_TT,tmp_output,Lstep,Lmin_out)
    n1ttee,n1tteb,n1ttte,n1tttb,n1eeeb,n1eete,n1eetb,n1ebte,n1ebtb,n1tetb=lensingbiases_f.compute_n1mix(clpp,normarray,cls,cltt,clee,clbb,clte,nells,nellsp,lmin,Lmaxout,lmax_TT,lcorr_TT,tmp_output,Lstep,Lmin_out)
    n1=[n1tt,n1te,n1tb,n1eb,n1ee,n1ttte,n1tttb,n1tteb,n1ttee,n1tetb,n1ebte,n1eete,n1ebtb,n1eetb,n1eeeb]
    n1_inter=[]

    for i in range(len(n1)):
        n1_inter.append(maps.interp(n1bins,n1[i])(bins))
    weights=[1/n0tt**2,1/n0te**2,1/n0tb**2,1/n0eb**2,1/n0ee**2,2/(n0tt*n0te),2/(n0tt*n0tb),2/(n0tt*n0eb),2/(n0tt*n0ee),
    2/(n0te*n0tb),2/(n0te*n0eb),2/(n0te*n0ee),2/(n0tb*n0eb),2/(n0tb*n0ee),2/(n0eb*n0ee)]
    mvn1=np.zeros(len(n1_inter[1]))
    sumc=np.zeros(len(n1_inter[1]))
    for i in range(len(weights)):
        mvn1+=weights[i]*n1_inter[i]
        sumc+=weights[i]
    mvn1=mvn1/sumc  

    mvn1=mvn1[Lmin_out:Lmaxout:Lstep]
    return mvn1 
    


def perturbe_clist(cl_array,bins,amount):
    """generate a list of cls where the cls at the position bins are perturbed by amount keeping other cls unperturbed"""
    cltt_list=[]
    for i in range(len(bins)):
        cl=cl_array.copy()
        cl[int(bins[i])]=amount*cl_array[int(bins[i])]
        cltt_list.append(cl)
    return cltt_list

def diff_cl(cl_array,bins,epsilon=0.001):
    """deltacls used in the denominator of finite difference derivative
        cls contain factor of ell*(ell+1)/2pi which is stripped off
    """
    ls=np.arange(2,len(cl_array)+2)
    cls=cl_array*2*np.pi/(ls*(ls+1))
    dcltt=[]
    for i in range(len(bins)):
        dcltt.append(2*epsilon*cls[int(bins[i])])
    return dcltt
    
def diff_clpy(cl_array,bins,epsilon=0.001):
    """deltacls used in the denominator of finite difference derivative
        cls is dimensionless
    """
    ls=np.arange(2,len(cl_array))
    cls=cl_array
    dcltt=[]
    for i in range(len(bins)):
        dcltt.append(2*epsilon*cls[int(bins[i])])
    return dcltt


def n1mv_dclkk(cl_array,bins,n1bins,clpp,norms,cls,cltt,clee,clbb,clte,nells,nellsp,lmin,Lmax_out,Lmax_TT,Lcorr_TT,tmp_output,Lstep,Lmin_out):
    """
    Compute derivative of N1 wrt clkappakappa
    Parameters
    ----------
    cl_array : 1d array
           Cltt to be perturbed
    bins : 1d array
           Multipoles in which derivatives are going to be calculated.
    n1bins: 1d array
            Multipoles of the N1 bias used.
    
    Returns
    Matrix corresponding to derivative of N1kk wrt convergence field
    with rows of L corresponding to N1 multipoles and columns of l the multipoles of Cl which derivatives are taken.
    """
    bins=bins-2
    array1001=perturbe_clist(cl_array,bins,1.001)
    array999=perturbe_clist(cl_array,bins,0.999)
    N1001=[] 
    N0999=[]
    delta=diff_cl(cl_array,bins)
    for i in range(len(array1001)):
        a=compute_n1mv(array1001[i],norms,cls,cltt,clee,clbb,clte,nells,nellsp,lmin,Lmax_out,Lmax_TT,Lcorr_TT,tmp_output,Lstep,Lmin_out)
        b=compute_n1mv(array999[i],norms,cls,cltt,clee,clbb,clte,nells,nellsp,lmin,Lmax_out,Lmax_TT,Lcorr_TT,tmp_output,Lstep,Lmin_out)
        N1001.append(a)
        N0999.append(b)
    derlist=[]
    diff=[n1bins]
    for i in range(len(N1001)):
        der=((N1001[i][:len(n1bins)]-N0999[i][:len(n1bins)])*(n1bins*(n1bins+1))**2)/(delta[i]*(bins[i]+2)*(bins[i]+3)) #strip off (l'*(l'+1)) because original lensed file has factor of (l*(l+1))**2/2pi
        diff.append(der)   
    der=np.insert(np.transpose(diff),0,np.insert(bins+2,0,0),axis=0)      
    derlist.append(der)
    np.savetxt('../data/n1mvdclkk.txt',der)
    return der


def n1mvderivative_clcmb(polcomb,bins,n1bins,clpp,norms,cls,cltt,clee,clbb,clte,nells,nellsp,lmin,Lmax_out,Lmax_TT,Lcorr_TT,tmp_output,Lstep,Lmin_out,use_mpi=True):
    """Compute the derivative of N1mv wrt CMB Cls.

    Args:
        polcomb : string, one of 'TT','EE','TE','BB'
        bins : 1d array
            Multipoles in which derivatives are going to be calculated.
        n1bins: 1d array
                Multipoles of the N1 bias used.
        clpp ([type]): [description]
        norms ([type]): [description]
        cltt ([type]): [description]
        clee ([type]): [description]
        clbb ([type]): [description]
        clte ([type]): [description]
        nells ([type]): [description]
        nellsp ([type]): [description]
        lmin ([type]): [description]
        Lmax_out ([type]): [description]
        Lmax_TT ([type]): [description]
        Lcorr_TT ([type]): [description]
        tmp_output ([type]): [description]
        Lstep ([type]): [description]
        Lmin_out ([type]): [description]
        use_mpi (bool, optional): [description]. Defaults to True.

    Returns
        List of arrays corresponding to the derivatives of N1_\kappa\kappa  with polcomb combinations [TT,EE,EB,TE,TB] wrt Cls
        with rows of L corresponding to N1 multipoles and columns of l the multipoles of Cl which derivatives are taken.
    """
    bins=bins-2
    pol_dict={'TT':cltt,'TE':clte,'EE':clee,'BB':clbb}
    array1001=perturbe_clist(pol_dict[polcomb],bins,1.001)
    array999=perturbe_clist(pol_dict[polcomb],bins,0.999)
    N1001=[] 
    N0999=[]
    delta=diff_cl(pol_dict[polcomb],bins)
    for i in range(len(array1001)):
        if polcomb=='TT':
            a=compute_n1mv(clpp,norms,cls,array1001[task],clee,clbb,clte,nells,nellsp,lmin,Lmax_out,Lmax_TT,Lcorr_TT,tmp_output,Lstep,Lmin_out)
            b=compute_n1mv(clpp,norms,cls,array999[task],clee,clbb,clte,nells,nellsp,lmin,Lmax_out,Lmax_TT,Lcorr_TT,tmp_output,Lstep,Lmin_out)
        elif polcomb=='TE':
            a=compute_n1mv(clpp,norms,cls,cltt,clee,clbb,array1001[task],nells,nellsp,lmin,Lmax_out,Lmax_TT,Lcorr_TT,tmp_output,Lstep,Lmin_out)
            b=compute_n1mv(clpp,norms,cls,cltt,clee,clbb,array999[task],nells,nellsp,lmin,Lmax_out,Lmax_TT,Lcorr_TT,tmp_output,Lstep,Lmin_out)
        elif polcomb=='EE':
            a=compute_n1mv(clpp,norms,cls,cltt,array1001[task],clbb,clte,nells,nellsp,lmin,Lmax_out,Lmax_TT,Lcorr_TT,tmp_output,Lstep,Lmin_out)
            b=compute_n1mv(clpp,norms,cls,cltt,array999[task],clbb,clte,nells,nellsp,lmin,Lmax_out,Lmax_TT,Lcorr_TT,tmp_output,Lstep,Lmin_out)
        elif polcomb=='BB':
            a=compute_n1mv(clpp,norms,cls,cltt,clee,array1001[task],clte,nells,nellsp,lmin,Lmax_out,Lmax_TT,Lcorr_TT,tmp_output,Lstep,Lmin_out)
            b=compute_n1mv(clpp,norms,cls,cltt,clee,array999[task],clte,nells,nellsp,lmin,Lmax_out,Lmax_TT,Lcorr_TT,tmp_output,Lstep,Lmin_out)
        N1001.append(a)
        N0999.append(b)
    derlist=[]
    diff=[n1bins]
    for i in range(len(N1001)):
        der=((N1001[i][:len(n1bins)]-N0999[i][:len(n1bins)])*(n1bins*(n1bins+1))**2*0.25)/delta[i]
        diff.append(der)   
    der=np.insert(np.transpose(diff),0,np.insert(bins+2,0,0),axis=0)      
    derlist.append(der)
    return der

def n1derivative_clcmb(polcomb,bins,n1bins,clpp,norms,cls,cltt,clee,clbb,clte,nells,nellsp,lmin,Lmax_out,Lmax_TT,Lcorr_TT,tmp_output,Lstep,Lmin_out,use_mpi=True):
    """
    Compute derivative of N1 wrt Cl^{polcomb}
    Parameters
    ----------
    polcomb : string, one of 'TT','EE','TE','BB'
    bins : 1d array
           Multipoles in which derivatives are going to be calculated.
    n1bins: 1d array
            Multipoles of the N1 bias used.
    
    Returns
    List of arrays corresponding to the derivatives of N1_\kappa\kappa  with polcomb combinations [TT,EE,EB,TE,TB] wrt Cls
    with rows of L corresponding to N1 multipoles and columns of l the multipoles of Cl which derivatives are taken.
    """
    bins=bins-2
    pol_dict={'TT':cltt,'TE':clte,'EE':clee,'BB':clbb}
    array1001=perturbe_clist(pol_dict[polcomb],bins,1.001)
    array999=perturbe_clist(pol_dict[polcomb],bins,0.999)
    N1001=[[],[],[],[],[]] #list of lists containing tt,ee,eb,te,tb
    N0999=[[],[],[],[],[]]
    delta=diff_cl(pol_dict[polcomb],bins)
    #this loop need to be mpi
    print(len(array1001))
    if use_mpi:
        comm,rank,my_tasks = mpi.distribute(len(array1001))
        print(my_tasks)
    else:
        comm,rank,my_tasks = FakeCommunicator(), 0, range(len(array1001))

    high=[]
    low=[]

    s = stats.Stats(comm)
    for task in my_tasks:
        print(task)
        if polcomb=='TT':
            a=compute_n1_py(clpp,norms,cls,array1001[task],clee,clbb,clte,nells,nellsp,lmin,Lmax_out,Lmax_TT,Lcorr_TT,tmp_output,Lstep,Lmin_out)
            b=compute_n1_py(clpp,norms,cls,array999[task],clee,clbb,clte,nells,nellsp,lmin,Lmax_out,Lmax_TT,Lcorr_TT,tmp_output,Lstep,Lmin_out)
        elif polcomb=='TE':
            a=compute_n1_py(clpp,norms,cls,cltt,clee,clbb,array1001[task],nells,nellsp,lmin,Lmax_out,Lmax_TT,Lcorr_TT,tmp_output,Lstep,Lmin_out)
            b=compute_n1_py(clpp,norms,cls,cltt,clee,clbb,array999[task],nells,nellsp,lmin,Lmax_out,Lmax_TT,Lcorr_TT,tmp_output,Lstep,Lmin_out)
        elif polcomb=='EE':
            a=compute_n1_py(clpp,norms,cls,cltt,array1001[task],clbb,clte,nells,nellsp,lmin,Lmax_out,Lmax_TT,Lcorr_TT,tmp_output,Lstep,Lmin_out)
            b=compute_n1_py(clpp,norms,cls,cltt,array999[task],clbb,clte,nells,nellsp,lmin,Lmax_out,Lmax_TT,Lcorr_TT,tmp_output,Lstep,Lmin_out)
        elif polcomb=='BB':
            a=compute_n1_py(clpp,norms,cls,cltt,clee,array1001[task],clte,nells,nellsp,lmin,Lmax_out,Lmax_TT,Lcorr_TT,tmp_output,Lstep,Lmin_out)
            b=compute_n1_py(clpp,norms,cls,cltt,clee,array999[task],clte,nells,nellsp,lmin,Lmax_out,Lmax_TT,Lcorr_TT,tmp_output,Lstep,Lmin_out)
        
        high.append(a)
        low.append(b)
    h = utils.allgatherv(high,comm)
    l = utils.allgatherv(low,comm)
    for j in range(len(h)):
        for k in range(len(N1001)):
            N1001[k].append(h[j][k])
            N0999[k].append(l[j][k])
    
    keys=['TT','EE','EB','TE','TB']
    
    derlist=[]

    for k in range(len(keys)):
        diff=[n1bins]
        for i in range(len(N1001[1])):
            der=((N1001[k][i][:len(n1bins)]-N0999[k][i][:len(n1bins)])*(n1bins*(n1bins+1))**2*0.25)/delta[i]
            diff.append(der)
        der=np.insert(np.transpose(diff),0,np.insert(bins+2,0,0),axis=0)
        derlist.append(der)
    return derlist

    
def n1_derclphiphi(
    x,
    y,
    clpp=None,
    normarray=None,
    cls=None,
    nells=None,
    nellsp=None,
    lmin=None,
    Lmaxout=None,
    lmax_TT=None,
    lcorr_TT=None,
    tmp_output=None,
    Lstep=None,
    Lmin_out=None
    ):

    
    """Calculation of the N1 lensing field derivative wrt clpp
    ----------
    #x= First set i.e 'TT'
    #y= Second set i.e 'EB'
    clpp : 1d array of lensing field phi starting at multipole L=2
     
    normarray : Array of Als arrays (Lensing potential N0s)
                np.array([N0TT,N0EE,N0EB,N0TE,N0TB,N0BB])
    cls : Array of CMB Cls arrays used for the weights F
        np.array([l,ClTT,ClEE,ClBB,ClTE])    
    cltt: 1d ClTT array used by the filters (cltt=cls[1])
    clee: cls[2]
    clbb: cls[3]
    clte: cls[4]
    nells: 1d array of the temperature noise
    nellsp: 1d array of the polarization noise
            Size of nells and nellsp (int) determine lmax the maximum multipole used to compute N1
    lmin: int
          minimum multipole used to compute N1
    Lmaxout: int
             Maximum multipole for the output
    lmax_TT: int
             Maximum multipole for temperature
    lcorr_TT: int
            Cut-off in ell for correlated noise ( zero if not wanted)
    Lstep: int
           Step size specifing the L's in which the N1 will be calculated
           
    Lmin_out: Minimum multipole for the output.
    
    Output:
        save and return a numpy matrix with first row being the L values of the derivative and the first column the L values of the n1
    """
    lensingbiases_f.compute_n1_derivatives(
        clpp,
        normarray,
        cls,
        nells,
        nellsp,
        lmin,
        Lmaxout,
        lmax_TT,
        lcorr_TT,
        tmp_output,
        Lstep,
        Lmin_out)
    n1 = np.loadtxt(os.path.join(tmp_output,'N1_%s%s_analytical_matrix.dat'% (x, y))).T  

    return n1

def compute_n0_py(clpp,cls,cltt,clee,clbb,clte,nells,nellsp,lmin,Lmaxout,lmax_TT,lcorr_TT,tmp_output,Lstep,Lmin_out):
    """Fortran Routine to calculate N0's
     Parameters
    ----------
    clpp : 1d array of lensing field phi starting at multipole L=2
     
    normarray : Array of Als arrays (Lensing potential N0s)
                np.array([N0TT,N0EE,N0EB,N0TE,N0TB,N0BB])
    cls : Array of CMB Cls arrays used for the weights F
        np.array([l,ClTT,ClEE,ClBB,ClTE])    
    cltt: 1d ClTT array used by the filters (cltt=cls[1])
    clee: cls[2]
    clbb: cls[3]
    clte: cls[4]
    nells: 1d array of the temperature noise 
    nellsp: 1d array of the polarization noise
            Size of nells and nellsp determine lmax the maximum multipole used to compute N1
    lmin: int
          minimum multipole used to compute N1
    Lmaxout: int
             Maximum multipole for the output
    lmax_TT: int
             Maximum multipole for temperature
    lcorr_TT: int
            Cut-off in ell for correlated noise ( zero if not wanted)
    Lstep: int
           Step size specifing the L's in which the N1 will be calculated
           
    Lmin_out: Minimum multipole for the output.
    Output:
        return n0tt,n0ee,n0eb,n0te,n0tb as 1D array of phi N0s
    
    """
    n0tt,n0ee,n0eb,n0te,n0tb=lensingbiases_f.compute_n0(clpp,cls,cltt,clee,clbb,clte,nells,nellsp,lmin,Lmaxout,lmax_TT,lcorr_TT,tmp_output,Lmin_out,Lstep)

    return n0tt,n0ee,n0eb,n0te,n0tb

	
def compute_n0mix_py(clpp,cls,cltt,clee,clbb,clte,nells,nellsp,lmin,Lmaxout,lmax_TT,lcorr_TT,tmp_output,Lstep,Lmin_out):
        
    """Fortran Routine to calculate N0's off diagonal terms
     Parameters
    ----------
    clpp : 1d array of lensing field phi starting at multipole L=2
     
    normarray : Array of Als arrays (Lensing potential N0s)
                np.array([N0TT,N0EE,N0EB,N0TE,N0TB,N0BB])
    cls : Array of CMB Cls arrays used for the weights F
        np.array([l,ClTT,ClEE,ClBB,ClTE])    
    cltt: 1d ClTT array used by the filters (cltt=cls[1])
    clee: cls[2]
    clbb: cls[3]
    clte: cls[4]
    nells: 1d array of the temperature noise 
    nellsp: 1d array of the polarization noise
            Size of nells and nellsp determine lmax the maximum multipole used to compute N1
    lmin: int
          minimum multipole used to compute N1
    Lmaxout: int
             Maximum multipole for the output
    lmax_TT: int
             Maximum multipole for temperature
    lcorr_TT: int
            Cut-off in ell for correlated noise ( zero if not wanted)
    Lstep: int
           Step size specifing the L's in which the N1 will be calculated
           
    Lmin_out: Minimum multipole for the output.
    Output:
        return n0ttee,n0ttte,n0eete,n0ebtb as 1D array of deflection N0s
    
    """
    n0ttee,n0ttte,n0eete,n0ebtb=lensingbiases_f.compute_n0mix(clpp,cls,cltt,clee,clbb,clte,nells,nellsp,lmin,Lmaxout,lmax_TT,lcorr_TT,tmp_output,Lmin_out,Lstep)
   
    return n0ttee,n0ttte,n0eete,n0ebtb

def compute_n0mv(clpp,cls,cltt,clee,clbb,clte,nells,nellsp,lmin,Lmaxout,lmax_TT,lcorr_TT,tmp_output,Lstep,Lmin_out):
        
    """Calculation of the theoretical N0 bias for the different off diagonal polcomb combinations
    Parameters
    ----------
    clpp : 1d array of lensing field phi starting at multipole L=2
     
    normarray : Array of Als arrays (Lensing potential N0s)
                np.array([N0TT,N0EE,N0EB,N0TE,N0TB,N0BB])
    cls : Array of CMB Cls arrays used for the weights F
        np.array([l,ClTT,ClEE,ClBB,ClTE])    
    cltt: 1d ClTT array used by the filters (cltt=cls[1])
    clee: cls[2]
    clbb: cls[3]
    clte: cls[4]
    nells: 1d array of the temperature noise 
    nellsp: 1d array of the polarization noise
            Size of nells and nellsp determine lmax the maximum multipole used to compute N1
    lmin: int
          minimum multipole used to compute N1
    Lmaxout: int
             Maximum multipole for the output
    lmax_TT: int
             Maximum multipole for temperature
    lcorr_TT: int
            Cut-off in ell for correlated noise ( zero if not wanted)
    Lstep: int
           Step size specifing the L's in which the N1 will be calculated
           
    Lmin_out: Minimum multipole for the output.
    Output:
        return n1mv as 1D arrays
    
    """

    n0tt,n0ee,n0eb,n0te,n0tb=compute_n0_py(clpp,cls,cltt,clee,clbb,clte,nells,nellsp,lmin,Lmaxout,lmax_TT,lcorr_TT,tmp_output,Lmin_out,Lstep)
    n0mv=1/(1/n0tt+1/n0ee+1/n0eb+1/n0te+1/n0tb)
    return n0mv 
    

def n0derivative_cmb(polN0,polcomb,bins,n0bins,clgrad,cltt,clee,clbb,clte,nells,nellsp,lmin,lmax,Lmax_out,use_mpi=True):
    """
    Compute derivative of N0 wrt cltt
    Parameters
    ----------
    cltt : 1d array
           Cltt to be perturbed
    bins : 1d array
           Multipoles in which derivatives are going to be calculated.
    n0bins: 1d array
            Multipoles of the N0 bias used.
    
    Returns
    List of arrays corresponding to the derivatives of the polcomb combinations [TT,EE,EB,TE,TB]
    with rows of L corresponding to N1 multipoles and columns of l the multipoles of Cl which derivatives are taken.
    """
    est_norm_list=[polN0]
    ells = np.arange(lmax+1)
    ucls = {}
    tcls = {}
    bins=bins-2
    pol_dict={'TT':clgrad[0],'TE':clte,'EE':clee,'BB':clbb}
    array1001=perturbe_clist(pol_dict[polcomb],bins,1.001)
    array999=perturbe_clist(pol_dict[polcomb],bins,0.999)

    N1001=[] 
    N0999=[]
    delta=diff_clpy(pol_dict[polcomb],bins)
    ucls['TT'] = clgrad[0][:8000] #otherwise tempura gives error
    ucls['TE'] = clgrad[1][:8000]
    ucls['EE'] = clgrad[2][:8000]
    ucls['BB'] = clgrad[3][:8000]
    
    tcls['TT'] = cltt[:8000]+maps.interp(np.arange(len(nells)),nells)(np.arange(8000))
    tcls['TE'] = clte[:8000]
    tcls['EE'] = clee[:8000]+maps.interp(np.arange(len(nellsp)),nellsp)(np.arange(8000))
    tcls['BB'] = clbb[:8000]+maps.interp(np.arange(len(nellsp)),nellsp)(np.arange(8000))

    if use_mpi:
        comm,rank,my_tasks = mpi.distribute(len(array1001))
        print(my_tasks)
    else:
        comm,rank,my_tasks = FakeCommunicator(), 0, range(len(array1001))

    high=[]
    low=[]

    s = stats.Stats(comm)
    for i in my_tasks:
        ucls[polcomb]=array1001[i]
        a = pytempura.get_norms(est_norm_list,ucls,tcls,lmin,lmax,k_ellmax=Lmax_out)[polN0][0]
        ucls[polcomb]=array999[i]
        b= pytempura.get_norms(est_norm_list,ucls,tcls,lmin,lmax,k_ellmax=Lmax_out)[polN0][0]
        high.append(a)
        low.append(b)

    h = utils.allgatherv(high,comm)
    l = utils.allgatherv(low,comm)

    for j in range(len(h)):
        print(j)
        N1001.append(h[j])
        N0999.append(l[j])

    derlist=[]
    diff=[n0bins]
    
    for i in range(len(N1001)):
        der=((N1001[i][:len(n0bins)]-N0999[i][:len(n0bins)])*(n0bins*(n0bins+1))**2*0.25)/delta[i]
        diff.append(der)   
    der=np.insert(np.transpose(diff),0,np.insert(bins,0,0),axis=0)      
    derlist.append(der)
    return der

def n0mvderivative_cltt(cl_array,bins,n0bins,clpp,norms,cls,cltt,clee,clbb,clte,NOISE_LEVEL,polnoise,lmin,LMAXOUT,LMAX_TT,LCORR_TT,TMP_OUTPUT,Lstep,Lmin_out):
    """
    Compute derivative of N0 wrt cltt
    Parameters
    ----------
    cl_array : 1d array
           cl_array to be perturbed
    bins : 1d array
           Multipoles in which derivatives are going to be calculated.
    n0bins: 1d array
            Multipoles of the N0 bias used.
    
    Returns
    List of arrays corresponding to the derivatives of the polcomb combinations [TT,EE,EB,TE,TB]
    with rows of L corresponding to N1 multipoles and columns of l the multipoles of Cl which derivatives are taken.
    """

    bins=bins-2
    array1001=perturbe_clist(cl_array,bins,1.001)
    array999=perturbe_clist(cl_array,bins,0.999)
    N1001=[] #list of lists containing tt,ee,eb,te,tb
    N0999=[]
    delta=diff_cl(cl_array,bins)
    input_path="/home/r/rbond/jiaqu/scratch/N1test/"
    np.savetxt(input_path+"delta.txt",delta)  

    for i in range(len(array1001)):
        a=compute_n0mv(clpp,cls,array1001[i],clee,clbb,clte,NOISE_LEVEL,polnoise,lmin,LMAXOUT,LMAX_TT,LCORR_TT,TMP_OUTPUT,Lmin_out,Lstep)
        np.savetxt(input_path+"high.txt",a)
        b=compute_n0mv(clpp,cls,array999[i],clee,clbb,clte,NOISE_LEVEL,polnoise,lmin,LMAXOUT,LMAX_TT,LCORR_TT,TMP_OUTPUT,Lmin_out,Lstep)
        np.savetxt(input_path+"low.txt",b)
        N1001.append(a)
        N0999.append(b)
    

    derlist=[]
    diff=[n0bins]
    for i in range(len(N1001)):
            der=((N1001[i][:len(n0bins)]-N0999[i][:len(n0bins)])*(n0bins*(n0bins+1))**2*0.25)/delta[i]
            np.savetxt(input_path+"diff.txt",(N1001[i][:len(n0bins)]-N0999[i][:len(n0bins)]))
            diff.append(der)
    der=np.insert(np.transpose(diff),0,np.insert(bins+2,0,0),axis=0)
    derlist.append(der)
    return der


    

def n0mvderivative_clee(cl_array,bins,n0bins,clpp,norms,cls,cltt,clee,clbb,clte,NOISE_LEVEL,polnoise,lmin,LMAXOUT,LMAX_TT,LCORR_TT,TMP_OUTPUT,Lstep,Lmin_out):
    """
    Compute derivative of N0 wrt clee
    Parameters
    ----------
    cl_array : 1d array
           Clee to be perturbed
    bins : 1d array
           Multipoles in which derivatives are going to be calculated.
    n1bins: 1d array
            Multipoles of the N1 bias used.
    
    Returns
    List of arrays corresponding to the derivatives of N1 convergence with the polcomb combinations [TT,EE,EB,TE,TB]
    with rows of L corresponding to N1 multipoles and columns of l the multipoles of Cl which derivatives are taken.
    """
 
    bins=bins-2
    array1001=perturbe_clist(cl_array,bins,1.001)
    array999=perturbe_clist(cl_array,bins,0.999)
    N1001=[] #list of lists containing tt,ee,eb,te,tb
    N0999=[]
    delta=diff_cl(cl_array,bins)

    for i in range(len(array1001)):
        a=compute_n0mv(clpp,cls,cltt,array1001[i],clbb,clte,NOISE_LEVEL,polnoise,lmin,LMAXOUT,LMAX_TT,LCORR_TT,TMP_OUTPUT,Lmin_out,Lstep)
        b=compute_n0mv(clpp,cls,cltt,array999[i],clbb,clte,NOISE_LEVEL,polnoise,lmin,LMAXOUT,LMAX_TT,LCORR_TT,TMP_OUTPUT,Lmin_out,Lstep)
        N1001.append(a)
        N0999.append(b)

    derlist=[]
    diff=[n0bins]
    for i in range(len(N1001)):
            der=((N1001[i][:len(n0bins)]-N0999[i][:len(n0bins)])*(n0bins*(n0bins+1))**2*0.25)/delta[i]
            diff.append(der)
    der=np.insert(np.transpose(diff),0,np.insert(bins+2,0,0),axis=0)
    derlist.append(der)
    np.savetxt('../data/n0mvdclee.txt',der)
    return der     
    
def n0mvderivative_clbb(cl_array,bins,n0bins,clpp,norms,cls,cltt,clee,clbb,clte,NOISE_LEVEL,polnoise,lmin,LMAXOUT,LMAX_TT,LCORR_TT,TMP_OUTPUT,Lstep,Lmin_out):
    """
    Compute derivative of N0 wrt clbb
    Parameters
    ----------
    cl_array : 1d array
           Clbb to be perturbed
    bins : 1d array
           Multipoles in which derivatives are going to be calculated.
    n0bins: 1d array
            Multipoles of the N0 bias used.
    
    Returns
    List of arrays corresponding to the derivatives of N1 convergence with the polcomb combinations [TT,EE,EB,TE,TB]
    with rows of L corresponding to N1 multipoles and columns of l the multipoles of Cl which derivatives are taken.
    """
    bins=bins-2
    array1001=perturbe_clist(cl_array,bins,1.001)
    array999=perturbe_clist(cl_array,bins,0.999)
    N1001=[] #list of lists containing tt,ee,eb,te,tb
    N0999=[]
    delta=diff_cl(cl_array,bins)

    for i in range(len(array1001)):
        a=compute_n0mv(clpp,cls,cltt,clee,array1001[i],clte,NOISE_LEVEL,polnoise,lmin,LMAXOUT,LMAX_TT,LCORR_TT,TMP_OUTPUT,Lmin_out,Lstep)
        b=compute_n0mv(clpp,cls,cltt,clee,array999[i],clte,NOISE_LEVEL,polnoise,lmin,LMAXOUT,LMAX_TT,LCORR_TT,TMP_OUTPUT,Lmin_out,Lstep)
        N1001.append(a)
        N0999.append(b)
    

    derlist=[]
    diff=[n0bins]
    for i in range(len(N1001)):
            der=((N1001[i][:len(n0bins)]-N0999[i][:len(n0bins)])*(n0bins*(n0bins+1))**2*0.25)/delta[i]
            diff.append(der)
    der=np.insert(np.transpose(diff),0,np.insert(bins+2,0,0),axis=0)
    derlist.append(der)
    np.savetxt('../data/n0mvdclbb.txt',der)
    return der 
    
def n0mvderivative_clte(cl_array,bins,n0bins,clpp,norms,cls,cltt,clee,clbb,clte,NOISE_LEVEL,polnoise,lmin,LMAXOUT,LMAX_TT,LCORR_TT,TMP_OUTPUT,Lstep,Lmin_out):
    """
    Compute derivative of N0 wrt clte
    Parameters
    ----------
    cl_array : 1d array
           Clte to be perturbed
    bins : 1d array
           Multipoles in which derivatives are going to be calculated.
    n0bins: 1d array
            Multipoles of the N1 bias used.
    
    Returns
    List of arrays corresponding to the derivatives of N1 convergence with the polcomb combinations [TT,EE,EB,TE,TB]
    with rows of L corresponding to N1 multipoles and columns of l the multipoles of Cl which derivatives are taken.
    """
    bins=bins-2
    array1001=perturbe_clist(cl_array,bins,1.001)
    array999=perturbe_clist(cl_array,bins,0.999)
    N1001=[] #list of lists containing tt,ee,eb,te,tb
    N0999=[]
    delta=diff_cl(cl_array,bins)

    for i in range(len(array1001)):
        a=compute_n0mv(clpp,cls,cltt,clee,clbb,array1001[i],NOISE_LEVEL,polnoise,lmin,LMAXOUT,LMAX_TT,LCORR_TT,TMP_OUTPUT,Lmin_out,Lstep)
        b=compute_n0mv(clpp,cls,cltt,clee,clbb,array999[i],NOISE_LEVEL,polnoise,lmin,LMAXOUT,LMAX_TT,LCORR_TT,TMP_OUTPUT,Lmin_out,Lstep)
        N1001.append(a)
        N0999.append(b)
    

    derlist=[]
    diff=[n0bins]
    for i in range(len(N1001)):
            der=((N1001[i][:len(n0bins)]-N0999[i][:len(n0bins)])*(n0bins*(n0bins+1))**2*0.25)/delta[i]
            diff.append(der)
    der=np.insert(np.transpose(diff),0,np.insert(bins+2,0,0),axis=0)
    derlist.append(der)
    np.savetxt('../data/n0mvdclte.txt',der)
    return der 
    
def n0derivative_cltt(cl_array,bins,n0bins,clpp,norms,cls,cltt,clee,clbb,clte,NOISE_LEVEL,polnoise,lmin,LMAXOUT,LMAX_TT,LCORR_TT,TMP_OUTPUT,Lstep,Lmin_out):
    """
    Compute derivative of N0 wrt cltt
    Parameters
    ----------
    cltt : 1d array
           Cltt to be perturbed
    bins : 1d array
           Multipoles in which derivatives are going to be calculated.
    n0bins: 1d array
            Multipoles of the N0 bias used.
    
    Returns
    List of arrays corresponding to the derivatives of the polcomb combinations [TT,EE,EB,TE,TB]
    with rows of L corresponding to N1 multipoles and columns of l the multipoles of Cl which derivatives are taken.
    """
    bins=bins-2
    array1001=perturbe_clist(cl_array,bins,1.001)
    array999=perturbe_clist(cl_array,bins,0.999)
    N1001=[[],[],[],[],[]] #list of lists containing tt,ee,eb,te,tb
    N0999=[[],[],[],[],[]]
    delta=diff_cl(cl_array,bins)

    for i in range(len(array1001)):
        print(i)
        a=compute_n0_py(clpp,cls,array1001[i],clee,clbb,clte,NOISE_LEVEL,polnoise,lmin,LMAXOUT,LMAX_TT,LCORR_TT,TMP_OUTPUT,Lmin_out,Lstep)
        b=compute_n0_py(clpp,cls,array999[i],clee,clbb,clte,NOISE_LEVEL,polnoise,lmin,LMAXOUT,LMAX_TT,LCORR_TT,TMP_OUTPUT,Lmin_out,Lstep)
        for j in range(len(N1001)):
            N1001[j].append(a[j])
            N0999[j].append(b[j])
    
    keys=['TT','EE','EB','TE','TB']
    
    derlist=[]
    for k in range(len(keys)):
        diff=[n0bins]
        for i in range(len(N1001[1])):
            der=((N1001[k][i][:len(n0bins)]-N0999[k][i][:len(n0bins)])*(n0bins*(n0bins+1))**2*0.25)/delta[i]
            diff.append(der)
        der=np.insert(np.transpose(diff),0,np.insert(bins+2,0,0),axis=0)
        derlist.append(der)
    return derlist


def n0derivative_clee(cl_array,bins,n0bins,clpp,norms,cls,cltt,clee,clbb,clte,NOISE_LEVEL,polnoise,lmin,LMAXOUT,LMAX_TT,LCORR_TT,TMP_OUTPUT,Lstep,Lmin_out):
    """
    Compute derivative of N0 wrt clee
    Parameters
    ----------
    cl_array : 1d array
           Clee to be perturbed
    bins : 1d array
           Multipoles in which derivatives are going to be calculated.
    n1bins: 1d array
            Multipoles of the N1 bias used.
    
    Returns
    List of arrays corresponding to the derivatives of N1 convergence with the polcomb combinations [TT,EE,EB,TE,TB]
    with rows of L corresponding to N1 multipoles and columns of l the multipoles of Cl which derivatives are taken.
    """
 
    bins=bins-2
    array1001=perturbe_clist(cl_array,bins,1.001)
    array999=perturbe_clist(cl_array,bins,0.999)
    
    N1001=[[],[],[],[],[]] #list of lists containing tt,ee,eb,te,tb
    N0999=[[],[],[],[],[]]
    
    for i in range(len(array1001)):
        print(i)
        a=compute_n0_py(clpp,cls,cltt,array1001[i],clbb,clte,NOISE_LEVEL,polnoise,lmin,LMAXOUT,LMAX_TT,LCORR_TT,TMP_OUTPUT,Lmin_out,Lstep)
        b=compute_n0_py(clpp,cls,cltt,array999[i],clbb,clte,NOISE_LEVEL,polnoise,lmin,LMAXOUT,LMAX_TT,LCORR_TT,TMP_OUTPUT,Lmin_out,Lstep)
        for j in range(len(N1001)):
            N1001[j].append(a[j])
            N0999[j].append(b[j])

    delta=diff_cl(cl_array,bins)

    
    keys=['TT','EE','EB','TE','TB']
    
    derlist=[]
    for k in range(len(keys)):
        diff=[n0bins]
        for i in range(len(N1001[1])):
            der=((N1001[k][i][:len(n0bins)]-N0999[k][i][:len(n0bins)])*(n0bins*(n0bins+1))**2*0.25)/delta[i]
            diff.append(der)
        der=np.insert(np.transpose(diff),0,np.insert(bins+2,0,0),axis=0)
        derlist.append(der)
        np.savetxt('../data/n0{}dclee.txt'.format(keys[k]),der)
    print(derlist)
    return derlist      
    
def n0derivative_clbb(cl_array,bins,n0bins,clpp,norms,cls,cltt,clee,clbb,clte,NOISE_LEVEL,polnoise,lmin,LMAXOUT,LMAX_TT,LCORR_TT,TMP_OUTPUT,Lstep,Lmin_out):
    """
    Compute derivative of N0 wrt clbb
    Parameters
    ----------
    cl_array : 1d array
           Clbb to be perturbed
    bins : 1d array
           Multipoles in which derivatives are going to be calculated.
    n0bins: 1d array
            Multipoles of the N0 bias used.
    
    Returns
    List of arrays corresponding to the derivatives of N1 convergence with the polcomb combinations [TT,EE,EB,TE,TB]
    with rows of L corresponding to N1 multipoles and columns of l the multipoles of Cl which derivatives are taken.
    """
    bins=bins-2
    array1001=perturbe_clist(cl_array,bins,1.001)
    array999=perturbe_clist(cl_array,bins,0.999)
    
    N1001=[[],[],[],[],[]] #list of lists containing tt,ee,eb,te,tb
    N0999=[[],[],[],[],[]]
    
    for i in range(len(array1001)):
        print(i)
        a=compute_n0_py(clpp,cls,cltt,clee,array1001[i],clte,NOISE_LEVEL,polnoise,lmin,LMAXOUT,LMAX_TT,LCORR_TT,TMP_OUTPUT,Lmin_out,Lstep)
        b=compute_n0_py(clpp,cls,cltt,clee,array999[i],clte,NOISE_LEVEL,polnoise,lmin,LMAXOUT,LMAX_TT,LCORR_TT,TMP_OUTPUT,Lmin_out,Lstep)
        for j in range(len(N1001)):
            N1001[j].append(a[j])
            N0999[j].append(b[j])

    delta=diff_cl(cl_array,bins)
    
    
    
    keys=['TT','EE','EB','TE','TB']
    
    derlist=[]
    for k in range(len(keys)):
        diff=[n0bins]
        for i in range(len(N1001[1])):
            der=((N1001[k][i][:len(n0bins)]-N0999[k][i][:len(n0bins)])*(n0bins*(n0bins+1))**2*0.25)/delta[i]
            diff.append(der)
        der=np.insert(np.transpose(diff),0,np.insert(bins+2,0,0),axis=0)
        derlist.append(der)
        np.savetxt('../data/n0{}dclbb.txt'.format(keys[k]),der)
    return derlist
    
def n0derivative_clte(cl_array,bins,n0bins,clpp,norms,cls,cltt,clee,clbb,clte,NOISE_LEVEL,polnoise,lmin,LMAXOUT,LMAX_TT,LCORR_TT,TMP_OUTPUT,Lstep,Lmin_out):
    """
    Compute derivative of N0 wrt clte
    Parameters
    ----------
    cl_array : 1d array
           Clte to be perturbed
    bins : 1d array
           Multipoles in which derivatives are going to be calculated.
    n0bins: 1d array
            Multipoles of the N1 bias used.
    
    Returns
    List of arrays corresponding to the derivatives of N1 convergence with the polcomb combinations [TT,EE,EB,TE,TB]
    with rows of L corresponding to N1 multipoles and columns of l the multipoles of Cl which derivatives are taken.
    """
    bins=bins-2
    array1001=perturbe_clist(cl_array,bins,1.001)
    array999=perturbe_clist(cl_array,bins,0.999)
    
    N1001=[[],[],[],[],[]] #list of lists containing tt,ee,eb,te,tb
    N0999=[[],[],[],[],[]]
    
    for i in range(len(array1001)):
        print(i)

        a=compute_n0_py(clpp,cls,cltt,clee,clbb,array1001[i],NOISE_LEVEL,polnoise,lmin,LMAXOUT,LMAX_TT,LCORR_TT,TMP_OUTPUT,Lmin_out,Lstep)
        b=compute_n0_py(clpp,cls,cltt,clee,clbb,array999[i],NOISE_LEVEL,polnoise,lmin,LMAXOUT,LMAX_TT,LCORR_TT,TMP_OUTPUT,Lmin_out,Lstep)
        for j in range(len(N1001)):
            N1001.append(a[j])
            N0999.append(b[j])

    delta=diff_cl(cl_array,bins)
    
    keys=['TT','EE','EB','TE','TB']

    derlist=[]
    for k in range(len(keys)):
        diff=[n0bins]
        for i in range(len(N1001[1])):
            der=((N1001[k][i][:len(n0bins)]-N0999[k][i][:len(n0bins)])*(n0bins*(n0bins+1))**2*0.25)/delta[i]
            diff.append(der)
        der=np.insert(np.transpose(diff),0,np.insert(bins+2,0,0),axis=0)
        derlist.append(der)
        np.savetxt('../data/n0{}dclte.txt'.format(keys[k]),der)
    return derlist
    

    
def extend_matrix(sizeL,_matrix):
    """Used to prepare the calculated derivative matrix into form used for the likelihood."""
    #unpacked=true
    #sizeL 3000 size of total unbinned clkk used
    #return sizeLxsizeL interpolated matrix
    matrix=_matrix
    n1bins=matrix[0][1:]
    derbins=matrix.transpose()[0][1:]
    bins=np.arange(sizeL)
    a=[]
    for i in range(1,len(matrix)):
        narray=maps.interp(n1bins,matrix[i][1:])(bins)
        a.append(narray)
    y=np.array(a).transpose()
    b=[]
    #the derivatives L' are calculated about the derbins values, this is the binned version, so set the other l' to 0
    for i in range(len(y)):
        ext=np.zeros(sizeL)
        ext[derbins.astype(int)]=y[i]
        b.append(ext)
    b=np.array(b)
    a=b.transpose()    
    return a
