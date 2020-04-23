from __future__ import print_function
from orphics import maps,io,cosmology
from pixell import enmap,lensing as plensing,curvedsky as cs,reproject
import numpy as np
import os,sys
import healpy as hp
from enlib import bench
from mapsims import noise,SOChannel
from falafel import qe
import solenspipe as s
from solenspipe._lensing_biases import lensingbiases as lensingbiases_f
from solenspipe._lensing_biases import checkproc as checkproc_f


def compute_n1_py(
    clpp=None,
    normarray=None,
    cls=None,
    cltt=None,
    clee=None,
    clbb=None,
    cleb=None,
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
    cleb: cls[4]
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
        return n1tt,n1ee,n1eb,n1te,n1tb as 1D arrays
    """
    n1tt,n1ee,n1eb,n1te,n1tb=lensingbiases_f.compute_n1(
        clpp,
        normarray,
        cls,
        cltt,
        clee,
        clbb,
        cleb,
        nells,
        nellsp,
        lmin,
        Lmaxout,
        lmax_TT,
        lcorr_TT,
        tmp_output,
        Lstep,
        Lmin_out)
    
    return n1tt,n1ee,n1eb,n1te,n1tb  
    
def compute_n1mix(
    phifile=None,
    normarray=None,
    lensedcmbfile=None,
    cltt=None,
    clee=None,
    clbb=None,
    cleb=None,
    noise_level=None,
    noisep=None,
    lmin=None,
    lmaxout=None,
    lmax_TT=None,
    lcorr_TT=None,
    tmp_output=None,
    Lstep=None,
    Lmin_out=None):
        
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
    cleb: cls[4]
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
        return n1ttee,n1tteb,n1ttte,n1tttb,n1eeeb,n1eete,n1eetb,n1ebte,n1ebtb,n1tetb as 1D arrays
    
    """        

    n1ttee,n1tteb,n1ttte,n1tttb,n1eeeb,n1eete,n1eetb,n1ebte,n1ebtb,n1tetb=lensingbiases_f.compute_n1mix(
        phifile,
        normarray,
        lensedcmbfile,
        cltt,
        clee,
        clbb,
        cleb,
        noise_level,
        noisep,
        lmin,
        lmaxout,
        lmax_TT,
        lcorr_TT,
        tmp_output,
        Lstep,
        Lmin_out)
    
    return n1ttee,n1tteb,n1ttte,n1tttb,n1eeeb,n1eete,n1eetb,n1ebte,n1ebtb,n1tetb  


def perturbe_clist(cl_array,bins,amount):
    """generate a list of cls where the cls at the position bins are perturbed by amount keeping other cls unperturbed"""
    cltt_list=[]
    for i in range(len(bins)):
        cl=cl_array.copy()
        cl[int(bins[i])]=amount*cl_array[int(bins[i])]
        cltt_list.append(cl)
    return cltt_list

def diff_cl(cl_array,bins):
    """deltacls used in the denominator of finite difference derivatives"""
    ls=np.arange(2,len(cl_array)+2)
    cls=cl_array*2*np.pi/(ls*(ls+1))
    dcltt=[]
    for i in range(len(bins)):
        dcltt.append(2*0.001*cls[int(bins[i])])
    return dcltt
    
def n1derivative_cltt(cl_array,bins,n1bins,clpp,norms,cls,cltt,clee,clbb,clte,NOISE_LEVEL,polnoise,lmin,LMAXOUT,LMAX_TT,LCORR_TT,TMP_OUTPUT,Lstep,Lmin_out):
    """
    Compute derivative of N1 wrt cltt
    Parameters
    ----------
    cltt : 1d array
           Cltt to be perturbed
    bins : 1d array
           Multipoles in which derivatives are going to be calculated.
    n1bins: 1d array
            Multipoles of the N1 bias used.
    
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
        
        a=compute_n1_py(clpp,norms,cls,array1001[i],clee,clbb,clte,NOISE_LEVEL,polnoise,lmin,LMAXOUT,LMAX_TT,LCORR_TT,TMP_OUTPUT,Lstep,Lmin_out)
        b=compute_n1_py(clpp,norms,cls,array999[i],clee,clbb,clte,NOISE_LEVEL,polnoise,lmin,LMAXOUT,LMAX_TT,LCORR_TT,TMP_OUTPUT,Lstep,Lmin_out)
        for j in range(len(N1001)):
            N1001[j].append(a[j])
            N0999[j].append(b[j])
    
    keys=['TT','EE','EB','TE','TB']
    
    derlist=[]
    for k in range(len(keys)):
        diff=[n1bins]
        for i in range(len(N1001[1])):
            der=((N1001[k][i][:len(n1bins)]-N0999[k][i][:len(n1bins)])*(n1bins*(n1bins+1))**2*0.25)/delta[i]
            diff.append(der)
        der=np.insert(np.transpose(diff),0,np.insert(bins,0,0),axis=0)
        derlist.append(der)
        np.savetxt('../data/n1{}dcltt.txt'.format(keys[k]),der)
    return derlist
    
def n1derivative_clee(cl_array,bins,n1bins,clpp,norms,cls,cltt,clee,clbb,clte,NOISE_LEVEL,polnoise,lmin,LMAXOUT,LMAX_TT,LCORR_TT,TMP_OUTPUT,Lstep,Lmin_out):
    """
    Compute derivative of N1 wrt clee
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
        print("i")
        
        a=compute_n1_py(clpp,norms,cls,cltt,array1001[i],clbb,clte,NOISE_LEVEL,polnoise,lmin,LMAXOUT,LMAX_TT,LCORR_TT,TMP_OUTPUT,Lstep,Lmin_out)
        b=compute_n1_py(clpp,norms,cls,cltt,array999[i],clbb,clte,NOISE_LEVEL,polnoise,lmin,LMAXOUT,LMAX_TT,LCORR_TT,TMP_OUTPUT,Lstep,Lmin_out)
        for j in range(len(N1001)):
            N1001[j].append(a[j])
            N0999[j].append(b[j])

    delta=diff_cl(cl_array,bins)

    
    keys=['TT','EE','EB','TE','TB']
    
    derlist=[]
    for k in range(len(keys)):
        diff=[n1bins]
        for i in range(len(N1001[1])):
            der=((N1001[k][i][:len(n1bins)]-N0999[k][i][:len(n1bins)])*(n1bins*(n1bins+1))**2*0.25)/delta[i]
            diff.append(der)
        der=np.insert(np.transpose(diff),0,np.insert(bins,0,0),axis=0)
        derlist.append(der)
        np.savetxt('../data/n1{}dclee.txt'.format(keys[k]),der)
    return derlist      
    
def n1derivative_clbb(cl_array,bins,n1bins,clpp,norms,cls,cltt,clee,clbb,clte,NOISE_LEVEL,polnoise,lmin,LMAXOUT,LMAX_TT,LCORR_TT,TMP_OUTPUT,Lstep,Lmin_out):
    """
    Compute derivative of N1 wrt clbb
    Parameters
    ----------
    cl_array : 1d array
           Clbb to be perturbed
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
        a=compute_n1_py(clpp,norms,cls,cltt,clee,array1001[i],clte,NOISE_LEVEL,polnoise,lmin,LMAXOUT,LMAX_TT,LCORR_TT,TMP_OUTPUT,Lstep,Lmin_out)
        b=compute_n1_py(clpp,norms,cls,cltt,clee,array999[i],clte,NOISE_LEVEL,polnoise,lmin,LMAXOUT,LMAX_TT,LCORR_TT,TMP_OUTPUT,Lstep,Lmin_out)
        for j in range(len(N1001)):
            N1001[j].append(a[j])
            N0999[j].append(b[j])

    delta=diff_cl(cl_array,bins)
    
    
    
    keys=['TT','EE','EB','TE','TB']
    
    derlist=[]
    for k in range(len(keys)):
        diff=[n1bins]
        for i in range(len(N1001[1])):
            der=((N1001[k][i][:len(n1bins)]-N0999[k][i][:len(n1bins)])*(n1bins*(n1bins+1))**2*0.25)/delta[i]
            diff.append(der)
        der=np.insert(np.transpose(diff),0,np.insert(bins,0,0),axis=0)
        derlist.append(der)
        np.savetxt('../data/n1{}dclbb.txt'.format(keys[k]),der)
    return derlist
    
def n1derivative_clte(cl_array,bins,n1bins,clpp,norms,cls,cltt,clee,clbb,clte,NOISE_LEVEL,polnoise,lmin,LMAXOUT,LMAX_TT,LCORR_TT,TMP_OUTPUT,Lstep,Lmin_out):
    """
    Compute derivative of N1 wrt clte
    Parameters
    ----------
    cl_array : 1d array
           Clte to be perturbed
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
        a=compute_n1_py(clpp,norms,cls,cltt,clee,clbb,array1001[i],NOISE_LEVEL,polnoise,lmin,LMAXOUT,LMAX_TT,LCORR_TT,TMP_OUTPUT,Lstep,Lmin_out)
        b=compute_n1_py(clpp,norms,cls,cltt,clee,clbb,array999[i],NOISE_LEVEL,polnoise,lmin,LMAXOUT,LMAX_TT,LCORR_TT,TMP_OUTPUT,Lstep,Lmin_out)
        for j in range(len(N1001)):
            N1001[j].append(a[j])
            N0999[j].append(b[j])

    delta=diff_cl(cl_array,bins)
    
    keys=['TT','EE','EB','TE','TB']

    derlist=[]
    for k in range(len(keys)):
        diff=[n1bins]
        for i in range(len(N1001[1])):
            der=((N1001[k][i][:len(n1bins)]-N0999[k][i][:len(n1bins)])*(n1bins*(n1bins+1))**2*0.25)/delta[i]
            diff.append(der)
        der=np.insert(np.transpose(diff),0,np.insert(bins,0,0),axis=0)
        derlist.append(der)
        np.savetxt('../data/n1{}dclte.txt'.format(keys[k]),der)
    return derlist

def n1_derclphiphi(
    x,
    y,
    clpp=None,
    normarray=None,
    cls=None,
    noise_level=None,
    noisep=None,
    lmin=None,
    lmaxout=None,
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
    cleb: cls[4]
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
        lmaxout,
        lmax_TT,
        lcorr_TT,
        tmp_output,
        Lstep,
        Lmin_out)
    n1 = np.loadtxt(os.path.join(tmp_output,'N1_%s%s_analytical_matrix.dat'% (x, y))).T  

    return n1

#N0 Functions

def compute_n0_py(
    phifile=None,
    lensedcmbfile=None,
    cltt=None,
    clee=None,
    clbb=None,
    cleb=None,
    noise_level=None,
    noisep=None,
    lmin=None,
    lmaxout=None,
    lmax_TT=None,
    lcorr_TT=None,
    tmp_output=None,
    Lmin_out=None,
    Lstep=None):
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
    cleb: cls[4]
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
        return n0tt,n0ee,n0eb,n0te,n0tb as 1D array of deflection N0s
    
    """
    n0tt,n0ee,n0eb,n0te,n0tb=lensingbiases_f.compute_n0(
        phifile,
        lensedcmbfile,
        cltt,
        clee,
        clbb,
        cleb,
        noise_level,
        noisep,
        lmin,
        lmaxout,
        lmax_TT,
        lcorr_TT,
        tmp_output,Lmin_out,Lstep)
    return n0tt,n0ee,n0eb,n0te,n0tb

	
def compute_n0mix_py(
    phifile=None,
    lensedcmbfile=None,
    cltt=None,
    clee=None,
    clbb=None,
    cleb=None,
    noise_level=None,
    noisep=None,
    lmin=None,
    lmaxout=None,
    lmax_TT=None,
    lcorr_TT=None,
    tmp_output=None,
    Lmin_out=None,
    Lstep=None):
        
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
    cleb: cls[4]
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
    n0ttee,n0ttte,n0eete,n0ebtb=lensingbiases_f.compute_n0mix(
        phifile,
        lensedcmbfile,
        cltt,
        clee,
        clbb,
        cleb,
        noise_level,
        noisep,
        lmin,
        lmaxout,
        lmax_TT,
        lcorr_TT,
        tmp_output,Lmin_out,Lstep)
        
    return n0ttee,n0ttte,n0eete,n0ebtb


def n0derivative_cltt(cl_array,bins,n1bins,clpp,norms,cls,cltt,clee,clbb,clte,NOISE_LEVEL,polnoise,lmin,LMAXOUT,LMAX_TT,LCORR_TT,TMP_OUTPUT,Lstep,Lmin_out):
    """
    Compute derivative of N0 wrt cltt
    Parameters
    ----------
    cltt : 1d array
           Cltt to be perturbed
    bins : 1d array
           Multipoles in which derivatives are going to be calculated.
    n1bins: 1d array
            Multipoles of the N1 bias used.
    
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
        
        a=compute_n0_py(clpp,cls,array1001[i],clee,clbb,clte,NOISE_LEVEL,polnoise,lmin,LMAXOUT,LMAX_TT,LCORR_TT,TMP_OUTPUT,Lmin_out,Lstep)
        b=compute_n0_py(clpp,cls,array999[i],clee,clbb,clte,NOISE_LEVEL,polnoise,lmin,LMAXOUT,LMAX_TT,LCORR_TT,TMP_OUTPUT,Lmin_out,Lstep)
        for j in range(len(N1001)):
            N1001[j].append(a[j])
            N0999[j].append(b[j])
    
    keys=['TT','EE','EB','TE','TB']
    
    derlist=[]
    for k in range(len(keys)):
        diff=[n1bins]
        for i in range(len(N1001[1])):
            der=((N1001[k][i][:len(n1bins)]-N0999[k][i][:len(n1bins)])*(n1bins*(n1bins+1))**2*0.25)/delta[i]
            diff.append(der)
        der=np.insert(np.transpose(diff),0,np.insert(bins,0,0),axis=0)
        derlist.append(der)
        np.savetxt('../data/n1{}dcltt.txt'.format(keys[k]),der)
    return derlist
    
def n0derivative_clee(cl_array,bins,n1bins,clpp,norms,cls,cltt,clee,clbb,clte,NOISE_LEVEL,polnoise,lmin,LMAXOUT,LMAX_TT,LCORR_TT,TMP_OUTPUT,Lstep,Lmin_out):
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
        print("i")
        
        a=compute_n0_py(clpp,cls,cltt,array1001[i],clbb,clte,NOISE_LEVEL,polnoise,lmin,LMAXOUT,LMAX_TT,LCORR_TT,TMP_OUTPUT,Lmin_out,Lstep)
        b=compute_n0_py(clpp,cls,cltt,array999[i],clbb,clte,NOISE_LEVEL,polnoise,lmin,LMAXOUT,LMAX_TT,LCORR_TT,TMP_OUTPUT,Lmin_out,Lstep)
        for j in range(len(N1001)):
            N1001[j].append(a[j])
            N0999[j].append(b[j])

    delta=diff_cl(cl_array,bins)

    
    keys=['TT','EE','EB','TE','TB']
    
    derlist=[]
    for k in range(len(keys)):
        diff=[n1bins]
        for i in range(len(N1001[1])):
            der=((N1001[k][i][:len(n1bins)]-N0999[k][i][:len(n1bins)])*(n1bins*(n1bins+1))**2*0.25)/delta[i]
            diff.append(der)
        der=np.insert(np.transpose(diff),0,np.insert(bins,0,0),axis=0)
        derlist.append(der)
        np.savetxt('../data/n1{}dclee.txt'.format(keys[k]),der)
    print(derlist)
    return derlist      
    
def n0derivative_clbb(cl_array,bins,n1bins,clpp,norms,cls,cltt,clee,clbb,clte,NOISE_LEVEL,polnoise,lmin,LMAXOUT,LMAX_TT,LCORR_TT,TMP_OUTPUT,Lstep,Lmin_out):
    """
    Compute derivative of N0 wrt clbb
    Parameters
    ----------
    cl_array : 1d array
           Clbb to be perturbed
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
        a=compute_n1_py(clpp,cls,cltt,clee,array1001[i],clte,NOISE_LEVEL,polnoise,lmin,LMAXOUT,LMAX_TT,LCORR_TT,TMP_OUTPUT,Lmin_out,Lstep)
        b=compute_n1_py(clpp,cls,cltt,clee,array999[i],clte,NOISE_LEVEL,polnoise,lmin,LMAXOUT,LMAX_TT,LCORR_TT,TMP_OUTPUT,Lmin_out,Lstep)
        for j in range(len(N1001)):
            N1001[j].append(a[j])
            N0999[j].append(b[j])

    delta=diff_cl(cl_array,bins)
    
    
    
    keys=['TT','EE','EB','TE','TB']
    
    derlist=[]
    for k in range(len(keys)):
        diff=[n1bins]
        for i in range(len(N1001[1])):
            der=((N1001[k][i][:len(n1bins)]-N0999[k][i][:len(n1bins)])*(n1bins*(n1bins+1))**2*0.25)/delta[i]
            diff.append(der)
        der=np.insert(np.transpose(diff),0,np.insert(bins,0,0),axis=0)
        derlist.append(der)
        np.savetxt('../data/n1{}dclbb.txt'.format(keys[k]),der)
    return derlist
    
def n0derivative_clte(cl_array,bins,n1bins,clpp,norms,cls,cltt,clee,clbb,clte,NOISE_LEVEL,polnoise,lmin,LMAXOUT,LMAX_TT,LCORR_TT,TMP_OUTPUT,Lstep,Lmin_out):
    """
    Compute derivative of N0 wrt clte
    Parameters
    ----------
    cl_array : 1d array
           Clte to be perturbed
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

        a=compute_n0_py(clpp,cls,cltt,clee,clbb,array1001[i],NOISE_LEVEL,polnoise,lmin,LMAXOUT,LMAX_TT,LCORR_TT,TMP_OUTPUT,Lmin_out,Lstep)
        b=compute_n0_py(clpp,cls,cltt,clee,clbb,array999[i],NOISE_LEVEL,polnoise,lmin,LMAXOUT,LMAX_TT,LCORR_TT,TMP_OUTPUT,Lmin_out,Lstep)
        for j in range(len(N1001)):
            N1001[j].append(a[j])
            N0999[j].append(b[j])

    delta=diff_cl(cl_array,bins)
    
    keys=['TT','EE','EB','TE','TB']

    derlist=[]
    for k in range(len(keys)):
        diff=[n1bins]
        for i in range(len(N1001[1])):
            der=((N1001[k][i][:len(n1bins)]-N0999[k][i][:len(n1bins)])*(n1bins*(n1bins+1))**2*0.25)/delta[i]
            diff.append(der)
        der=np.insert(np.transpose(diff),0,np.insert(bins,0,0),axis=0)
        derlist.append(der)
        np.savetxt('../data/n1{}dclte.txt'.format(keys[k]),der)
    return derlist
