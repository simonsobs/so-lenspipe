from __future__ import print_function
from orphics import maps,io,cosmology
from orphics import stats,mpi
from pixell import utils # These are needed for MPI.
from pixell import enmap,lensing as plensing,curvedsky as cs,reproject
import numpy as np
import os,sys
import time
import healpy as hp
from enlib import bench
from falafel import qe
from solenspipe._lensing_biases import lensingbiases as lensingbiases_f
from solenspipe._lensing_biases import checkproc as checkproc_f
import pytempura

# ---------------------------------------------------------------------------
# Interface to the post-2021 build of fortran/LensingBiases.f90.
#
# The compiled module takes raw spectra (no ell factors) with these index
# conventions (numpy index i, 0-based):
#   c_phi_phi[i]           C_L^{phiphi} at L = i+1
#   *_response[i]          C_l at l = i+2
#   c_cmb_fiducial[1:5,i]  C_l at l = i+2 (row 0, the l column, is ignored)
#   *_total_filter[i]      C_l^{total} at l = i+1
#   lens_norm_phi[:,i]     A_L^{phi} at L = i+2, rows TT,EE,EB,TE,TB,BB
# Every rank-1 array must have length len(c_phi_phi). Outputs are N1^{phi}
# sampled at L = Lmin_out + Lstep*k, L <= Lmax_out.
#
# The n1_pairs/n1_mv/n1mv_dclkk_kernel functions below take spectra indexed
# by multipole (x[l] = value at multipole l) and apply the offsets
# internally; the legacy compute_n1_py/compute_n1mix/compute_n1mv wrappers
# keep the pre-2021 calling convention (CAMB lensedCls D_l tables, dd-scaled
# clpp starting at L=2, separate noise curves) and delegate to them.
# ---------------------------------------------------------------------------

_N1_ESTS = ['TT','EE','EB','TE','TB']
_N1_MIX_KEYS = ['TTEE','TTEB','TTTE','TTTB','EEEB','EETE','EETB','EBTE','EBTB','TETB']


def _place(arr, first_l, lmaxmax):
    """Pack a by-multipole array (arr[l] = value at multipole l) into the
    Fortran convention where numpy element i corresponds to multipole
    i+first_l, zero-padded/truncated to length lmaxmax."""
    arr = np.asarray(arr, dtype=np.float64)
    out = np.zeros(lmaxmax)
    seg = arr[first_l:first_l + lmaxmax]
    out[:seg.size] = seg
    return out


def n1_output_bins(Lmin_out=2, Lmax_out=3000, Lstep=20):
    """Output multipoles of the Fortran N1 routines."""
    nout = (Lmax_out - Lmin_out) // Lstep + 1
    return Lmin_out + Lstep * np.arange(nout)


def phi_sample(lmaxout):
    """Python copy of SetPhiSampling(sampling=.true.) in LensingBiases.f90:
    the non-uniform L' grid on which the N1 phi-integral samples C_L^{phiphi},
    and the corresponding midpoint quadrature weights dPh."""
    nodes = list(range(2, 111, 10))
    for dl in [(30, 580), (100, lmaxout // 2), (300, lmaxout)]:
        step, top = dl
        nodes += list(range(nodes[-1] + step, top + 1, step))
    nodes = np.array(nodes)
    dph = np.zeros(nodes.size)
    dph[0] = (nodes[1] - nodes[0]) / 2.
    dph[1:-1] = (nodes[2:] - nodes[:-2]) / 2.
    dph[-1] = nodes[-1] - nodes[-2]
    return nodes, dph


def _n1_fortran_args(clpp, norms, cl_fid, cl_resp, cl_total, lmin, lmax,
                     Lmin_out, Lmax_out, Lstep):
    """Assemble the positional argument tuple of compute_n1/compute_n1mix.
    All input spectra are indexed by multipole and are raw power spectra:
    clpp is C_L^{phiphi}, CMB spectra are C_l in muK^2, norms are A_L^{phi}
    (pytempura Als[est][0]); norms may omit 'BB' (set to unity)."""
    lmaxmax = max(lmax, Lmax_out) + 2
    cphi = _place(clpp, 1, lmaxmax)
    fid = np.zeros((5, lmaxmax))
    for i, k in enumerate(['TT', 'EE', 'BB', 'TE']):
        fid[i + 1] = _place(cl_fid[k], 2, lmaxmax)
    resp = [_place(cl_resp[k], 2, lmaxmax) for k in ['TT', 'EE', 'BB', 'TE']]
    tot = [_place(cl_total[k], 1, lmaxmax) for k in ['TT', 'EE', 'BB']]
    normarray = np.ones((6, lmaxmax))
    for i, k in enumerate(_N1_ESTS):
        normarray[i] = _place(norms[k], 2, lmaxmax)
    if 'BB' in norms:
        normarray[5] = _place(norms['BB'], 2, lmaxmax)
    return (cphi, normarray, fid, resp[0], resp[1], resp[2], resp[3],
            tot[0], tot[1], tot[2], lmin, Lmax_out, lmax, Lstep, Lmin_out)


def n1_pairs(clpp, norms, cl_fid, cl_resp, cl_total, lmin, lmax,
             Lmin_out=2, Lmax_out=3000, Lstep=20):
    """N1^{phi} for the 5 diagonal and 10 mixed estimator pairs.

    Parameters
    ----------
    clpp : raw C_L^{phiphi} indexed by multipole (clpp[L] = value at L)
    norms : dict of A_L^{phi} gradient normalizations indexed by L, keys
            TT,EE,EB,TE,TB (BB optional, defaults to unity)
    cl_fid : dict TT,EE,BB,TE of raw C_l; the fiducial spectra in the QE
             weight numerators (what pytempura used as ucls)
    cl_resp : dict TT,EE,BB,TE of raw C_l; the lensed response spectra in
              the trispectrum contractions
    cl_total : dict TT,EE,BB of raw total (signal+noise) C_l; the filter
               denominators (what the pipeline used as tcls)
    lmin,lmax : CMB multipole range of the estimator

    Returns a dict with 'L' (output multipoles) and the 15 pair keys, each
    an N1^{phi} array; multiply by (L(L+1)/2)^2 for N1^{kappa}."""
    args = _n1_fortran_args(clpp, norms, cl_fid, cl_resp, cl_total,
                            lmin, lmax, Lmin_out, Lmax_out, Lstep)
    n1tt, n1ee, n1eb, n1te, n1tb = lensingbiases_f.compute_n1(*args)
    mix = lensingbiases_f.compute_n1mix(*args)
    out = {'L': n1_output_bins(Lmin_out, Lmax_out, Lstep),
           'TT': n1tt, 'EE': n1ee, 'EB': n1eb, 'TE': n1te, 'TB': n1tb}
    out.update(dict(zip(_N1_MIX_KEYS, mix)))
    return out


def n1_mv(clpp, norms, cl_fid, cl_resp, cl_total, lmin, lmax,
          Lmin_out=2, Lmax_out=3000, Lstep=20, return_pairs=False):
    """Minimum-variance N1^{phi} on the output bins, combining the 15 pair
    N1s with inverse-noise weights w_i = 1/A_i^{phi} (common kappa factors
    cancel between numerator and normalization). Inputs as in n1_pairs.

    Returns (L, n1mv_phi) or (L, n1mv_phi, pairs)."""
    pairs = n1_pairs(clpp, norms, cl_fid, cl_resp, cl_total, lmin, lmax,
                     Lmin_out, Lmax_out, Lstep)
    L = pairs['L']
    with np.errstate(divide='ignore', invalid='ignore'):
        w = {k: 1. / np.asarray(norms[k], dtype=np.float64)[L] for k in _N1_ESTS}
        s = np.zeros(L.size)
        num = np.zeros(L.size)
        for i, e1 in enumerate(_N1_ESTS):
            s += w[e1]
            num += pairs[e1] * w[e1]**2
            for e2 in _N1_ESTS[i + 1:]:
                num += 2. * pairs[e1 + e2] * w[e1] * w[e2]
        mv = np.nan_to_num(num / s**2)
    if return_pairs:
        return L, mv, pairs
    return L, mv


def n1mv_dclkk_kernel(clpp, norms, cl_fid, cl_resp, cl_total, lmin, lmax,
                      Lmin_out=2, Lmax_out=3000, Lstep=20, eps=0.001,
                      comm=None):
    """Finite-difference derivative of the MV N1 with respect to C_L^{kappa
    kappa}, as a per-unit-multipole response kernel.

    The Fortran phi-integral samples C^{phiphi} only on the phi_sample(L')
    nodes with radial quadrature weight dPh, so clpp is perturbed by +-eps
    at each node and the quadrature weight is divided out of the finite
    difference; the result is the continuous kernel
        K(L,L') = dN1^{kappa}(L) / dC^{kappakappa}(L')
    at the node columns, such that N1^{kappa}(L) ~= sum_{L'} K(L,L')
    C^{kappakappa}(L') once interpolated to all L' (see kernel_to_matrix).

    If comm is given the node loop is MPI-distributed (allgatherv order is
    preserved by the contiguous task split of mpi.distribute).

    Returns (Lout, nodes, K) with K of shape (len(Lout), len(nodes))."""
    nodes, dph = phi_sample(Lmax_out)
    Lout = n1_output_bins(Lmin_out, Lmax_out, Lstep).astype(np.float64)
    clpp = np.asarray(clpp, dtype=np.float64)
    if comm is not None:
        # contiguous split (rank order preserved by allgatherv); tolerates
        # more ranks than nodes (empty chunks)
        my_tasks = np.array_split(np.arange(nodes.size), comm.Get_size())[comm.Get_rank()]
    else:
        my_tasks = range(nodes.size)
    cols = []
    for j in my_tasks:
        node = nodes[j]
        up = clpp.copy(); up[node] = (1. + eps) * clpp[node]
        dn = clpp.copy(); dn[node] = (1. - eps) * clpp[node]
        _, a = n1_mv(up, norms, cl_fid, cl_resp, cl_total, lmin, lmax,
                     Lmin_out, Lmax_out, Lstep)
        _, b = n1_mv(dn, norms, cl_fid, cl_resp, cl_total, lmin, lmax,
                     Lmin_out, Lmax_out, Lstep)
        # dN1^phi/dC^pp at the node, with the quadrature weight divided out
        dphi = (a - b) / (2. * eps * clpp[node] * dph[j])
        # phi -> kappa on both axes
        cols.append(dphi * (Lout * (Lout + 1.))**2 / (node * (node + 1.))**2)
    cols = np.asarray(cols).reshape(len(cols), Lout.size)
    if comm is not None:
        cols = utils.allgatherv(cols, comm)
    return Lout.astype(int), nodes, np.asarray(cols).T


def kernel_to_matrix(Lout, nodes, K, sizeL=3000):
    """Bilinearly interpolate the (Lout, nodes) kernel from n1mv_dclkk_kernel
    to a dense (sizeL, sizeL) matrix M with M[L,L'] = dN1^{kappa}(L)/
    dC^{kappakappa}(L'), so that M @ clkk ~= N1^{kappa} for the fiducial."""
    ls = np.arange(sizeL)
    rows = np.array([maps.interp(nodes, K[i])(ls) for i in range(len(Lout))])
    out = np.array([maps.interp(Lout, rows[:, j])(ls) for j in range(sizeL)]).T
    out[:2, :] = 0.
    out[:, :2] = 0.
    return out


def _dl_to_cl_by_l(dl, first_l=2):
    """Convert a D_l = l(l+1)C_l/2pi array starting at multipole first_l
    (CAMB lensedCls column) into a raw C_l array indexed by multipole."""
    dl = np.asarray(dl, dtype=np.float64)
    ls = np.arange(first_l, first_l + dl.size, dtype=np.float64)
    out = np.zeros(first_l + dl.size)
    out[first_l:] = dl * (2. * np.pi) / (ls * (ls + 1.))
    return out


def _legacy_n1_inputs(clpp, normarray, cls, cltt, clee, clbb, clte, nells, nellsp):
    """Convert the pre-2021 wrapper inputs (dd-scaled clpp starting at L=2,
    CAMB lensedCls D_l tables starting at l=2, raw noise curves with element
    i at multipole i+1, CTobs = CTf + NT) to the by-multipole raw-spectrum
    dicts of n1_pairs/n1_mv. Returns (clpp, norms, cl_fid, cl_resp,
    cl_total, lmax) with lmax = len(nells) as in the old build."""
    clpp = np.asarray(clpp, dtype=np.float64)
    ls = np.arange(2, 2 + clpp.size, dtype=np.float64)
    clpp_raw = np.zeros(2 + clpp.size)
    clpp_raw[2:] = clpp * (2. * np.pi) / (ls * (ls + 1.))**2
    cl_fid = {'TT': _dl_to_cl_by_l(cls[1]), 'EE': _dl_to_cl_by_l(cls[2]),
              'BB': _dl_to_cl_by_l(cls[3]), 'TE': _dl_to_cl_by_l(cls[4])}
    cl_resp = {'TT': _dl_to_cl_by_l(cltt), 'EE': _dl_to_cl_by_l(clee),
               'BB': _dl_to_cl_by_l(clbb), 'TE': _dl_to_cl_by_l(clte)}
    lmax = len(nells)
    noise_t = np.zeros(lmax + 2); noise_t[1:lmax + 1] = np.asarray(nells, dtype=np.float64)
    noise_p = np.zeros(lmax + 2); noise_p[1:lmax + 1] = np.asarray(nellsp, dtype=np.float64)
    cl_total = {}
    for k, noi in [('TT', noise_t), ('EE', noise_p), ('BB', noise_p)]:
        tot = noi.copy()
        n = min(tot.size, cl_fid[k].size)
        tot[:n] += cl_fid[k][:n]
        cl_total[k] = tot
    norms = {}
    for i, k in enumerate(_N1_ESTS + ['BB']):
        row = np.atleast_2d(np.asarray(normarray[i], dtype=np.float64))[0]
        arr = np.zeros(row.size + 2)
        arr[2:] = row
        norms[k] = arr
    return clpp_raw, norms, cl_fid, cl_resp, cl_total, lmax


def compute_n1_py(clpp,normarray,cls,cltt,clee,clbb,clte,nells,nellsp,lmin,Lmaxout,lmax_TT,lcorr_TT,tmp_output,Lstep,Lmin_out):

    """Calculation of the theoretical N1 bias for the different polcomb combinations
    Parameters
    ----------
    clpp : 1d array of lensing field phi starting at multipole L=2,
           in the CAMB lenspotentialCls convention [L(L+1)]^2 C_L^{phiphi}/2pi

    normarray : Array of Als arrays (Lensing potential N0s)
                np.array([N0TT,N0EE,N0EB,N0TE,N0TB,N0BB]), columns from L=2
    cls : Array of CMB Cls arrays used for the weights F
        np.array([l,ClTT,ClEE,ClBB,ClTE]), lensedCls D_l convention from l=2
    cltt: 1d ClTT array used by the response (cltt=cls[1])
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
    lmax_TT: unused (kept for backward compatibility; the current Fortran
             build has a single CMB lmax set by len(nells))
    lcorr_TT: unused (dead parameter of the historical noise model)
    tmp_output: unused (the current Fortran build writes no files)
    Lstep: int
           Step size specifing the L's in which the N1 will be calculated

    Lmin_out: Minimum multipole for the output.

    Output:
        return n1tt,n1ee,n1eb,n1te,n1tb as 1D arrays for phi (multiply by (ell*(ell+1))/2)**2 for kappa N1)
    """
    clpp_raw, norms, cl_fid, cl_resp, cl_total, lmax = _legacy_n1_inputs(
        clpp, normarray, cls, cltt, clee, clbb, clte, nells, nellsp)
    pairs = n1_pairs(clpp_raw, norms, cl_fid, cl_resp, cl_total, lmin, lmax,
                     Lmin_out=Lmin_out, Lmax_out=Lmaxout, Lstep=Lstep)
    return pairs['TT'], pairs['EE'], pairs['EB'], pairs['TE'], pairs['TB']

def compute_n1mix(clpp,normarray,cls,cltt,clee,clbb,clte,nells,nellsp,lmin,Lmaxout,lmax_TT,lcorr_TT,tmp_output,Lstep,Lmin_out):
    """Calculation of the theoretical N1 bias for the different off diagonal polcomb combinations
    Inputs as in compute_n1_py.
    Output:
        return n1ttee,n1tteb,n1ttte,n1tttb,n1eeeb,n1eete,n1eetb,n1ebte,n1ebtb,n1tetb for phi as 1D arrays (multiply by (ell*(ell+1))/2)**2 for kappa N1)

    """
    clpp_raw, norms, cl_fid, cl_resp, cl_total, lmax = _legacy_n1_inputs(
        clpp, normarray, cls, cltt, clee, clbb, clte, nells, nellsp)
    pairs = n1_pairs(clpp_raw, norms, cl_fid, cl_resp, cl_total, lmin, lmax,
                     Lmin_out=Lmin_out, Lmax_out=Lmaxout, Lstep=Lstep)
    return tuple(pairs[k] for k in _N1_MIX_KEYS)

def compute_n1mv(clpp,normarray,cls,cltt,clee,clbb,clte,nells,nellsp,lmin,Lmaxout,lmax_TT,lcorr_TT,tmp_output,Lstep,Lmin_out):

    """Calculation of the theoretical N1 bias for MV combination.
    Inputs as in compute_n1_py.
    Output:
        return n1mv (phi convention) as a 1D array on the output bins
        np.arange(Lmin_out,Lmaxout,Lstep)
    """
    clpp_raw, norms, cl_fid, cl_resp, cl_total, lmax = _legacy_n1_inputs(
        clpp, normarray, cls, cltt, clee, clbb, clte, nells, nellsp)
    _, mvn1 = n1_mv(clpp_raw, norms, cl_fid, cl_resp, cl_total, lmin, lmax,
                    Lmin_out=Lmin_out, Lmax_out=Lmaxout, Lstep=Lstep)
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
    dcl=[]
    for i in range(len(bins)):
        dcl.append(2*epsilon*cl_array[int(bins[i])])
    return dcl


def n1mv_dclkk(cl_array,bins,n1bins,clpp,norms,cls,cltt,clee,clbb,clte,nells,nellsp,lmin,Lmax_out,Lmax_TT,Lcorr_TT,tmp_output,Lstep,Lmin_out):
    """
    Compute derivative of N1 wrt clkappakappa
    Parameters
    ----------
    cl_array : 1d array
           Cls to be perturbed
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
        print('derivative')
        print(der)
        diff.append(der)   
    der=np.insert(np.transpose(diff),0,np.insert(bins+2,0,0),axis=0)      
    derlist.append(der)
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

    if use_mpi:
        comm,rank,my_tasks = mpi.distribute(len(array1001))
        print(my_tasks)
    else:
        comm,rank,my_tasks = FakeCommunicator(), 0, range(len(array1001))
    s = stats.Stats(comm)
    for task in my_tasks:
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

    
def n1_derclphiphi(x,y,clpp,norms,cls,nells,nellsp,lmin,Lmax_out,Lmax_TT,Lcorr_TT,tmp_output,Lstep,Lmin_out):
  

    
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
    lensingbiases_f.compute_n1_derivatives(clpp,norms,cls,nells,nellsp,lmin,Lmax_out,Lmax_TT,Lcorr_TT,tmp_output,Lstep,Lmin_out)
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

    raise NotImplementedError(
        "n1derivative_clcmb was an unfinished stub that returned the MV N0; "
        "use n1mv_derivative_cmb (by-multipole inputs, MPI) or the legacy "
        "n1mvderivative_clcmb.")
    

def n0derivative_cmb(polN0,polcomb,bins,n0bins,ucls,tcls,clgrad,cltt,clee,clbb,clte,lmin,lmax,Lmax_out,use_mpi=True):
    """
    Compute derivative of N0[polN0] wrt Cl^{polcomb}
    Parameters
    ----------
    cltt : 1d array
           Cltt to be perturbed
    bins : 1d array
           Multipoles in which derivatives are going to be calculated.
    n0bins: 1d array
            Multipoles of the N0 bias used.
    
    Returns
    array of shape (lmax,len(bins)): 
    List of arrays corresponding to the derivatives of the polcomb combinations [TT,EE,EB,TE,TB]
    with rows of L corresponding to the N0 multipoles and columns of l the multipoles of Cl which derivatives are taken.
    First row corresponds to the ells where derivatives are taken
    First column is the L bins
    """
   
    est_norm_list=[polN0]
    ells = np.arange(lmax+1)
    ucls['TT'] = clgrad[0][:8000] #otherwise tempura gives error
    ucls['TE'] = clgrad[1][:8000]
    ucls['EE'] = clgrad[2][:8000]
    ucls['BB'] = clgrad[3][:8000]
    tcls['TT'] = np.interp(np.arange(8000),np.arange(len(tcls['TT'])),tcls['TT'])
    tcls['TE'] = np.interp(np.arange(8000),np.arange(len(tcls['TE'])),tcls['TE'])
    tcls['EE'] = np.interp(np.arange(8000),np.arange(len(tcls['EE'])),tcls['EE'])
    tcls['BB'] = np.interp(np.arange(8000),np.arange(len(tcls['BB'])),tcls['BB'])
    # Weight leg pinned to the fiducial response spectra (2026-09-22): the
    # pre-2021 three-argument get_norms call below perturbed both legs of the
    # normalization, which doubles the derivative; the cross-leg convention
    # is the one of the shipped DR6 like_corrs matrix (see norm_derivative_cmb).
    ucls_fid = {k: np.array(v, copy=True) for k, v in ucls.items()}
    bins=bins-2
    pol_dict={'TT':clgrad[0],'TE':clte,'EE':clee,'BB':clbb}
    array1001=perturbe_clist(pol_dict[polcomb],bins,1.001)
    array999=perturbe_clist(pol_dict[polcomb],bins,0.999)

    N1001=[] 
    N0999=[]
    delta=diff_clpy(pol_dict[polcomb],bins)


    comm,rank,my_tasks = mpi.distribute(len(array1001))
    print(my_tasks)

    high=[]
    low=[]

    s = stats.Stats(comm)
    for i in my_tasks:
        ucls[polcomb]=array1001[i]
        a = pytempura.get_norms(est_norm_list,ucls,ucls_fid,tcls,lmin,lmax,k_ellmax=Lmax_out)[polN0][0]
        ucls[polcomb]=array999[i]
        b= pytempura.get_norms(est_norm_list,ucls,ucls_fid,tcls,lmin,lmax,k_ellmax=Lmax_out)[polN0][0]
        high.append(a)
        low.append(b)

    h = utils.allgatherv(high,comm)
    l = utils.allgatherv(low,comm)

    for j in range(len(h)):
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

def extend_matrix(sizeL,_matrix):
    """Used to prepare the calculated derivative matrix into form used for the likelihood. Return (L,L') matrix"""
    #sizeL 3000 size of total unbinned clkk used
    #return sizeLxsizeL interpolated matrix
    matrix=_matrix
    derbins=matrix[0][1:]
    ellbins=matrix.transpose()[0][1:]
    bins=np.arange(sizeL)
    a=[]
    for i in range(1,len(matrix)):
        narray=maps.interp(derbins,matrix[i][1:])(bins)
        a.append(narray)
    y=np.array(a).transpose()
    b=[]
    #interpolate the Ls
    for i in range(len(y)):
        narray=maps.interp(ellbins,y[i])(bins)
        b.append(narray)
    b=np.array(b)
    a=b.transpose()    
    return a


# ---------------------------------------------------------------------------
# Derivatives of the lensing normalization and of the MV N1 with respect to
# the CMB power spectra of the map: the like_corrs inputs of the DR6 lens-only
# likelihood (norm_correction_matrix, N1der_{TT,EE,BB,TE}). Added 2026-09-22.
# ---------------------------------------------------------------------------
NORM_SPEC_ESTS = {'TT': ['TT'], 'EE': ['EE', 'EB'], 'TE': ['TE', 'TB'], 'BB': []}
_NORM_ESTS = ['TT', 'TE', 'EE', 'EB', 'TB']
_SPECS = ('TT', 'EE', 'BB', 'TE')


def spectrum_bands(lmin, lmax, band):
    """Contiguous multipole bands [lo, hi) of width `band` covering lmin..lmax.
    A trailing band shorter than band/2 is merged into the previous one, so
    no band is a sliver whose per-multipole derivative would dominate the
    interpolation. Returns (lo, hi, centres); band=1 is one band per l."""
    lo = np.arange(lmin, lmax + 1, band)
    hi = np.minimum(lo + band, lmax + 1)
    if band > 1 and len(lo) > 1 and (hi[-1] - lo[-1]) < 0.5 * band:
        lo, hi = lo[:-1], hi[:-1].copy()
        hi[-1] = lmax + 1
    centres = 0.5 * (lo + hi - 1)
    return lo, hi, centres


def bands_to_matrix(gbar, lo, hi, centres, S, lmin, lmax, lmax_in, renorm=True):
    """Per-multipole additive derivative matrix from band-averaged fractional
    response densities.

    gbar (nb, n): for band j and output index (L) i, the average over l in
    the band of g_l = (dF/dC_l) S_l, i.e. gbar[j, i] = delta_j(i) / (2 eps
    n_j) for a +-eps S perturbation of the whole band. g is smooth in l (it
    is the response to a fractional change), so it is what gets interpolated
    linearly between band centres (flat to lmin/lmax, zero outside); with
    renorm=True the residual n_j gbar_j - sum_{l in band} g_hat of every band
    is then spread uniformly over the band, so the measured band responses
    are preserved exactly (sum_l M_{il} S_l equals the exact global response)
    without the instability of a multiplicative factor where g_hat crosses
    zero. The additive derivative is M_{il} = g_hat(i, l) / S_l. Bands of
    width 1 need no interpolation. Returns (M (n, lmax_in+1), the largest
    residual relative to the band's mean |g_hat|, a measure of how far the
    linear interpolation was from the measured band responses)."""
    gbar = np.asarray(gbar, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    nb, n = gbar.shape
    lmax_eff = min(lmax, lmax_in)
    if np.all(hi - lo == 1):
        G = np.zeros((n, lmax_in + 1))
        cols = lo[lo <= lmax_eff]
        G[:, cols] = gbar[:len(cols)].T
        fmax = 0.
    else:
        G = interp_bands_to_ell(gbar, centres, lmin, lmax_eff, lmax_in)
        fmax = 0.
        if renorm:
            for j in range(nb):
                a, b = int(lo[j]), int(min(hi[j], lmax_eff + 1))
                if b <= a:
                    continue
                target = gbar[j] * (b - a)
                got = G[:, a:b].sum(axis=1)
                resid = target - got
                scale = np.abs(G[:, a:b]).mean(axis=1) * (b - a)
                with np.errstate(divide='ignore', invalid='ignore'):
                    rel = np.where(scale > 0, np.abs(resid) / scale, 0.)
                fmax = max(fmax, float(np.max(rel)))
                G[:, a:b] += (resid / (b - a))[:, None]
    with np.errstate(divide='ignore', invalid='ignore'):
        M = np.where(S[None, :lmax_in + 1] > 0, G / S[None, :lmax_in + 1], 0.)
    return M, fmax


def perturbation_scale(cl, spec):
    """Additive perturbation amplitude S^X_l: |C^X_l| for TT, EE, BB and
    sqrt(C^TT_l C^EE_l) for TE, which stays finite where C^TE crosses zero."""
    if spec == 'TE':
        return np.sqrt(np.abs(cl['TT'] * cl['EE']))
    return np.abs(cl[spec])


def interp_bands_to_ell(values, centres, lmin, lmax, lmax_in):
    """values (nb, n) at the band centres -> (n, lmax_in+1): linear in l
    between centres, flat from the first/last centre to lmin/lmax, zero
    outside [lmin, lmax]."""
    values = np.asarray(values, dtype=np.float64)
    nb = values.shape[0]
    ls = np.arange(lmax_in + 1)
    W = np.zeros((lmax_in + 1, nb))
    sel = (ls >= lmin) & (ls <= min(lmax, lmax_in))
    l_in = ls[sel].astype(np.float64)
    if nb == 1:
        W[sel, 0] = 1.
    else:
        j = np.clip(np.searchsorted(centres, l_in, side='right') - 1, 0, nb - 2)
        w = np.clip((l_in - centres[j]) / (centres[j + 1] - centres[j]), 0., 1.)
        W[sel, j] = 1. - w
        W[sel, j + 1] += w
    return values.T @ W.T


def _split_tasks(ntasks, comm):
    if comm is None:
        return np.arange(ntasks)
    return np.array_split(np.arange(ntasks), comm.Get_size())[comm.Get_rank()]


def norm_derivative_cmb(ucls, tcls, lmin, lmax, k_ellmax=4000, lmax_in=3000,
                        band=10, eps=0.05, specs=_SPECS, comm=None, verbose=True):
    """Derivative of the plain MV lensing normalization with respect to the
    CMB power spectra of the map, dA_L/dC^X_l, in the convention of the DR6
    like_corrs matrix.

    Convention. The normalization is the cross-leg one, A^{-1} = sum f(C_map)
    f(C_fid) / (2 C_tot C_tot): pytempura.get_norms(est, response_cls=C_map,
    response_cls_weights=C_fid, total_cls=C_tot) with the weight leg pinned
    to the fiducial gradient spectra the filters assume. The shipped DR6
    matrix satisfies sum_X M^X C^X_grad = -A_MV to 1e-4 at every L, which
    fixes this convention (both legs perturbed would give -2 A_MV). A^{-1} is
    exactly linear in C_map, so the central finite difference is exact and
    eps only sets the numerical precision. The MV norm is the harmonic sum
    of the TT, TE, EE, EB, TB gradient norms (pytempura no_corr=True); each
    spectrum enters a fixed subset (TT -> TT; EE -> EE, EB; TE -> TE, TB;
    BB -> none), so only that subset is recomputed per perturbation and
    dA_MV/dC = -A_MV^2 sum_i d(A_i^{-1})/dC. Bias hardening is not included
    (plain MV, as the fiducial A_MV this pairs with).

    Perturbations are additive on bands of `band` multipoles from lmin to
    lmax, C^X -> C^X +- eps S^X (S from perturbation_scale); the per-multipole
    derivative at the band centre is (delta_up - delta_dn) / (2 eps sum_band S),
    interpolated linearly in l between centres (interp_bands_to_ell).

    Parameters
    ----------
    ucls, tcls : dicts TT, EE, BB, TE indexed by multipole (>= k_ellmax+1),
        the pipeline normalization inputs (falafel get_theory_dicts(grad=True)
        ucls in muK^2; tcls overridden by stage_filter/filters.txt)
    lmin, lmax : CMB multipole range of the estimator
    k_ellmax : output L range (0..k_ellmax); lmax_in : output l range
    comm : MPI communicator; bands are split contiguously over ranks and
        gathered, so every rank returns the full result

    Returns dict: 'L', 'A_mv' (fiducial A_L^phi), 'A_est' (per estimator),
    'M_phi' (X -> (k_ellmax+1, lmax_in+1) dA_MV^phi/dC^X_l), 'centres',
    'dinv_bands' (raw band derivatives of A^{-1}), 'tasks'. Multiply A and M
    by (L(L+1)/2)^2 for kappa units (the like_corrs layout)."""
    import pytempura
    ucls = {k: np.asarray(ucls[k], dtype=np.float64)[:k_ellmax + 1] for k in _SPECS}
    tcls = {k: np.asarray(tcls[k], dtype=np.float64)[:k_ellmax + 1] for k in _SPECS}
    rank0 = (comm is None) or (comm.Get_rank() == 0)

    def norms(ests, resp):
        r = pytempura.get_norms(ests, resp, ucls, tcls, lmin, lmax, k_ellmax=k_ellmax)
        return {e: np.asarray(r[e], dtype=np.float64)[0] for e in ests}

    def inv(a):
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(a > 0, 1. / a, 0.)

    A0 = norms(_NORM_ESTS, ucls)
    s = sum(inv(A0[e]) for e in _NORM_ESTS)
    A_mv = inv(s)

    lo, hi, centres = spectrum_bands(lmin, lmax, band)
    tasks = [(X, j) for X in specs if NORM_SPEC_ESTS[X] for j in range(len(centres))]
    my = _split_tasks(len(tasks), comm)
    out = np.zeros((len(my), k_ellmax + 1))
    t0 = time.time()
    for n, it in enumerate(my):
        X, j = tasks[it]
        S = perturbation_scale(ucls, X)
        sl = slice(int(lo[j]), int(hi[j]))
        up = dict(ucls); dn = dict(ucls)
        up[X] = ucls[X].copy(); up[X][sl] += eps * S[sl]
        dn[X] = ucls[X].copy(); dn[X][sl] -= eps * S[sl]
        ests = NORM_SPEC_ESTS[X]
        Au, Ad = norms(ests, up), norms(ests, dn)
        d = np.zeros(k_ellmax + 1)
        for e in ests:
            d += inv(Au[e]) - inv(Ad[e])
        out[n] = d / (2. * eps * (hi[j] - lo[j]))          # band-averaged g = d(1/A)/dC * S
        if verbose and rank0 and (n % 50 == 0 or n == len(my) - 1):
            print(f'  norm derivative: task {n + 1}/{len(my)} on rank 0 ({X}, band {j}), '
                  f'{time.time() - t0:.1f} s elapsed', flush=True)
    if comm is not None:
        out = utils.allgatherv(out, comm)
    M, renorm = {}, {}
    for X in _SPECS:
        rows = [i for i, (Y, j) in enumerate(tasks) if Y == X]
        if rows:
            m, renorm[X] = bands_to_matrix(out[rows], lo, hi, centres, perturbation_scale(ucls, X),
                                           lmin, lmax, lmax_in)
            m *= -(A_mv ** 2)[:, None]
            m[:2, :] = 0.
        else:
            m, renorm[X] = np.zeros((k_ellmax + 1, lmax_in + 1)), 0.
        M[X] = m
    return {'L': np.arange(k_ellmax + 1), 'A_mv': A_mv, 'A_est': A0, 'M_phi': M,
            'bands': (lo, hi, centres), 'gbar': out, 'tasks': tasks, 'renorm_max_dev': renorm}


def n1mv_derivative_cmb(clpp, norms, cl_fid, cl_resp, cl_total, lmin, lmax,
                        Lmin_out=2, Lmax_out=3000, Lstep=40, band=40, eps=0.05,
                        specs=_SPECS, comm=None, verbose=True):
    """Finite-difference derivative of the MV N1^kappa with respect to the
    CMB spectra of the map, dN1^kappa(L)/dC^X_l, on bands of `band`
    multipoles. The map spectra are the response spectra cl_resp of the
    trispectrum contractions (perturbed); the QE weights cl_fid, the filters
    cl_total and the normalizations norms stay fixed, as in the estimator.
    N1 is quadratic in cl_resp, so the central difference is exact in eps.
    Perturbations and the per-multipole reduction are as in
    norm_derivative_cmb. Inputs otherwise as in n1_mv.

    Returns (Lout, centres, D) with D[X] of shape (len(Lout), nbands): the
    derivative at the band centre, in kappa units per unit C_l."""
    Lout = n1_output_bins(Lmin_out, Lmax_out, Lstep).astype(np.float64)
    kfac = (Lout * (Lout + 1.))**2 / 4.
    cl_resp = {k: np.asarray(cl_resp[k], dtype=np.float64) for k in _SPECS}
    lo, hi, centres = spectrum_bands(lmin, lmax, band)
    tasks = [(X, j) for X in specs for j in range(len(centres))]
    my = _split_tasks(len(tasks), comm)
    rank0 = (comm is None) or (comm.Get_rank() == 0)
    out = np.zeros((len(my), Lout.size))
    t0 = time.time()
    for n, it in enumerate(my):
        X, j = tasks[it]
        S = perturbation_scale(cl_resp, X)
        sl = slice(int(lo[j]), int(hi[j]))
        up = dict(cl_resp); dn = dict(cl_resp)
        up[X] = cl_resp[X].copy(); up[X][sl] += eps * S[sl]
        dn[X] = cl_resp[X].copy(); dn[X][sl] -= eps * S[sl]
        _, a = n1_mv(clpp, norms, cl_fid, up, cl_total, lmin, lmax, Lmin_out, Lmax_out, Lstep)
        _, b = n1_mv(clpp, norms, cl_fid, dn, cl_total, lmin, lmax, Lmin_out, Lmax_out, Lstep)
        out[n] = (a - b) / (2. * eps * (hi[j] - lo[j])) * kfac   # band-averaged g in kappa units
        if verbose and rank0:
            print(f'  N1 derivative: task {n + 1}/{len(my)} on rank 0 ({X}, band {j}), '
                  f'{time.time() - t0:.1f} s elapsed', flush=True)
    if comm is not None:
        out = utils.allgatherv(out, comm)
    D = {}
    for X in specs:
        rows = [i for i, (Y, j) in enumerate(tasks) if Y == X]
        D[X] = out[rows].T
    return Lout.astype(int), (lo, hi, centres), D


def n1_derivative_to_matrix(Lout, bands, D_X, S, lmin, lmax, sizeL=3000, lmax_in=2999):
    """Dense (sizeL, lmax_in+1) matrix dN1^kappa(L)/dC^X_l from the band
    densities D_X (len(Lout), nbands) of n1mv_derivative_cmb: bands_to_matrix
    in l (S = perturbation_scale of the response spectra), then linear in L
    over Lout, rows L<2 zero. The DR6 like_corrs N1der files are the
    (3000, 3000) case. Returns (matrix, max renormalization deviation)."""
    lo, hi, centres = bands
    rows, fmax = bands_to_matrix(np.asarray(D_X).T, lo, hi, centres, S, lmin, lmax, lmax_in)
    ls = np.arange(sizeL, dtype=np.float64)
    out = np.empty((sizeL, lmax_in + 1))
    Lf = np.asarray(Lout, dtype=np.float64)
    for j in range(lmax_in + 1):
        out[:, j] = np.interp(ls, Lf, rows[:, j], left=0., right=0.)
    out[:2, :] = 0.
    return out, fmax
