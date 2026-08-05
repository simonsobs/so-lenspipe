from orphics import stats
import numpy as np
from scipy.stats import chi2


def get_chi_Squared_cov(data,cov,theory,size_cov,exp=False):
    obs=data-theory
    invcov=np.linalg.inv(cov)
    corr=(size_cov-len(obs)-2)/(size_cov-1)
    invcov*=corr
    if exp==True:
        invcov=PME(cov,size_cov,order=2)
    chi_sq=obs.transpose().dot(invcov.dot(obs))
    return chi_sq

def get_scatter(n_array,error_bin,mean=False,multiplicative=None):
    if multiplicative is not None:
        mb=error_bin@multiplicative
    s_n1=[]
    for i in range(len(n_array)):
        n_array[i][:2]=0
        if len(n_array[i][:3000])!=3000:
            n_array_extend=np.zeros(3000)
            n_array_extend[:len(n_array[i])]=n_array[i]
            b=error_bin@n_array_extend

        else:
            b=error_bin@n_array[i][:3000]

        if multiplicative is not None:
            s_n1.append(b*mb)
        else:
            s_n1.append(b)    
    s_n1=np.array(s_n1)
    if mean==True:
        std_n1=np.std(s_n1,axis=0,ddof=1)/np.sqrt(len(n_array))
        cov_n1=np.cov(np.transpose(s_n1))/len(n_array)
    else:
        std_n1=np.std(s_n1,axis=0,ddof=1)
        cov_n1=np.cov(np.transpose(s_n1))
    return std_n1,cov_n1

def scatter_error_n1(auto_scatter,curl_scatter,n1,rd,data_g,data_c,exp=False,diag=False,plot_matrix=None,binning_matrix=None,multiplicative=None):
    """return scatter error and pte of gradient and curl"""
    
    std_g,_=get_scatter(auto_scatter,plot_matrix)
    std_c,_=get_scatter(curl_scatter,plot_matrix)
    std_gn1,_=get_scatter(n1[:,0],plot_matrix,mean=True)
    std_cn1,_=get_scatter(n1[:,1],plot_matrix,mean=True)
    std_grd,_=get_scatter(rd[:,0],plot_matrix,mean=True)
    std_crd,_=get_scatter(rd[:,1],plot_matrix,mean=True)
 
    
    std_gchi,cov_g=get_scatter(auto_scatter,binning_matrix,multiplicative=multiplicative)
    std_cchi,cov_c=get_scatter(curl_scatter,binning_matrix,multiplicative=multiplicative)
    stdg_n1chi,covg_n1=get_scatter(n1[:,0],binning_matrix,mean=True,multiplicative=multiplicative)
    stdc_n1chi,covc_n1=get_scatter(n1[:,1],binning_matrix,mean=True,multiplicative=multiplicative)
    std_grdchi,covg_rd=get_scatter(rd[:,0],binning_matrix,mean=True,multiplicative=multiplicative)
    std_crdchi,covc_rd=get_scatter(rd[:,1],binning_matrix,mean=True,multiplicative=multiplicative)


    theory=np.zeros(len(data_g))
    dof=len(data_g)


    if diag:
        chisquare=get_chi_Squared(data_g,std_gchi,theory)
        chisquarec=get_chi_Squared(data_c,np.sqrt(std_cchi**2),theory)
    else:
        chisquare=get_chi_Squared_cov(data_g,cov_g+covg_rd+covg_n1,theory,len(auto_scatter),exp=exp)
        chisquarec=get_chi_Squared_cov(data_c,cov_c+covc_rd+covc_n1,theory,len(auto_scatter),exp=exp)

    pte = round(1.-chi2.cdf(chisquare, dof),4)
    ptec = round(1.-chi2.cdf(chisquarec, dof),4)
    
    cov=cov_g+covg_n1+covg_rd
    

    return (np.sqrt(std_g**2+std_gn1**2+std_grd**2),np.sqrt(std_c**2+std_cn1**2)),(pte,ptec),(chisquare,chisquarec),cov

def get_mc_auto(grad,curl,aclg_m,aclc_m,cross,clkk,binning_matrix=None,multiplicative=None):


    cross[0]=0
    aclg_m[0]=0
    aclc_m[0]=0
    grad[0]=0
    curl[0]=0
    xclb=binning_matrix@cross
    clkkb=binning_matrix@clkk
    facb=(clkkb/xclb)**2
    mean=binning_matrix@aclg_m
    aclc_mb=binning_matrix@aclc_m
    mcbias=facb*mean-clkkb
    mccurl=facb*aclc_mb
    autobl=facb*(binning_matrix@grad)-mcbias
    autocbl=facb*(binning_matrix@curl)-mccurl
    if multiplicative is not None:
        mb=binning_matrix@multiplicative
        autobl*=mb
        autocbl*=mb
    
    return autobl,autocbl,facb*mean-mcbias


def get_auto_bandpowers(fname,nobh,pol,diag=False,exp=False,Lmax=3000,sd=800,rd=800,null=False,plot_matrix=None,binning_matrix=None,multiplicative=None,fname_save=None):
    

    acl=np.load(f"{fname}/stage_auto/auto_{nobh}_{pol}.npy")[0][:Lmax]
    aclc=np.load(f"{fname}/stage_auto/autocurl_{nobh}_{pol}.npy")[0][:Lmax]

    #load the rdn0
    rdn0g=np.load(f"{fname}/stage_rdn0/rdn0_{nobh}_{pol}revtotalt.npy").mean(axis=0)[:][0][:Lmax]
    rdn0c=np.load(f"{fname}/stage_rdn0/rdn0_{nobh}_{pol}revtotalt.npy").mean(axis=0)[:][1][:Lmax]
    mcn0g=np.load(f"{fname}/stage_rdn0/mcn0_{nobh}_{pol}revtotalt.npy").mean(axis=0)[0][:Lmax]
    mcn0c=np.load(f"{fname}/stage_rdn0/mcn0_{nobh}_{pol}revtotalt.npy").mean(axis=0)[1][:Lmax]

    #load the realizations for the covariance
    acl_scatter=np.load(f"{fname}/stage_scatter/acl_grad_scatter.npy")[:,:Lmax]
    acl_scatter1=np.load(f"{fname}/stage_scatter/acl_grad_scatter1.npy")[:,:Lmax]
    acl_scatter2=np.load(f"{fname}/stage_scatter/acl_grad_scatter2.npy")[:,:Lmax]
    acl_scatter3=np.load(f"{fname}/stage_scatter/acl_grad_scatter3.npy")[:,:Lmax]

    aclc_scatter=np.load(f"{fname}/stage_scatter/acl_curl_scatter.npy")[:,:Lmax]
    aclc_scatter1=np.load(f"{fname}/stage_scatter/acl_curl_scatter1.npy")[:,:Lmax]
    aclc_scatter2=np.load(f"{fname}/stage_scatter/acl_curl_scatter2.npy")[:,:Lmax]
    aclc_scatter3=np.load(f"{fname}/stage_scatter/acl_curl_scatter3.npy")[:,:Lmax]


    #load the semianalytic realization dependent N0
    dumbg=np.loadtxt(f"{fname}/stage_scatter/dumbg.txt")
    dumbg1=np.loadtxt(f"{fname}/stage_scatter/dumbg1.txt")
    dumbg2=np.loadtxt(f"{fname}/stage_scatter/dumbg2.txt")
    dumbg3=np.loadtxt(f"{fname}/stage_scatter/dumbg3.txt")

    dumbc=np.loadtxt(f"{fname}/stage_scatter/dumbc.txt")
    dumbc1=np.loadtxt(f"{fname}/stage_scatter/dumbc1.txt")
    dumbc2=np.loadtxt(f"{fname}/stage_scatter/dumbc2.txt")
    dumbc3=np.loadtxt(f"{fname}/stage_scatter/dumbc3.txt")

    acl_scatter=np.concatenate((acl_scatter,acl_scatter1,acl_scatter2,acl_scatter3), axis=0)
    aclc_scatter=np.concatenate((aclc_scatter,aclc_scatter1,aclc_scatter2,aclc_scatter3), axis=0)
    dumbg=np.concatenate((dumbg,dumbg1,dumbg2,dumbg3), axis=0)
    dumbc=np.concatenate((dumbc,dumbc1,dumbc2,dumbc3), axis=0)

    #load the noiseless reconstructions for additive mc bias estimation
    acl_scattert=np.load(f"{fname}/stage_scatter/acl_grad_scattert.npy")[:,:Lmax]
    aclc_scattert=np.load(f"{fname}/stage_scatter/acl_curl_scattert.npy")[:,:Lmax]
    acl_scatter1t=np.load(f"{fname}/stage_scatter/acl_grad_scatter1t.npy")[:,:Lmax]
    aclc_scatter1t=np.load(f"{fname}/stage_scatter/acl_curl_scatter1t.npy")[:,:Lmax]  
    acl_scattert=np.concatenate((acl_scattert,acl_scatter1t), axis=0)
    aclc_scattert=np.concatenate((aclc_scattert,aclc_scatter1t), axis=0)


    #load the mcn1 bias
    mcn1g=np.load(f"{fname}/stage_mcn1/mcn1_{nobh}_{pol}n.npy")[:].mean(axis=0)[0][:Lmax]
    mcn1c=np.load(f"{fname}/stage_mcn1/mcn1_{nobh}_{pol}n.npy")[:].mean(axis=0)[1][:Lmax]

    #load the mean field bandpowers (for plotting only)
    mfg=np.loadtxt(f"{fname}/stage_auto/mfg_cl{pol}.txt")[:Lmax]
    mfc=np.loadtxt(f"{fname}/stage_auto/mfc_cl{pol}.txt")[:Lmax]



    aclg_m=np.mean(acl_scattert,axis=0)[:Lmax]-mcn1g-mcn0g
    aclc_m=np.mean(aclc_scattert,axis=0)[:Lmax]-mcn1c-mcn0c

    clkk=np.loadtxt("/home/r/rbond/jiaqu/so-lenspipe/data/clkk.txt")[:Lmax]
    xcl=np.load(f"{fname}/stage_scatter/xcl_scatter.npy")
    icl=np.load(f"{fname}/stage_scatter/icl_scatter.npy")

    xclm=np.mean(xcl,axis=0)[:Lmax]
    iclm=np.mean(icl,axis=0)[:Lmax]

    #GET THE MULTIPLICATIVE MC BIAS
    fac=(iclm/xclm)**2
    xclb=plot_matrix@xclm
    clkkb=plot_matrix@clkk
    facb=(clkkb/xclb)**2


    #estimate the additive MC bias
    aclg_m[0]=0
    mean=(plot_matrix@aclg_m)
    mcbias=facb*mean-clkkb

    l_upper=len(dumbg[0])
    acl_scatter=fac[:l_upper]*(acl_scatter[:,:l_upper]-dumbg[:,:Lmax]-mcn1g[:l_upper])
    aclc_scatter=fac[:l_upper]*(aclc_scatter[:,:l_upper]-dumbc[:,:Lmax]-mcn1c[:l_upper])


    auto=(acl-rdn0g-mcn1g)
    autocl=(aclc-rdn0c-mcn1c)
    autobl,autocbl,mc=get_mc_auto(auto,autocl,aclg_m,aclc_m,xclm,iclm,binning_matrix=plot_matrix,multiplicative=multiplicative)
    autoblt,autocblt,mct=get_mc_auto(auto,autocl,aclg_m,aclc_m,xclm,iclm,binning_matrix=binning_matrix,multiplicative=multiplicative)

    rd=fac**2*np.load(f"{fname}/stage_rdn0/rdn0_{nobh}_{pol}revtotalt.npy")[:,:,:Lmax]
    n1=fac**2*np.load(f"{fname}/stage_mcn1/mcn1_{nobh}_{pol}n.npy")[:,:,:Lmax]

    autoblchi,autocblchi,_=get_mc_auto(auto,autocl,aclg_m,aclc_m,xclm,iclm,binning_matrix=binning_matrix,multiplicative=multiplicative)
    error,pte,chi,cov=scatter_error_n1(acl_scatter,aclc_scatter,n1,rd,autoblchi,autocblchi,exp=exp,diag=diag,plot_matrix=plot_matrix,binning_matrix=binning_matrix)
    error_chi,_,_,_=scatter_error_n1(acl_scatter,aclc_scatter,n1,rd,autoblchi,autocblchi,exp=exp,diag=diag,plot_matrix=binning_matrix,binning_matrix=binning_matrix)
    cents=plot_matrix@np.arange(Lmax)
   
    return cents,autobl,autocbl,error,pte,chi,mc,facb,cov,(autoblchi,autocblchi,error_chi),mct,binning_matrix@np.nan_to_num(fac**2*mcn1g)


#get alens

from scipy.stats import binned_statistic as binnedstat,chi2

def fit_linear_model(x,y,ycov,funcs,dofs=None,deproject=False,Cinv=None,Cy=None):
    """
    Given measurements with known uncertainties, this function fits those to a linear model:
    y = a0*funcs[0](x) + a1*funcs[1](x) + ...
    and returns the best fit coefficients a0,a1,... and their uncertainties as a covariance matrix
    """
    s = solve if deproject else np.linalg.solve
    C = ycov
    y = y[:,None] 
    A = np.zeros((y.size,len(funcs)))
    print(A.shape)
    for i,func in enumerate(funcs):
        A[:,i] = func
    CA = s(C,A) if Cinv is None else np.dot(Cinv,A)
    cov = np.linalg.inv(np.dot(A.T,CA))
    if Cy is None: Cy = s(C,y) if Cinv is None else np.dot(Cinv,y)
    b = np.dot(A.T,Cy)
    X = np.dot(cov,b)
    YAX = y - np.dot(A,X)
    CYAX = s(C,YAX) if Cinv is None else np.dot(Cinv,YAX)
    chisquare = np.dot(YAX.T,CYAX)
    dofs = len(x)-len(funcs)-1 if dofs is None else dofs
    pte = 1 - chi2.cdf(chisquare, dofs)    
    return X,cov,chisquare/dofs,pte

def get_inv_covmat(covmat,nsims,obs=10,exp=False,cov1=None):
    
    if exp==True:
        invcov=PME(covmat,nsims,order=1,cov1=cov1)
    else:
        invcov=np.linalg.inv(covmat)
        corr=(nsims-obs-2)/(nsims-1)
        invcov*=corr
    return invcov