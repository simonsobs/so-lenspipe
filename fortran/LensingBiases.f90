! Copyright (C) 2016 Lewis, Peloton
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Fortran script to compute CMB weak lensing biases (N0, N1)
! and derivatives. f2py friendly.
! Authors: Original script by Antony Lewis, adapted by Julien Peloton.
! Contact: j.peloton@sussex.ac.uk
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Modified by Frank Qu and Mathew Madhavacheril
module LensingBiases
!$ use omp_lib
implicit none

contains

    subroutine SetPhiSampling(LMin,lmx,lmaxmax,sampling,nPhiSample,Phi_Sample,dPhi_Sample)
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        ! Define the sampling to be used to compute biases
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        integer, parameter :: DP = 8
        integer, parameter :: I4B = 4
        integer,intent(in) :: LMin,lmx, lmaxmax
        logical, intent(in) :: sampling
        integer(I4B), intent(out) :: Phi_Sample(lmaxmax)
        real(dp), intent(out) :: dPhi_Sample(lmaxmax)
        integer(I4B), intent(out) :: nPhiSample

        integer(I4B) :: Lstep = 20
        integer i,ix, dL,Lix, L
        real :: acc=1

        ! print *, 'sampling', sampling

        if (.not. sampling) then
            Lix=0
            do L=LMin, lmx, Lstep
                LIx=Lix+1
                Phi_Sample(Lix)=L
            end do
            nPhiSample = Lix
        else
            ix=0

            do i=2, 110, nint(10/acc)
                ix=ix+1
                Phi_Sample(ix)=i
            end do

            dL =nint(30/acc)
            do i=Phi_Sample(ix)+dL, 580, dL
                ix=ix+1
                Phi_Sample(ix)=i
            end do
            dL =nint(100/acc)
            do i=Phi_Sample(ix)+dL, lmx/2, dL
                ix=ix+1
                Phi_Sample(ix)=i
            end do
            dL =nint(300/acc)
            do i=Phi_Sample(ix)+dL, lmx, dL
                ix=ix+1
                Phi_Sample(ix)=i
            end do

            nPhiSample =  ix
        end if

        dPhi_Sample(1) = (Phi_Sample(2)-Phi_Sample(1))/2.
        do i=2, nPhiSample-1
            dPhi_Sample(i) = (Phi_Sample(i+1)-Phi_Sample(i-1))/2.
        end do
        dPhi_Sample(nPhiSample) = (Phi_Sample(nPhiSample)-Phi_Sample(nPhiSample-1))

    end subroutine SetPhiSampling


    


    subroutine ReadPowert(Filename,Lmax, lmaxmax,CT,CE,CB,CX,CTf,CEf,CBf,CXf)
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        ! Read input file and return lensed CMB spectra (and weights)
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        integer, parameter :: DP = 8
        integer, parameter :: I4B = 4
        real(dp), parameter :: pi =  3.1415927, twopi=2*pi
        real, intent(in) :: Filename(5,lmaxmax)
        integer, intent(in) :: Lmax, lmaxmax
        real(dp),intent(out) :: CX(lmaxmax), CE(lmaxmax),CB(lmaxmax), CT(lmaxmax)
        real(dp),intent(out) :: CXf(lmaxmax), CEf(lmaxmax),CBf(lmaxmax), CTf(lmaxmax)
        real(dp) :: CPhi(lmaxmax)
        integer L, status
        logical :: newform = .false.
        CT=0
        CE=0
        CB=0
        CX=0
        do L=2, lmax
        CT(L) = Filename(2,L-1) * twopi/(Filename(1,L-1)*(Filename(1,L-1)+1))
        CE(L) = Filename(3,L-1) * twopi/(Filename(1,L-1)*(Filename(1,L-1)+1))
        CB(L) = Filename(4,L-1) * twopi/(Filename(1,L-1)*(Filename(1,L-1)+1))
        CX(L) = Filename(5,L-1) * twopi/(Filename(1,L-1)*(Filename(1,L-1)+1))
        
        end do
 
        CTf=CT
        CEf=CE
        CBf=CB
        CXf=CX       
        
        


    end subroutine ReadPowert
    !
    
    subroutine getNorm(WantIntONly,sampling,doCurl,lmin_filter,lmax,lmaxout,lmaxmax,n_est, CPhi,&
                        & CT, CE, CX, CB, CTf, CEf, CXf, CBf, CTobs, CEobs, CBobs, n0tt,n0ee,n0eb,n0te,n0tb,L_min, Lstep)
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        ! Main routine to compute N0 bias.
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        integer, parameter :: DP = 8
        integer, parameter :: I4B = 4
        real(dp), parameter :: pi =  3.1415927, twopi=2*pi

        integer,  intent(in) :: lmin_filter,lmax,lmaxout,lmaxmax,n_est,L_min, Lstep
        real(dp), intent(in) :: CPhi(lmaxmax)
        real(dp), intent(in) :: CX(lmaxmax), CE(lmaxmax),CB(lmaxmax), CT(lmaxmax)
        real(dp), intent(in) :: CXf(lmaxmax), CEf(lmaxmax),CBf(lmaxmax), CTf(lmaxmax)
        real(dp), intent(in) :: CEobs(lmaxmax), CTobs(lmaxmax), CBobs(lmaxmax)
        logical, intent(in) :: WantIntONly, sampling, doCurl

        integer(I4B), parameter :: i_TT=1,i_EE=2,i_EB=3,i_TE=4,i_TB=5, i_BB=6
        integer L, Lix, l1, nphi, phiIx, L2int
        real(dp) dphi
        real(dp) L1Vec(2), L2vec(2), LVec(2)
        real(dp) phi, cos2L1L2, sin2
        real(dP) norm, L2
        real(dp) cosfac,f12(n_est),f21(n_est),Win21(n_est),Win12(n_est), fac
        integer, parameter :: dL1=1
        integer file_id, i,j, icurl, nPhiSample,Phi_Sample(lmaxmax)
        logical isCurl
        real(dp) N0(n_est,n_est), N0_L(n_est,n_est),dPhi_Sample(lmaxmax)
        
        real(dp),  DIMENSION((lmaxout-L_min)/Lstep+1),intent(out) ::  n0tt,n0ee,n0eb,n0te,n0tb
        CHARACTER(LEN=13) :: creturn

        call SetPhiSampling(lmin_filter,lmaxout,lmaxmax,sampling,nPhiSample,Phi_Sample,dPhi_Sample)
        ! print *,nPhiSample

        !Order TT, EE, EB, TE, TB, BB
        do icurl = 0,1
            isCurl = icurl==1
            if (IsCurl) then
                if (.not. doCurl) cycle
                print *,''
                print *,'N0 computation (curl)'
            else
                print *,'N0 computation (phi)'
            end if

            Lix=0
            do L=L_min, lmaxout, Lstep
                Lix=Lix+1
                write(*,*) Lix
                creturn = achar(13)
                WRITE( * , 101 , ADVANCE='NO' ) creturn , int(real(Lix,kind=dp)/nPhiSample*100.,kind=I4B)
                101     FORMAT( a , 'Progression : ',i7,' % ')
                Lvec(1) = L
                LVec(2)= 0
                N0_L=0

!$OMP           PARALLEL DO&
!$OMP&          default(shared) private(L1,nphi,dphi,N0, &
!$OMP&          PhiIx,phi,L1vec,L2vec,L2,L2int,cos2L1L2, &
!$OMP&          sin2,cosfac,f12,f21,Win12,Win21,i,j) &
!$OMP&          reduction(+:N0_L)

                do L1=lmin_filter, lmax, dL1

                nphi=(2*L1+1)
                dphi=(2*Pi/nphi)
                N0=0

                do PhiIx=0,(nphi-1)/2
                    phi= dphi*PhiIx
                    L1vec(1)=L1*cos(phi)
                    L1vec(2)=L1*sin(phi)
                    L2vec = Lvec-L1vec
                    L2=(sqrt(L2vec(1)**2+L2vec(2)**2))
                    if (L2<lmin_filter .or. L2>lmax) cycle
                    L2int=nint(L2)

      
                    call getResponseFull(n_est,lmaxmax,L1vec(1)*L,L2vec(1)*L, L1vec,real(L1,dp),&
                    & L1, L2vec,L2, L2int,f12,f21, CT, CE, CX, CB)
                    call getWins(n_est,lmaxmax,L1vec(1)*L,L2vec(1)*L, L1vec,real(L1,dp),&
                    & L1, L2vec,L2, L2int, CX, CTf, CEf, CXf, CBf,CTobs, CEobs, CBobs,Win12, Win21)
                   

                    do i=1,n_est
                        N0(i,i) = N0(i,i) + f12(i)*Win12(i)
                    end do


                    !Important to use symmetric form here if only doing half PhiIx integral
                    N0(i_TT,i_TE) = N0(i_TT,i_TE) + Win12(i_TT)*(Win12(i_TE)*&
                    &CTobs(L1)*CXf(L2int) + Win21(i_TE)*CXf(L1)*CTobs(L2int))

                    N0(i_TT,i_EE) = N0(i_TT,i_EE) + 2*Win12(i_TT)*Win12(i_EE)*CXf(L1)*CXf(L2int)

                    N0(i_EE,i_TE) = N0(i_EE,i_TE) + Win12(i_EE)*(Win12(i_TE)*&
                    &CXf(L1)*CEobs(L2int) + Win21(i_TE)*CEobs(L1)*CXf(L2int))

                    N0(i_EB,i_TB) = N0(i_EB,i_TB) + (Win12(i_EB)*(Win12(i_TB)*CXf(L1)*CBobs(L2int)) + &
                        & Win21(i_EB)*(Win21(i_TB)*CXf(L2int)*CBobs(L1)))/2

                    if (PhiIx==0) N0 = N0/2
                end do
                fac = dphi* L1*dL1 *2
                N0_L = N0_L + N0 * fac

            end do
            !$OMP END PARALLEL DO
            N0_L = N0_L/(twopi**2)

            N0=0
            do i=1,n_est
                do j=i,n_est
                    if (WantIntONly) then
                        N0(i,j) = N0_L(i,j)
                    else
                        N0(i,j) = N0_L(i,j)/N0_L(i,i)/N0_L(j,j)
                    end if
                    N0(j,i) = N0(i,j)
                end do
            end do
            n0tt(Lix)=N0(1,1)
            n0ee(Lix)=N0(2,2)
            n0eb(Lix)=N0(3,3)
            n0te(Lix)=N0(4,4)
            n0tb(Lix)=N0(5,5)

        

            norm = real(L*(L+1),dp)**2/twopi
            ! write (file_id,'(1I5, 1E16.6)',Advance='NO') L, CPhi(L)*norm

            ! call WriteMatrixLine(file_id,N0,n_est)

        end do
        ! close(file_id)
        end do
        print *,''

    end subroutine getNorm
    !
    
    

    

    !
    !
    

    
    !
    !
    subroutine getResponseFull(n_est,lmaxmax,L_dot_L1,L_dot_L2, L1vec,L1,L1int, L2vec,L2, L2int,f12,f21, CT, CE, CX, CB)
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        ! Response function (signal + noise)
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        integer, parameter :: DP = 8
        integer, parameter :: I4B = 4
        real(dp), parameter :: pi =  3.1415927, twopi=2*pi
        integer, intent(in) :: n_est,lmaxmax,L1int, L2int
        real(dp), intent(in) :: L1vec(2),L2vec(2), L1,L2, L_dot_L1,L_dot_L2
        real(dp), intent(in) :: CX(lmaxmax), CE(lmaxmax),CB(lmaxmax), CT(lmaxmax)
        real(dp),intent(out) :: f12(n_est), f21(n_est)
        real(dp) cosfac, sin2, cos2L1L2
        integer(I4B), parameter :: i_TT=1,i_EE=2,i_EB=3,i_TE=4,i_TB=5, i_BB=6

        f12(i_TT)= (L_dot_L1*CT(L1int) + L_dot_L2*CT(L2int))

        cosfac= dot_product(L1vec,L2vec)/real(L1*L2,dp)
        cos2L1L2 =2*cosfac**2-1
        f12(i_EE)= (L_dot_L1*CE(L1int) + L_dot_L2*CE(L2int))*cos2L1L2

        if (n_est>=i_BB) then
            f12(i_BB) = (L_dot_L1*CB(L1int) + L_dot_L2*CB(L2int))*cos2L1L2
        end if
        f21=f12

        sin2=  2*cosfac*(L1vec(2)*L2vec(1)-L1vec(1)*L2vec(2))/(L2*L1)
        !Note sign typo for f^EB in in Hu and Okamoto 2002 (BB term)
        f12(i_EB) = (L_dot_L1*CE(L1int) + L_dot_L2*CB(L2int))*sin2
        f21(i_EB) = -(L_dot_L2*CE(L2int) + L_dot_L1*CB(L1int))*sin2

        f12(i_TE)=  L_dot_L1*CX(L1int)*cos2L1L2 + L_dot_L2*CX(L2int)
        f21(i_TE)=  L_dot_L2*CX(L2int)*cos2L1L2 + L_dot_L1*CX(L1int)

        f12(i_TB) = L_dot_L1*CX(L1int)*sin2
        f21(i_TB) = -L_dot_L2*CX(L2int)*sin2

    end subroutine getResponseFull
    
    subroutine getResponsefid(n_est,lmaxmax,L_dot_L1,L_dot_L2, L1vec,L1,L1int, L2vec,L2, L2int,CTf, CEf, CXf, CBf,f12,f21)
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        ! Response function (using fiducial model)
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        integer, parameter :: DP = 8
        real(dp), parameter :: pi =  3.1415927, twopi=2*pi
        integer, intent(in) :: n_est,L1int, L2int,lmaxmax
        real(dp), intent(in) :: L1vec(2),L2vec(2), L1,L2, L_dot_L1,L_dot_L2
        real(dp), intent(in) :: CXf(lmaxmax), CEf(lmaxmax),CBf(lmaxmax), CTf(lmaxmax)
        real(dp),intent(out) :: f12(n_est), f21(n_est)

        call getResponseFull(n_est,lmaxmax,L_dot_L1,L_dot_L2, L1vec,L1,L1int, L2vec,L2, L2int,f12,f21, CTf, CEf, CXf, CBf)

    end subroutine getResponsefid
    !
    subroutine getResponse(n_est,lmaxmax,L_dot_L1,L_dot_L2, L1vec,L1,L1int, L2vec,L2, L2int,CT, CE, CX, CB,f12,f21)
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        ! Response function (lensed CMB spectra)
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        integer, parameter :: DP = 8
        integer, intent(in) :: L1int, L2int,n_est,lmaxmax
        real(dp), intent(in) :: L1vec(2),L2vec(2), L1,L2, L_dot_L1,L_dot_L2
        real(dp), intent(in) :: CX(lmaxmax), CE(lmaxmax),CB(lmaxmax), CT(lmaxmax)
        real(dp),intent(out) :: f12(n_est), f21(n_est)

        call getResponseFull(n_est,lmaxmax,L_dot_L1,L_dot_L2, L1vec,L1,L1int, L2vec,L2, L2int,f12,f21, CT, CE, CX, CB)

    end subroutine getResponse
    !
    subroutine getWins(n_est,lmaxmax,L_dot_L1,L_dot_L2, L1vec,L1,L1int, L2vec,L2, L2int, &
                    & CX,CTf, CEf, CXf, CBf, CTobs, CEobs, CBobs, Win12, Win21)
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        ! Compute filters f/cl^2.
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        integer, parameter :: DP = 8
        integer, parameter :: I4B = 4
        real(dp), parameter :: pi =  3.1415927, twopi=2*pi
        integer, intent(in) :: n_est,L1int, L2int,lmaxmax
        real(dp), intent(in) :: L1vec(2),L2vec(2), L1,L2, L_dot_L1,L_dot_L2
        real(dp), intent(in) :: CX(lmaxmax)
        real(dp), intent(in) :: CEobs(lmaxmax), CTobs(lmaxmax), CBobs(lmaxmax)
        real(dp), intent(in) :: CXf(lmaxmax), CEf(lmaxmax),CBf(lmaxmax), CTf(lmaxmax)
        real(dp), intent(out) :: Win12(n_est)
        real(dp), intent(out), optional :: Win21(n_est)
        real(dp) f12(n_est), f21(n_est)
        integer(I4B), parameter :: i_TT=1,i_EE=2,i_EB=3,i_TE=4,i_TB=5, i_BB=6

        call getResponsefid(n_est,lmaxmax,L_dot_L1,L_dot_L2, L1vec,L1,L1int, L2vec,L2, L2int, CTf, CEf, CXf, CBf,f12, f21)

        Win12(i_TT) = f12(i_TT)/(2*CTobs(L1int)*CTobs(L2int))

        Win12(i_EE) = f12(i_EE)/(2*CEobs(L1int)*CEobs(L2int))

        Win12(i_EB) = f12(i_EB)/(CEobs(L1int)*CBobs(L2int))

        Win12(i_TE) = (f12(i_TE)* CEobs(L1int)*CTobs(L2int) - f21(i_TE)*CX(L1int)*CX(L2int))&
            /(CTobs(L1int)*CEobs(L2int)*CTobs(L2int)*CEobs(L1int) - (CX(L1int)*CX(L2int))**2)

        Win12(i_TB) = f12(i_TB)/(CTobs(L1int)*CBobs(L2int))

        if (n_est>=i_BB) then
            Win12(i_BB) = f12(i_BB)/(2*CBobs(L1int)*CBobs(L2int))
        end if

        if (present(Win21)) then
            Win21=Win12

            Win21(i_TE) = (f21(i_TE)* CTobs(L1int)*CEobs(L2int) - f12(i_TE)*CX(L1int)*CX(L2int))&
                /(CTobs(L1int)*CEobs(L2int)*CTobs(L2int)*CEobs(L1int) - (CX(L1int)*CX(L2int))**2)

            Win21(i_EB) = f21(i_EB)/(CEobs(L2int)*CBobs(L1int))
            Win21(i_TB) = f21(i_TB)/(CTobs(L2int)*CBobs(L1int))
        end if

    end subroutine getWins
    !
    function responseFor(n_est,i,j, f12,f21)
        integer, parameter :: DP = 8
        integer, parameter :: I4B = 4
        integer, intent(in) :: n_est,i,j
        real(dp), intent(in) :: f12(n_est), f21(n_est)
        integer ix
        real(dp) responseFor
        integer, DIMENSION(3, 3) :: lumpsFor
        integer(I4B), parameter :: i_TT=1,i_EE=2,i_EB=3,i_TE=4,i_TB=5, i_BB=6

        lumpsFor = transpose( &
         & reshape((/ i_TT, i_TE, i_TB, i_TE, i_EE, i_EB, i_TB, i_EB, i_BB /), (/ 3,3/) ))

        if (j>=i) then
            ix= lumpsFor(i,j)
            if (ix<=n_est) then
                responseFor = f12(ix)
            else
                responseFor=0
            end if
        else
            ix= lumpsFor(j,i)
            if (ix<=n_est) then
                responseFor = f21(ix)
            else
                responseFor=0
            end if
        end if

    end function responseFor
    
    !
    
 
    
    subroutine loadNorm(normarray,n_est,lmin_filter,lmaxout, Lstep,Norms)
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        ! Load N0 bias from the disk
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        integer, parameter :: DP = 8
        integer, parameter :: I4B = 4
        real(dp), parameter :: pi =  3.1415927, twopi=2*pi
        real, intent(in) :: normarray(6,*)
        integer,  intent(in) :: lmin_filter,lmaxout,n_est,Lstep
        real(dp),intent(out) :: Norms(lmaxout,n_est)
        integer  file_id, L,i
        real(dp) ell,TT,EE,EB,TE,TB,BB
        real(dp) N0(n_est,n_est), dum !array size 5 i.e n_est=5

        
        do L=lmin_filter, lmaxout, Lstep
            !read(file_id,*) ell, dum, N0
            !do i=1,n_est
                !Norms(L,i) = N0(i,i)
            !end do
            !read(file_id,*) ell,
            
            Norms(L,1)=normarray(1,L-1)
            Norms(L,2)=normarray(2,L-1)
            Norms(L,3)=normarray(3,L-1)
            Norms(L,4)=normarray(4,L-1)
            Norms(L,5)=normarray(5,L-1)
            Norms(L,6)=normarray(6,L-1)
            
                
    
        end do
    end subroutine loadNorm
    
    
     subroutine getmixNorm(WantIntONly,sampling,doCurl,lmin_filter,lmax,lmaxout,lmaxmax,n_est, CPhi,&
                        & CT, CE, CX, CB, CTf, CEf, CXf, CBf, CTobs, CEobs, CBobs, n0ttee,n0ttte,n0eete,n0ebtb,L_min, Lstep)
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        ! Main routine to compute N0 bias.
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        integer, parameter :: DP = 8
        integer, parameter :: I4B = 4
        real(dp), parameter :: pi =  3.1415927, twopi=2*pi

        integer,  intent(in) :: lmin_filter,lmax,lmaxout,lmaxmax,n_est,L_min, Lstep
        real(dp), intent(in) :: CPhi(lmaxmax)
        real(dp), intent(in) :: CX(lmaxmax), CE(lmaxmax),CB(lmaxmax), CT(lmaxmax)
        real(dp), intent(in) :: CXf(lmaxmax), CEf(lmaxmax),CBf(lmaxmax), CTf(lmaxmax)
        real(dp), intent(in) :: CEobs(lmaxmax), CTobs(lmaxmax), CBobs(lmaxmax)
        logical, intent(in) :: WantIntONly, sampling, doCurl

        integer(I4B), parameter :: i_TT=1,i_EE=2,i_EB=3,i_TE=4,i_TB=5, i_BB=6
        integer L, Lix, l1, nphi, phiIx, L2int
        real(dp) dphi
        real(dp) L1Vec(2), L2vec(2), LVec(2)
        real(dp) phi, cos2L1L2, sin2
        real(dP) norm, L2
        real(dp) cosfac,f12(n_est),f21(n_est),Win21(n_est),Win12(n_est), fac
        integer, parameter :: dL1=1
        integer file_id, i,j, icurl, nPhiSample,Phi_Sample(lmaxmax)
        logical isCurl
        real(dp) N0(n_est,n_est), N0_L(n_est,n_est),dPhi_Sample(lmaxmax)
        real(dp),  DIMENSION((lmaxout-L_min)/Lstep+1),intent(out) ::  n0ttee,n0ttte,n0eete,n0ebtb
        real(dp) Norms(lmaxmax,n_est)
        CHARACTER(LEN=13) :: creturn

   call SetPhiSampling(lmin_filter,lmaxout,lmaxmax,sampling,nPhiSample,Phi_Sample,dPhi_Sample)
        ! print *,nPhiSample

        !Order TT, EE, EB, TE, TB, BB
        do icurl = 0,1
            isCurl = icurl==1
            if (IsCurl) then
                if (.not. doCurl) cycle
                print *,''
                print *,'N0 computation (curl)'
            else
                print *,'N0 computation (phi)'
            end if


            Lix=0
            do L=L_min, lmaxout, Lstep
                Lix=Lix+1
                creturn = achar(13)
                WRITE( * , 101 , ADVANCE='NO' ) creturn , int(real(Lix,kind=dp)/nPhiSample*100.,kind=I4B)
                101     FORMAT( a , 'Progression : ',i7,' % ')
                Lvec(1) = L
                LVec(2)= 0
                N0_L=0

!$OMP           PARALLEL DO&
!$OMP&          default(shared) private(L1,nphi,dphi,N0, &
!$OMP&          PhiIx,phi,L1vec,L2vec,L2,L2int,cos2L1L2, &
!$OMP&          sin2,cosfac,f12,f21,Win12,Win21,i,j) &
!$OMP&          reduction(+:N0_L)

                do L1=lmin_filter, lmax, dL1

                nphi=(2*L1+1)
                dphi=(2*Pi/nphi)
                N0=0

                do PhiIx=0,(nphi-1)/2
                    phi= dphi*PhiIx
                    L1vec(1)=L1*cos(phi)
                    L1vec(2)=L1*sin(phi)
                    L2vec = Lvec-L1vec
                    L2=(sqrt(L2vec(1)**2+L2vec(2)**2))
                    if (L2<lmin_filter .or. L2>lmax) cycle
                    L2int=nint(L2)

      
                    call getResponseFull(n_est,lmaxmax,L1vec(1)*L,L2vec(1)*L, L1vec,real(L1,dp),&
                    & L1, L2vec,L2, L2int,f12,f21, CT, CE, CX, CB)
                    call getWins(n_est,lmaxmax,L1vec(1)*L,L2vec(1)*L, L1vec,real(L1,dp),&
                    & L1, L2vec,L2, L2int, CX, CTf, CEf, CXf, CBf,CTobs, CEobs, CBobs,Win12, Win21)
                   

                    do i=1,n_est
                        N0(i,i) = N0(i,i) + f12(i)*Win12(i)
                    end do


                    !Important to use symmetric form here if only doing half PhiIx integral
                    N0(i_TT,i_TE) = N0(i_TT,i_TE) + Win12(i_TT)*(Win12(i_TE)*&
                    &CTobs(L1)*CXf(L2int) + Win21(i_TE)*CXf(L1)*CTobs(L2int))
                    

                    N0(i_TT,i_EE) = N0(i_TT,i_EE) + 2*Win12(i_TT)*Win12(i_EE)*CXf(L1)*CXf(L2int)

                    N0(i_EE,i_TE) = N0(i_EE,i_TE) + Win12(i_EE)*(Win12(i_TE)*&
                    &CXf(L1)*CEobs(L2int) + Win21(i_TE)*CEobs(L1)*CXf(L2int))

                    N0(i_EB,i_TB) = N0(i_EB,i_TB) + (Win12(i_EB)*(Win12(i_TB)*CXf(L1)*CBobs(L2int)) + &
                        & Win21(i_EB)*(Win21(i_TB)*CXf(L2int)*CBobs(L1)))/2

                    if (PhiIx==0) N0 = N0/2
                end do
                fac = dphi* L1*dL1 *2
                N0_L = N0_L + N0 * fac

            end do
            !$OMP END PARALLEL DO
            N0_L = N0_L/(twopi**2)

            N0=0
            do i=1,n_est
                do j=i,n_est
                    if (WantIntONly) then
                        N0(i,j) = N0_L(i,j)
                    else
                        N0(i,j) = N0_L(i,j)/N0_L(i,i)/N0_L(j,j)
                    end if
                    N0(j,i) = N0(i,j)
                end do
            end do
            n0ttee(Lix)=N0(1,2)
            n0ttte(Lix)=N0(1,4)
            n0eete(Lix)=N0(2,4)
            n0ebtb(Lix)=N0(3,5)


            ! print *, L, CPhi(L)*norm, N0(i_TT,i_TT)*norm
        end do
        end do
        print *,''

    end subroutine getmixNorm 
    
    subroutine GetN1General(normarray,sampling,lmin_filter,lmax,lmaxout,lmaxmax,n_est, CPhi,&
                        & CT, CE, CX, CB, CTf, CEf, CXf, CBf, CTobs, CEobs, CBobs, n1theta,n1ee,n1eb,n1te,n1tb,Lstep,L_min)
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        ! Main routine to compute N1 bias.
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        integer, parameter :: DP = 8
        integer, parameter :: I4B = 4
        real(dp), parameter :: pi =  3.1415927, twopi=2*pi

        integer,  intent(in) :: lmin_filter,lmax,lmaxout,lmaxmax,n_est,Lstep,L_min
        real(dp), intent(in) :: CPhi(lmaxmax)
        real(dp), intent(in) :: CX(lmaxmax), CE(lmaxmax),CB(lmaxmax), CT(lmaxmax)
        real(dp), intent(in) :: CXf(lmaxmax), CEf(lmaxmax),CBf(lmaxmax), CTf(lmaxmax)
        real(dp), intent(in) :: CEobs(lmaxmax), CTobs(lmaxmax), CBobs(lmaxmax)
        logical, intent(in) :: sampling

        integer(I4B), parameter :: i_TT=1,i_EE=2,i_EB=3,i_TE=4,i_TB=5, i_BB=6
        integer(I4B), parameter ::  dL = 20
        integer  :: lumped_indices(2,n_est)
        integer L, Lix, l1, nphi, phiIx, L2int,PhiL_nphi,PhiL_phi_ix,L3int,L4int
        integer PhiL
        real(dp) dphi,PhiL_phi_dphi
        real(dp) L1Vec(2), L2vec(2), LVec(2), L3Vec(2),L4Vec(2), phiLVec(2)
        real(dp) phi, PhiL_phi
        real(dP) L2, L4, L3
        real(dp) dPh
        real(dp) phiL_dot_L2, phiL_dot_L3, phiL_dot_L1, phiL_dot_L4
        real(dp) fact(n_est,n_est),tmp(n_est,n_est), N1(n_est,n_est), N1_L1(n_est,n_est),N1_PhiL(n_est,n_est)
        real(dp) Win12(n_est), Win34(n_est), Win43(n_est)
        real(dp) WinCurl12(n_est), WinCurl34(n_est), WinCurl43(n_est), tmpCurl(n_est,n_est), &
            factCurl(n_est,n_est),N1_PhiL_Curl(n_est,n_est), N1_L1_Curl(n_est,n_est),  N1_Curl(n_est,n_est)
        real(dp) f24(n_est), f13(n_est),f31(n_est), f42(n_est)
        integer file_id, nPhiSample,Phi_Sample(lmaxmax)
        integer file_id_Curl, PhiLix
        integer ij(2),pq(2), est1, est2
        real(dp) tmpPS, tmpPSCurl, N1_PhiL_PS, N1_PhiL_PS_Curl, N1_L1_PS_Curl, N1_L1_PS, N1_PS, N1_PS_Curl
        real(dp) dPhi_Sample(lmaxmax)
        real, intent(in) :: normarray(6,*)
        real(dp):: Norms(lmaxout,n_est)
        integer file_id_PS
        real(dp) this13, this24
        real(dp),  DIMENSION((lmaxout-L_min)/Lstep+1),intent(out) ::  n1theta,n1ee,n1eb,n1te,n1tb
        character(LEN=10) outtag
        CHARACTER(LEN=13) :: creturn

        lumped_indices = transpose(reshape((/ 1,2,2,1,1,3,1,2,3,2,3,3 /), (/ n_est, 2 /)  ))
        call SetPhiSampling(lmin_filter,lmaxout,lmaxmax,sampling,nPhiSample,Phi_Sample,dPhi_Sample)

        outtag = 'N1_All'
        
        call loadNorm(normarray,n_est,l_min,lmaxout, Lstep,Norms)
        Lix=0
        print *,'N1 computation (phi, curl, PS)'
        do L=L_min, lmaxout, Lstep
            creturn = achar(13)
            WRITE( * , 101 , ADVANCE='NO' ) creturn , int(real(L,kind=dp)/lmaxout*100.,kind=I4B)
            101     FORMAT( a , 'Progression : ',i7,' % ')
            Lix=Lix+1
            write(*,*) Lix
            Lvec(1) = L
            LVec(2)= 0
            N1=0

            do L1=max(lmin_filter,dL/2), lmax, dL
                N1_L1 = 0



                nphi=(2*L1+1)
                if (L1>3*dL) nphi=2*nint(L1/real(2*dL))+1
                dphi=(2*Pi/nphi)

                !$OMP PARALLEL DO default(shared), private(PhiIx,phi,PhiL_nphi, PhiL_phi_dphi, PhiL_phi_ix, PhiL_phi,PhiLix, dPh), &
                !$OMP private(L1vec,L2,L2vec, L2int,  L3, L3vec, L3int, L4, L4vec, L4int),&
                !$OMP private(tmp, Win12, Win34, Win43, fact,phiL_dot_L1, phiL_dot_L2, phiL_dot_L3, phiL_dot_L4), &
                !$OMP private(tmpCurl, WinCurl12, WinCurl34, WinCurl43, factCurl, N1_PhiL_Curl), &
                !$OMP private(f13, f31, f42, f24, ij, pq, est1, est2,this13,this24), &
                !$OMP private(tmpPS, tmpPSCurl, N1_PhiL_PS, N1_PhiL_PS_Curl), &

                !$OMP private(PhiL, PhiLVec, N1_PhiL), schedule(STATIC), reduction(+:N1_L1), reduction(+:N1_L1_Curl), &
                !$OMP reduction(+:N1_L1_PS), reduction(+:N1_L1_PS_Curl)
                
                do phiIx=0,(nphi-1)/2 !
                phi= dphi*PhiIx
                L1vec(1)=L1*cos(phi)
                L1vec(2)=L1*sin(phi)
                L2vec = Lvec-L1vec
                L2=(sqrt(L2vec(1)**2+L2vec(2)**2))
                if (L2<lmin_filter .or. L2>lmax) cycle
                L2int=nint(L2)

                call getWins(n_est,lmaxmax,L*L1vec(1),L*L2vec(1), L1vec,real(L1,dp),L1, L2vec,L2, L2int,  &
                & CX, CTf, CEf, CXf, CBf,CTobs, CEobs, CBobs, Win12)


                N1_PhiL=0
                N1_PhiL_PS=0
                
                do PhiLIx = 1, nPhiSample
                    PhiL = Phi_Sample(PhiLIx)
                    dPh = dPhi_Sample(PhiLIx)
                    PhiL_nphi=(2*PhiL+1)
                    if (phiL>20) PhiL_nphi=2*nint(real(PhiL_nphi)/dPh/2)+1
                    PhiL_phi_dphi=(2*Pi/PhiL_nphi)
                    tmp=0
                    do PhiL_phi_ix=-(PhiL_nphi-1)/2, (PhiL_nphi-1)/2
                        PhiL_phi= PhiL_phi_dphi*PhiL_phi_ix
                        PhiLvec(1)=PhiL*cos(PhiL_phi)
                        PhiLvec(2)=PhiL*sin(PhiL_phi)
                        L3vec= PhiLvec - L1vec
                        L3 = sqrt(L3vec(1)**2+L3vec(2)**2)
                        if (L3>=lmin_filter .and. L3<=lmax) then
                            L3int = nint(L3)
                            L4vec = -Lvec-L3vec
                            L4 = sqrt(L4vec(1)**2+L4vec(2)**2)
                            L4int=nint(L4)
                            if (L4>=lmin_filter .and. L4<=lmax) then
                                call getWins(n_est,lmaxmax,-L*L3vec(1),-L*L4vec(1), L3vec,L3,L3int, L4vec,L4, L4int,  &
                                & CX, CTf, CEf, CXf, CBf,CTobs, CEobs, CBobs, Win34, Win43)

                                phiL_dot_L1=dot_product(PhiLVec,L1vec)
                                phiL_dot_L2=-dot_product(PhiLVec,L2vec)
                                phiL_dot_L3=dot_product(PhiLVec,L3vec)
                                phiL_dot_L4=-dot_product(PhiLVec,L4vec)

                                call getResponse(n_est,lmaxmax,phiL_dot_L1,phiL_dot_L3, L1vec,real(L1,dp),L1, L3vec,L3, &
                                & L3int, CT, CE, CX, CB, f13, f31)
                                call getResponse(n_est,lmaxmax,phiL_dot_L2,phiL_dot_L4, L2vec,L2,L2int, L4vec,L4, &
                                & L4int, CT, CE, CX, CB, f24, f42)

                                do est1=1,n_est
                                    ij=lumped_indices(:,est1)
                                    do est2=est1,n_est
                                        pq=lumped_indices(:,est2)
                                        this13 = responseFor(n_est,ij(1),pq(1),f13,f31)
                                        this24 = responseFor(n_est,ij(2),pq(2),f24,f42)
                                        tmp(est1,est2)=tmp(est1,est2)+this13*this24*Win34(est2)
                                        !tmp(est1,est2)=tmp(est1,est2)+this13*this24
                                        !tmp(est1,est2)=tmp(est1,est2)+Win34(est2)
                               

                                        this13 = responseFor(n_est,ij(1),pq(2),f13,f31)
                                        this24 = responseFor(n_est,ij(2),pq(1),f24,f42)
                                        tmp(est1,est2)=tmp(est1,est2)+this13*this24*Win43(est2)

                                    end do
                                end do
                            end if
                        end if
                    end do
                    if (phiIx/=0) tmp=tmp*2 !integrate 0-Pi for phi_L1
                    fact = tmp* PhiL_phi_dphi* PhiL
                    N1_PhiL= N1_PhiL + fact * Cphi(PhiL)*dPh
    

                end do
                do est1=1,n_est
                    N1_PhiL(est1,:)=N1_PhiL(est1,:)*Win12(est1)

                end do
                N1_L1 = N1_L1+N1_PhiL

            end do
            !$OMP END PARALLEL DO
            N1= N1 + N1_L1 * dphi* L1*dL


        end do
        do est1=1,n_est
            do est2=est1,n_est
                N1(est1,est2) = norms(L,est1)*norms(L,est2)*N1(est1,est2) / (twopi**4)
                N1(est2,est1) = N1(est1,est2)
            end do
        
        end do
        
        n1theta(Lix)=N1(1,1)
        n1ee(Lix)=N1(2,2)
        n1eb(Lix)=N1(3,3)
        n1te(Lix)=N1(4,4)
        n1tb(Lix)=N1(5,5)
        end do
        print *,''

    end subroutine GetN1General
    


    !
    subroutine GetN1MatrixGeneral(normarray,sampling,lmin_filter,lmax,lmaxout,lmaxmax,n_est, CPhi,&
                        & CT, CE, CX, CB, CTf, CEf, CXf, CBf, CTobs, CEobs, CBobs, Lstep,L_min)
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        ! Main routine to compute N1 derivatives.
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        integer, parameter :: DP = 8
        integer, parameter :: I4B = 4
        real(dp), parameter :: pi =  3.1415927, twopi=2*pi

        integer,  intent(in) :: lmin_filter,lmax,lmaxout,lmaxmax,n_est,Lstep,L_min
        real(dp), intent(in) :: CPhi(lmaxmax)
        real(dp), intent(in) :: CX(lmaxmax), CE(lmaxmax),CB(lmaxmax), CT(lmaxmax)
        real(dp), intent(in) :: CXf(lmaxmax), CEf(lmaxmax),CBf(lmaxmax), CTf(lmaxmax)
        real(dp), intent(in) :: CEobs(lmaxmax), CTobs(lmaxmax), CBobs(lmaxmax)
        logical, intent(in) :: sampling

        integer(I4B), parameter :: i_TT=1,i_EE=2,i_EB=3,i_TE=4,i_TB=5, i_BB=6
        integer(I4B), parameter :: dL = 20
        integer  :: lumped_indices(2,n_est)
        integer L, Lix, l1, nphi, phiIx, L2int,PhiL_nphi,PhiL_phi_ix,L3int,L4int
        integer PhiL
        real(dp) dphi,PhiL_phi_dphi
        real(dp) L1Vec(2), L2vec(2), LVec(2), L3Vec(2),L4Vec(2), phiLVec(2)
        real(dp) phi, PhiL_phi
        real(dP) L2, L4, L3
        real(dp) dPh
        real(dp) phiL_dot_L2, phiL_dot_L3, phiL_dot_L1, phiL_dot_L4
        real(dp) fact(n_est,n_est),tmp(n_est,n_est), N1(n_est,n_est), N1_L1(n_est,n_est),N1_PhiL(n_est,n_est)
        real(dp) matrixfact(n_est,n_est)
        real(dp) Win12(n_est), Win34(n_est), Win43(n_est)
        real(dp) WinCurl12(n_est), WinCurl34(n_est), WinCurl43(n_est), tmpCurl(n_est,n_est), &
            factCurl(n_est,n_est),N1_PhiL_Curl(n_est,n_est), N1_L1_Curl(n_est,n_est),  N1_Curl(n_est,n_est)
        real(dp) f24(n_est), f13(n_est),f31(n_est), f42(n_est)
        integer file_id, nPhiSample,Phi_Sample(lmaxmax)
        integer file_id_Curl, PhiLix
        integer ij(2),pq(2), est1, est2
        real(dp) tmpPS, tmpPSCurl, N1_PhiL_PS, N1_PhiL_PS_Curl, N1_L1_PS_Curl, N1_L1_PS, N1_PS, N1_PS_Curl
        real(dp) dPhi_Sample(lmaxmax)
        real(dp):: Norms(lmaxout,n_est)
        real, intent(in) :: normarray(6,*)
        integer file_id_PS
        real(dp) this13, this24
        real(dp), allocatable :: Matrix(:,:,:,:), MatrixL1(:,:,:)
        character(LEN=10) outtag
        CHARACTER(LEN=13) :: creturn
        character(2) :: estnames(n_est)
        estnames = ['TT','EE','EB','TE','TB','BB']

        lumped_indices = transpose(reshape((/ 1,2,2,1,1,3,1,2,3,2,3,3 /), (/ n_est, 2 /)  ))
        call SetPhiSampling(lmin_filter,lmaxout,lmaxmax,sampling,nPhiSample,Phi_Sample,dPhi_Sample)

        outtag = 'N1_All'
        write(*,*)   nPhiSample
        allocate(matrix((lmaxout-L_min)/Lstep+1,nPhiSample, n_est,n_est))
        allocate(matrixL1(nPhiSample,n_est,n_est))
        matrix=0
        call loadNorm(normarray,n_est,L_min,lmaxout, Lstep,Norms) !load N0

        Lix=0
        print *,'Derivatives of N1 computation'
        do L=L_min, lmaxout, Lstep
            creturn = achar(13)
            WRITE( * , 101 , ADVANCE='NO' ) creturn , int(real(L,kind=dp)/lmaxout*100.,kind=I4B)
            101     FORMAT( a , 'Progression : ',i7,' % ')
            write(*,*) L
            Lix=Lix+1
            Lvec(1) = L
            LVec(2)= 0
            N1=0
            do L1=max(lmin_filter,dL/2), lmax, dL
                N1_L1 = 0

                matrixL1=0

                nphi=(2*L1+1)
                if (L1>3*dL) nphi=2*nint(L1/real(2*dL))+1
                dphi=(2*Pi/nphi)

                !$OMP PARALLEL DO default(shared), private(PhiIx,phi,PhiL_nphi, PhiL_phi_dphi, PhiL_phi_ix, PhiL_phi,PhiLix, dPh), &
                !$OMP private(L1vec,L2,L2vec, L2int,  L3, L3vec, L3int, L4, L4vec, L4int),&
                !$OMP private(tmp, Win12, Win34, Win43, matrixfact,fact,phiL_dot_L1, phiL_dot_L2, phiL_dot_L3, phiL_dot_L4), &
                !$OMP private(f13, f31, f42, f24, ij, pq, est1, est2,this13,this24), &
                !$OMP private(PhiL, PhiLVec, N1_PhiL), schedule(STATIC), reduction(+:N1_L1)
                do phiIx=0,(nphi-1)/2
                phi= dphi*PhiIx
                L1vec(1)=L1*cos(phi)
                L1vec(2)=L1*sin(phi)
                L2vec = Lvec-L1vec
                L2=(sqrt(L2vec(1)**2+L2vec(2)**2))
                if (L2<lmin_filter .or. L2>lmax) cycle
                L2int=nint(L2)

                call getWins(n_est,lmaxmax,L*L1vec(1),L*L2vec(1), L1vec,real(L1,dp),L1, L2vec,L2, L2int,  &
                & CX, CTf, CEf, CXf, CBf,CTobs, CEobs, CBobs, Win12)  !used to generate the window functions
                ! call getWins(L*L1vec(1),L*L2vec(1), L1vec,real(L1,dp),L1, L2vec,L2, L2int,  Win12)
                
                N1_PhiL=0
                do PhiLIx = 1, nPhiSample
                    PhiL = Phi_Sample(PhiLIx) !2,12,22,32...
                    dPh = dPhi_Sample(PhiLIx)
                    PhiL_nphi=(2*PhiL+1)
                    if (phiL>20) then
                        PhiL_nphi=2*nint(0.5d0*real(PhiL_nphi)/dPh)+1
                    end if
                    PhiL_phi_dphi=(2*Pi/PhiL_nphi)
                    tmp=0
                    do PhiL_phi_ix=-(PhiL_nphi-1)/2, (PhiL_nphi-1)/2  !approximately from -pi to pi
                        PhiL_phi= PhiL_phi_dphi*PhiL_phi_ix
                        PhiLvec(1)=PhiL*cos(PhiL_phi)
                        PhiLvec(2)=PhiL*sin(PhiL_phi)
                        L3vec= PhiLvec - L1vec
                        L3 = sqrt(L3vec(1)**2+L3vec(2)**2)
                        if (L3>=lmin_filter .and. L3<=lmax) then
                            L3int = nint(L3)

                            L4vec = -Lvec-L3vec
                            L4 = sqrt(L4vec(1)**2+L4vec(2)**2)
                            L4int=nint(L4)
                            if (L4>=lmin_filter .and. L4<=lmax) then
                                ! call getWins(-L*L3vec(1),-L*L4vec(1), L3vec,L3,L3int, L4vec,L4, L4int,  Win34, Win43)
                                call getWins(n_est,lmaxmax,-L*L3vec(1),-L*L4vec(1), L3vec,L3,L3int, L4vec,L4, L4int,  &
                                & CX, CTf, CEf, CXf, CBf,CTobs, CEobs, CBobs, Win34, Win43)

                                phiL_dot_L1=dot_product(PhiLVec,L1vec)
                                phiL_dot_L2=-dot_product(PhiLVec,L2vec)
                                phiL_dot_L3=dot_product(PhiLVec,L3vec)
                                phiL_dot_L4=-dot_product(PhiLVec,L4vec)

                                ! call getResponse(phiL_dot_L1,phiL_dot_L3, L1vec,real(L1,dp),L1, L3vec,L3, L3int,  f13, f31)
                                ! call getResponse(phiL_dot_L2,phiL_dot_L4, L2vec,L2,L2int, L4vec,L4, L4int,  f24, f42)
                                call getResponse(n_est,lmaxmax,phiL_dot_L1,phiL_dot_L3, L1vec,real(L1,dp),L1, L3vec,L3, &
                                & L3int, CT, CE, CX, CB, f13, f31)
                                call getResponse(n_est,lmaxmax,phiL_dot_L2,phiL_dot_L4, L2vec,L2,L2int, L4vec,L4, &
                                & L4int, CT, CE, CX, CB, f24, f42)

                                do est1=1,n_est
                                    ij=lumped_indices(:,est1)
                                    do est2=est1,n_est
                                        pq=lumped_indices(:,est2)
                                        this13 = responseFor(n_est,ij(1),pq(1),f13,f31)
                                        this24 = responseFor(n_est,ij(2),pq(2),f24,f42)
                                        tmp(est1,est2)=tmp(est1,est2)+this13*this24*Win34(est2)*Win12(est1)

                                        this13 = responseFor(n_est,ij(1),pq(2),f13,f31)
                                        this24 = responseFor(n_est,ij(2),pq(1),f24,f42)
                                        tmp(est1,est2)=tmp(est1,est2)+this13*this24*Win43(est2)*Win12(est1)
                                    end do
                                end do
                            end if
                        end if
                    end do
                    if (phiIx/=0) tmp=tmp*2 !integrate 0-Pi for phi_L1
                    fact = tmp* PhiL_phi_dphi* PhiL
                    do est1=1,n_est
                        matrixfact(est1,:) = fact(est1,:)*dPh
                    end do

                    !$OMP CRITICAL
                    matrixL1(phiLix,:,:)=matrixL1(phiLix,:,:) + matrixfact
                    !$OMP END CRITICAL
                    N1_PhiL= N1_PhiL + fact * Cphi(PhiL)*dPh
                end do
                do est1=1,n_est
                    N1_PhiL(est1,:)=N1_PhiL(est1,:)*Win12(est1)
                end do
                N1_L1 = N1_L1+N1_PhiL
            end do
            !$OMP END PARALLEL DO

            matrix(Lix,:,:,:)=matrix(Lix,:,:,:) + matrixL1*dphi*L1*dL
            N1= N1 + N1_L1 * dphi* L1*dL

        end do !L1

        do est1=1,n_est
            do est2=est1,n_est
                matrix(Lix,:,est1,est2) = matrix(Lix,:,est1,est2)*norms(L,est1)*norms(L,est2) / (twopi**4)
                N1(est1,est2) = norms(L,est1)*norms(L,est2)*N1(est1,est2) / (twopi**4)
                N1(est2,est1) = N1(est1,est2)
            end do
        end do

        end do
    end subroutine GetN1MatrixGeneral


    
  
    subroutine compute_n0(Cphi,lensedcmbfile,Tfile,Efile,Bfile,Xfile,lmin_filter,lmaxout,lmax,lmax_TT,lmaxmax,n0tt, &
         n0ee,n0eb,n0te,n0tb,L_min, Lstep)
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        ! Interface to python to compute N0 bias
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        implicit none
        integer, parameter :: DP = 8
        integer, parameter :: I4B = 4
        real(dp), parameter :: pi =  3.1415927, twopi=2*pi
        ! Order 1 2 3 = T E B
        ! Estimator order TT, EE, EB, TE, TB, BB
        integer(I4B), parameter :: n_est = 6
        integer, intent(in) ::  lmaxmax 
        integer, intent(in)      :: lmin_filter, lmaxout, lmax, lmax_TT,L_min, Lstep
        real, intent(in) :: lensedcmbfile(5,*)
        real, intent(in) :: Tfile(lmaxmax),Efile(lmaxmax),Bfile(lmaxmax),Xfile(lmaxmax)
        logical :: doCurl = .True.
        real(dp), intent(in) :: CPhi(lmaxmax)
        real(dp) :: CX(lmaxmax), CE(lmaxmax),CB(lmaxmax), CT(lmaxmax)
        real(dp) :: CXf(lmaxmax), CEf(lmaxmax),CBf(lmaxmax), CTf(lmaxmax)
        real(dp) :: NT(lmaxmax), NP(lmaxmax)
        real(dp) :: CEobs(lmaxmax), CTobs(lmaxmax), CBobs(lmaxmax)
        integer(I4B) :: LMin
        real(dp),  DIMENSION((lmaxout-L_min)/Lstep+1),intent(out) ::  n0tt,n0ee,n0eb,n0te,n0tb
 

        LMin = lmin_filter

        call ReadPowernum(lensedcmbfile,Tfile,Efile,Bfile,Xfile,lmax,lmaxmax,CT,CE,CB,CX,CTf,CEf,CBf,CXf)

        CTobs = CTf
        CEobs = CEf
        CBobs = CBf

        call getNorm( .false. , .false. ,.False.,lmin_filter,lmax,lmaxout,lmaxmax,n_est, CPhi,&
                            & CT, CE, CX, CB, CTf, CEf, CXf, CBf, CTobs, CEobs, CBobs, n0tt,n0ee,n0eb,n0te,n0tb,L_min, Lstep)

    end subroutine compute_n0
    
    subroutine ReadPowernum(Filename,Tfile,Efile,Bfile,Xfile,Lmax, lmaxmax,CT,CE,CB,CX,CTf,CEf,CBf,CXf)
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        ! Read input file and return lensed CMB spectra (and weights)
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        integer, parameter :: DP = 8
        integer, parameter :: I4B = 4
        real(dp), parameter :: pi =  3.1415927, twopi=2*pi
        real, intent(in) :: Filename(5,lmaxmax)
        real, intent(in) :: Tfile(lmaxmax),Efile(lmaxmax),Bfile(lmaxmax),Xfile(lmaxmax)
        integer, intent(in) :: Lmax, lmaxmax
        real(dp),intent(out) :: CX(lmaxmax), CE(lmaxmax),CB(lmaxmax), CT(lmaxmax)
        real(dp),intent(out) :: CXf(lmaxmax), CEf(lmaxmax),CBf(lmaxmax), CTf(lmaxmax)
        real(dp) :: CPhi(lmaxmax)
        integer L, status
        logical :: newform = .false.
        CTf=0
        CEf=0
        CBf=0
        CXf=0
        CT=0
        CE=0
        CB=0
        CX=0
        do L=2, lmax  !we keep the fiducial ones unchanged
        CTf(L) = Filename(2,L-1) 
        CEf(L) = Filename(3,L-1) 
        CBf(L) = Filename(4,L-1) 
        CXf(L) = Filename(5,L-1) 
        CT(L)=Tfile(L-1) 
        CE(L)=Efile(L-1) 
        CB(L)=Bfile(L-1) 
        CX(L)=Xfile(L-1) 
        
        end do
        !we will vary these ones for test purposes only vary CT used in f

    end subroutine ReadPowernum

    subroutine compute_n1(C_phi_phi,Lens_norm_phi,C_CMB_fiducial,C_TT_response, &
         C_EE_response,C_BB_response,C_TE_response,C_TT_total_filter,C_EE_total_filter, &
         C_BB_total_filter,lmin_CMB,kappa_Lmax,lmax_CMB,kappa_L_step,kappa_Lmin,n1theta, &
         n1ee,n1eb,n1te,n1tb,lmaxmax)
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        ! Interface to python to compute N1 bias
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        implicit none
        integer, parameter :: DP = 8
        integer, parameter :: I4B = 4
        real(dp), parameter :: pi =  3.1415927, twopi=2*pi
        ! Order 1 2 3 = T E B
        ! Estimator order TT, EE, EB, TE, TB, BB
        integer(I4B), parameter :: n_est = 6
        integer ::  lmaxmax 
        integer, intent(in)      :: lmin_CMB, kappa_Lmax, lmax_CMB,kappa_L_step,kappa_Lmin
        real, intent(in) :: C_CMB_fiducial(5,*)
        real(dp), intent(in) :: C_phi_phi(lmaxmax)
        real, intent(in) :: Lens_norm_phi(6,*)
        real, intent(in) :: C_TT_response(lmaxmax),C_EE_response(lmaxmax),C_BB_response(lmaxmax),C_TE_response(lmaxmax)
        real(dp) :: CX(lmaxmax), CE(lmaxmax),CB(lmaxmax), CT(lmaxmax)
        real(dp) :: CXf(lmaxmax), CEf(lmaxmax),CBf(lmaxmax), CTf(lmaxmax)
        real(dp), intent(in) :: C_EE_total_filter(lmaxmax), C_TT_total_filter(lmaxmax), C_BB_total_filter(lmaxmax)
        real(dp),dimension((kappa_Lmax-kappa_Lmin)/kappa_L_step+1), intent(out) ::  n1theta,n1ee,n1eb,n1te,n1tb

        call ReadPowernum(C_CMB_fiducial,C_TT_response,C_EE_response,C_BB_response,C_TE_response,&
             & lmax_CMB,lmaxmax,CT,CE,CB,CX,CTf,CEf,CBf,CXf)

        call GetN1General(Lens_norm_phi, .true. ,lmin_CMB,lmax_CMB,kappa_Lmax,lmaxmax,n_est, C_phi_phi,&
                            & CT, CE, CX, CB, CTf, CEf, CXf, CBf, C_TT_total_filter, C_EE_total_filter,&
                            & C_BB_total_filter, n1theta,n1ee,n1eb,n1te,n1tb,kappa_L_step,kappa_Lmin)

    end subroutine compute_n1

    subroutine compute_n1_derivatives(CPhi,normarray,lensedcmbfile,&
        & lmin_filter,lmaxout,lmax,lmax_TT,lmaxmax,Lstep,L_min)
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        ! Interface to python to compute
        ! derivatives of N1 bias wrt phiphi
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        implicit none
        integer, parameter :: DP = 8
        integer, parameter :: I4B = 4
        real(dp), parameter :: pi =  3.1415927, twopi=2*pi
        ! Order 1 2 3 = T E B
        ! Estimator order TT, EE, EB, TE, TB, BB
        integer(I4B), parameter :: n_est = 6
        integer, intent(in) ::  lmaxmax 
        integer, intent(in)      :: lmin_filter, lmaxout, lmax, lmax_TT,Lstep,L_min
        real, intent(in) :: normarray(6,*)
        real, intent(in) :: lensedcmbfile(5,*)
        real(dp), intent(in) :: CPhi(lmaxmax)
        real(dp) :: CX(lmaxmax), CE(lmaxmax),CB(lmaxmax), CT(lmaxmax)
        real(dp) :: CXf(lmaxmax), CEf(lmaxmax),CBf(lmaxmax), CTf(lmaxmax)
        real(dp) :: NT(lmaxmax), NP(lmaxmax)
        real(dp) :: CEobs(lmaxmax), CTobs(lmaxmax), CBobs(lmaxmax)
        integer(I4B) :: LMin
 

        LMin = L_min


        call ReadPowert(lensedcmbfile,lmax,lmaxmax,CT,CE,CB,CX,CTf,CEf,CBf,CXf)

        CTobs = CT
        CEobs = CE
        CBobs = CB

        call GetN1MatrixGeneral( normarray,.true. ,lmin_filter,lmax,lmaxout,lmaxmax,n_est, CPhi,&
                            & CT, CE, CX, CB, CTf, CEf, CXf, CBf, CTobs, CEobs, CBobs, Lstep,L_min)

    end subroutine compute_n1_derivatives


    subroutine compute_n0mix(Cphi,lensedcmbfile,Tfile,Efile,Bfile,Xfile,lmin_filter,lmaxout,lmax,lmax_TT,lmaxmax,&
    &  n0ttee,n0ttte,n0eete,n0ebtb,L_min, Lstep)
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        ! Interface to python to compute N0 bias
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        implicit none
        integer, parameter :: DP = 8
        integer, parameter :: I4B = 4
        real(dp), parameter :: pi =  3.1415927, twopi=2*pi
        ! Order 1 2 3 = T E B
        ! Estimator order TT, EE, EB, TE, TB, BB
        integer(I4B), parameter :: n_est = 6
        integer, intent(in) ::  lmaxmax 
        integer, intent(in)      :: lmin_filter, lmaxout, lmax, lmax_TT,L_min, Lstep
        real, intent(in) :: lensedcmbfile(5,*)
        real, intent(in) :: Tfile(lmaxmax),Efile(lmaxmax),Bfile(lmaxmax),Xfile(lmaxmax)
        logical :: doCurl = .True.
        real(dp), intent(in) :: CPhi(lmaxmax)
        real(dp) :: CX(lmaxmax), CE(lmaxmax),CB(lmaxmax), CT(lmaxmax)
        real(dp) :: CXf(lmaxmax), CEf(lmaxmax),CBf(lmaxmax), CTf(lmaxmax)
        real(dp) :: NT(lmaxmax), NP(lmaxmax)
        real(dp) :: CEobs(lmaxmax), CTobs(lmaxmax), CBobs(lmaxmax)
        integer(I4B) :: LMin
        real(dp),  DIMENSION((lmaxout-L_min)/Lstep+1),intent(out) ::  n0ttee,n0ttte,n0eete,n0ebtb 
 
        LMin = lmin_filter

        call ReadPowernum(lensedcmbfile,Tfile,Efile,Bfile,Xfile,lmax,lmaxmax,CT,CE,CB,CX,CTf,CEf,CBf,CXf)

        CTobs = CTf
        CEobs = CEf
        CBobs = CBf

        call getmixNorm( .false. , .false. ,.False.,lmin_filter,lmax,lmaxout,lmaxmax,n_est, CPhi,&
                            & CT, CE, CX, CB, CTf, CEf, CXf, CBf, CTobs, CEobs, CBobs, n0ttee,&
                        & n0ttte,n0eete,n0ebtb,L_min, Lstep)

    end subroutine compute_n0mix

   subroutine GetN1mix(normarray,sampling,lmin_filter,lmax,lmaxout,lmaxmax,n_est, CPhi,&
                        & CT, CE, CX, CB, CTf, CEf, CXf, CBf, CTobs, CEobs, CBobs, n1ttee,&
                        &n1tteb,n1ttte,n1tttb,n1eeeb,n1eete,n1eetb,n1ebte,n1ebtb,n1tetb,Lstep,L_min)
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        ! Main routine to compute N1 bias.
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        integer, parameter :: DP = 8
        integer, parameter :: I4B = 4
        real(dp), parameter :: pi =  3.1415927, twopi=2*pi

        integer,  intent(in) :: lmin_filter,lmax,lmaxout,lmaxmax,n_est,Lstep,L_min
        real(dp), intent(in) :: CPhi(lmaxmax)
        real(dp), intent(in) :: CX(lmaxmax), CE(lmaxmax),CB(lmaxmax), CT(lmaxmax)
        real(dp), intent(in) :: CXf(lmaxmax), CEf(lmaxmax),CBf(lmaxmax), CTf(lmaxmax)
        real(dp), intent(in) :: CEobs(lmaxmax), CTobs(lmaxmax), CBobs(lmaxmax)
        logical, intent(in) :: sampling

        integer(I4B), parameter :: i_TT=1,i_EE=2,i_EB=3,i_TE=4,i_TB=5, i_BB=6
        integer(I4B), parameter ::  dL = 20
        integer  :: lumped_indices(2,n_est)
        integer L, Lix, l1, nphi, phiIx, L2int,PhiL_nphi,PhiL_phi_ix,L3int,L4int
        integer PhiL
        real(dp) dphi,PhiL_phi_dphi
        real(dp) L1Vec(2), L2vec(2), LVec(2), L3Vec(2),L4Vec(2), phiLVec(2)
        real(dp) phi, PhiL_phi
        real(dP) L2, L4, L3
        real(dp) dPh
        real(dp) phiL_dot_L2, phiL_dot_L3, phiL_dot_L1, phiL_dot_L4
        real(dp) fact(n_est,n_est),tmp(n_est,n_est), N1(n_est,n_est), N1_L1(n_est,n_est),N1_PhiL(n_est,n_est)
        real(dp) Win12(n_est), Win34(n_est), Win43(n_est)
        real(dp) WinCurl12(n_est), WinCurl34(n_est), WinCurl43(n_est), tmpCurl(n_est,n_est), &
            factCurl(n_est,n_est),N1_PhiL_Curl(n_est,n_est), N1_L1_Curl(n_est,n_est),  N1_Curl(n_est,n_est)
        real(dp) f24(n_est), f13(n_est),f31(n_est), f42(n_est)
        integer file_id, nPhiSample,Phi_Sample(lmaxmax)
        integer file_id_Curl, PhiLix
        integer ij(2),pq(2), est1, est2
        real(dp) tmpPS, tmpPSCurl, N1_PhiL_PS, N1_PhiL_PS_Curl, N1_L1_PS_Curl, N1_L1_PS, N1_PS, N1_PS_Curl
        real(dp) dPhi_Sample(lmaxmax)
        real, intent(in) :: normarray(6,*)
        real(dp):: Norms(lmaxout,n_est)
        integer file_id_PS
        real(dp) this13, this24
        real(dp),  DIMENSION((lmaxout-L_min)/Lstep+1),intent(out) ::  n1ttee,n1tteb,n1ttte, &
             n1tttb,n1eeeb,n1eete,n1eetb,n1ebte,n1ebtb,n1tetb
        character(LEN=10) outtag
        CHARACTER(LEN=13) :: creturn

        lumped_indices = transpose(reshape((/ 1,2,2,1,1,3,1,2,3,2,3,3 /), (/ n_est, 2 /)  ))
        call SetPhiSampling(lmin_filter,lmaxout,lmaxmax,sampling,nPhiSample,Phi_Sample,dPhi_Sample)

        outtag = 'N1_All'
        
        call loadNorm(normarray,n_est,L_min,lmaxout, Lstep,Norms)
        Lix=0
        print *,'N1 computation (phi, curl, PS)'
        do L=L_min, lmaxout, Lstep
            creturn = achar(13)
            WRITE( * , 101 , ADVANCE='NO' ) creturn , int(real(L,kind=dp)/lmaxout*100.,kind=I4B)
            101     FORMAT( a , 'Progression : ',i7,' % ')
            Lix=Lix+1
            write(*,*) Lix
            Lvec(1) = L
            LVec(2)= 0
            N1=0

            do L1=max(lmin_filter,dL/2), lmax, dL
                N1_L1 = 0



                nphi=(2*L1+1)
                if (L1>3*dL) nphi=2*nint(L1/real(2*dL))+1
                dphi=(2*Pi/nphi)

                !$OMP PARALLEL DO default(shared), private(PhiIx,phi,PhiL_nphi, PhiL_phi_dphi, PhiL_phi_ix, PhiL_phi,PhiLix, dPh), &
                !$OMP private(L1vec,L2,L2vec, L2int,  L3, L3vec, L3int, L4, L4vec, L4int),&
                !$OMP private(tmp, Win12, Win34, Win43, fact,phiL_dot_L1, phiL_dot_L2, phiL_dot_L3, phiL_dot_L4), &
                !$OMP private(tmpCurl, WinCurl12, WinCurl34, WinCurl43, factCurl, N1_PhiL_Curl), &
                !$OMP private(f13, f31, f42, f24, ij, pq, est1, est2,this13,this24), &
                !$OMP private(tmpPS, tmpPSCurl, N1_PhiL_PS, N1_PhiL_PS_Curl), &

                !$OMP private(PhiL, PhiLVec, N1_PhiL), schedule(STATIC), reduction(+:N1_L1), reduction(+:N1_L1_Curl), &
                !$OMP reduction(+:N1_L1_PS), reduction(+:N1_L1_PS_Curl)
                !do phiIx= -(nphi-1)/2, (nphi-1)/2
                do phiIx=0,(nphi-1)/2 !
                phi= dphi*PhiIx
                L1vec(1)=L1*cos(phi)
                L1vec(2)=L1*sin(phi)
                L2vec = Lvec-L1vec
                L2=(sqrt(L2vec(1)**2+L2vec(2)**2))
                if (L2<lmin_filter .or. L2>lmax) cycle
                L2int=nint(L2)

                call getWins(n_est,lmaxmax,L*L1vec(1),L*L2vec(1), L1vec,real(L1,dp),L1, L2vec,L2, L2int,  &
                & CX, CTf, CEf, CXf, CBf,CTobs, CEobs, CBobs, Win12)


                N1_PhiL=0
                N1_PhiL_PS=0
                
                do PhiLIx = 1, nPhiSample
                    PhiL = Phi_Sample(PhiLIx)
                    dPh = dPhi_Sample(PhiLIx)
                    PhiL_nphi=(2*PhiL+1)
                    if (phiL>20) PhiL_nphi=2*nint(real(PhiL_nphi)/dPh/2)+1
                    PhiL_phi_dphi=(2*Pi/PhiL_nphi)
                    tmp=0
                    do PhiL_phi_ix=-(PhiL_nphi-1)/2, (PhiL_nphi-1)/2
                        PhiL_phi= PhiL_phi_dphi*PhiL_phi_ix
                        PhiLvec(1)=PhiL*cos(PhiL_phi)
                        PhiLvec(2)=PhiL*sin(PhiL_phi)
                        L3vec= PhiLvec - L1vec
                        L3 = sqrt(L3vec(1)**2+L3vec(2)**2)
                        if (L3>=lmin_filter .and. L3<=lmax) then
                            L3int = nint(L3)
                            L4vec = -Lvec-L3vec
                            L4 = sqrt(L4vec(1)**2+L4vec(2)**2)
                            L4int=nint(L4)
                            if (L4>=lmin_filter .and. L4<=lmax) then
                                call getWins(n_est,lmaxmax,-L*L3vec(1),-L*L4vec(1), L3vec,L3,L3int, L4vec,L4, L4int,  &
                                & CX, CTf, CEf, CXf, CBf,CTobs, CEobs, CBobs, Win34, Win43)

                                phiL_dot_L1=dot_product(PhiLVec,L1vec)
                                phiL_dot_L2=-dot_product(PhiLVec,L2vec)
                                phiL_dot_L3=dot_product(PhiLVec,L3vec)
                                phiL_dot_L4=-dot_product(PhiLVec,L4vec)

                                call getResponse(n_est,lmaxmax,phiL_dot_L1,phiL_dot_L3, L1vec,real(L1,dp),L1, L3vec,L3, &
                                & L3int, CT, CE, CX, CB, f13, f31)
                                call getResponse(n_est,lmaxmax,phiL_dot_L2,phiL_dot_L4, L2vec,L2,L2int, L4vec,L4, &
                                & L4int, CT, CE, CX, CB, f24, f42)

                                do est1=1,n_est
                                    ij=lumped_indices(:,est1)
                                    do est2=est1,n_est
                                        pq=lumped_indices(:,est2)
                                        this13 = responseFor(n_est,ij(1),pq(1),f13,f31)
                                        this24 = responseFor(n_est,ij(2),pq(2),f24,f42)
                                        tmp(est1,est2)=tmp(est1,est2)+this13*this24*Win34(est2)
                                        !tmp(est1,est2)=tmp(est1,est2)+this13*this24
                                        !tmp(est1,est2)=tmp(est1,est2)+Win34(est2)
                               

                                        this13 = responseFor(n_est,ij(1),pq(2),f13,f31)
                                        this24 = responseFor(n_est,ij(2),pq(1),f24,f42)
                                        !tmp(est1,est2)=tmp(est1,est2)+this13*this24
                                        tmp(est1,est2)=tmp(est1,est2)+this13*this24*Win43(est2)
                                        !tmp(est1,est2)=tmp(est1,est2)+Win43(est2)
                                 
                                    end do
                                end do
                            end if
                        end if
                    end do
                    if (phiIx/=0) tmp=tmp*2 !integrate 0-Pi for phi_L1
                    fact = tmp* PhiL_phi_dphi* PhiL
                    N1_PhiL= N1_PhiL + fact * Cphi(PhiL)*dPh
    

                end do
                do est1=1,n_est
                    N1_PhiL(est1,:)=N1_PhiL(est1,:)*Win12(est1)
                    !N1_PhiL(est1,:)=N1_PhiL(est1,:)
    
                end do
                N1_L1 = N1_L1+N1_PhiL

            end do
            !$OMP END PARALLEL DO
            N1= N1 + N1_L1 * dphi* L1*dL


        end do
        do est1=1,n_est
            do est2=est1,n_est
                N1(est1,est2) = norms(L,est1)*norms(L,est2)*N1(est1,est2) / (twopi**4)
                N1(est2,est1) = N1(est1,est2)
            end do
        
        end do
        
        ! write(file_id,'(1I5)',advance='NO') L
        ! call WriteMatrixLine(file_id, N1,n_est)


        ! print *,L, N1(i_TT,i_TT), N1(i_eb,i_eb), N1(i_eb, i_ee)
        ! print *, 'Psi',L, N1_Curl(i_TT,i_TT), N1_Curl(i_tb,i_eb), N1_Curl(i_eb, i_ee)
        ! print *, 'PS', L, N1_PS, N1_PS_Curl
        
        n1ttee(Lix)=N1(1,2)
        n1tteb(Lix)=N1(1,3)
        n1ttte(Lix)=N1(1,4)
        n1tttb(Lix)=N1(1,5)
        n1eeeb(Lix)=N1(2,3)
        n1eete(Lix)=N1(2,4)
        n1eetb(Lix)=N1(2,5)
        n1ebte(Lix)=N1(3,4)
        n1ebtb(Lix)=N1(3,5)
        n1tetb(Lix)=N1(4,5)
        end do
        print *,''
        ! close(file_id)

    end subroutine GetN1mix

    subroutine compute_n1mix(CPhi,Lens_norm_phi,C_CMB_fiducial,C_TT_response,&
         & C_EE_response,C_BB_response,C_TE_response,C_TT_total_filter,C_EE_total_filter,&
         & C_BB_total_filter,lmin_CMB,kappa_Lmax,lmax_CMB,kappa_L_step,kappa_Lmin,&
         & n1ttee,n1tteb,n1ttte,n1tttb,n1eeeb,n1eete,n1eetb,n1ebte,n1ebtb,n1tetb,lmaxmax)
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        ! Interface to python to compute N1 bias
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        implicit none
        integer, parameter :: DP = 8
        integer, parameter :: I4B = 4
        real(dp), parameter :: pi =  3.1415927, twopi=2*pi
        ! Order 1 2 3 = T E B
        ! Estimator order TT, EE, EB, TE, TB, BB
        integer(I4B), parameter :: n_est = 6
        integer, intent(in) ::  lmaxmax 
        integer, intent(in)      :: lmin_CMB, kappa_Lmax, lmax_CMB,kappa_L_step,kappa_Lmin
        real(dp), intent(in) :: CPhi(lmaxmax)
        real, intent(in) :: C_CMB_fiducial(5,*)
        real, intent(in) :: Lens_norm_phi(6,*)
        real, intent(in) :: C_TT_response(lmaxmax),C_EE_response(lmaxmax),C_BB_response(lmaxmax),C_TE_response(lmaxmax)
        real(dp) :: CX(lmaxmax), CE(lmaxmax),CB(lmaxmax), CT(lmaxmax),CTm(lmaxmax)
        real(dp) :: CXf(lmaxmax), CEf(lmaxmax),CBf(lmaxmax), CTf(lmaxmax)
        real(dp) :: CEobs(lmaxmax), CTobs(lmaxmax), CBobs(lmaxmax)
        real(dp), intent(in) :: C_EE_total_filter(lmaxmax), C_TT_total_filter(lmaxmax), C_BB_total_filter(lmaxmax)
        real(dp),dimension((kappa_Lmax-kappa_Lmin)/kappa_L_step+1), intent(out) ::  n1ttee,n1tteb, &
             n1ttte,n1tttb,n1eeeb,n1eete,n1eetb,n1ebte,n1ebtb,n1tetb
        

        call ReadPowernum(C_CMB_fiducial,C_TT_response,C_EE_response,C_BB_response,C_TE_response,&
             & lmax_CMB,lmaxmax,CT,CE,CB,CX,CTf,CEf,CBf,CXf)

        call GetN1mix(Lens_norm_phi, .true. ,lmin_CMB,lmax_CMB,kappa_Lmax,lmaxmax,n_est, CPhi,&
                            & CT, CE, CX, CB, CTf, CEf, CXf, CBf, C_TT_total_filter, C_EE_total_filter, &
                            & C_BB_total_filter, n1ttee,n1tteb,n1ttte,n1tttb,n1eeeb,n1eete, &
                            n1eetb,n1ebte,n1ebtb,n1tetb,kappa_L_step,kappa_Lmin)

    end subroutine compute_n1mix












end module

module checkproc
  !$ use omp_lib
  implicit none
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  ! Module to check if you are correctly using openmp capability
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  public :: get_threads

contains

  function get_threads() result(nt)
    integer :: nt

    nt = 0
    !$ nt = omp_get_max_threads()

  end function get_threads

end module checkproc
! Copyright (C) 2016 Lewis, Peloton  
