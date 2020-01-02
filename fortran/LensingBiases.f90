! Copyright (C) 2016 Lewis, Peloton
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Fortran script to compute CMB weak lensing biases (N0, N1)
! and derivatives. f2py friendly.
! Authors: Original script by Antony Lewis, adapted by Julien Peloton.
! Contact: j.peloton@sussex.ac.uk
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
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

    subroutine NoiseInit(AN,ANP,noise_fwhm_deg,lmax,lmax_TT,lcorr_TT,lmaxmax,NT,NP)
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        ! Compute noise power spectra (temp and polar)
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        integer, parameter :: DP = 8
        integer, parameter :: I4B = 4
        real(dp), parameter :: pi =  3.1415927, twopi=2*pi
        integer, intent(in) :: lmax, lmax_TT,lcorr_TT,lmaxmax
        real(dp), intent(in) ::noise_fwhm_deg
        real(dp),dimension(lmaxmax), intent(in) :: AN, ANP 
        real(dp), intent(out) :: NT(lmaxmax), NP(lmaxmax)
        real(dp) xlc, sigma2
        integer l

        xlc= 180*sqrt(8.*log(2.))/pi
        sigma2 = (noise_fwhm_deg/xlc)**2
        do l=2, lmax
            NT(L) = AN(l)*exp(l*(l+1)*sigma2)
            if (l>lmax_TT) NT(L) = NT(L) + ( 0.000001*pi/180/60.)**2 *exp(l*(l+1)*(15./60./xlc)**2)
            if (l<lcorr_TT) NT(L) = NT(L) * (1 + (real(lcorr_TT,dp)/real(l,dp))**4)
            NP(L) = ANP(l)*exp(l*(l+1)*sigma2)
        end do

    end subroutine NoiseInit

    subroutine ReadPhiPhi(phifile,Lmax,lmaxmax,CPhi)
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        ! Read input file and return lensing potential power-spectrum
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        integer, parameter :: DP = 8
        integer, parameter :: I4B = 4
        real(dp), parameter :: pi =  3.1415927, twopi=2*pi
        integer, intent(in) :: Lmax, lmaxmax
        real(dp), intent(out) :: CPhi(lmaxmax)
        real(dp), intent(in) :: phifile(lmaxmax)
        integer file_id
        character(LEN=1024) InLine
        integer L, status
        real(dp) T, E, B, TE, phi

        CPhi=0
        do L=1, lmaxmax
        CPhi(L) = phifile(L)* twopi/real(L*(L+1),dp)**2
        end do

    end subroutine ReadPhiPhi

    subroutine ReadPower(Filename,Lmax, lmaxmax,CT,CE,CB,CX,CTf,CEf,CBf,CXf)
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        ! Read input file and return lensed CMB spectra (and weights)
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        integer, parameter :: DP = 8
        integer, parameter :: I4B = 4
        real(dp), parameter :: pi =  3.1415927, twopi=2*pi
        character(LEN=*), intent(in) :: filename
        integer, intent(in) :: Lmax, lmaxmax
        real(dp),intent(out) :: CX(lmaxmax), CE(lmaxmax),CB(lmaxmax), CT(lmaxmax)
        real(dp),intent(out) :: CXf(lmaxmax), CEf(lmaxmax),CBf(lmaxmax), CTf(lmaxmax)
        real(dp) :: CPhi(lmaxmax)
        integer file_id
        character(LEN=1024) InLine
        integer L, status
        real(dp) T, E, B, TE, Phi
        logical :: newform = .false.

        open(file=Filename, newunit = file_id, form='formatted', status='old', iostat=status)
        if (status/=0) stop 'error opening Cl'
        CT=0
        CE=0
        CB=0
        CX=0
        do
            read(file_id, '(a)', iostat=status) InLine
            if (status/=0) exit
            if (InLine=='') cycle
            if (InLIne(1:1)=='#') then
                !!  newform = .true.
                cycle
            end if
            if (newform) then
                read(InLine,*, iostat=status)  l, T, E, TE, Phi
                B=0
                if (status/=0) stop 'error reading power'
            else
                read(InLine,*, iostat=status)  l, T, E, B , TE
                if (status/=0) then
                    B=1
                    read(InLine,*, iostat=status)  l, T, E, TE
                end if
            end if
            if (L> Lmax) exit
            if (newform .and. L>=1) then
                CPhi(L) = phi * twopi/real(L*(L+1),dp)**2
            end if
            if (L<2) cycle
            CT(L) = T * twopi/(l*(l+1))
            CE(L) = E * twopi/(l*(l+1))
            CB(L) = B * twopi/(l*(l+1))
            CX(L) = TE * twopi/(l*(l+1))
        end do
        CTf=CT
        CEf=CE
        CBf=CB
        CXf=CX
        close(file_id)

    end subroutine ReadPower
    !
    subroutine WriteMatrixLine(file_id,N0,n_est)
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        ! Write matrix in a .dat file
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        integer, parameter :: DP = 8
        integer, parameter :: I4B = 4
        integer(I4B), intent(in) :: file_id,n_est
        real(dp),intent(in) :: N0(n_est,n_est)
        integer i,j

        do i=1, n_est
            do j=1,n_est
                write (file_id,'(1E16.6)',Advance='NO') N0(i,j)
            end do
        end do
        write(file_id,'(a)') ''

    end subroutine WriteMatrixLine
    !
    subroutine WriteMatrix(name, vartag, dir,matrix,lmin_filter, lmaxout, lmaxmax,Lstep,nPhiSample,Phi_Sample)
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        ! Write matrix in a .dat file
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        integer, parameter :: DP = 8
        integer, parameter :: I4B = 4
        real(dp), parameter :: pi =  3.1415927, twopi=2*pi
        integer(I4B), intent(in) :: nPhiSample,lmin_filter, lmaxout, Lstep,lmaxmax
        integer(I4B), intent(in) :: Phi_Sample(lmaxmax)
        character(LEN=*), intent(in) :: name, vartag, dir
        real(dp), intent(in) :: matrix(:,:)
        integer Lix, L, file_id, PhiLix, L1

        Lix=0
        open(file=trim(dir)//'/'//trim(name)//trim(vartag)//'_matrix.dat', newunit = file_id, form='formatted', status='replace')

        write (file_id, '(1I6)', advance='NO') 0
        do PhiLix=1, nPhiSample
            write (file_id, '(1I16)', advance='NO') Phi_Sample(PhiLIx)
        end do
        write(file_id,'(a)') ''
        do L=lmin_filter, lmaxout, Lstep
            Lix=Lix+1
            write (file_id, '(1I6)', advance='NO') L
            do PhiLix=1, nPhiSample
                L1 = Phi_Sample(PhiLIx)
                write (file_id, '(1E16.6)', advance='NO') matrix(Lix,PhiLix)/((L1*(L1+1.d0))**2/twopi) * (L*(L+1.d0))**2/twopi
            end do
            write(file_id,'(a)') ''
        end do
        close(file_id)

    end subroutine WriteMatrix
    !
    subroutine getNorm(WantIntONly,sampling,doCurl,lmin_filter,lmax,lmaxout,lmaxmax,n_est, CPhi,&
                        & CT, CE, CX, CB, CTf, CEf, CXf, CBf, CTobs, CEobs, CBobs, dir,vartag)
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        ! Main routine to compute N0 bias.
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        integer, parameter :: DP = 8
        integer, parameter :: I4B = 4
        real(dp), parameter :: pi =  3.1415927, twopi=2*pi

        integer,  intent(in) :: lmin_filter,lmax,lmaxout,lmaxmax,n_est
        real(dp), intent(in) :: CPhi(lmaxmax)
        real(dp), intent(in) :: CX(lmaxmax), CE(lmaxmax),CB(lmaxmax), CT(lmaxmax)
        real(dp), intent(in) :: CXf(lmaxmax), CEf(lmaxmax),CBf(lmaxmax), CTf(lmaxmax)
        real(dp), intent(in) :: CEobs(lmaxmax), CTobs(lmaxmax), CBobs(lmaxmax)
        character(LEN=50), intent(in) :: dir
        character(LEN=50), intent(in) :: vartag
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
                open(file=trim(dir)//'/'//'N0'//trim(vartag)//'_Curl.dat', newunit = file_id, form='formatted', status='replace')
            else
                print *,'N0 computation (phi)'
                open(file=trim(dir)//'/'//'N0'//trim(vartag)//'.dat', newunit = file_id, form='formatted', status='replace')
            end if


            do Lix =1, nPhiSample
                creturn = achar(13)
                WRITE( * , 101 , ADVANCE='NO' ) creturn , int(real(Lix,kind=dp)/nPhiSample*100.,kind=I4B)
                101     FORMAT( a , 'Progression : ',i7,' % ')
                L = Phi_Sample(Lix)
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

                    if (isCurl) then
                        call getResponseFull(n_est,lmaxmax,L1vec(2)*L,L2vec(2)*L, L1vec,real(L1,dp),&
                        & L1, L2vec,L2, L2int,f12,f21, CT, CE, CX, CB)
                        call getWins(n_est,lmaxmax,L1vec(2)*L,L2vec(2)*L, L1vec,real(L1,dp),&
                        & L1, L2vec,L2, L2int, CX, CTf, CEf, CXf, CBf,CTobs, CEobs, CBobs,Win12, Win21)
                    else
                        call getResponseFull(n_est,lmaxmax,L1vec(1)*L,L2vec(1)*L, L1vec,real(L1,dp),&
                        & L1, L2vec,L2, L2int,f12,f21, CT, CE, CX, CB)
                        call getWins(n_est,lmaxmax,L1vec(1)*L,L2vec(1)*L, L1vec,real(L1,dp),&
                        & L1, L2vec,L2, L2int, CX, CTf, CEf, CXf, CBf,CTobs, CEobs, CBobs,Win12, Win21)
                    end if

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

            do i=1,n_est
                Norms(L,i) = N0(i,i)
            end do

            norm = real(L*(L+1),dp)**2/twopi
            write (file_id,'(1I5, 1E16.6)',Advance='NO') L, CPhi(L)*norm

            call WriteMatrixLine(file_id,N0,n_est)

            ! print *, L, CPhi(L)*norm, N0(i_TT,i_TT)*norm
        end do
        close(file_id)
        end do
        print *,''

    end subroutine getNorm
    !
    subroutine loadNorm(n_est,lmin_filter, lmaxmax,lmaxout, Lstep,Norms,vartag,dir)
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        ! Load N0 bias from the disk
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        integer, parameter :: DP = 8
        integer, parameter :: I4B = 4
        real(dp), parameter :: pi =  3.1415927, twopi=2*pi

        integer,  intent(in) :: lmin_filter,lmaxmax,lmaxout,n_est,Lstep
        real(dp),intent(out) :: Norms(lmaxmax,n_est)
        character(LEN=50), intent(in) :: dir
        character(LEN=50), intent(in) :: vartag
        integer  file_id, L,i
        real(dp) ell,TT,EE,EB,TE,TB,BB
        real(dp) N0(n_est,n_est), dum !array size 5 i.e n_est=5

        !open(file=trim(dir)//'/'//'N0'//trim(vartag)//'.dat', newunit = file_id, form='formatted', status='old')
        open(file=trim(dir)//'/'//'N0'//trim(vartag)//'.txt', newunit = file_id, form='formatted', status='old')
        do L=lmin_filter, lmaxout, Lstep
            !read(file_id,*) ell, dum, N0
            read(file_id,*) ell, TT, EE,EB,TE,TB,BB
            if (L/=ell) stop 'wrong N0 file'
            !do i=1,n_est
                !Norms(L,i) = N0(i,i)
            !end do
            !read(file_id,*) ell,
            
            
            
            Norms(L,1) = TT
            Norms(L,2)=EE
            Norms(L,3)=EB
            Norms(L,4)=TE
            Norms(L,5)=TB
            Norms(L,6)=BB
                
                
    
        end do
        close(file_id)
    end subroutine loadNorm
    !
    !
    
    subroutine loadNormCurl(n_est,lmin_filter, lmaxmax,lmaxout, Lstep,NormsCurl,vartag,dir)
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        ! Load N0 bias (curl) from the disk
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        integer, parameter :: DP = 8
        integer, parameter :: I4B = 4
        real(dp), parameter :: pi =  3.1415927, twopi=2*pi

        integer,  intent(in) :: lmin_filter,lmaxmax,lmaxout,n_est,Lstep
        real(dp),intent(out) :: NormsCurl(lmaxmax,n_est)
        character(LEN=50), intent(in) :: dir
        character(LEN=50), intent(in) :: vartag
        integer ell, file_id, L,i
        real(dp) N0(n_est,n_est), dum



    end subroutine loadNormCurl
    
    !
    subroutine WriteRanges(LMin, lmaxout,lmaxmax, Lstep,Phi_Sample,dPhi_Sample,nPhiSample,name,vartag,dir)
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        ! Write sampling, bins, etc on the disk
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        integer, parameter :: DP = 8
        integer PhiLix, L, file_id
        integer, intent(in) :: LMin, lmaxout,lmaxmax, Lstep
        character(LEN=*), intent(in) :: name,vartag,dir
        integer, intent(in) :: nPhiSample,Phi_Sample(lmaxmax)
        real(dp),intent(in) :: dPhi_Sample(lmaxmax)

        open(file=trim(dir)//'/'//trim(name)//trim(vartag)//'_Lin.dat', newunit = file_id, form='formatted', status='replace')
        do PhiLix=1, nPhiSample
            write (file_id, '(1I6,1E16.6)') Phi_Sample(PhiLix), dPhi_Sample(PhiLix)
        end do
        close(file_id)

        open(file=trim(dir)//'/'//trim(name)//trim(vartag)//'_Lout.dat', newunit = file_id, form='formatted', status='replace')
        do L=LMin, lmaxout, Lstep
            write (file_id, *) L
        end do
        close(file_id)

    end subroutine WriteRanges
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
    
    subroutine getResponseDerivatives(n_est,lmaxmax,L1_dot_L2,L1_dot_L3,L1_dot_L4, L2vec,L2,L2int,L3vec,L3, L3int,L4vec,L4, L4int,f12,CE,CB)
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        ! Derivatives of f wrt to C_ls cmb
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        integer, parameter :: DP = 8
        integer, parameter :: I4B = 4
        real(dp), parameter :: pi =  3.1415927, twopi=2*pi
        integer, intent(in) :: n_est,lmaxmax,L2int, L3int,L4int
        real(dp), intent(in) :: L2vec(2),L3vec(2),L4vec(2),L2,L3,L4,L1_dot_L2,L1_dot_L3,L1_dot_L4
        real(dp), intent(in) :: CE(lmaxmax),CB(lmaxmax)
        real(dp),intent(out) :: f12(n_est,n_est)
        real(dp) cosfac34,cosfac24, sin2L4L3, cos2L4L3,sin2L2L4,cos2L2L4
        integer(I4B), parameter :: i_TT=1,i_EE=2,i_EB=3,i_TE=4,i_TB=5, i_BB=6

        cosfac34= dot_product(L4vec,L3vec)/real(L4*L3,dp)
        cos2L4L3 =2*cosfac34**2-1
        sin2L4L3=  2*cosfac34*(L4vec(2)*L3vec(1)-L4vec(1)*L3vec(2))/(L4*L3)
        cosfac24= dot_product(L4vec,L2vec)/real(L4*L2,dp)
        cos2L2L4 =2*cosfac24**2-1
        sin2L2L4=  2*cosfac24*(L2vec(2)*L4vec(1)-L2vec(1)*L4vec(2))/(L2*L4)
        f12(i_TT,i_TT)= 2*L1_dot_L4
        f12(i_EE,i_EE)= 2*L1_dot_L4       
        f12(i_BB,i_BB)= 2*L1_dot_L4
        f12(i_TE,i_TE) = L1_dot_L4*(1+cos2L4L3)
        f12(i_TB,i_TE)=L1_dot_L4*sin2L4L3
        f12(i_EB,i_EE) = (L1_dot_L4+CB(L3int)*L1_dot_L3)*sin2L4L3
        f12(i_EB,i_BB)=(L1_dot_L4+CE(L2int)*L1_dot_L2)*sin2L2L4

    end subroutine getResponseDerivatives
    !
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
    subroutine GetN1General(sampling,lmin_filter,lmax,lmaxout,lmaxmax,n_est, CPhi,&
                        & CT, CE, CX, CB, CTf, CEf, CXf, CBf, CTobs, CEobs, CBobs, dir,vartag)
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        ! Main routine to compute N1 bias.
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        integer, parameter :: DP = 8
        integer, parameter :: I4B = 4
        real(dp), parameter :: pi =  3.1415927, twopi=2*pi

        integer,  intent(in) :: lmin_filter,lmax,lmaxout,lmaxmax,n_est
        real(dp), intent(in) :: CPhi(lmaxmax)
        real(dp), intent(in) :: CX(lmaxmax), CE(lmaxmax),CB(lmaxmax), CT(lmaxmax)
        real(dp), intent(in) :: CXf(lmaxmax), CEf(lmaxmax),CBf(lmaxmax), CTf(lmaxmax)
        real(dp), intent(in) :: CEobs(lmaxmax), CTobs(lmaxmax), CBobs(lmaxmax)
        character(LEN=50), intent(in) :: dir
        character(LEN=50), intent(in) :: vartag
        logical, intent(in) :: sampling

        integer(I4B), parameter :: i_TT=1,i_EE=2,i_EB=3,i_TE=4,i_TB=5, i_BB=6
        integer(I4B), parameter :: Lstep = 20, dL = 20
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
        real(dp) Norms(lmaxmax,n_est), NormsCurl(lmaxmax,n_est)
        integer file_id_PS
        real(dp) this13, this24
        character(LEN=10) outtag
        CHARACTER(LEN=13) :: creturn

        lumped_indices = transpose(reshape((/ 1,2,2,1,1,3,1,2,3,2,3,3 /), (/ n_est, 2 /)  ))

        call SetPhiSampling(lmin_filter,lmaxout,lmaxmax,sampling,nPhiSample,Phi_Sample,dPhi_Sample)

        outtag = 'N1_All'
        call loadNorm(n_est,lmin_filter, lmaxmax,lmaxout, Lstep,Norms,vartag,dir)
        !call loadNormCurl(n_est,lmin_filter, lmaxmax,lmaxout, Lstep,NormsCurl,vartag,dir)

        call WriteRanges(lmin_filter, lmaxout,lmaxmax, Lstep,Phi_Sample,dPhi_Sample,nPhiSample,outtag,vartag,dir)
        open(file=trim(dir)//'/'//trim(outtag)//trim(vartag)//'.dat', newunit = file_id, form='formatted',&
        & status='replace')
        !open(file=trim(dir)//'/'//trim(outtag)//trim(vartag)//'_Curl.dat', newunit = file_id_Curl, form='formatted',&
        !& status='replace')
        open(file=trim(dir)//'/'//trim(outtag)//trim(vartag)//'_PS.dat', newunit = file_id_PS, form='formatted',&
        & status='replace')

        Lix=0
        print *,'N1 computation (phi, curl, PS)'
        do L=lmin_filter, lmaxout, Lstep
            creturn = achar(13)
            WRITE( * , 101 , ADVANCE='NO' ) creturn , int(real(L,kind=dp)/lmaxout*100.,kind=I4B)
            101     FORMAT( a , 'Progression : ',i7,' % ')
            Lix=Lix+1
            Lvec(1) = L
            LVec(2)= 0
            N1=0
            !N1_Curl = 0
            N1_PS=0
            !N1_PS_Curl=0
            do L1=max(lmin_filter,dL/2), lmax, dL
                N1_L1 = 0
                !N1_L1_Curl = 0
                N1_L1_PS = 0
                !N1_L1_PS_Curl = 0


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
                call getWins(n_est,lmaxmax,L*L1vec(2),L*L2vec(2), L1vec,real(L1,dp),L1, L2vec,L2, L2int,  &
                & CX, CTf, CEf, CXf, CBf,CTobs, CEobs, CBobs, WinCurl12)

                N1_PhiL=0
                N1_PhiL_Curl=0
                N1_PhiL_PS=0
                N1_PhiL_PS_Curl=0
                do PhiLIx = 1, nPhiSample
                    PhiL = Phi_Sample(PhiLIx)
                    dPh = dPhi_Sample(PhiLIx)
                    PhiL_nphi=(2*PhiL+1)
                    if (phiL>20) PhiL_nphi=2*nint(real(PhiL_nphi)/dPh/2)+1
                    PhiL_phi_dphi=(2*Pi/PhiL_nphi)
                    tmp=0
                    tmpCurl=0
                    tmpPS = 0
                    tmpPSCurl = 0
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
                                call getWins(n_est,lmaxmax,-L*L3vec(2),-L*L4vec(2), L3vec,L3,L3int, L4vec,L4, L4int,  &
                                & CX, CTf, CEf, CXf, CBf,CTobs, CEobs, CBobs, WinCurl34, WinCurl43)

                                phiL_dot_L1=dot_product(PhiLVec,L1vec)
                                phiL_dot_L2=-dot_product(PhiLVec,L2vec)
                                phiL_dot_L3=dot_product(PhiLVec,L3vec)
                                phiL_dot_L4=-dot_product(PhiLVec,L4vec)

                                call getResponse(n_est,lmaxmax,phiL_dot_L1,phiL_dot_L3, L1vec,real(L1,dp),L1, L3vec,L3, &
                                & L3int,CT, CE, CX, CB, f13, f31)
                                call getResponse(n_est,lmaxmax,phiL_dot_L2,phiL_dot_L4, L2vec,L2,L2int, L4vec,L4, &
                                & L4int, CT, CE, CX, CB, f24, f42)

                                do est1=1,n_est
                                    ij=lumped_indices(:,est1)
                                    do est2=est1,n_est
                                        pq=lumped_indices(:,est2)
                                        this13 = responseFor(n_est,ij(1),pq(1),f13,f31)
                                        this24 = responseFor(n_est,ij(2),pq(2),f24,f42)
                                        tmp(est1,est2)=tmp(est1,est2)+this13*this24*Win34(est2)
                                        !tmpCurl(est1,est2)=tmpCurl(est1,est2)+this13*this24*WinCurl34(est2)

                                        this13 = responseFor(n_est,ij(1),pq(2),f13,f31)
                                        this24 = responseFor(n_est,ij(2),pq(1),f24,f42)
                                        tmp(est1,est2)=tmp(est1,est2)+this13*this24*Win43(est2)
                                        !tmpCurl(est1,est2)=tmpCurl(est1,est2)+this13*this24*WinCurl43(est2)
                                    end do
                                end do
                                tmpPS = tmpPS + Win43(1) + Win34(1)
                                !tmpPSCurl = tmpPSCurl + WinCurl43(1) + WinCurl34(1)
                            end if
                        end if
                    end do
                    if (phiIx/=0) tmp=tmp*2 !integrate 0-Pi for phi_L1
                    !if (phiIx/=0) tmpCurl=tmpCurl*2 !integrate 0-Pi for phi_L1
                    if (phiIx/=0) tmpPS=tmpPS*2 !integrate 0-Pi for phi_L1
                    !if (phiIx/=0) tmpPSCurl=tmpPSCurl*2 !integrate 0-Pi for phi_L1
                    fact = tmp* PhiL_phi_dphi* PhiL
                    !factCurl = tmpCurl* PhiL_phi_dphi* PhiL
                    N1_PhiL= N1_PhiL + fact * Cphi(PhiL)*dPh
                    !N1_PhiL_Curl= N1_PhiL_Curl + factCurl * Cphi(PhiL)*dPh

                    N1_PhiL_PS = N1_PhiL_PS + tmpPS* PhiL_phi_dphi* PhiL*dPh  !/ PhiL**2
                    !N1_PhiL_PS_Curl = N1_PhiL_PS_CUrl + tmpPSCurl* PhiL_phi_dphi* PhiL*dPh  !/ PhiL**2

                end do
                do est1=1,n_est
                    N1_PhiL(est1,:)=N1_PhiL(est1,:)*Win12(est1)
                    !N1_PhiL_Curl(est1,:)=N1_PhiL_Curl(est1,:)*WinCurl12(est1)
                end do
                N1_L1 = N1_L1+N1_PhiL
                !N1_L1_Curl = N1_L1_Curl+N1_PhiL_Curl
                N1_L1_PS = N1_L1_PS + N1_PhiL_PS * Win12(1)
                !N1_L1_PS_Curl = N1_L1_PS_Curl + N1_PhiL_PS_Curl * WinCurl12(1)
            end do
            !$OMP END PARALLEL DO
            N1= N1 + N1_L1 * dphi* L1*dL
            !N1_Curl= N1_Curl + N1_L1_Curl * dphi* L1*dL
            N1_PS = N1_PS + N1_L1_PS *dphi*L1*dL
            !N1_PS_Curl = N1_PS_Curl + N1_L1_PS_Curl *dphi*L1*dL

        end do

        do est1=1,n_est
            do est2=est1,n_est
                N1(est1,est2) = norms(L,est1)*norms(L,est2)*N1(est1,est2) / (twopi**4)
                N1(est2,est1) = N1(est1,est2)
                !N1_Curl(est1,est2) = normsCurl(L,est1)*normsCurl(L,est2)*N1_Curl(est1,est2) / (twopi**4)
                !N1_Curl(est2,est1) = N1_Curl(est1,est2)
            end do
        end do
        N1_PS = norms(L,1)*norms(L,1)*N1_PS / (twopi**4)
        !N1_PS_Curl = normsCurl(L,1)*normsCurl(L,1)*N1_PS_Curl / (twopi**4)

        write(file_id,'(1I5)',advance='NO') L
        call WriteMatrixLine(file_id, N1,n_est)
        !write(file_id_Curl,'(1I5)',advance='NO') L
        !call WriteMatrixLine(file_id_Curl, N1_Curl,n_est)

        write(file_id_PS,*) L, N1_PS, N1_PS_Curl


        ! print *,L, N1(i_TT,i_TT), N1(i_eb,i_eb), N1(i_eb, i_ee)
        ! print *, 'Psi',L, N1_Curl(i_TT,i_TT), N1_Curl(i_tb,i_eb), N1_Curl(i_eb, i_ee)
        ! print *, 'PS', L, N1_PS, N1_PS_Curl

        end do
        print *,''
        close(file_id)
        !close(file_id_Curl)
        close(file_id_PS)

    end subroutine GetN1General
    !
    subroutine GetN1MatrixGeneral(sampling,lmin_filter,lmax,lmaxout,lmaxmax,n_est, CPhi,&
                        & CT, CE, CX, CB, CTf, CEf, CXf, CBf, CTobs, CEobs, CBobs, dir,vartag)
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        ! Main routine to compute N1 derivatives.
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        integer, parameter :: DP = 8
        integer, parameter :: I4B = 4
        real(dp), parameter :: pi =  3.1415927, twopi=2*pi

        integer,  intent(in) :: lmin_filter,lmax,lmaxout,lmaxmax,n_est
        real(dp), intent(in) :: CPhi(lmaxmax)
        real(dp), intent(in) :: CX(lmaxmax), CE(lmaxmax),CB(lmaxmax), CT(lmaxmax)
        real(dp), intent(in) :: CXf(lmaxmax), CEf(lmaxmax),CBf(lmaxmax), CTf(lmaxmax)
        real(dp), intent(in) :: CEobs(lmaxmax), CTobs(lmaxmax), CBobs(lmaxmax)
        character(LEN=50), intent(in) :: dir
        character(LEN=50), intent(in) :: vartag
        logical, intent(in) :: sampling

        integer(I4B), parameter :: i_TT=1,i_EE=2,i_EB=3,i_TE=4,i_TB=5, i_BB=6
        integer(I4B), parameter :: Lstep = 20, dL = 20
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
        real(dp) Norms(lmaxmax,n_est), NormsCurl(lmaxmax,n_est)
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
        allocate(matrix((lmaxout-lmin_filter)/Lstep+1,nPhiSample, n_est,n_est))
        allocate(matrixL1(nPhiSample,n_est,n_est))
        matrix=0
        call loadNorm(n_est,lmin_filter, lmaxmax,lmaxout, Lstep,Norms,vartag,dir) !load N0
        call WriteRanges(lmin_filter, lmaxout,lmaxmax, Lstep,Phi_Sample,dPhi_Sample,nPhiSample,outtag,vartag,dir)

        open(file=trim(dir)//'/'//trim(outtag)//trim(vartag)//'.dat', newunit = file_id, form='formatted', status='replace')

        Lix=0
        print *,'Derivatives of N1 computation'
        do L=lmin_filter, lmaxout, Lstep
            creturn = achar(13)
            WRITE( * , 101 , ADVANCE='NO' ) creturn , int(real(L,kind=dp)/lmaxout*100.,kind=I4B)
            101     FORMAT( a , 'Progression : ',i7,' % ')
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
                                
                             

                                do est1=1,1
                                    ij=lumped_indices(:,est1)
                                    do est2=est1,1
                                        pq=lumped_indices(:,est2)
                                        this13 = responseFor(n_est,ij(1),pq(1),f13,f31)
                                        this24 = responseFor(n_est,ij(2),pq(2),f24,f42)
                                        tmp(est1,est2)=tmp(est1,est2)+this13*this24*Win34(est2)

                                        this13 = responseFor(n_est,ij(1),pq(2),f13,f31)
                                        this24 = responseFor(n_est,ij(2),pq(1),f24,f42)
                                        tmp(est1,est2)=tmp(est1,est2)+this13*this24*Win43(est2)
                                    end do
                                end do
                            end if
                        end if
                    end do
                    if (phiIx/=0) tmp=tmp*2!integrate 0-Pi for phi_L1
                    fact = tmp* PhiL_phi_dphi* PhiL
        
                    do est1=1,1
                        matrixfact(est1,:) = fact(est1,:)*Win12(est1)*dPh
                    end do

                    !$OMP CRITICAL
                    matrixL1(phiLix,:,:)=matrixL1(phiLix,:,:) + matrixfact
                    !$OMP END CRITICAL
                   
                end do
          
            end do
            !!$OMP END PARALLEL DO

            matrix(Lix,:,:,:)=matrix(Lix,:,:,:) + matrixL1*dphi*L1*dL
            N1= N1 + N1_L1 * dphi* L1*dL

        end do !L1
        matrix(Lix,:,1,1)=matrix(Lix,:,1,1)*norms(L,1)*norms(L,1)/(twopi**4)


        ! print *, 'N1 L, TTTT, EBEB: ',L, N1(i_TT,i_TT), N1(i_eb,i_eb)
        write(file_id,'(1I5)',advance='NO') L
        call WriteMatrixLine(file_id,N1,n_est)
        end do
        close(file_id)
        outtag = 'N1_'//estnames(1)//estnames(1)
        call WriteMatrix(outtag, vartag,dir, matrix(:,:,1,1),lmin_filter, lmaxout, &
            & lmaxmax,Lstep,nPhiSample,Phi_Sample)
        print *,''

    end subroutine GetN1MatrixGeneral

   
    subroutine N1tt_tt(sampling,lmin_filter,lmax,lmaxout,lmaxmax,n_est, CPhi,&
                        & CT, CE, CX, CB, CTf, CEf, CXf, CBf, CTobs, CEobs, CBobs, dir,vartag)
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        ! N1 tt derivative wrt to cl_tt
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        integer, parameter :: DP = 8
        integer, parameter :: I4B = 4
        real(dp), parameter :: pi =  3.1415927, twopi=2*pi

        integer,  intent(in) :: lmin_filter,lmax,lmaxout,lmaxmax,n_est
        real(dp), intent(in) :: CPhi(lmaxmax)
        real(dp), intent(in) :: CX(lmaxmax), CE(lmaxmax),CB(lmaxmax), CT(lmaxmax)
        real(dp), intent(in) :: CXf(lmaxmax), CEf(lmaxmax),CBf(lmaxmax), CTf(lmaxmax)
        real(dp), intent(in) :: CEobs(lmaxmax), CTobs(lmaxmax), CBobs(lmaxmax)
        character(LEN=50), intent(in) :: dir
        character(LEN=50), intent(in) :: vartag
        logical, intent(in) :: sampling

        integer(I4B), parameter :: i_TT=1,i_EE=2,i_EB=3,i_TE=4,i_TB=5, i_BB=6
        integer(I4B), parameter :: Lstep = 20, dL = 20
        integer  :: lumped_indices(2,n_est)
        integer L, Lix, l1, nphi, phiIx, L2int,PhiL_nphi,PhiL_phi_ix,L3int,L4int,L1int,Lphi1int,Lphi2int,L5int,Lphi3int,Lphi4int
        integer PhiL
        real(dp) dphi,PhiL_phi_dphi
        real(dp) L1Vec(2), L2vec(2), LVec(2), L3Vec(2),L4Vec(2),L5Vec(2), phiLVec(2),Lphi1vec(2),Lphi2vec(2),Lphi3vec(2),Lphi4vec(2)
        real(dp) phi, PhiL_phi
        real(dP) L2, L4, L3,Lphi1,Lphi2,L5,Lphi3,Lphi4
        real(dp) dPh
        real(dp) phiL_dot_L2, phiL_dot_L3, phiL_dot_L1, phiL_dot_L4,phiL_dot_L5,phiL_dot_L6,phiL_dot_L7,phiL_dot_L8
        real(dp) fact(n_est,n_est),tmp(n_est,n_est), N1(n_est,n_est), N1_L1(n_est,n_est),N1_PhiL(n_est,n_est)
        real(dp) matrixfact(n_est,n_est)
        real(dp) Win12(n_est), Win34(n_est), Win43(n_est),Win45(n_est),Win56(n_est),Win89(n_est),Win65(n_est)
        real(dp) WinCurl12(n_est), WinCurl34(n_est), WinCurl43(n_est), tmpCurl(n_est,n_est), &
            factCurl(n_est,n_est),N1_PhiL_Curl(n_est,n_est), N1_L1_Curl(n_est,n_est),  N1_Curl(n_est,n_est)
        real(dp) f12(n_est), f34(n_est),f43(n_est), f21(n_est),f1(n_est),f2(n_est),Ff1(n_est),Ff2(n_est),f56(n_est),f65(n_est),f78(n_est),f87(n_est)
        integer file_id, nPhiSample,Phi_Sample(lmaxmax)
        integer file_id_Curl, PhiLix
        integer ij(2),pq(2), est1, est2
        real(dp) tmpPS, tmpPSCurl, N1_PhiL_PS, N1_PhiL_PS_Curl, N1_L1_PS_Curl, N1_L1_PS, N1_PS, N1_PS_Curl
        real(dp) dPhi_Sample(lmaxmax)
        real(dp) Norms(lmaxmax,n_est), NormsCurl(lmaxmax,n_est)
        integer file_id_PS
        real(dp) this12, this34
        real(dp), allocatable :: Matrix(:,:,:,:), MatrixL1(:,:,:)
        character(LEN=10) outtag
        CHARACTER(LEN=13) :: creturn
        character(2) :: estnames(n_est)
        estnames = ['TT','EE','EB','TE','TB','BB']

        lumped_indices = transpose(reshape((/ 1,2,2,1,1,3,1,2,3,2,3,3 /), (/ n_est, 2 /)  ))
        call SetPhiSampling(lmin_filter,lmaxout,lmaxmax,sampling,nPhiSample,Phi_Sample,dPhi_Sample)

        outtag = 'N1_All'
        
        allocate(matrix((lmaxout-lmin_filter)/Lstep+1,nPhiSample, 1,1))
        allocate(matrixL1(nPhiSample,1,1))
        matrix=0
        call loadNorm(n_est,lmin_filter, lmaxmax,lmaxout, Lstep,Norms,vartag,dir) !load N0
        call WriteRanges(lmin_filter, lmaxout,lmaxmax, Lstep,Phi_Sample,dPhi_Sample,nPhiSample,outtag,vartag,dir)
        open(file=trim(dir)//'/'//trim(outtag)//trim(vartag)//'.dat', newunit = file_id, form='formatted', status='replace')
        Lix=0
        print *,'Derivatives wrt cltt of N1 computation'
        do L=lmin_filter, lmaxout, Lstep   !Perform the derivative of N1(L) from lmin=2 to L output
            WRITE(*,*) L
            creturn = achar(13)
            WRITE( * , 101 , ADVANCE='NO' ) creturn , int(real(L,kind=dp)/lmaxout*100.,kind=I4B)
            101     FORMAT( a , 'Progression : ',i7,' % ')
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
                !!$OMP PARALLEL DO default(shared), private(PhiIx,phi,PhiL_nphi, PhiL_phi_dphi, PhiL_phi_ix, PhiL_phi,PhiLix, dPh), &
                !!$OMP private(L1vec,L2,L2vec, L2int,  L3, L3vec, L3int, L4, L4vec, L4int,Lphi1int,Lphi2int,Lphi1,Lphi2),&
                !!$OMP private(tmp, Win12, Win34, Win43, matrixfact,fact,phiL_dot_L1, phiL_dot_L2, phiL_dot_L3, phiL_dot_L4), &
                !!$OMP private(f12, f21, f34, f43, ij, pq,this12,this34), &
                !!$OMP private(PhiL, PhiLVec, N1_PhiL), schedule(STATIC), reduction(+:N1_L1)
                do phiIx=0,(nphi-1)/2
                phi= dphi*PhiIx
                L1vec(1)=L1*cos(phi)
                L1vec(2)=L1*sin(phi)
                L2vec = Lvec-L1vec  !L1+L2=L
                L2=(sqrt(L2vec(1)**2+L2vec(2)**2))
                if (L2<lmin_filter .or. L2>lmax) cycle
                L2int=nint(L2)

                call getWins(n_est,lmaxmax,L*L1vec(1),L*L2vec(1), L1vec,real(L1,dp),L1, L2vec,L2, L2int,  &
                & CX, CTf, CEf, CXf, CBf,CTobs, CEobs, CBobs, Win12)  !used to generate the window functions F(l_1,l_2)
                Ff1(1)=(CTf(L1)*L*L1vec(1)+CTf(L2int)*L*L2vec(1))/(2*CTobs(L1)*CTobs(L2int))

                N1_PhiL=0
                do PhiLIx = 1, nPhiSample   !derivative wrt L'
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
                        L3vec= PhiLvec  !L3=L'
                        L3 = sqrt(L3vec(1)**2+L3vec(2)**2)
                        if (L3>=lmin_filter .and. L3<=lmax) then
                            L3int = nint(L3)
                            L4vec = Lvec-L3vec  !Convention where L4vec+L3vec=Lvec
                            L4 = sqrt(L4vec(1)**2+L4vec(2)**2)
                            L5vec=Lvec+L3vec
                            L5 = sqrt(L5vec(1)**2+L5vec(2)**2)
                            Lphi3vec=L3vec+L1vec   !used for the l inside clphiphi
                            Lphi3=sqrt(Lphi3vec(1)**2+Lphi3vec(2)**2)
                            Lphi3int=nint(Lphi3)
                            Lphi4vec=L2vec+L3vec
                            Lphi4=sqrt(Lphi4vec(1)**2+Lphi4vec(2)**2)
                            Lphi4int=nint(Lphi4)
                            if (L5>=lmin_filter .and. L5<=lmax) then
                                L5int=nint(L5)
                                call getWins(n_est,lmaxmax,-L*L3vec(1),L*L5vec(1), L3vec,L3,L3int, L5vec,L5, L5int,  &
                                & CX, CTf, CEf, CXf, CBf,CTobs, CEobs, CBobs, Win56, Win65)
                                phiL_dot_L5=-dot_product(L1vec+L3vec,L2vec)
                                phiL_dot_L6=dot_product(L1vec+L3vec,L5vec)
                                phiL_dot_L7=-dot_product(L2vec+L3vec,L1vec)
                                phiL_dot_L8=dot_product(L2vec+L3vec,L5vec)
                                call getResponse(n_est,lmaxmax,phiL_dot_L5,phiL_dot_L6, -L2vec,L2,L2int, L5vec,L5, &
                                & L5int, CT, CE, CX, CB, f56, f65) !calculate ftt
                                call getResponse(n_est,lmaxmax,phiL_dot_L7,phiL_dot_L8, -L1vec,real(L1,dp),L1, L5vec,L5, &
                                & L5int, CT, CE, CX, CB, f78, f87) 
                                if(Lphi3int>=lmin_filter .and. Lphi3int<=lmax) then
                                    tmp(1,1)=tmp(1,1)+(f56(1)*Cphi(Lphi3int)*dot_product(L3vec,L3vec+L1vec))*Win56(1)
                                end if
                                if(Lphi4int>=lmin_filter .and. Lphi4int<=lmax) then
                                    tmp(1,1)=tmp(1,1)+(f78(1)*Cphi(Lphi4int)*dot_product(L3vec,L3vec+L2vec))*Win56(1)
                                end if   
                            end if

                            
                            !L integers for the phi
                            
                            if (L4>=lmin_filter .and. L4<=lmax) then
                                L4int=nint(L4)
                                Lphi1vec=L3vec-L1vec   !used for the l inside clphiphi
                                Lphi1=sqrt(Lphi1vec(1)**2+Lphi1vec(2)**2)
                                Lphi1int=nint(Lphi1)
                                Lphi2vec=L3vec-L2vec
                                Lphi2=sqrt(Lphi2vec(1)**2+Lphi2vec(2)**2)
                                Lphi2int=nint(Lphi2)
                                call getWins(n_est,lmaxmax,L*L3vec(1),L*L4vec(1), L3vec,L3,L3int, L4vec,L4, L4int,  &
                                & CX, CTf, CEf, CXf, CBf,CTobs, CEobs, CBobs, Win34, Win43) !F(L',L-L')
                                Ff2(1)=(CTf(L3)*L*L3vec(1)+CTf(L4int)*L*L4vec(1))/(2*CTobs(L4int)*CTobs(L3int))
                                !build the first response f(-l2,L-L') with total l1-L'
                                phiL_dot_L1=-dot_product(L1vec-L3vec,L2vec) !leg 1 
                                phiL_dot_L2=dot_product(L1vec-L3vec,L4vec) !leg 2
                                !build the second response f(-l1,L-L') with total l2-L'
                                phiL_dot_L3=-dot_product(L2vec-L3vec,L1vec) !leg 3
                                phiL_dot_L4=dot_product(L2vec-L3vec,L4vec) !leg 4

                                
                                f1(1)=CT(L2int)*phiL_dot_L1+CT(L4int)*phiL_dot_L2
                                f2(1)=CT(L1)*phiL_dot_L3+CT(L4int)*phiL_dot_L4
                                !f1(1)=0
                                !f2(1)=0
                                
                                
                                
                                
                                
                                if(Lphi1int>=lmin_filter .and. Lphi1int<=lmax) then
                                    tmp(1,1)=tmp(1,1)+(f1(1)*Cphi(Lphi1int)*dot_product(L3vec,L3vec-L1vec))*Ff2(1)
                                end if
                                if(Lphi2int>=lmin_filter .and. Lphi2int<=lmax) then
                                    tmp(1,1)=tmp(1,1)+(f2(1)*Cphi(Lphi2int)*dot_product(L3vec,L3vec-L2vec))*Ff2(1)
                                end if
                                !tmp(1,1)=tmp(1,1)+(f12(1)*Cphi(Lphi1int)*dot_product(L3vec,L3vec-L1vec)+f34(1)*Cphi(Lphi2int)*dot_product(L3vec,L3vec-L2vec))*Win34(1)
                                !tmp(1,1)=tmp(1,1)+(this12*Cphi(Lphi1int)*dot_product(L3vec,L3vec-L1vec)+this34*Cphi(Lphi2int)*dot_product(L3vec,L3vec-L2vec))*Win34(1)
                                !this12 = responseFor(n_est,ij(1),pq(2),f12,f21)
                                !this34 = responseFor(n_est,ij(2),pq(1),f34,f43)
                                !tmp(1,1)=tmp(1,1)+this13*this24*Win43(1)
                                
                            end if
                        end if
                    end do
                    
                    if (phiIx/=0) tmp=tmp*2 !integrate 0-Pi for phi_L1
                    fact(1,1) = tmp(1,1)* PhiL_phi_dphi*PhiL
                    matrixfact(1,1) = fact(1,1)*Ff1(1)*dPh
                   

                    !$OMP CRITICAL
                    matrixL1(phiLix,1,1)=matrixL1(phiLix,1,1) + matrixfact(1,1)
                    !$OMP END CRITICAL
                 
                end do
            end do
            !!$OMP END PARALLEL DO

            matrix(Lix,:,1,1)=matrix(Lix,:,1,1) + matrixL1(:,1,1)*dphi*L1*dL
        
        end do !L1

  
        matrix(Lix,:,1,1) = matrix(Lix,:,1,1)*norms(L,1)*norms(L,1)/ (twopi**4)
            
      

        write(file_id,'(1I5)',advance='NO') L
        call WriteMatrixLine(file_id, N1,n_est)

    

        end do
        close(file_id)

  
        outtag = 'N1t1_'//estnames(1)//estnames(1)

        call WriteMatrix(outtag, vartag,dir, matrix(:,:,1,1),lmin_filter, lmaxout,lmaxmax,Lstep,nPhiSample,Phi_Sample)
        print *,''

    end subroutine N1tt_tt

    subroutine N1tt_tto(sampling,lmin_filter,lmax,lmaxout,lmaxmax,n_est, CPhi,&
                        & CT, CE, CX, CB, CTf, CEf, CXf, CBf, CTobs, CEobs, CBobs, dir,vartag)
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        ! N1 tt derivative wrt to cl_tt
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        integer, parameter :: DP = 8
        integer, parameter :: I4B = 4
        real(dp), parameter :: pi =  3.1415927, twopi=2*pi

        integer,  intent(in) :: lmin_filter,lmax,lmaxout,lmaxmax,n_est
        real(dp), intent(in) :: CPhi(lmaxmax)
        real(dp), intent(in) :: CX(lmaxmax), CE(lmaxmax),CB(lmaxmax), CT(lmaxmax)
        real(dp), intent(in) :: CXf(lmaxmax), CEf(lmaxmax),CBf(lmaxmax), CTf(lmaxmax)
        real(dp), intent(in) :: CEobs(lmaxmax), CTobs(lmaxmax), CBobs(lmaxmax)
        character(LEN=50), intent(in) :: dir
        character(LEN=50), intent(in) :: vartag
        logical, intent(in) :: sampling

        integer(I4B), parameter :: i_TT=1,i_EE=2,i_EB=3,i_TE=4,i_TB=5, i_BB=6
        integer(I4B), parameter :: Lstep = 20, dL = 20
        integer  :: lumped_indices(2,n_est)
        integer L, Lix, l1, nphi, phiIx, L2int,PhiL_nphi,PhiL_phi_ix,L3int,L4int,L1int,Lphi1int,Lphi2int
        integer PhiL
        real(dp) dphi,PhiL_phi_dphi
        real(dp) L1Vec(2), L2vec(2), LVec(2), L3Vec(2),L4Vec(2), phiLVec(2),Lphi1vec(2),Lphi2vec(2)
        real(dp) phi, PhiL_phi
        real(dP) L2, L4, L3,Lphi1,Lphi2
        real(dp) dPh
        real(dp) phiL_dot_L2, phiL_dot_L3, phiL_dot_L1, phiL_dot_L4
        real(dp) fact(n_est,n_est),tmp(n_est,n_est), N1(n_est,n_est), N1_L1(n_est,n_est),N1_PhiL(n_est,n_est)
        real(dp) matrixfact(n_est,n_est)
        real(dp) Win12(n_est), Win34(n_est), Win43(n_est)
        real(dp) WinCurl12(n_est), WinCurl34(n_est), WinCurl43(n_est), tmpCurl(n_est,n_est), &
            factCurl(n_est,n_est),N1_PhiL_Curl(n_est,n_est), N1_L1_Curl(n_est,n_est),  N1_Curl(n_est,n_est)
        real(dp) f12(n_est), f34(n_est),f43(n_est), f21(n_est)
        integer file_id, nPhiSample,Phi_Sample(lmaxmax)
        integer file_id_Curl, PhiLix
        integer ij(2),pq(2), est1, est2
        real(dp) tmpPS, tmpPSCurl, N1_PhiL_PS, N1_PhiL_PS_Curl, N1_L1_PS_Curl, N1_L1_PS, N1_PS, N1_PS_Curl
        real(dp) dPhi_Sample(lmaxmax)
        real(dp) Norms(lmaxmax,n_est), NormsCurl(lmaxmax,n_est)
        integer file_id_PS
        real(dp) this12, this34
        real(dp), allocatable :: Matrix(:,:,:,:), MatrixL1(:,:,:)
        character(LEN=10) outtag
        CHARACTER(LEN=13) :: creturn
        character(2) :: estnames(n_est)
        estnames = ['TT','EE','EB','TE','TB','BB']

        lumped_indices = transpose(reshape((/ 1,2,2,1,1,3,1,2,3,2,3,3 /), (/ n_est, 2 /)  ))
        call SetPhiSampling(lmin_filter,lmaxout,lmaxmax,sampling,nPhiSample,Phi_Sample,dPhi_Sample)

        outtag = 'N1_All'
        
        allocate(matrix((lmaxout-lmin_filter)/Lstep+1,nPhiSample, n_est,n_est))
        allocate(matrixL1(nPhiSample,n_est,n_est))
        matrix=0
        call loadNorm(n_est,lmin_filter, lmaxmax,lmaxout, Lstep,Norms,vartag,dir) !load N0
        call WriteRanges(lmin_filter, lmaxout,lmaxmax, Lstep,Phi_Sample,dPhi_Sample,nPhiSample,outtag,vartag,dir)
        open(file=trim(dir)//'/'//trim(outtag)//trim(vartag)//'.dat', newunit = file_id, form='formatted', status='replace')
        Lix=0
        print *,'Derivatives wrt cltt of N1 computation'
        L=1582
        WRITE(*,*) L
        WRITE(*,*) nPhiSample
        creturn = achar(13)
        WRITE( * , 101 , ADVANCE='NO' ) creturn , int(real(L,kind=dp)/lmaxout*100.,kind=I4B)
        101     FORMAT( a , 'Progression : ',i7,' % ')
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
            !!$OMP PARALLEL DO default(shared), private(PhiIx,phi,PhiL_nphi, PhiL_phi_dphi, PhiL_phi_ix, PhiL_phi,PhiLix, dPh), &
            !!$OMP private(L1vec,L2,L2vec, L2int,  L3, L3vec, L3int, L4, L4vec, L4int,Lphi1int,Lphi2int,Lphi1,Lphi2),&
            !!$OMP private(tmp, Win12, Win34, Win43, matrixfact,fact,phiL_dot_L1, phiL_dot_L2, phiL_dot_L3, phiL_dot_L4), &
            !!$OMP private(f12, f21, f34, f43, ij, pq,this12,this34), &
            !!$OMP private(PhiL, PhiLVec, N1_PhiL), schedule(STATIC), reduction(+:N1_L1)
            do phiIx=0,(nphi-1)/2
            phi= dphi*PhiIx
            L1vec(1)=L1*cos(phi)
            L1vec(2)=L1*sin(phi)
            L2vec = Lvec-L1vec
            L2=(sqrt(L2vec(1)**2+L2vec(2)**2))
            if (L2<lmin_filter .or. L2>lmax) cycle
            L2int=nint(L2)

            call getWins(n_est,lmaxmax,L*L1vec(1),L*L2vec(1), L1vec,real(L1,dp),L1, L2vec,L2, L2int,  &
            & CX, CTf, CEf, CXf, CBf,CTobs, CEobs, CBobs, Win12)  !used to generate the window functions F(l_1,l_2)
            
            N1_PhiL=0
            !PhiLIx = 32   !derivative wrt L'
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
                    L3vec= PhiLvec
                    L3 = sqrt(L3vec(1)**2+L3vec(2)**2)
                    if (L3>=lmin_filter .and. L3<=lmax) then
                        L3int = nint(L3)
                        L4vec = Lvec-L3vec
                        L4 = sqrt(L4vec(1)**2+L4vec(2)**2)
                        
                        !L integers for the phi
                        
                        if (L4>=lmin_filter .and. L4<=lmax) then
                            L4int=nint(L4)
                            Lphi1vec=L3vec-L1vec
                            Lphi1=sqrt(Lphi1vec(1)**2+Lphi1vec(2)**2)
                            Lphi1int=nint(Lphi1)
                            Lphi2vec=L3vec-L2vec
                            Lphi2=sqrt(Lphi2vec(1)**2+Lphi2vec(2)**2)
                            Lphi2int=nint(Lphi2)
                            call getWins(n_est,lmaxmax,L*L3vec(1),L*L4vec(1), L3vec,L3,L3int, L4vec,L4, L4int,  &
                            & CX, CTf, CEf, CXf, CBf,CTobs, CEobs, CBobs, Win34, Win43) !F(L',L-L')
                            !build the first response f(-l2,L-L') with total l1-L'
                            phiL_dot_L1=-dot_product(L1vec-L3vec,L2vec) !leg 1 
                            phiL_dot_L2=dot_product(L1vec-L3vec,L4vec) !leg 2
                            !build the second response f(-l1,L-L') with total l2-L'
                            phiL_dot_L3=-dot_product(L2vec-L3vec,L1vec) !leg 3
                            phiL_dot_L4=dot_product(L2vec-L3vec,L4vec) !leg 4

                            call getResponse(n_est,lmaxmax,phiL_dot_L1,phiL_dot_L2, -L2vec,L2,L2int, L4vec,L4, &
                            & L4int, CT, CE, CX, CB, f12, f21) !signs
                            call getResponse(n_est,lmaxmax,phiL_dot_L3,phiL_dot_L4, -L1vec,real(L1,dp),L1, L4vec,L4, &
                            & L4int, CT, CE, CX, CB, f34, f43)

                            

                            if(Lphi1int>=lmin_filter .and. Lphi1int<=lmax) then
                                tmp(1,1)=tmp(1,1)+(f12(1)*Cphi(Lphi1int)*dot_product(L3vec,L3vec-L1vec))*Win34(1)
                            end if
                            if(Lphi2int>=lmin_filter .and. Lphi2int<=lmax) then
                                tmp(1,1)=tmp(1,1)+(f34(1)*Cphi(Lphi2int)*dot_product(L3vec,L3vec-L2vec))*Win34(1)
                            end if
                
                            !tmp(1,1)=tmp(1,1)+(f12(1)*Cphi(Lphi1int)*dot_product(L3vec,L3vec-L1vec)+f34(1)*Cphi(Lphi2int)*dot_product(L3vec,L3vec-L2vec))*Win34(1)
                            !tmp(1,1)=tmp(1,1)+(this12*Cphi(Lphi1int)*dot_product(L3vec,L3vec-L1vec)+this34*Cphi(Lphi2int)*dot_product(L3vec,L3vec-L2vec))*Win34(1)
                            !this12 = responseFor(n_est,ij(1),pq(2),f12,f21)
                            !this34 = responseFor(n_est,ij(2),pq(1),f34,f43)
                            !tmp(1,1)=tmp(1,1)+this13*this24*Win43(1)
                        
                        end if
                    end if
                end do
                
                if (phiIx/=0) tmp=tmp*2 !integrate 0-Pi for phi_L1
                fact = tmp* PhiL_phi_dphi*PhiL
                
                matrixfact(1,:) = fact(1,:)*Win12(1)*dPh
               

                !$OMP CRITICAL
                matrixL1(phiLix,:,:)=matrixL1(phiLix,:,:) + matrixfact
                !$OMP END CRITICAL
            end do
            
        end do
            !!$OMP END PARALLEL DO

        matrix(Lix,:,:,:)=matrix(Lix,:,:,:) + matrixL1*dphi*L1*dL
        
    end do   !L1

  
    matrix(Lix,:,1,1) = matrix(Lix,:,1,1)*norms(L,1)*norms(L,1)/ (twopi**4)
            
      

    write(file_id,'(1I5)',advance='NO') L
    call WriteMatrixLine(file_id, N1,n_est)

    

        
    close(file_id)

  
    outtag = 'N1t1only_'//estnames(1)//estnames(1)

    call WriteMatrix(outtag, vartag,dir, matrix(:,:,1,1),lmin_filter, lmaxout,lmaxmax,Lstep,nPhiSample,Phi_Sample)
    print *,''

    end subroutine N1tt_tto
    !
     subroutine N1tt_ttn(sampling,lmin_filter,lmax,lmaxout,lmaxmax,n_est, CPhi,&
                        & CT, CE, CX, CB, CTf, CEf, CXf, CBf, CTobs, CEobs, CBobs, dir,vartag)
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        ! N1 tt derivative wrt to cl_tt test that Julien convention works
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        integer, parameter :: DP = 8
        integer, parameter :: I4B = 4
        real(dp), parameter :: pi =  3.1415927, twopi=2*pi

        integer,  intent(in) :: lmin_filter,lmax,lmaxout,lmaxmax,n_est
        real(dp), intent(in) :: CPhi(lmaxmax)
        real(dp), intent(in) :: CX(lmaxmax), CE(lmaxmax),CB(lmaxmax), CT(lmaxmax)
        real(dp), intent(in) :: CXf(lmaxmax), CEf(lmaxmax),CBf(lmaxmax), CTf(lmaxmax)
        real(dp), intent(in) :: CEobs(lmaxmax), CTobs(lmaxmax), CBobs(lmaxmax)
        character(LEN=50), intent(in) :: dir
        character(LEN=50), intent(in) :: vartag
        logical, intent(in) :: sampling

        integer(I4B), parameter :: i_TT=1,i_EE=2,i_EB=3,i_TE=4,i_TB=5, i_BB=6
        integer(I4B), parameter :: Lstep = 20, dL = 20
        integer  :: lumped_indices(2,n_est)
        integer L, Lix, l1, nphi, phiIx, L2int,PhiL_nphi,PhiL_phi_ix,L3int,L4int,L1int,Lphi1int,Lphi2int,Lphi3int,Lphi4int,L5int
        integer PhiL
        real(dp) dphi,PhiL_phi_dphi
        real(dp) L1Vec(2), L2vec(2), LVec(2), L3Vec(2),L4Vec(2), phiLVec(2),Lphi1vec(2),Lphi2vec(2),L5vec(2),Lphi3vec(2),Lphi4vec(2)
        real(dp) phi, PhiL_phi
        real(dP) L2, L4, L3,L5,Lphi1,Lphi2,Lphi3,Lphi4
        real(dp) dPh
        real(dp) phiL_dot_L2, phiL_dot_L3, phiL_dot_L1, phiL_dot_L4,phiL_dot_L5,phiL_dot_L6,phiL_dot_L7,phiL_dot_L8
        real(dp) fact(n_est,n_est),tmp(n_est,n_est), N1(n_est,n_est), N1_L1(n_est,n_est),N1_PhiL(n_est,n_est)
        real(dp) matrixfact(n_est,n_est)
        real(dp) Win12(n_est), Win34(n_est), Win43(n_est),Win56(n_est),Win65(n_est)
        real(dp) WinCurl12(n_est), WinCurl34(n_est), WinCurl43(n_est), tmpCurl(n_est,n_est), &
            factCurl(n_est,n_est),N1_PhiL_Curl(n_est,n_est), N1_L1_Curl(n_est,n_est),  N1_Curl(n_est,n_est)
        real(dp) f12(n_est), f34(n_est),f43(n_est), f21(n_est),f56(n_est),f65(n_est),f78(n_est),f87(n_est)
        integer file_id, nPhiSample,Phi_Sample(lmaxmax)
        integer file_id_Curl, PhiLix
        integer ij(2),pq(2), est1, est2
        real(dp) tmpPS, tmpPSCurl, N1_PhiL_PS, N1_PhiL_PS_Curl, N1_L1_PS_Curl, N1_L1_PS, N1_PS, N1_PS_Curl
        real(dp) dPhi_Sample(lmaxmax)
        real(dp) Norms(lmaxmax,n_est), NormsCurl(lmaxmax,n_est)
        integer file_id_PS
        real(dp) this12, this34
        real(dp), allocatable :: Matrix(:,:,:,:), MatrixL1(:,:,:)
        character(LEN=10) outtag
        CHARACTER(LEN=13) :: creturn
        character(2) :: estnames(n_est)
        estnames = ['TT','EE','EB','TE','TB','BB']

        lumped_indices = transpose(reshape((/ 1,2,2,1,1,3,1,2,3,2,3,3 /), (/ n_est, 2 /)  ))
        call SetPhiSampling(lmin_filter,lmaxout,lmaxmax,sampling,nPhiSample,Phi_Sample,dPhi_Sample)

        outtag = 'N1_All'
        
        allocate(matrix((lmaxout-lmin_filter)/Lstep+1,nPhiSample, n_est,n_est))
        allocate(matrixL1(nPhiSample,n_est,n_est))
        matrix=0
        call loadNorm(n_est,lmin_filter, lmaxmax,lmaxout, Lstep,Norms,vartag,dir) !load N0
        call WriteRanges(lmin_filter, lmaxout,lmaxmax, Lstep,Phi_Sample,dPhi_Sample,nPhiSample,outtag,vartag,dir)
        open(file=trim(dir)//'/'//trim(outtag)//trim(vartag)//'.dat', newunit = file_id, form='formatted', status='replace')
        Lix=0
        print *,'Derivatives wrt clttn of N1 computation'
        do L=lmin_filter, lmaxout, Lstep
            creturn = achar(13)
            WRITE( * , 101 , ADVANCE='NO' ) creturn , int(real(L,kind=dp)/lmaxout*100.,kind=I4B)
            101     FORMAT( a , 'Progression : ',i7,' % ')
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
                do phiIx=0,(nphi-1)/2
                phi= dphi*PhiIx
                L1vec(1)=L1*cos(phi)
                L1vec(2)=L1*sin(phi)
                L2vec = Lvec-L1vec
                L2=(sqrt(L2vec(1)**2+L2vec(2)**2))
                if (L2<lmin_filter .or. L2>lmax) cycle
                L2int=nint(L2)

                call getWins(n_est,lmaxmax,L*L1vec(1),L*L2vec(1), L1vec,real(L1,dp),L1, L2vec,L2, L2int,  &
                & CX, CTf, CEf, CXf, CBf,CTobs, CEobs, CBobs, Win12)  !used to generate the window functions F(l_1,l_2)
                
                N1_PhiL=0
                do PhiLIx = 1, nPhiSample   !derivative wrt L'
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
                        L3vec= PhiLvec
                        L3 = sqrt(L3vec(1)**2+L3vec(2)**2)
                        if (L3>=lmin_filter .and. L3<=lmax) then
                            L3int = nint(L3)
                            L4vec = -Lvec-L3vec
                            L4 = sqrt(L4vec(1)**2+L4vec(2)**2)
                            L4int=nint(L4)
                            L5vec=Lvec-L3vec
                            L5 = sqrt(L5vec(1)**2+L5vec(2)**2)
                            L5int=nint(L5)
                            
                            !L integers for the phi

                            
                            if (L4>=lmin_filter .and. L4<=lmax) then
                                Lphi1vec=L1vec+L3vec
                                Lphi1=sqrt(Lphi1vec(1)**2+Lphi1vec(2)**2)
                                Lphi1int=nint(Lphi1)
                                Lphi2vec=L2vec+L3vec
                                Lphi2=sqrt(Lphi2vec(1)**2+Lphi2vec(2)**2)
                                Lphi2int=nint(Lphi2)
                                Lphi3vec=L3vec-L1vec
                                Lphi3=sqrt(Lphi3vec(1)**2+Lphi3vec(2)**2)
                                Lphi3int=nint(Lphi3)
                                Lphi4vec=L3vec-L2vec
                                Lphi4=sqrt(Lphi4vec(1)**2+Lphi4vec(2)**2)
                                Lphi4int=nint(Lphi4)
                                
                                
                                call getWins(n_est,lmaxmax,-L*L3vec(1),-L*L4vec(1), L3vec,L3,L3int, L4vec,L4, L4int,  &
                                & CX, CTf, CEf, CXf, CBf,CTobs, CEobs, CBobs, Win34, Win43) !F(L',L-L')
                                call getWins(n_est,lmaxmax,L*L3vec(1),L*L5vec(1), L3vec,L3,L3int, L5vec,L5, L5int,  &
                                & CX, CTf, CEf, CXf, CBf,CTobs, CEobs, CBobs, Win56, Win65) 
                                !build the first response f(-l2,L-L') with total l1-L'
                                phiL_dot_L1=dot_product(-L1vec-L3vec,L2vec) !leg 1 
                                phiL_dot_L2=dot_product(-L1vec-L3vec,L4vec) !leg 2
                                !build the second response f(-l1,L-L') with total l2-L'
                                phiL_dot_L3=dot_product(-L2vec-L3vec,L1vec) !leg 3
                                phiL_dot_L4=dot_product(-L2vec-L3vec,L4vec) !leg 4
                                phiL_dot_L5=dot_product(L3vec-L1vec,L2vec)
                                phiL_dot_L6=-dot_product(L3vec-L1vec,L5vec)
                                phiL_dot_L7=dot_product(L3vec-L2vec,L1vec)
                                phiL_dot_L8=-dot_product(L3vec-L2vec,L5vec)
                                call getResponse(n_est,lmaxmax,phiL_dot_L1,phiL_dot_L2, L2vec,L2,L2int, L4vec,L4, &
                                & L4int, CT, CE, CX, CB, f12, f21) !signs
                                call getResponse(n_est,lmaxmax,phiL_dot_L3,phiL_dot_L4, L1vec,real(L1,dp),L1, L4vec,L4, &
                                & L4int, CT, CE, CX, CB, f34, f43)
                                call getResponse(n_est,lmaxmax,phiL_dot_L5,phiL_dot_L6, L2vec,L2,L2int, L5vec,L5, &
                                & L5int, CT, CE, CX, CB, f56, f65)
                                call getResponse(n_est,lmaxmax,phiL_dot_L7,phiL_dot_L8, L1vec,real(L1,dp),L1, L5vec,L5, &
                                & L5int, CT, CE, CX, CB, f78, f87)
                                

                      
                                if(Lphi1int>=lmin_filter .and. Lphi1int<=lmax) then
                                    tmp(1,1)=tmp(1,1)+(f12(1)*Cphi(Lphi1int)*dot_product(L3vec,Lphi1vec))*Win34(1)
                                end if
                                if(Lphi2int>=lmin_filter .and. Lphi2int<=lmax) then
                                    tmp(1,1)=tmp(1,1)+(f34(1)*Cphi(Lphi2int)*dot_product(L3vec,Lphi2vec))*Win34(1)
                                end if
                                tmp(1,1)=tmp(1,1)+(f56(1)*Cphi(Lphi3int)*dot_product(L3vec,Lphi3vec))*Win56(1)
                                tmp(1,1)=tmp(1,1)+(f78(1)*Cphi(Lphi4int)*dot_product(L3vec,Lphi4vec))*Win56(1)
                                !if (f12(1)*dot_product(L3vec,L3vec+L1vec)*Win34(1)<0) then 
    
                                !tmp(1,1)=tmp(1,1)+(f12(1)*Cphi(Lphi1int)*dot_product(L3vec,L3vec+L1vec)+f34(1)*Cphi(Lphi2int)*dot_product(L3vec,L3vec+L2vec))*Win34(1)
                                
                                !tmp(1,1)=tmp(1,1)+(this12*Cphi(Lphi1int)*dot_product(L3vec,L3vec-L1vec)+this34*Cphi(Lphi2int)*dot_product(L3vec,L3vec-L2vec))*Win34(1)
                                !this12 = responseFor(n_est,ij(1),pq(2),f12,f21)
                                !this34 = responseFor(n_est,ij(2),pq(1),f34,f43)
                                !tmp(1,1)=tmp(1,1)+this13*this24*Win43(1)
                            
                            end if
                        end if
                    end do
                    
                    if (phiIx/=0) tmp=tmp*2 !integrate 0-Pi for phi_L1
                    fact = tmp* PhiL_phi_dphi* PhiL
                    
                    matrixfact(1,:) = fact(1,:)*Win12(1)*dPh
           
                    matrixL1(phiLix,:,:)=matrixL1(phiLix,:,:) + matrixfact
               
                    N1_PhiL= N1_PhiL + fact * Cphi(PhiL)*dPh
                end do
            end do


            matrix(Lix,:,:,:)=matrix(Lix,:,:,:) + matrixL1*dphi*L1*dL
        
        end do !L1

  
        matrix(Lix,:,1,1) = (matrix(Lix,:,1,1)*norms(L,1)*norms(L,1)*L*L*L*L)/ (twopi**4)
            
      

        write(file_id,'(1I5)',advance='NO') L
        call WriteMatrixLine(file_id, N1,n_est)

    

        end do
        close(file_id)

  
        outtag = 'N1tn_'//estnames(1)//estnames(1)

        call WriteMatrix(outtag, vartag,dir, matrix(:,:,1,1),lmin_filter, lmaxout,lmaxmax,Lstep,nPhiSample,Phi_Sample)
        print *,''

    end subroutine N1tt_ttn

    subroutine N1pptest(sampling,lmin_filter,lmax,lmaxout,lmaxmax,n_est, CPhi,&
                        & CT, CE, CX, CB, CTf, CEf, CXf, CBf, CTobs, CEobs, CBobs, dir,vartag)
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        ! Main routine to compute N1 derivatives.
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        integer, parameter :: DP = 8
        integer, parameter :: I4B = 4
        real(dp), parameter :: pi =  3.1415927, twopi=2*pi

        integer,  intent(in) :: lmin_filter,lmax,lmaxout,lmaxmax,n_est
        real(dp), intent(in) :: CPhi(lmaxmax)
        real(dp), intent(in) :: CX(lmaxmax), CE(lmaxmax),CB(lmaxmax), CT(lmaxmax)
        real(dp), intent(in) :: CXf(lmaxmax), CEf(lmaxmax),CBf(lmaxmax), CTf(lmaxmax)
        real(dp), intent(in) :: CEobs(lmaxmax), CTobs(lmaxmax), CBobs(lmaxmax)
        character(LEN=50), intent(in) :: dir
        character(LEN=50), intent(in) :: vartag
        logical, intent(in) :: sampling

        integer(I4B), parameter :: i_TT=1,i_EE=2,i_EB=3,i_TE=4,i_TB=5, i_BB=6
        integer(I4B), parameter :: Lstep = 20, dL = 20
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
        real(dp) Norms(lmaxmax,n_est), NormsCurl(lmaxmax,n_est)
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
        allocate(matrix((lmaxout-lmin_filter)/Lstep+1,nPhiSample, 1,1))
        allocate(matrixL1(nPhiSample,1,1))
        matrix=0
        call loadNorm(n_est,lmin_filter, lmaxmax,lmaxout, Lstep,Norms,vartag,dir) !load N0
        call WriteRanges(lmin_filter, lmaxout,lmaxmax, Lstep,Phi_Sample,dPhi_Sample,nPhiSample,outtag,vartag,dir)

        open(file=trim(dir)//'/'//trim(outtag)//trim(vartag)//'.dat', newunit = file_id, form='formatted', status='replace')

        Lix=0
        print *,'Derivativestest of N1 computation'
        do L=lmin_filter, lmaxout, Lstep
            creturn = achar(13)
            WRITE( * , 101 , ADVANCE='NO' ) creturn , int(real(L,kind=dp)/lmaxout*100.,kind=I4B)
            101     FORMAT( a , 'Progression : ',i7,' % ')
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

                !!$OMP PARALLEL DO default(shared), private(PhiIx,phi,PhiL_nphi, PhiL_phi_dphi, PhiL_phi_ix, PhiL_phi,PhiLix, dPh), &
                !!$OMP private(L1vec,L2,L2vec, L2int,  L3, L3vec, L3int, L4, L4vec, L4int),&
                !!$OMP private(tmp, Win12, Win34, Win43, matrixfact,fact,phiL_dot_L1, phiL_dot_L2, phiL_dot_L3, phiL_dot_L4), &
                !!$OMP private(f13, f31, f42, f24, ij, pq, est1, est2,this13,this24), &
                !!$OMP private(PhiL, PhiLVec, N1_PhiL), schedule(STATIC), reduction(+:N1_L1)
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

                                do est1=1,1
                                    ij=lumped_indices(:,est1)
                                    do est2=est1,1
                                        pq=lumped_indices(:,est2)
                                        this13 = responseFor(n_est,ij(1),pq(1),f13,f31)
                                        this24 = responseFor(n_est,ij(2),pq(2),f24,f42)
                                        tmp(est1,est2)=tmp(est1,est2)+this13*this24*Win34(est2)

                                        this13 = responseFor(n_est,ij(1),pq(2),f13,f31)
                                        this24 = responseFor(n_est,ij(2),pq(1),f24,f42)
                                        tmp(est1,est2)=tmp(est1,est2)+this13*this24*Win43(est2)
                                    end do
                                end do
                            end if
                        end if
                    end do
                    if (phiIx/=0) tmp=tmp*2!integrate 0-Pi for phi_L1
                    fact = tmp* PhiL_phi_dphi* PhiL
                    do est1=1,1
                        matrixfact(est1,:) = fact(est1,:)*Win12(est1)*dPh
                    end do

                    !$OMP CRITICAL
                    matrixL1(phiLix,:,:)=matrixL1(phiLix,:,:) + matrixfact
                    !$OMP END CRITICAL
                   
                end do
          
            end do
            !!$OMP END PARALLEL DO

            matrix(Lix,:,:,:)=matrix(Lix,:,:,:) + matrixL1*dphi*L1*dL
            N1= N1 + N1_L1 * dphi* L1*dL

        end do !L1
        matrix(Lix,:,1,1)=matrix(Lix,:,1,1)*norms(L,1)*norms(L,1)/(twopi**4)


        ! print *, 'N1 L, TTTT, EBEB: ',L, N1(i_TT,i_TT), N1(i_eb,i_eb)
        write(file_id,'(1I5)',advance='NO') L
        call WriteMatrixLine(file_id,N1,n_est)
        end do
        close(file_id)
        outtag = 'N1pptest_'//estnames(1)//estnames(1)
        call WriteMatrix(outtag, vartag,dir, matrix(:,:,1,1),lmin_filter, lmaxout, &
            & lmaxmax,Lstep,nPhiSample,Phi_Sample)
        print *,''

    end subroutine N1pptest
    
    subroutine compute_n0(phifile,lensedcmbfile,noise_fwhm_deg,nll,nlp,lmin_filter,lmaxout,lmax,lmax_TT,lcorr_TT,dir)
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
        integer, parameter :: lmaxmax = 8000
        real(dp), intent(in)     :: noise_fwhm_deg
        integer, intent(in)      :: lmin_filter, lmaxout, lmax, lmax_TT, lcorr_TT
        character(LEN=50), intent(in) :: dir
        character(LEN=200), intent(in) ::  lensedcmbfile
        real(dp), intent(in) :: phifile(lmaxmax)
        character(LEN=:), allocatable :: root
        logical :: doCurl = .True.
        character(LEN=50) vartag
        real(dp) :: CPhi(lmaxmax)
        real(dp) :: CX(lmaxmax), CE(lmaxmax),CB(lmaxmax), CT(lmaxmax)
        real(dp) :: CXf(lmaxmax), CEf(lmaxmax),CBf(lmaxmax), CTf(lmaxmax)
        real(dp) :: NT(lmaxmax), NP(lmaxmax)
        real(dp) :: CEobs(lmaxmax), CTobs(lmaxmax), CBobs(lmaxmax)
        integer(I4B) :: LMin
        real(dp),dimension(lmax), intent(in) :: nll,nlp
        real(dp),dimension(lmax):: NoiseVar, NoiseVarP
 

        NoiseVar =  nll  !muKArcmin becomes the input array
        NoiseVarP=nlp
        LMin = lmin_filter

        call system('mkdir -p '//dir)

        call ReadPhiPhi(phifile,lmax,lmaxmax,CPhi)
        call ReadPower(lensedcmbfile,lmax,lmaxmax,CT,CE,CB,CX,CTf,CEf,CBf,CXf)

        call NoiseInit(NoiseVar, NoiseVarP,noise_fwhm_deg,lmax,lmax_TT,lcorr_TT,lmaxmax,NT,NP)
        CTobs = CT + NT
        CEobs = CE + NP
        CBobs = CB + NP

        root = 'analytical'
        vartag = '_'//root

        call getNorm( .false. , .false. ,.False.,lmin_filter,lmax,lmaxout,lmaxmax,n_est, CPhi,&
                            & CT, CE, CX, CB, CTf, CEf, CXf, CBf, CTobs, CEobs, CBobs, dir, vartag)

    end subroutine compute_n0

    subroutine compute_n1(phifile,lensedcmbfile,noise_fwhm_deg,nll,nlp,lmin_filter,lmaxout,lmax,lmax_TT,lcorr_TT,dir)
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
        integer, parameter :: lmaxmax = 8000
        real(dp), intent(in)     :: noise_fwhm_deg
        integer, intent(in)      :: lmin_filter, lmaxout, lmax, lmax_TT, lcorr_TT
        character(LEN=50), intent(in) :: dir
        character(LEN=200), intent(in) :: lensedcmbfile
        real(dp), intent(in) :: phifile(lmaxmax)
        character(LEN=:), allocatable :: root
        character(LEN=50) vartag
        real(dp) :: CPhi(lmaxmax)
        real(dp) :: CX(lmaxmax), CE(lmaxmax),CB(lmaxmax), CT(lmaxmax),CTm(lmaxmax)
        real(dp) :: CXf(lmaxmax), CEf(lmaxmax),CBf(lmaxmax), CTf(lmaxmax)
        real(dp) :: NT(lmaxmax), NP(lmaxmax)
        real(dp) :: CEobs(lmaxmax), CTobs(lmaxmax), CBobs(lmaxmax)
        integer(I4B) :: LMin,L
        real(dp),dimension(lmax), intent(in) :: nll,nlp
        real(dp),dimension(lmax):: NoiseVar, NoiseVarP
 

        NoiseVar =  nll  !nll is the temperature noise power spectrum from so-obs
        NoiseVarP=nlp    !nlp is the polarization noise power 
        LMin = lmin_filter
        LMin = lmin_filter

        call system('mkdir -p '//dir)

        call ReadPhiPhi(phifile,lmax,lmaxmax,CPhi)
        call ReadPower(lensedcmbfile,lmax,lmaxmax,CT,CE,CB,CX,CTf,CEf,CBf,CXf)

        call NoiseInit(NoiseVar, NoiseVarP,noise_fwhm_deg,lmax,lmax_TT,lcorr_TT,lmaxmax,NT,NP)
        CTobs = CT + NT
        CEobs = CE + NP
        CBobs = CB + NP

        root = 'analytical'
        vartag = '_'//root
        CTm=CT
        !Ctm(1052)=0.5*CT(1052)
        write(*,*) 'hELLO'
        !write(*,*) Ctm(1052)
        !write(*,*) CT(1052)
        
        call GetN1General( .true. ,lmin_filter,lmax,lmaxout,lmaxmax,n_est, CPhi,&
                            & 1.1*CTm, CE, CX, CB, CTf, CEf, CXf, CBf, CTobs, CEobs, CBobs, dir, vartag)

    end subroutine compute_n1

    subroutine compute_n1_derivatives(phifile,lensedcmbfile,noise_fwhm_deg,nll,nlp,&
        & lmin_filter,lmaxout,lmax,lmax_TT,lcorr_TT,dir)
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
        integer, parameter :: lmaxmax = 9999
        real(dp), intent(in)     :: noise_fwhm_deg
        integer, intent(in)      :: lmin_filter, lmaxout, lmax, lmax_TT, lcorr_TT
        character(LEN=50), intent(in) :: dir
        character(LEN=200), intent(in) ::  lensedcmbfile
        real(dp), intent(in) :: phifile(lmaxmax)
        character(LEN=:), allocatable :: root
        character(LEN=50) vartag
        real(dp) :: CPhi(lmaxmax)
        real(dp) :: CX(lmaxmax), CE(lmaxmax),CB(lmaxmax), CT(lmaxmax)
        real(dp) :: CXf(lmaxmax), CEf(lmaxmax),CBf(lmaxmax), CTf(lmaxmax)
        real(dp) :: NT(lmaxmax), NP(lmaxmax)
        real(dp) :: CEobs(lmaxmax), CTobs(lmaxmax), CBobs(lmaxmax)
        integer(I4B) :: LMin
        real(dp),dimension(lmax), intent(in) :: nll,nlp
        real(dp),dimension(lmax):: NoiseVar, NoiseVarP
 

        NoiseVar =  nll  !nll is the temperature noise power spectrum from so-obs
        NoiseVarP=nlp  
        LMin = lmin_filter

        call system('mkdir -p '//dir)

        call ReadPhiPhi(phifile,lmax,lmaxmax,CPhi)
        call ReadPower(lensedcmbfile,lmax,lmaxmax,CT,CE,CB,CX,CTf,CEf,CBf,CXf)

        call NoiseInit(NoiseVar, NoiseVarP,noise_fwhm_deg,lmax,lmax_TT,lcorr_TT,lmaxmax,NT,NP)
        CTobs = CT + NT
        CEobs = CE + NP
        CBobs = CB + NP

        root = 'analytical'
        vartag = '_'//root

        call GetN1MatrixGeneral( .true. ,lmin_filter,lmax,lmaxout,lmaxmax,n_est, CPhi,&
                            & CT, CE, CX, CB, CTf, CEf, CXf, CBf, CTobs, CEobs, CBobs, dir, vartag)

    end subroutine compute_n1_derivatives

    subroutine compute_n1_TT(phifile,lensedcmbfile,noise_fwhm_deg,nll,nlp,&
        & lmin_filter,lmaxout,lmax,lmax_TT,lcorr_TT,dir)
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        ! Interface to python to compute
        ! derivatives of N1 bias wrt Cltt
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        implicit none
        integer, parameter :: DP = 8
        integer, parameter :: I4B = 4
        real(dp), parameter :: pi =  3.1415927, twopi=2*pi
        ! Order 1 2 3 = T E B
        ! Estimator order TT, EE, EB, TE, TB, BB
        integer(I4B), parameter :: n_est = 6
        integer, parameter :: lmaxmax = 10000
        real(dp), intent(in)     :: noise_fwhm_deg
        integer, intent(in)      :: lmin_filter, lmaxout, lmax, lmax_TT, lcorr_TT
        character(LEN=50), intent(in) :: dir
        character(LEN=200), intent(in) ::lensedcmbfile
        real(dp), intent(in) :: phifile(lmaxmax)
        character(LEN=:), allocatable :: root
        character(LEN=50) vartag
        real(dp) :: CPhi(lmaxmax)
        real(dp) :: CX(lmaxmax), CE(lmaxmax),CB(lmaxmax), CT(lmaxmax)
        real(dp) :: CXf(lmaxmax), CEf(lmaxmax),CBf(lmaxmax), CTf(lmaxmax)
        real(dp) :: NT(lmaxmax), NP(lmaxmax)
        real(dp) :: CEobs(lmaxmax), CTobs(lmaxmax), CBobs(lmaxmax)
        integer(I4B) :: LMin
        real(dp),dimension(lmax), intent(in) :: nll,nlp
        real(dp),dimension(lmax):: NoiseVar, NoiseVarP
 

        NoiseVar =  nll  !nll is the temperature noise power spectrum from so-obs
        NoiseVarP=nlp  
        LMin = lmin_filter

        call system('mkdir -p '//dir)

        call ReadPhiPhi(phifile,lmax,lmaxmax,CPhi)
        call ReadPower(lensedcmbfile,lmax,lmaxmax,CT,CE,CB,CX,CTf,CEf,CBf,CXf)

        call NoiseInit(NoiseVar, NoiseVarP,noise_fwhm_deg,lmax,lmax_TT,lcorr_TT,lmaxmax,NT,NP)
        CTobs = CT + NT
        CEobs = CE + NP
        CBobs = CB + NP

        root = 'analytical'
        vartag = '_'//root

        call N1tt_tt( .true. ,lmin_filter,lmax,lmaxout,lmaxmax,n_est, CPhi,&
                            & CT, CE, CX, CB, CTf, CEf, CXf, CBf, CTobs, CEobs, CBobs, dir, vartag)

    end subroutine compute_n1_TT
    
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
