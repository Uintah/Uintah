! This module is token from dl-poly. 
! There are so many 3D-fft I really didn't felt like I have to rewrite them
! SO I just took one (any one, no particular reason for coosing dl-poly one)
! FFT is NOT paralelized 



module fft_3D_modified
 implicit none

public :: direct_d2_fft_1d_Int_fft_1_ARR
public :: direct_d2_fft_1d_Int_fft_2_ARR
public :: direct_d2_fft_1d_Int_fft_4_ARR
public :: inverse_d2_fft_1d_Int_fft_1_ARR
public :: inverse_d2_fft_1d_Int_fft_2_ARR
public :: inverse_d2_fft_1d_Int_fft_4_ARR
public :: dlpfft3_1_ARR
public :: dlpfft3_2_ARR
public :: dlpfft3_4_ARR

 contains


 subroutine direct_d2_fft_1d_Int_fft_1_ARR
! does direct 2d-fft and 1d-fourier-integral on 1 array charge
  use variables_smpe, only : nfftx,nffty,nfftz, &
                             qqq1_Re,qqq1_Im
  implicit none
  integer ix,iy,iz,iz1,i_adress
   call dlpfft3_1_ARR(0,+1)
! I do not need to multiply them with -1.
! I double checked this comparing non-fft vs. vvs
! That is due to fft symetries properties.
!   do ix = 1, nfftx; do iy = 1, nffty
!   do iz = 0, nfftz-1
!    iz1 = iz + 1
!    i_adress = (iy-1+iz*nffty)*nfftx + ix
!    if (mod(iz,2) == 1 ) then
!      qqq1_Re(i_adress) = -qqq1_Re(i_adress)
!      qqq1_Im(i_adress) = -qqq1_Im(i_adress)
!!     qqq(:,:,iz1) = -qqq(:,:,iz1)
!    endif
!   enddo
!  enddo ; enddo
 
 end subroutine direct_d2_fft_1d_Int_fft_1_ARR

  subroutine direct_d2_fft_1d_Int_fft_2_ARR
! does direct 2d-fft and 1d-fourier-integral on 2 array charge
  use variables_smpe, only : nfftx,nffty,nfftz, &
                             qqq1_Re,qqq1_Im,qqq2_Re,qqq2_Im
  implicit none
  integer ix,iy,iz,iz1,i_adress
   call dlpfft3_2_ARR(0,+1)
!   do ix = 1, nfftx; do iy = 1, nffty
!   do iz = 0, nfftz-1
!    iz1 = iz + 1
!    i_adress = (iy-1+iz*nffty)*nfftx + ix
!    if (mod(iz,2) == 1 ) then
!      qqq1_Re(i_adress) = -qqq1_Re(i_adress)
!      qqq1_Im(i_adress) = -qqq1_Im(i_adress)
!      qqq2_Re(i_adress) = -qqq2_Re(i_adress)
!      qqq2_Im(i_adress) = -qqq2_Im(i_adress)
!!     qqq(:,:,iz1) = -qqq(:,:,iz1)
!    endif
!   enddo
!  enddo ; enddo
!
 end subroutine direct_d2_fft_1d_Int_fft_2_ARR

   subroutine direct_d2_fft_1d_Int_fft_4_ARR
! does direct 2d-fft and 1d-fourier-integral on 2 array charge
  use variables_smpe, only : nfftx,nffty,nfftz, &
                             qqq1_Re,qqq1_Im,qqq2_Re,qqq2_Im, qqq3_Im,qqq4_Im,&
                             qqq3_Re,qqq4_Re
  implicit none
  integer ix,iy,iz,iz1,i_adress
   call dlpfft3_4_ARR(0,+1)
!   do ix = 1, nfftx; do iy = 1, nffty
!   do iz = 0, nfftz-1
!    iz1 = iz + 1
!    i_adress = (iy-1+iz*nffty)*nfftx + ix
!    if (mod(iz,2) == 1 ) then
!      qqq1_Re(i_adress) = -qqq1_Re(i_adress)
!      qqq1_Im(i_adress) = -qqq1_Im(i_adress)
!      qqq2_Re(i_adress) = -qqq2_Re(i_adress)
!      qqq2_Im(i_adress) = -qqq2_Im(i_adress)
!!     qqq(:,:,iz1) = -qqq(:,:,iz1)
!    endif
!   enddo
!  enddo ; enddo
!
 end subroutine direct_d2_fft_1d_Int_fft_4_ARR



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

 subroutine inverse_d2_fft_1d_Int_fft_1_ARR
 use variables_smpe, only : nfftx,nffty,nfftz, &
                             qqq1_Re,qqq1_Im

 implicit none
 integer ix,iy,iz,iz1,i_adress, ndx, ndx1
 
 call dlpfft3_1_ARR(0,-1)
! do ix = 1, nfftx; do iy = 1, nffty 
!  do iz = 0, nfftz-1
!    iz1 = iz + 1
!    i_adress = (iy-1+iz*nffty)*nfftx + ix 
!    if (mod(iz,2) == 1 ) then
!      qqq1_Re(i_adress) = -qqq1_Re(i_adress)
!      qqq1_Im(i_adress) = -qqq1_Im(i_adress)
!!      qqq(:,:,iz1) = -qqq(:,:,iz1)
!    endif
!  enddo
!  enddo ; enddo

 end subroutine inverse_d2_fft_1d_Int_fft_1_ARR


 subroutine inverse_d2_fft_1d_Int_fft_2_ARR
 use variables_smpe, only : nfftx,nffty,nfftz, &
                             qqq1_Re,qqq1_Im,qqq2_Re,qqq2_Im

 implicit none
 integer ix,iy,iz,iz1,i_adress, ndx, ndx1

 call dlpfft3_2_ARR(0,-1)
! do ix = 1, nfftx; do iy = 1, nffty
!  do iz = 0, nfftz-1
!    iz1 = iz + 1
!    i_adress = (iy-1+iz*nffty)*nfftx + ix
!    if (mod(iz,2) == 1 ) then
!      qqq1_Re(i_adress) = -qqq1_Re(i_adress)
!      qqq1_Im(i_adress) = -qqq1_Im(i_adress)
!      qqq2_Re(i_adress) = -qqq2_Re(i_adress)
!      qqq2_Im(i_adress) = -qqq2_Im(i_adress)
!!      qqq(:,:,iz1) = -qqq(:,:,iz1)
!    endif
!  enddo
!  enddo ; enddo

 end subroutine inverse_d2_fft_1d_Int_fft_2_ARR

 subroutine inverse_d2_fft_1d_Int_fft_4_ARR
 use variables_smpe, only : nfftx,nffty,nfftz, &
                             qqq1_Re,qqq1_Im,qqq2_Re,qqq2_Im,qqq3_Im,qqq4_Im,&
                             qqq3_Re,qqq4_Re


 implicit none
 integer ix,iy,iz,iz1,i_adress, ndx, ndx1

 call dlpfft3_4_ARR(0,-1)
! do ix = 1, nfftx; do iy = 1, nffty
!  do iz = 0, nfftz-1
!    iz1 = iz + 1
!    i_adress = (iy-1+iz*nffty)*nfftx + ix
!    if (mod(iz,2) == 1 ) then
!      qqq1_Re(i_adress) = -qqq1_Re(i_adress)
!      qqq1_Im(i_adress) = -qqq1_Im(i_adress)
!      qqq2_Re(i_adress) = -qqq2_Re(i_adress)
!      qqq2_Im(i_adress) = -qqq2_Im(i_adress)
!!      qqq(:,:,iz1) = -qqq(:,:,iz1)
!    endif
!  enddo
!  enddo ; enddo

 end subroutine inverse_d2_fft_1d_Int_fft_4_ARR
 

 subroutine dlpfft3_1_ARR(ind,isw)
     use variables_smpe, only : nfftx,nffty,nfftz,&
                                ww1_Re,ww1_Im,ww2_Re,ww2_Im,ww3_Re,ww3_Im,key1,key2,key3,&
                                qqq1_Re,qqq1_Im
     implicit none
!   Modified version to work with 1D-array and 1-array.
!***********************************************************************
!     
!     dl-poly 3D fast fourier transform routine (in place)
!     
!     copyright daresbury laboratory 1998
!     
!     author w smith july 1998
!     
!     wl
!     2002/05/31 13:58:07
!     1.2
!     Exp
!     
!***********************************************************************
      
      integer, intent(IN) :: ind,isw
 
      logical lkx,lky,lkz
      real(8) ttt, ttt1_Re, ttt1_Im, ttt_Re, ttt_Im
      integer, save :: nu1,nu2,nu3
      integer idm
      integer kkk,iii,jjj,ii,kk,jj,i,j,k,jj2,jjj2,num,kk1,k12,l
      integer ndx, ndx1
      real(8) arg
      real(8) q0,q1,w0,w1
      integer i1,j1,k1
       

      real(8), parameter ::  tpi = 6.283185307179586d0

      if(ind.gt.0)then

!     check FFT array dimensions

        idm=1 ; lkx=.true.  ;  lky=.true.  ;  lkz=.true.

        do i=1,30
          idm=2*idm
          if(idm.eq.nfftx)then
            lkx=.false.
            nu1=i
          endif
          if(idm.eq.nffty)then
            lky=.false.
            nu2=i
          endif
          if(idm.eq.nfftz)then
            lkz=.false.
            nu3=i
          endif
        enddo ! i=1,30
        
        if(lkx.or.lky.or.lkz)then
          write(*,*)'error -dl-poly FFT array not 2**N',nfftx,nffty,nfftz
          stop
        endif
        
        do kkk=1,nfftx
          iii=0
          jjj=kkk-1
          do j=1,nu1
            jj2=jjj/2
            iii=2*(iii-jj2)+jjj
            jjj=jj2
          enddo
          key1(kkk)=iii+1
        enddo

        do kkk=1,nffty
          iii=0
          jjj=kkk-1
          do j=1,nu2
            jj2=jjj/2
            iii=2*(iii-jj2)+jjj
            jjj=jj2
          enddo
          key2(kkk)=iii+1
        enddo

        do kkk=1,nfftz
          iii=0
          jjj=kkk-1
          do j=1,nu3
            jj2=jjj/2
            iii=2*(iii-jj2)+jjj
            jjj=jj2
          enddo
          key3(kkk)=iii+1
        enddo

!     initialise complex exponential factors
        
        ww1_Re(1)=1.0d0
        ww1_Im(1)=0.0d0 
        do i=1,nfftx/2
          arg=(tpi/dble(nfftx))*dble(i)
          ww1_Re(i+1) = dcos(arg)
          ww1_Im(i+1) = dsin(arg)
          ww1_Re(nfftx+1-i) =   ww1_Re(i+1)
          ww1_Im(nfftx+1-i) = - ww1_Im(i+1)
!          ww1(i+1)=cmplx(cos(arg),sin(arg),kind=8)
!          ww1(nfftx+1-i)=conjg(ww1(i+1))
        enddo
        
        ww2_Re(1)=1.d0
        ww2_Im(1)=0.d0
        do i=1,nffty/2
          arg=(tpi/dble(nffty))*dble(i)
          ww2_Re(i+1) = dcos(arg)
          ww2_Im(i+1) = dsin(arg)
          ww2_Re(nffty+1-i) =   ww2_Re(i+1)
          ww2_Im(nffty+1-i) = - ww2_Im(i+1)
!          ww2(i+1)=cmplx(cos(arg),sin(arg),kind=8)
!          ww2(nffty+1-i)=conjg(ww2(i+1))
        enddo
        
        ww3_Re(1)=1.d0
        ww3_Im(1)=0.d0
        do i=1,nfftz/2
          arg=(tpi/dble(nfftz))*dble(i)
          ww3_Re(i+1) = dcos(arg)
          ww3_Im(i+1) = dsin(arg)
          ww3_Re(nfftz+1-i) =   ww3_Re(i+1)
          ww3_Im(nfftz+1-i) = - ww3_Im(i+1)
!          ww3(i+1)=cmplx(cos(arg),sin(arg),kind=8)
!          ww3(nfftz+1-i)=conjg(ww3(i+1))
        enddo
        return
      endif ! ind ==0

!     take conjugate of exponentials if required
      
      if(isw.lt.0)then
          ww1_Im(1:nfftx) = - ww1_Im(1:nfftx)
          ww2_Im(1:nffty) = - ww2_Im(1:nffty)
          ww3_Im(1:nfftz) = - ww3_Im(1:nfftz)
!          ww1(1:nfftx)=conjg(ww1(1:nfftx))
!          ww2(1:nffty)=conjg(ww2(1:nffty))
!          ww3(1:nfftz)=conjg(ww3(1:nfftz))
      endif ! isw.lt.0

!     perform fourier transform in X direction
      
      kkk=0
      num=nfftx/2
      do l=1,nu1
        do while(kkk.lt.nfftx)
          do i=1,num
            iii=key1(kkk/num+1)
            kk1=kkk+1
            k12=kk1+num
            do j=1,nffty
              j1 = j - 1
              do k=1,nfftz
                k1 = k - 1    ! aaa(k12,j,k)
                ndx = (j1+k1*nffty)*nfftx + k12 
                ndx1 = (j1+k1*nffty)*nfftx + kk1 
!print*, 'in fft j k ndx=',j,k,k12,ndx
!print*, 'in fft j k ndx1=',j,k,kk1,ndx1
                q0 = qqq1_Re(ndx) ; q1 = qqq1_Im(ndx) ;
                w0 = ww1_Re(iii)  ; w1 =  ww1_Im(iii)
                ttt_Re = q0*w0 - q1*w1
                ttt_Im = q0*w1 + w0*q1
                qqq1_Re(ndx) = qqq1_Re(ndx1) - ttt_Re
                qqq1_Im(ndx) = qqq1_Im(ndx1) - ttt_Im
                qqq1_Re(ndx1) = qqq1_Re(ndx1) + ttt_Re
                qqq1_Im(ndx1) = qqq1_Im(ndx1) + ttt_Im
!                ttt=aaa(k12,j,k)*ww1(iii)
!                aaa(k12,j,k)=aaa(kk1,j,k)-ttt
!                aaa(kk1,j,k)=aaa(kk1,j,k)+ttt
              enddo  !  k=1,nfftz
            enddo    ! j=1,nffty
            kkk=kkk+1
          enddo  !   i=1,num
          kkk=kkk+num
        enddo  !  while(kkk.lt.nfftx)
        kkk=0
        num=num/2
      enddo !  l=1,nu1
!     unscramble the fft using bit address array
      do kkk=1,nfftx
        iii=key1(kkk)
        if(iii.gt.kkk)then
          do j=1,nffty
            j1 = j - 1
            do k=1,nfftz   ! aaa(kkk,j,k)=aaa(iii,j,k)
                k1 = k - 1
                ndx = (j1+k1*nffty)*nfftx + kkk 
                ndx1 = (j1+k1*nffty)*nfftx + iii 
                ttt_Re = qqq1_Re(ndx)
                ttt_Im = qqq1_Im(ndx)
                qqq1_Re(ndx) = qqq1_Re(ndx1) 
                qqq1_Im(ndx) = qqq1_Im(ndx1) 
                qqq1_Re(ndx1) = ttt_Re 
                qqq1_Im(ndx1) = ttt_Im
!              ttt=aaa(kkk,j,k)
!              aaa(kkk,j,k)=aaa(iii,j,k)
!              aaa(iii,j,k)=ttt
            enddo !k=1,nfftz
          enddo   !j=1,nffty
        endif     !(iii.gt.kkk)
      enddo       !kkk=1,nfftx
!     perform fourier transform in Y direction
 
      kkk=0
      num=nffty/2
      do l=1,nu2
        do while(kkk.lt.nffty)
          do i=1,num
            iii=key2(kkk/num+1)
            kk1=kkk+1
            k12=kk1+num
            do j=1,nfftx
              do k=1,nfftz
                k1 = k - 1
                ndx = (k12-1+k1*nffty)*nfftx + j     ! aaa(j,k12,k)
                ndx1 = (kk1-1+k1*nffty)*nfftx + j
                q0 = qqq1_Re(ndx) ; q1 = qqq1_Im(ndx) ; 
                w0 = ww2_Re(iii); w1 =  ww2_Im(iii)
                ttt_Re = q0*w0 - q1*w1
                ttt_Im = q0*w1 + w0*q1
                qqq1_Re(ndx) = qqq1_Re(ndx1) - ttt_Re
                qqq1_Im(ndx) = qqq1_Im(ndx1) - ttt_Im
                qqq1_Re(ndx1) = qqq1_Re(ndx1) + ttt_Re
                qqq1_Im(ndx1) = qqq1_Im(ndx1) + ttt_Im
!                ttt=aaa(j,k12,k)*ww2(iii)
!                aaa(j,k12,k)=aaa(j,kk1,k)-ttt
!                aaa(j,kk1,k)=aaa(j,kk1,k)+ttt
              enddo
            enddo
            kkk=kkk+1
          enddo
          kkk=kkk+num
        enddo
        kkk=0
        num=num/2
      enddo

!     unscramble the fft using bit address array
      
      do kkk=1,nffty
        iii=key2(kkk)
        if(iii.gt.kkk)then
          do j=1,nfftx
            do k=1,nfftz   !  aaa(j,kkk,k)=aaa(j,iii,k
                k1 = k - 1
                ndx = (kkk-1+k1*nffty)*nfftx + j 
                ndx1 = (iii-1+k1*nffty)*nfftx + j 
                ttt_Re = qqq1_Re(ndx)
                ttt_Im = qqq1_Im(ndx)
                qqq1_Re(ndx) = qqq1_Re(ndx1)
                qqq1_Im(ndx) = qqq1_Im(ndx1)
                qqq1_Re(ndx1) = ttt_Re
                qqq1_Im(ndx1) = ttt_Im

!              ttt=aaa(j,kkk,k)
!              aaa(j,kkk,k)=aaa(j,iii,k)
!              aaa(j,iii,k)=ttt
            enddo
          enddo
        endif
      enddo

!     perform fourier transform in Z direction

      kkk=0
      num=nfftz/2
      do l=1,nu3
        do while(kkk.lt.nfftz)
          do i=1,num
            iii=key3(kkk/num+1)
            kk1=kkk+1
            k12=kk1+num
            do j=1,nfftx
              do k=1,nffty  ! aaa(j,k,k12)
                k1 = k - 1
                ndx = (k1+(k12-1)*nffty)*nfftx + j
                ndx1 = (k1+(kk1-1)*nffty)*nfftx + j
                q0 = qqq1_Re(ndx) ; q1 = qqq1_Im(ndx) ;
                w0 = ww3_Re(iii); w1 =  ww3_Im(iii)
                ttt_Re = q0*w0 - q1*w1
                ttt_Im = q0*w1 + w0*q1
                qqq1_Re(ndx) = qqq1_Re(ndx1) - ttt_Re
                qqq1_Im(ndx) = qqq1_Im(ndx1) - ttt_Im
                qqq1_Re(ndx1) = qqq1_Re(ndx1) + ttt_Re
                qqq1_Im(ndx1) = qqq1_Im(ndx1) + ttt_Im
!                ttt=aaa(j,k,k12)*ww3(iii)
!                aaa(j,k,k12)=aaa(j,k,kk1)-ttt
!                aaa(j,k,kk1)=aaa(j,k,kk1)+ttt
              enddo
            enddo
            kkk=kkk+1
          enddo
          kkk=kkk+num
        enddo
        kkk=0
        num=num/2
      enddo

!     unscramble the fft using bit address array
      
      do kkk=1,nfftz
        iii=key3(kkk)
        if(iii.gt.kkk)then
          do j=1,nfftx
            do k=1,nffty  ! =aaa(j,k,kkk)
                k1 = k - 1
                ndx = (k1+(kkk-1)*nffty)*nfftx + j
                ndx1 = (k1+(iii-1)*nffty)*nfftx + j
                ttt_Re = qqq1_Re(ndx)
                ttt_Im = qqq1_Im(ndx)
                qqq1_Re(ndx) = qqq1_Re(ndx1)
                qqq1_Im(ndx) = qqq1_Im(ndx1)
                qqq1_Re(ndx1) = ttt_Re
                qqq1_Im(ndx1) = ttt_Im
!              ttt=aaa(j,k,kkk)
!              aaa(j,k,kkk)=aaa(j,k,iii)
!              aaa(j,k,iii)=ttt
            enddo
          enddo
        endif
      enddo

!     restore exponentials to unconjugated values if necessary
      
      if(isw.lt.0)then
         ww1_Im(:) = -ww1_Im(:) 
         ww2_Im(:) = -ww2_Im(:)
         ww3_Im(:) = -ww3_Im(:)
!         ww3(1:nfftz) = conjg(ww3(1:nfftz))
      endif ! isw.lt.0
    
  end subroutine  dlpfft3_1_ARR


!!!!!!!!!!!!!!!!!!!!!


 subroutine dlpfft3_2_ARR(ind,isw)
     use variables_smpe, only : nfftx,nffty,nfftz,&
                                ww1_Re,ww1_Im,ww2_Re,ww2_Im,ww3_Re,ww3_Im,key1,key2,key3,&
                                qqq1_Re,qqq1_Im,qqq2_Re,qqq2_Im
     implicit none
!   Modified version to work with 1D-array and 2-arrays.
!***********************************************************************
!
!     dl-poly 3D fast fourier transform routine (in place)
!
!     copyright daresbury laboratory 1998
!
!     author w smith july 1998
!
!     wl
!     2002/05/31 13:58:07
!     1.2
!     Exp
!
!***********************************************************************

      integer, intent(IN) :: ind,isw

      logical lkx,lky,lkz
      real(8) ttt, ttt1_Re, ttt1_Im, ttt_Re, ttt_Im
      integer, save :: nu1,nu2,nu3
      integer idm
      integer kkk,iii,jjj,ii,kk,jj,i,j,k,jj2,jjj2,num,kk1,k12,l
      integer ndx, ndx1
      real(8) arg
      real(8) q0,q1,w0,w1
      integer i1,j1,k1
      real(8), parameter ::  tpi = 6.283185307179586d0
      logical, save :: l_very_first_pass = .true.

      if(ind.gt.0)then

!     check FFT array dimensions

        idm=1 ; lkx=.true.  ;  lky=.true.  ;  lkz=.true.

        do i=1,30
          idm=2*idm
          if(idm.eq.nfftx)then
            lkx=.false.
            nu1=i
          endif
          if(idm.eq.nffty)then
            lky=.false.
            nu2=i
          endif
          if(idm.eq.nfftz)then
            lkz=.false.
            nu3=i
          endif
        enddo ! i=1,30

        if(lkx.or.lky.or.lkz)then
          write(*,*)'error -dl-poly FFT array not 2**N',nfftx,nffty,nfftz
          stop
        endif
        do kkk=1,nfftx
          iii=0
          jjj=kkk-1
          do j=1,nu1
            jj2=jjj/2
            iii=2*(iii-jj2)+jjj
            jjj=jj2
          enddo
          key1(kkk)=iii+1
        enddo

        do kkk=1,nffty
          iii=0
          jjj=kkk-1
          do j=1,nu2
            jj2=jjj/2
            iii=2*(iii-jj2)+jjj
            jjj=jj2
          enddo
          key2(kkk)=iii+1
        enddo

        do kkk=1,nfftz
          iii=0
          jjj=kkk-1
          do j=1,nu3
            jj2=jjj/2
            iii=2*(iii-jj2)+jjj
            jjj=jj2
          enddo
          key3(kkk)=iii+1
        enddo

!     initialise complex exponential factors
        ww1_Re(1)=1.0d0
        ww1_Im(1)=0.0d0
        do i=1,nfftx/2
          arg=(tpi/dble(nfftx))*dble(i)
          ww1_Re(i+1) = dcos(arg)
          ww1_Im(i+1) = dsin(arg)
          ww1_Re(nfftx+1-i) =   ww1_Re(i+1)
          ww1_Im(nfftx+1-i) = - ww1_Im(i+1)
!          ww1(i+1)=cmplx(cos(arg),sin(arg),kind=8)
!          ww1(nfftx+1-i)=conjg(ww1(i+1))
        enddo

        ww2_Re(1)=1.d0
        ww2_Im(1)=0.d0
        do i=1,nffty/2
          arg=(tpi/dble(nffty))*dble(i)
          ww2_Re(i+1) = dcos(arg)
          ww2_Im(i+1) = dsin(arg)
          ww2_Re(nffty+1-i) =   ww2_Re(i+1)
          ww2_Im(nffty+1-i) = - ww2_Im(i+1)
!          ww2(i+1)=cmplx(cos(arg),sin(arg),kind=8)
!          ww2(nffty+1-i)=conjg(ww2(i+1))
        enddo

        ww3_Re(1)=1.d0
        ww3_Im(1)=0.d0
        do i=1,nfftz/2
          arg=(tpi/dble(nfftz))*dble(i)
          ww3_Re(i+1) = dcos(arg)
          ww3_Im(i+1) = dsin(arg)
          ww3_Re(nfftz+1-i) =   ww3_Re(i+1)
          ww3_Im(nfftz+1-i) = - ww3_Im(i+1)
!          ww3(i+1)=cmplx(cos(arg),sin(arg),kind=8)
!          ww3(nfftz+1-i)=conjg(ww3(i+1))
        enddo
        return
      endif ! ind ==0
!     take conjugate of exponentials if required

      if(isw.lt.0)then
          ww1_Im(1:nfftx) = - ww1_Im(1:nfftx)
          ww2_Im(1:nffty) = - ww2_Im(1:nffty)
          ww3_Im(1:nfftz) = - ww3_Im(1:nfftz)
!          ww1(1:nfftx)=conjg(ww1(1:nfftx))
!          ww2(1:nffty)=conjg(ww2(1:nffty))
!          ww3(1:nfftz)=conjg(ww3(1:nfftz))
      endif ! isw.lt.0

!     perform fourier transform in X direction

      kkk=0
      num=nfftx/2
      do l=1,nu1
        do while(kkk.lt.nfftx)
          do i=1,num
            iii=key1(kkk/num+1)
            kk1=kkk+1
            k12=kk1+num
            do j=1,nffty
              j1 = j - 1
              do k=1,nfftz
                k1 = k - 1    ! aaa(k12,j,k)
                ndx = (j1+k1*nffty)*nfftx + k12
                ndx1 = (j1+k1*nffty)*nfftx + kk1
!print*, 'in fft j k ndx=',j,k,k12,ndx
!print*, 'in fft j k ndx1=',j,k,kk1,ndx1
                q0 = qqq1_Re(ndx) ; q1 = qqq1_Im(ndx) ;
                w0 = ww1_Re(iii)  ; w1 =  ww1_Im(iii)
                ttt_Re = q0*w0 - q1*w1
                ttt_Im = q0*w1 + w0*q1
                qqq1_Re(ndx) = qqq1_Re(ndx1) - ttt_Re
                qqq1_Im(ndx) = qqq1_Im(ndx1) - ttt_Im
                qqq1_Re(ndx1) = qqq1_Re(ndx1) + ttt_Re
                qqq1_Im(ndx1) = qqq1_Im(ndx1) + ttt_Im

                q0 = qqq2_Re(ndx) ; q1 = qqq2_Im(ndx) ;
                ttt_Re = q0*w0 - q1*w1
                ttt_Im = q0*w1 + w0*q1
                qqq2_Re(ndx) = qqq2_Re(ndx1) - ttt_Re
                qqq2_Im(ndx) = qqq2_Im(ndx1) - ttt_Im
                qqq2_Re(ndx1) = qqq2_Re(ndx1) + ttt_Re
                qqq2_Im(ndx1) = qqq2_Im(ndx1) + ttt_Im

!                ttt=aaa(k12,j,k)*ww1(iii)
!                aaa(k12,j,k)=aaa(kk1,j,k)-ttt
!                aaa(kk1,j,k)=aaa(kk1,j,k)+ttt
              enddo  !  k=1,nfftz
            enddo    ! j=1,nffty
            kkk=kkk+1
          enddo  !   i=1,num
          kkk=kkk+num
        enddo  !  while(kkk.lt.nfftx)
        kkk=0
        num=num/2
      enddo !  l=1,nu1
!     unscramble the fft using bit address array
      do kkk=1,nfftx
        iii=key1(kkk)
        if(iii.gt.kkk)then
          do j=1,nffty
            j1 = j - 1
            do k=1,nfftz   ! aaa(kkk,j,k)=aaa(iii,j,k)
                k1 = k - 1
                ndx = (j1+k1*nffty)*nfftx + kkk
                ndx1 = (j1+k1*nffty)*nfftx + iii
                ttt_Re = qqq1_Re(ndx)
                ttt_Im = qqq1_Im(ndx)
                qqq1_Re(ndx) = qqq1_Re(ndx1)
                qqq1_Im(ndx) = qqq1_Im(ndx1)
                qqq1_Re(ndx1) = ttt_Re
                qqq1_Im(ndx1) = ttt_Im

                ttt_Re = qqq2_Re(ndx)
                ttt_Im = qqq2_Im(ndx)
                qqq2_Re(ndx) = qqq2_Re(ndx1)
                qqq2_Im(ndx) = qqq2_Im(ndx1)
                qqq2_Re(ndx1) = ttt_Re
                qqq2_Im(ndx1) = ttt_Im

!              ttt=aaa(kkk,j,k)
!              aaa(kkk,j,k)=aaa(iii,j,k)
!              aaa(iii,j,k)=ttt
            enddo !k=1,nfftz
          enddo   !j=1,nffty
        endif     !(iii.gt.kkk)
      enddo       !kkk=1,nfftx
!     perform fourier transform in Y direction
      kkk=0
      num=nffty/2
      do l=1,nu2
        do while(kkk.lt.nffty)
          do i=1,num
            iii=key2(kkk/num+1)
            kk1=kkk+1
            k12=kk1+num
            do j=1,nfftx
              do k=1,nfftz
                k1 = k - 1
                ndx = (k12-1+k1*nffty)*nfftx + j     ! aaa(j,k12,k)
                ndx1 = (kk1-1+k1*nffty)*nfftx + j
                q0 = qqq1_Re(ndx) ; q1 = qqq1_Im(ndx) ;
                w0 = ww2_Re(iii); w1 =  ww2_Im(iii)
                ttt_Re = q0*w0 - q1*w1
                ttt_Im = q0*w1 + w0*q1
                qqq1_Re(ndx) = qqq1_Re(ndx1) - ttt_Re
                qqq1_Im(ndx) = qqq1_Im(ndx1) - ttt_Im
                qqq1_Re(ndx1) = qqq1_Re(ndx1) + ttt_Re
                qqq1_Im(ndx1) = qqq1_Im(ndx1) + ttt_Im

                q0 = qqq2_Re(ndx) ; q1 = qqq2_Im(ndx) ;
                ttt_Re = q0*w0 - q1*w1
                ttt_Im = q0*w1 + w0*q1
                qqq2_Re(ndx) = qqq2_Re(ndx1) - ttt_Re
                qqq2_Im(ndx) = qqq2_Im(ndx1) - ttt_Im
                qqq2_Re(ndx1) = qqq2_Re(ndx1) + ttt_Re
                qqq2_Im(ndx1) = qqq2_Im(ndx1) + ttt_Im

!                ttt=aaa(j,k12,k)*ww2(iii)
!                aaa(j,k12,k)=aaa(j,kk1,k)-ttt
!                aaa(j,kk1,k)=aaa(j,kk1,k)+ttt
              enddo
            enddo
            kkk=kkk+1
          enddo
          kkk=kkk+num
        enddo
        kkk=0
        num=num/2
      enddo

!     unscramble the fft using bit address array
      do kkk=1,nffty
        iii=key2(kkk)
        if(iii.gt.kkk)then
          do j=1,nfftx
            do k=1,nfftz   !  aaa(j,kkk,k)=aaa(j,iii,k
                k1 = k - 1
                ndx = (kkk-1+k1*nffty)*nfftx + j
                ndx1 = (iii-1+k1*nffty)*nfftx + j
                ttt_Re = qqq1_Re(ndx)
                ttt_Im = qqq1_Im(ndx)
                qqq1_Re(ndx) = qqq1_Re(ndx1)
                qqq1_Im(ndx) = qqq1_Im(ndx1)
                qqq1_Re(ndx1) = ttt_Re
                qqq1_Im(ndx1) = ttt_Im

                ttt_Re = qqq2_Re(ndx)
                ttt_Im = qqq2_Im(ndx)
                qqq2_Re(ndx) = qqq2_Re(ndx1)
                qqq2_Im(ndx) = qqq2_Im(ndx1)
                qqq2_Re(ndx1) = ttt_Re
                qqq2_Im(ndx1) = ttt_Im


!              ttt=aaa(j,kkk,k)
!              aaa(j,kkk,k)=aaa(j,iii,k)
!              aaa(j,iii,k)=ttt
            enddo
          enddo
        endif
      enddo

!     perform fourier transform in Z direction
      kkk=0
      num=nfftz/2
      do l=1,nu3
        do while(kkk.lt.nfftz)
          do i=1,num
            iii=key3(kkk/num+1)
            kk1=kkk+1
            k12=kk1+num
            do j=1,nfftx
              do k=1,nffty  ! aaa(j,k,k12)
                k1 = k - 1
                ndx = (k1+(k12-1)*nffty)*nfftx + j
                ndx1 = (k1+(kk1-1)*nffty)*nfftx + j
                q0 = qqq1_Re(ndx) ; q1 = qqq1_Im(ndx) ;
                w0 = ww3_Re(iii); w1 =  ww3_Im(iii)
                ttt_Re = q0*w0 - q1*w1
                ttt_Im = q0*w1 + w0*q1
                qqq1_Re(ndx) = qqq1_Re(ndx1) - ttt_Re
                qqq1_Im(ndx) = qqq1_Im(ndx1) - ttt_Im
                qqq1_Re(ndx1) = qqq1_Re(ndx1) + ttt_Re
                qqq1_Im(ndx1) = qqq1_Im(ndx1) + ttt_Im

                q0 = qqq2_Re(ndx) ; q1 = qqq2_Im(ndx) ;
                ttt_Re = q0*w0 - q1*w1
                ttt_Im = q0*w1 + w0*q1
                qqq2_Re(ndx) = qqq2_Re(ndx1) - ttt_Re
                qqq2_Im(ndx) = qqq2_Im(ndx1) - ttt_Im
                qqq2_Re(ndx1) = qqq2_Re(ndx1) + ttt_Re
                qqq2_Im(ndx1) = qqq2_Im(ndx1) + ttt_Im

!                ttt=aaa(j,k,k12)*ww3(iii)
!                aaa(j,k,k12)=aaa(j,k,kk1)-ttt
!                aaa(j,k,kk1)=aaa(j,k,kk1)+ttt
              enddo
            enddo
            kkk=kkk+1
          enddo
          kkk=kkk+num
        enddo
        kkk=0
        num=num/2
      enddo

!     unscramble the fft using bit address array
      do kkk=1,nfftz
        iii=key3(kkk)
        if(iii.gt.kkk)then
          do j=1,nfftx
            do k=1,nffty  ! =aaa(j,k,kkk)
                k1 = k - 1
                ndx = (k1+(kkk-1)*nffty)*nfftx + j
                ndx1 = (k1+(iii-1)*nffty)*nfftx + j
                ttt_Re = qqq1_Re(ndx)
                ttt_Im = qqq1_Im(ndx)
                qqq1_Re(ndx) = qqq1_Re(ndx1)
                qqq1_Im(ndx) = qqq1_Im(ndx1)
                qqq1_Re(ndx1) = ttt_Re
                qqq1_Im(ndx1) = ttt_Im

                ttt_Re = qqq2_Re(ndx)
                ttt_Im = qqq2_Im(ndx)
                qqq2_Re(ndx) = qqq2_Re(ndx1)
                qqq2_Im(ndx) = qqq2_Im(ndx1)
                qqq2_Re(ndx1) = ttt_Re
                qqq2_Im(ndx1) = ttt_Im

!              ttt=aaa(j,k,kkk)
!              aaa(j,k,kkk)=aaa(j,k,iii)
!              aaa(j,k,iii)=ttt
            enddo
          enddo
        endif
      enddo

!     restore exponentials to unconjugated values if necessary

      if(isw.lt.0)then
         ww1_Im(:) = -ww1_Im(:)
         ww2_Im(:) = -ww2_Im(:)
         ww3_Im(:) = -ww3_Im(:)
!         ww3(1:nfftz) = conjg(ww3(1:nfftz))
      endif ! isw.lt.0

  end subroutine  dlpfft3_2_ARR

 subroutine dlpfft3_4_ARR(ind,isw)
     use variables_smpe, only : nfftx,nffty,nfftz,&
                                ww1_Re,ww1_Im,ww2_Re,ww2_Im,ww3_Re,ww3_Im,key1,key2,key3,&
                                qqq1_Re,qqq1_Im,qqq2_Re,qqq2_Im,qqq3_Re,qqq3_Im,qqq4_Re,qqq4_Im
     implicit none
!   Modified version to work with 1D-array and 4-arrays.
!***********************************************************************
!
!     dl-poly 3D fast fourier transform routine (in place)
!
!     copyright daresbury laboratory 1998
!
!     author w smith july 1998
!
!     wl
!     2002/05/31 13:58:07
!     1.2
!     Exp
!
!***********************************************************************

      integer, intent(IN) :: ind,isw

      logical lkx,lky,lkz
      real(8) ttt, ttt1_Re, ttt1_Im, ttt_Re, ttt_Im
      integer, save :: nu1,nu2,nu3
      integer idm
      integer kkk,iii,jjj,ii,kk,jj,i,j,k,jj2,jjj2,num,kk1,k12,l
      integer ndx, ndx1
      real(8) arg
      real(8) q0,q1,w0,w1
      integer i1,j1,k1
      real(8), parameter ::  tpi = 6.283185307179586d0
      logical, save :: l_very_first_pass = .true.

      if(ind.gt.0)then

!     check FFT array dimensions

        idm=1 ; lkx=.true.  ;  lky=.true.  ;  lkz=.true.

        do i=1,30
          idm=2*idm
          if(idm.eq.nfftx)then
            lkx=.false.
            nu1=i
          endif
          if(idm.eq.nffty)then
            lky=.false.
            nu2=i
          endif
          if(idm.eq.nfftz)then
            lkz=.false.
            nu3=i
          endif
        enddo ! i=1,30

        if(lkx.or.lky.or.lkz)then
          write(*,*)'error -dl-poly FFT array not 2**N',nfftx,nffty,nfftz
          stop
        endif
        do kkk=1,nfftx
          iii=0
          jjj=kkk-1
          do j=1,nu1
            jj2=jjj/2
            iii=2*(iii-jj2)+jjj
            jjj=jj2
          enddo
          key1(kkk)=iii+1
        enddo

        do kkk=1,nffty
          iii=0
          jjj=kkk-1
          do j=1,nu2
            jj2=jjj/2
            iii=2*(iii-jj2)+jjj
            jjj=jj2
          enddo
          key2(kkk)=iii+1
        enddo

        do kkk=1,nfftz
          iii=0
          jjj=kkk-1
          do j=1,nu3
            jj2=jjj/2
            iii=2*(iii-jj2)+jjj
            jjj=jj2
          enddo
          key3(kkk)=iii+1
        enddo

!     initialise complex exponential factors
        ww1_Re(1)=1.0d0
        ww1_Im(1)=0.0d0
        do i=1,nfftx/2
          arg=(tpi/dble(nfftx))*dble(i)
          ww1_Re(i+1) = dcos(arg)
          ww1_Im(i+1) = dsin(arg)
          ww1_Re(nfftx+1-i) =   ww1_Re(i+1)
          ww1_Im(nfftx+1-i) = - ww1_Im(i+1)
!          ww1(i+1)=cmplx(cos(arg),sin(arg),kind=8)
!          ww1(nfftx+1-i)=conjg(ww1(i+1))
        enddo

        ww2_Re(1)=1.d0
        ww2_Im(1)=0.d0
        do i=1,nffty/2
          arg=(tpi/dble(nffty))*dble(i)
          ww2_Re(i+1) = dcos(arg)
          ww2_Im(i+1) = dsin(arg)
          ww2_Re(nffty+1-i) =   ww2_Re(i+1)
          ww2_Im(nffty+1-i) = - ww2_Im(i+1)
!          ww2(i+1)=cmplx(cos(arg),sin(arg),kind=8)
!          ww2(nffty+1-i)=conjg(ww2(i+1))
        enddo

        ww3_Re(1)=1.d0
        ww3_Im(1)=0.d0
        do i=1,nfftz/2
          arg=(tpi/dble(nfftz))*dble(i)
          ww3_Re(i+1) = dcos(arg)
          ww3_Im(i+1) = dsin(arg)
          ww3_Re(nfftz+1-i) =   ww3_Re(i+1)
          ww3_Im(nfftz+1-i) = - ww3_Im(i+1)
!          ww3(i+1)=cmplx(cos(arg),sin(arg),kind=8)
!          ww3(nfftz+1-i)=conjg(ww3(i+1))
        enddo
        return
      endif ! ind ==0
!     take conjugate of exponentials if required

      if(isw.lt.0)then
          ww1_Im(1:nfftx) = - ww1_Im(1:nfftx)
          ww2_Im(1:nffty) = - ww2_Im(1:nffty)
          ww3_Im(1:nfftz) = - ww3_Im(1:nfftz)
!          ww1(1:nfftx)=conjg(ww1(1:nfftx))
!          ww2(1:nffty)=conjg(ww2(1:nffty))
!          ww3(1:nfftz)=conjg(ww3(1:nfftz))
      endif ! isw.lt.0

!     perform fourier transform in X direction

      kkk=0
      num=nfftx/2
      do l=1,nu1
        do while(kkk.lt.nfftx)
          do i=1,num
            iii=key1(kkk/num+1)
            kk1=kkk+1
            k12=kk1+num
            do j=1,nffty
              j1 = j - 1
              do k=1,nfftz
                k1 = k - 1    ! aaa(k12,j,k)
                ndx = (j1+k1*nffty)*nfftx + k12
                ndx1 = (j1+k1*nffty)*nfftx + kk1
!print*, 'in fft j k ndx=',j,k,k12,ndx
!print*, 'in fft j k ndx1=',j,k,kk1,ndx1
                q0 = qqq1_Re(ndx) ; q1 = qqq1_Im(ndx) ;
                w0 = ww1_Re(iii)  ; w1 =  ww1_Im(iii)
                ttt_Re = q0*w0 - q1*w1
                ttt_Im = q0*w1 + w0*q1
                qqq1_Re(ndx) = qqq1_Re(ndx1) - ttt_Re
                qqq1_Im(ndx) = qqq1_Im(ndx1) - ttt_Im
                qqq1_Re(ndx1) = qqq1_Re(ndx1) + ttt_Re
                qqq1_Im(ndx1) = qqq1_Im(ndx1) + ttt_Im

                q0 = qqq2_Re(ndx) ; q1 = qqq2_Im(ndx) ;
                ttt_Re = q0*w0 - q1*w1
                ttt_Im = q0*w1 + w0*q1
                qqq2_Re(ndx) = qqq2_Re(ndx1) - ttt_Re
                qqq2_Im(ndx) = qqq2_Im(ndx1) - ttt_Im
                qqq2_Re(ndx1) = qqq2_Re(ndx1) + ttt_Re
                qqq2_Im(ndx1) = qqq2_Im(ndx1) + ttt_Im

                q0 = qqq3_Re(ndx) ; q1 = qqq3_Im(ndx) ;
                ttt_Re = q0*w0 - q1*w1
                ttt_Im = q0*w1 + w0*q1
                qqq3_Re(ndx) = qqq3_Re(ndx1) - ttt_Re
                qqq3_Im(ndx) = qqq3_Im(ndx1) - ttt_Im
                qqq3_Re(ndx1) = qqq3_Re(ndx1) + ttt_Re
                qqq3_Im(ndx1) = qqq3_Im(ndx1) + ttt_Im

                q0 = qqq4_Re(ndx) ; q1 = qqq4_Im(ndx) ;
                ttt_Re = q0*w0 - q1*w1
                ttt_Im = q0*w1 + w0*q1
                qqq4_Re(ndx) = qqq4_Re(ndx1) - ttt_Re
                qqq4_Im(ndx) = qqq4_Im(ndx1) - ttt_Im
                qqq4_Re(ndx1) = qqq4_Re(ndx1) + ttt_Re
                qqq4_Im(ndx1) = qqq4_Im(ndx1) + ttt_Im


!                ttt=aaa(k12,j,k)*ww1(iii)
!                aaa(k12,j,k)=aaa(kk1,j,k)-ttt
!                aaa(kk1,j,k)=aaa(kk1,j,k)+ttt
              enddo  !  k=1,nfftz
            enddo    ! j=1,nffty
            kkk=kkk+1
          enddo  !   i=1,num
          kkk=kkk+num
        enddo  !  while(kkk.lt.nfftx)
        kkk=0
        num=num/2
      enddo !  l=1,nu1
!     unscramble the fft using bit address array
      do kkk=1,nfftx
        iii=key1(kkk)
        if(iii.gt.kkk)then
          do j=1,nffty
            j1 = j - 1
            do k=1,nfftz   ! aaa(kkk,j,k)=aaa(iii,j,k)
                k1 = k - 1
                ndx = (j1+k1*nffty)*nfftx + kkk
                ndx1 = (j1+k1*nffty)*nfftx + iii
                ttt_Re = qqq1_Re(ndx)
                ttt_Im = qqq1_Im(ndx)
                qqq1_Re(ndx) = qqq1_Re(ndx1)
                qqq1_Im(ndx) = qqq1_Im(ndx1)
                qqq1_Re(ndx1) = ttt_Re
                qqq1_Im(ndx1) = ttt_Im

                ttt_Re = qqq2_Re(ndx)
                ttt_Im = qqq2_Im(ndx)
                qqq2_Re(ndx) = qqq2_Re(ndx1)
                qqq2_Im(ndx) = qqq2_Im(ndx1)
                qqq2_Re(ndx1) = ttt_Re
                qqq2_Im(ndx1) = ttt_Im

                ttt_Re = qqq3_Re(ndx)
                ttt_Im = qqq3_Im(ndx)
                qqq3_Re(ndx) = qqq3_Re(ndx1)
                qqq3_Im(ndx) = qqq3_Im(ndx1)
                qqq3_Re(ndx1) = ttt_Re
                qqq3_Im(ndx1) = ttt_Im

                ttt_Re = qqq4_Re(ndx)
                ttt_Im = qqq4_Im(ndx)
                qqq4_Re(ndx) = qqq4_Re(ndx1)
                qqq4_Im(ndx) = qqq4_Im(ndx1)
                qqq4_Re(ndx1) = ttt_Re
                qqq4_Im(ndx1) = ttt_Im

!              ttt=aaa(kkk,j,k)
!              aaa(kkk,j,k)=aaa(iii,j,k)
!              aaa(iii,j,k)=ttt
            enddo !k=1,nfftz
          enddo   !j=1,nffty
        endif     !(iii.gt.kkk)
      enddo       !kkk=1,nfftx
!     perform fourier transform in Y direction
      kkk=0
      num=nffty/2
      do l=1,nu2
        do while(kkk.lt.nffty)
          do i=1,num
            iii=key2(kkk/num+1)
            kk1=kkk+1
            k12=kk1+num
            do j=1,nfftx
              do k=1,nfftz
                k1 = k - 1
                ndx = (k12-1+k1*nffty)*nfftx + j     ! aaa(j,k12,k)
                ndx1 = (kk1-1+k1*nffty)*nfftx + j
                q0 = qqq1_Re(ndx) ; q1 = qqq1_Im(ndx) ;
                w0 = ww2_Re(iii); w1 =  ww2_Im(iii)
                ttt_Re = q0*w0 - q1*w1
                ttt_Im = q0*w1 + w0*q1
                qqq1_Re(ndx) = qqq1_Re(ndx1) - ttt_Re
                qqq1_Im(ndx) = qqq1_Im(ndx1) - ttt_Im
                qqq1_Re(ndx1) = qqq1_Re(ndx1) + ttt_Re
                qqq1_Im(ndx1) = qqq1_Im(ndx1) + ttt_Im

                q0 = qqq2_Re(ndx) ; q1 = qqq2_Im(ndx) ;
                ttt_Re = q0*w0 - q1*w1
                ttt_Im = q0*w1 + w0*q1
                qqq2_Re(ndx) = qqq2_Re(ndx1) - ttt_Re
                qqq2_Im(ndx) = qqq2_Im(ndx1) - ttt_Im
                qqq2_Re(ndx1) = qqq2_Re(ndx1) + ttt_Re
                qqq2_Im(ndx1) = qqq2_Im(ndx1) + ttt_Im

                q0 = qqq3_Re(ndx) ; q1 = qqq3_Im(ndx) ;
                ttt_Re = q0*w0 - q1*w1
                ttt_Im = q0*w1 + w0*q1
                qqq3_Re(ndx) = qqq3_Re(ndx1) - ttt_Re
                qqq3_Im(ndx) = qqq3_Im(ndx1) - ttt_Im
                qqq3_Re(ndx1) = qqq3_Re(ndx1) + ttt_Re
                qqq3_Im(ndx1) = qqq3_Im(ndx1) + ttt_Im

                q0 = qqq4_Re(ndx) ; q1 = qqq4_Im(ndx) ;
                ttt_Re = q0*w0 - q1*w1
                ttt_Im = q0*w1 + w0*q1
                qqq4_Re(ndx) = qqq4_Re(ndx1) - ttt_Re
                qqq4_Im(ndx) = qqq4_Im(ndx1) - ttt_Im
                qqq4_Re(ndx1) = qqq4_Re(ndx1) + ttt_Re
                qqq4_Im(ndx1) = qqq4_Im(ndx1) + ttt_Im

!                ttt=aaa(j,k12,k)*ww2(iii)
!                aaa(j,k12,k)=aaa(j,kk1,k)-ttt
!                aaa(j,kk1,k)=aaa(j,kk1,k)+ttt
              enddo
            enddo
            kkk=kkk+1
          enddo
          kkk=kkk+num
        enddo
        kkk=0
        num=num/2
      enddo

!     unscramble the fft using bit address array
      do kkk=1,nffty
        iii=key2(kkk)
        if(iii.gt.kkk)then
          do j=1,nfftx
            do k=1,nfftz   !  aaa(j,kkk,k)=aaa(j,iii,k
                k1 = k - 1
                ndx = (kkk-1+k1*nffty)*nfftx + j
                ndx1 = (iii-1+k1*nffty)*nfftx + j
                ttt_Re = qqq1_Re(ndx)
                ttt_Im = qqq1_Im(ndx)
                qqq1_Re(ndx) = qqq1_Re(ndx1)
                qqq1_Im(ndx) = qqq1_Im(ndx1)
                qqq1_Re(ndx1) = ttt_Re
                qqq1_Im(ndx1) = ttt_Im

                ttt_Re = qqq2_Re(ndx)
                ttt_Im = qqq2_Im(ndx)
                qqq2_Re(ndx) = qqq2_Re(ndx1)
                qqq2_Im(ndx) = qqq2_Im(ndx1)
                qqq2_Re(ndx1) = ttt_Re
                qqq2_Im(ndx1) = ttt_Im

                ttt_Re = qqq3_Re(ndx)
                ttt_Im = qqq3_Im(ndx)
                qqq3_Re(ndx) = qqq3_Re(ndx1)
                qqq3_Im(ndx) = qqq3_Im(ndx1)
                qqq3_Re(ndx1) = ttt_Re
                qqq3_Im(ndx1) = ttt_Im

                ttt_Re = qqq4_Re(ndx)
                ttt_Im = qqq4_Im(ndx)
                qqq4_Re(ndx) = qqq4_Re(ndx1)
                qqq4_Im(ndx) = qqq4_Im(ndx1)
                qqq4_Re(ndx1) = ttt_Re
                qqq4_Im(ndx1) = ttt_Im

!              ttt=aaa(j,kkk,k)
!              aaa(j,kkk,k)=aaa(j,iii,k)
!              aaa(j,iii,k)=ttt
            enddo
          enddo
        endif
      enddo

!     perform fourier transform in Z direction
      kkk=0
      num=nfftz/2
      do l=1,nu3
        do while(kkk.lt.nfftz)
          do i=1,num
            iii=key3(kkk/num+1)
            kk1=kkk+1
            k12=kk1+num
            do j=1,nfftx
              do k=1,nffty  ! aaa(j,k,k12)
                k1 = k - 1
                ndx = (k1+(k12-1)*nffty)*nfftx + j
                ndx1 = (k1+(kk1-1)*nffty)*nfftx + j
                q0 = qqq1_Re(ndx) ; q1 = qqq1_Im(ndx) ;
                w0 = ww3_Re(iii); w1 =  ww3_Im(iii)
                ttt_Re = q0*w0 - q1*w1
                ttt_Im = q0*w1 + w0*q1
                qqq1_Re(ndx) = qqq1_Re(ndx1) - ttt_Re
                qqq1_Im(ndx) = qqq1_Im(ndx1) - ttt_Im
                qqq1_Re(ndx1) = qqq1_Re(ndx1) + ttt_Re
                qqq1_Im(ndx1) = qqq1_Im(ndx1) + ttt_Im

                q0 = qqq2_Re(ndx) ; q1 = qqq2_Im(ndx) ;
                ttt_Re = q0*w0 - q1*w1
                ttt_Im = q0*w1 + w0*q1
                qqq2_Re(ndx) = qqq2_Re(ndx1) - ttt_Re
                qqq2_Im(ndx) = qqq2_Im(ndx1) - ttt_Im
                qqq2_Re(ndx1) = qqq2_Re(ndx1) + ttt_Re
                qqq2_Im(ndx1) = qqq2_Im(ndx1) + ttt_Im

                q0 = qqq3_Re(ndx) ; q1 = qqq3_Im(ndx) ;
                ttt_Re = q0*w0 - q1*w1
                ttt_Im = q0*w1 + w0*q1
                qqq3_Re(ndx) = qqq3_Re(ndx1) - ttt_Re
                qqq3_Im(ndx) = qqq3_Im(ndx1) - ttt_Im
                qqq3_Re(ndx1) = qqq3_Re(ndx1) + ttt_Re
                qqq3_Im(ndx1) = qqq3_Im(ndx1) + ttt_Im

                q0 = qqq4_Re(ndx) ; q1 = qqq4_Im(ndx) ;
                ttt_Re = q0*w0 - q1*w1
                ttt_Im = q0*w1 + w0*q1
                qqq4_Re(ndx) = qqq4_Re(ndx1) - ttt_Re
                qqq4_Im(ndx) = qqq4_Im(ndx1) - ttt_Im
                qqq4_Re(ndx1) = qqq4_Re(ndx1) + ttt_Re
                qqq4_Im(ndx1) = qqq4_Im(ndx1) + ttt_Im


!                ttt=aaa(j,k,k12)*ww3(iii)
!                aaa(j,k,k12)=aaa(j,k,kk1)-ttt
!                aaa(j,k,kk1)=aaa(j,k,kk1)+ttt
              enddo
            enddo
            kkk=kkk+1
          enddo
          kkk=kkk+num
        enddo
        kkk=0
        num=num/2
      enddo

!     unscramble the fft using bit address array
      do kkk=1,nfftz
        iii=key3(kkk)
        if(iii.gt.kkk)then
          do j=1,nfftx
            do k=1,nffty  ! =aaa(j,k,kkk)
                k1 = k - 1
                ndx = (k1+(kkk-1)*nffty)*nfftx + j
                ndx1 = (k1+(iii-1)*nffty)*nfftx + j
                ttt_Re = qqq1_Re(ndx)
                ttt_Im = qqq1_Im(ndx)
                qqq1_Re(ndx) = qqq1_Re(ndx1)
                qqq1_Im(ndx) = qqq1_Im(ndx1)
                qqq1_Re(ndx1) = ttt_Re
                qqq1_Im(ndx1) = ttt_Im

                ttt_Re = qqq2_Re(ndx)
                ttt_Im = qqq2_Im(ndx)
                qqq2_Re(ndx) = qqq2_Re(ndx1)
                qqq2_Im(ndx) = qqq2_Im(ndx1)
                qqq2_Re(ndx1) = ttt_Re
                qqq2_Im(ndx1) = ttt_Im

                ttt_Re = qqq3_Re(ndx)
                ttt_Im = qqq3_Im(ndx)
                qqq3_Re(ndx) = qqq3_Re(ndx1)
                qqq3_Im(ndx) = qqq3_Im(ndx1)
                qqq3_Re(ndx1) = ttt_Re
                qqq3_Im(ndx1) = ttt_Im

                ttt_Re = qqq4_Re(ndx)
                ttt_Im = qqq4_Im(ndx)
                qqq4_Re(ndx) = qqq4_Re(ndx1)
                qqq4_Im(ndx) = qqq4_Im(ndx1)
                qqq4_Re(ndx1) = ttt_Re
                qqq4_Im(ndx1) = ttt_Im

!              ttt=aaa(j,k,kkk)
!              aaa(j,k,kkk)=aaa(j,k,iii)
!              aaa(j,k,iii)=ttt
            enddo
          enddo
        endif
      enddo

!     restore exponentials to unconjugated values if necessary

      if(isw.lt.0)then
         ww1_Im(:) = -ww1_Im(:)
         ww2_Im(:) = -ww2_Im(:)
         ww3_Im(:) = -ww3_Im(:)
!         ww3(1:nfftz) = conjg(ww3(1:nfftz))
      endif ! isw.lt.0

  end subroutine  dlpfft3_4_ARR

end module fft_3D_modified

