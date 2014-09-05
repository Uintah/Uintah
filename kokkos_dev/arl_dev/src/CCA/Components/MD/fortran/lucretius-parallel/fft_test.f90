
 include 'fft-3d.f90'

 program main
 use fft_3D
 implicit none
 integer, parameter :: nfftx = 128 , nffty = 128 , nfftz = 64
 integer i,j,k 
 real(8) V(nfftx,nffty,nfftz),V0(nfftx,nffty,nfftz)
 integer key1(nfftx),key2(nffty),key3(nfftz)
 complex(8) ww1(nfftx),ww2(nffty),ww3(nfftz)
 complex(8) VQ(nfftx,nffty,nfftz), VQ0(nfftx,nffty,nfftz), VQ1(nfftx,nffty,nfftz)
 real(8) pref,dn

 do i = 1, nfftx
   do j = 1, nffty
     do k = 1, nfftz
          pref = 0.00025d0*(dble(i-nfftx/2)**2+dble(j-nffty/2)**2+dble(k-nfftz/2)**2)
          V(i,j,k) = dexp(-pref**2)
          VQ(i,j,k) = cmplx(V(i,j,k),0.0d0,kind=8)
     enddo
   enddo
 enddo

 V0 = V ; VQ0 = VQ
 ! initialize
 call  dlpfft3(1,+1,nfftx,nffty,nfftz,key1,key2,key3,ww1,ww2,ww3,VQ)

 call dlpfft3(0,+1,nfftx,nffty,nfftz,key1,key2,key3,ww1,ww2,ww3,VQ)
 VQ1 = VQ
 call dlpfft3(0,-1,nfftx,nffty,nfftz,key1,key2,key3,ww1,ww2,ww3,VQ)
 
 do i = 1, nfftx
   do j = 1, nffty
     do k = 1, nfftz
     dn = dble(nfftx)*dble(nffty)*dble(nfftz)
print*,V(i,j,k)
print*, 'VQ1=',VQ1(i,j,k)
print*,i,j,k,'VQ VQ0=',VQ(i,j,k),VQ0(i,j,k)
print*, Real(VQ(i,j,k)),Real(VQ0(i,j,k)),Real(VQ(i,j,k))/Real(VQ0(i,j,k))/dn
read(*,*)
     enddo
   enddo
 enddo
 
! call fftwnd_f77_one(fplan,V,0)

!
 end program main 
 
 
 
