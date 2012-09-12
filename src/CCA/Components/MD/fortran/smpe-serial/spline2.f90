 module spline2
 implicit none
 public :: complex_spline2_xy ! complex splines for 3D and 2D-xy
 public :: complex_spline2_z  ! complex splines for 2D
 public :: real_spline2_and_deriv ! read splines + their derivatives
 public :: real_spline2_pp        ! real splines only

contains  
 subroutine complex_spline2_xy(order,nfft,spline) ! compute the complex (Fourier space) splines
 implicit none
 integer , intent(in) :: order, nfft
 real(8), intent(out) :: spline(0:nfft-1)
 real(8) knot(0:order-1),idi,invfft
 real(8) , parameter :: SMALL = 1.0d-10
 real(8) , parameter :: twopi = 6.28318530717959d0
 integer i,j,k
 real(8) a,b,arg

 knot(0) = 1.0d0 ; knot(1) = 0.0d0
 do i = 2, order-1
   idi = 1.0d0 / dble(i)
   knot(i) = 0.0d0
   do j = 1, i-1
     knot(i-j) = dble(j)*knot(i-j-1)+dble(i-j+1)*knot(i-j)
     knot(i-j) = knot(i-j) * idi
   enddo
  knot(0) = idi * knot(0)
 enddo
 
 invfft = 1.0d0 / dble(nfft)
 do i = 0, nfft-1
   a=0.0d0 ; b = 0.0d0
   do j = 0, order-1
     arg = twopi*dble(i)*dble(j)*invfft
     a = a + knot(j) * dcos(arg)
     b = b + knot(j) * dsin(arg)
   enddo
   spline(i) = a*a+b*b
 enddo

 do i = 1, nfft-1
   if (spline(i) < SMALL ) then
     spline(i) = 0.5d0*(spline(i+1)-spline(i-1))
   endif
 enddo 

 end subroutine complex_spline2_xy

 subroutine complex_spline2_z(order,h_cut,nfft,spline) ! compute the Fourier splines along z-direction for 2D geometry
 implicit none
 integer , intent(in) :: order, h_cut,nfft ! for OZ h_cut = nfft/2
 real(8), intent(out) :: spline(0:nfft-1)
 real(8) knot(0:order-1),idi,invfft
 real(8) , parameter :: SMALL = 1.0d-10
 real(8) , parameter :: twopi = 6.28318530717959d0
 integer i,j,k, i1
 real(8) a,b,arg
! invdz = Pi2/nfftz
! h_cut = nfftz/2

 knot(0) = 1.0d0 ; knot(1) = 0.0d0
 do i = 2, order-1
   idi = 1.0d0 / dble(i)
   knot(i) = 0.0d0
   do j = 1, i-1
     knot(i-j) = dble(j)*knot(i-j-1)+dble(i-j+1)*knot(i-j)
     knot(i-j) = knot(i-j) * idi
   enddo
  knot(0) = idi * knot(0)
 enddo

 invfft = 1.0d0 / dble(nfft)
 do i = -h_cut, h_cut-1
   a=0.0d0 ; b = 0.0d0
   do j = 0, order-1
     arg = dble(i)*dble(j)*twopi*invfft
     a = a + knot(j) * dcos(arg)
     b = b + knot(j) * dsin(arg)
   enddo
 i1 = i + h_cut
   spline(i1) = a*a+b*b
 enddo
 do i = -h_cut+1 , h_cut-1
 i1 = i + h_cut
   if (spline(i1) < SMALL ) then
     spline(i1) = 0.5d0*(spline(i1+1)-spline(i1-1))
   endif
 enddo
 end subroutine complex_spline2_z

 subroutine real_spline2_and_deriv(order,x,spline,spline_DERIV) ! compute real space splines and their derivatives
 implicit none
 integer, intent(in) :: order
 real(8) x ! fractional coordinate 
 real(8), intent(OUT) :: spline(0:order-1),spline_DERIV(0:order-1)
 integer i,j,k,order1,jjj
 real(8) idi
  
  order1 = order - 1
  spline(0) = 1.0d0 - x
  spline(1) = x
  spline(order1) = 0.0d0
  do i = 2, order1-1
    idi = 1.0d0 / dble(i)
    spline(i) = idi * x * spline(i-1)
    do j = 1, i-1
    jjj = i-j
      spline(jjj) = (x+dble(j)) *spline(jjj-1)+(dble(jjj+1)-x)*spline(jjj)
      spline(jjj) = spline(jjj) * idi
    enddo ! j
    spline(0) = spline(0)*idi* (1.0d0-x)
  enddo ! i

  spline_DERIV(0) = - spline(0)
  do i = 1,order1
    spline_DERIV(i) = spline(i-1) - spline(i)
  enddo
  idi = 1.0d0 / dble(order1)
  spline(order1) = x*spline(order1-1)*idi 

  do j = 1, order1-1
    jjj = order1-j
    spline(jjj) = (x+dble(j))*spline(jjj-1)+(dble(jjj+1)-x)*spline(jjj)
    spline(jjj) = spline(jjj) * idi
  enddo

  spline(0) = idi * (1.0d0-x) * spline(0)

 end subroutine real_spline2_and_deriv 

 subroutine real_spline2_pp(order,x,spline) ! only the spline; no deriv
 implicit none
 integer, intent(in) :: order
 real(8) x ! fractional coordinate
 real(8), intent(OUT) :: spline(0:order-1)
 integer i,j,k,order1,jjj
 real(8) idi

  order1 = order - 1
  spline(0) = 1.0d0 - x
  spline(1) = x
  spline(order1) = 0.0d0
  do i = 2, order1-1
    idi = 1.0d0 / dble(i)
    spline(i) = idi * x * spline(i-1)
    do j = 1, i-1
    jjj = i-j
      spline(jjj) = (x+dble(j)) *spline(jjj-1)+(dble(jjj+1)-x)*spline(jjj)
      spline(jjj) = spline(jjj) * idi
    enddo ! j
    spline(0) = spline(0)*idi* (1.0d0-x)
  enddo ! i

  idi = 1.0d0 / dble(order1)
  spline(order1) = x*spline(order1-1)*idi

  do j = 1, order1-1
    jjj = order1-j
    spline(jjj) = (x+dble(j))*spline(jjj-1)+(dble(jjj+1)-x)*spline(jjj)
    spline(jjj) = spline(jjj) * idi
  enddo

  spline(0) = idi * (1.0d0-x) * spline(0)

 end subroutine real_spline2_pp

 end module spline2


