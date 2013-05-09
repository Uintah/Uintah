 module spline_module
 implicit none
! I made them as 'autonomous' subroutines
 public :: beta_spline_REAL_coef_dd_2
 public :: beta_spline_REAL_coef
 public :: beta_spline_REAL_coef_pp
 public :: beta_spline_COMPLEX_coef
 public :: spline_cexponentials
 public :: spline_cexponentials_zz
 public :: spline_coef_REAL_2

 contains

   subroutine beta_spline_REAL_coef_dd_2(order,x,spline,spline_DERIV,spline_DERIV_2)
! get spline coeficient with first and second derivatives
   integer, intent(IN) :: order
   real(8) , intent(OUT) :: spline(order),spline_DERIV(order), spline_DERIV_2(order)
   real(8), intent(IN) :: x
   real(8) frac
   integer i,j,k
   real(8) idi,aaa,epsiloni


   spline_DERIV(1) = 1.0d0
   spline_DERIV(2) = -1.0d0   ! carefull here ; I have splines between -1 .. +1
   spline(1) = x-dble(int(x))
   spline(2) = 1.0d0 - spline(1)
   epsiloni = spline(2)

   frac = x-dble(int(x))
   do k = 3,order
    spline(k) = 0.0d0
    idi = 1.0d0/dble(k-1)
     do j=k,2,-1
       if(k.eq.order) then
          spline_DERIV(j)=spline(j)-spline(j-1)
       endif
       if(k==order-1) then
          if (j>2) then
             spline_DERIV_2(j)=spline(j)-2.0d0*spline(j-1)+spline(j-2);
          else ! j==2
             spline_DERIV_2(j)=spline(j)-2.0d0*spline(j-1)
          endif
       endif
        aaa=frac+dble(j-1)
        spline(j)=(aaa*spline(j)+(dble(k)-aaa)*spline(j-1))*idi
     enddo ! j=k,2,-1
     if (k.eq.order) then
        spline_DERIV(1)= spline(1)
     endif
     if (k==order-1) then
       spline_DERIV_2(1) = spline(1)
       spline_DERIV_2(order) = epsiloni        
     endif
     spline(1)=frac * spline(1)*idi
     epsiloni =(-1+frac) * epsiloni *idi 
   enddo ! k=3,N_splines

  end subroutine beta_spline_REAL_coef_dd_2

   subroutine beta_spline_REAL_coef(order,x,spline,spline_DERIV)
   integer, intent(IN) :: order
   real(8) , intent(OUT) :: spline(order),spline_DERIV(order)
   real(8), intent(IN) :: x
   real(8) frac
   integer i,j,k
   real(8) idi,aaa
  

   spline_DERIV(1) = 1.0d0
   spline_DERIV(2) = -1.0d0   ! carefull here ; I have splines between -1 .. +1
   spline(1) = x-dble(int(x))
   spline(2) = 1.0d0 - spline(1)

   frac = x-dble(int(x))
   do k = 3,order
    spline(k) = 0.0d0
    idi = 1.0d0/dble(k-1)
     do j=k,2,-1
       if(k.eq.order) then
          spline_DERIV(j)=spline(j)-spline(j-1)
       endif
        aaa=frac+dble(j-1)
        spline(j)=(aaa*spline(j)+(dble(k)-aaa)*spline(j-1))*idi
     enddo ! j=k,2,-1
     if (k.eq.order) then
        spline_DERIV(1)= spline(1)
     endif
     spline(1)=frac * spline(1)*idi
   enddo ! k=3,N_splines

  end subroutine beta_spline_REAL_coef


   subroutine beta_spline_REAL_coef_pp(order,x,spline)
   integer, intent(IN) :: order
   real(8) , intent(OUT) :: spline(order)
   real(8), intent(IN) :: x
   real(8) frac
   integer i,j,k
   real(8) idi,aaa


   frac = x-dble(int(x))
   spline(1) = frac
   spline(2) = 1.0d0 - spline(1)

   do k = 3,order
    spline(k) = 0.0d0
    idi = 1.0d0/dble(k-1)
     do j=k,2,-1
        aaa=frac+dble(j-1)
        spline(j)=(aaa*spline(j)+(dble(k)-aaa)*spline(j-1))*idi
     enddo ! j=k,2,-1
     spline(1)=frac * spline(1)*idi
   enddo ! k=3,N_splines

  end subroutine beta_spline_REAL_coef_pp


  subroutine beta_spline_COMPLEX_coef(order,nfft,ww,spline_out)
  integer, intent(IN) :: order, nfft
  complex(8), intent(IN) :: ww(nfft)
  real(8), intent(OUT) :: spline_out(nfft)
  complex(8) spline(nfft)
 
  complex(8) auxi
  integer i,j,k
  real(8) spline_knots(order)
     spline_knots(1)=0.0d0
     spline_knots(2)=1.0d0

     do k=3,order
       spline_knots(k)=0.0d0
        do j=k,2,-1
          spline_knots(j)=(dble(j-1)*spline_knots(j)+dble(k-j+1)*spline_knots(j-1))/dble(k-1)
        enddo
     enddo

     do i=0,nfft-1
        auxi=(0.d0,0.d0)
        do k=0,order-2
          auxi=auxi+spline_knots(k+2)*ww(mod(i*k,nfft)+1)
        enddo
        spline(i+1)=ww(mod(i*(order-1),nfft)+1)/auxi
     enddo

     do i = 1, nfft
       spline_out(i) = 1.0d0/REAL(spline(i)*conjg(spline(i)),kind=8)
     enddo

   end subroutine beta_spline_COMPLEX_coef

!  subroutine beta_spline_COMPLEX_coef_ZZ(order,nfft,ww,spline)
!  integer, intent(IN) :: order, nfft
!  complex(8), intent(IN) :: ww(nfft)
!  complex(8),intent(OUT) :: spline(nfft)
!  complex(8) auxi
!  integer i,j,k,nfft2,ii
!  real(8) spline_knots(order)
!
!  nfft2 = nfft/2
!
!     spline_knots(1)=0.0d0
!     spline_knots(2)=1.0d0
!
!     do k=3,order
!       spline_knots(k)=0.0d0
!        do j=k,2,-1
!          spline_knots(j)=(dble(j-1)*spline_knots(j)+dble(k-j+1)*spline_knots(j-1))/dble(k-1)
!        enddo
!     enddo
!
!     do i=-nfft2+1,nfft2
!     ii = i + nfft2 + 1
!        auxi=(0.d0,0.d0)
!        do k=0,order-2
!          auxi=auxi+spline_knots(k+2)*ww(mod(ii*k,nfft)+1)
!        enddo
!        spline(ii)=ww(mod(ii*(order-1),nfft)+1)/auxi
!     enddo
!
!   end subroutine beta_spline_COMPLEX_coef_ZZ



   subroutine spline_cexponentials(nfftx,nffty,nfftz,ww_x,ww_y,ww_z)
     real(8), parameter :: twopi = 6.28318530717959d0
     integer , intent(in) :: nfftx,nffty,nfftz
     complex(8), intent(OUT) :: ww_x(nfftx),ww_y(nffty),ww_z(nfftz)
     integer           :: i
     real(8) :: arg

     ww_x(1)=(1.0d0,0.0d0)
      do i=1,nfftx/2
         arg=(twopi/dble(nfftx))*dble(i)
         ww_x(i+1)=cmplx(dcos(arg),dsin(arg), kind = 8)
         ww_x(nfftx+1-i)=conjg(ww_x(i+1))
      enddo

     ww_y(1)=(1.0d0,0.0d0)
      do i=1,nffty/2
         arg=(twopi/dble(nffty))*dble(i)
         ww_y(i+1)=cmplx(dcos(arg),dsin(arg), kind = 8)
         ww_y(nffty+1-i)=conjg(ww_y(i+1))
      enddo

     ww_z(1)=(1.0d0,0.0d0)
      do i=1,nfftz/2
          arg=(twopi/dble(nfftz)) * dble(i)
          ww_z(i+1)=cmplx(dcos(arg),dsin(arg), kind = 8)
          ww_z(nfftz+1-i)=conjg(ww_z(i+1))
     enddo

    end subroutine spline_cexponentials

    subroutine spline_cexponentials_zz(nfftz,ww_z)
     real(8), parameter :: twopi = 6.28318530717959d0
    integer , intent(in) :: nfftz
    complex(8), intent(OUT) ::ww_z(nfftz)
    integer          i,ii
    real(8) :: arg
   
    do i = -nfftz/2+1-1,nfftz/2-1  ! -1
     ii = i + nfftz/2 + 1
     arg=(1.0d0/dble(nfftz)) * dble(i)
     ww_z(ii) =  cmplx(dcos(arg),dsin(arg), kind = 8)
    enddo
   end subroutine spline_cexponentials_zz


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

 subroutine spline_coef_REAL_2(order,x,beta,beta_deriv)
 integer , intent(in) :: order
 real(8), intent(IN) :: x
 real(8) , intent(out) :: beta(0:order-1), beta_deriv(0:order-1)
 integer iii,i,j,order1
 real(8) idi

   order1 = order-1
   beta(1) = 1.0d0 - x
   beta(2) = x

   beta(order) = 0.0d0

   do i = 2, (order1)
    idi = 1.0d0/dble(i)
    beta(i) = x*beta(i-1)*idi
    do j = 1,i
       beta(i-j) = ((x+dble(j))*beta(i-j+1)+(dble(i-j+1)-x)*beta(i-j))*idi
    enddo
    beta(0) = (1.0d0-x)*beta(0)
   enddo ! i=3,order
! NOW THE DERIVATIEVS
   beta_deriv(0) = - beta(0)
   do i = 1,(order1)
     beta_deriv(i) = beta(i-1)-beta(i)
   enddo
   iii = order1
   idi = 1.0d0/dble(iii)
   beta(iii) = x*beta(iii-1)*idi
   do j = 1,iii-1
      beta(iii-j) = ((x+dble(j))*beta(iii-j-1)+(dble(iii-j+1)-x)*beta(iii-j))*idi
   enddo
   beta(0) = beta(0)*idi*(1.0d0-x)

 end subroutine spline_coef_REAL_2

! subroutine spline_coef_COMPLEX_2(order, Ngrid, beta)
! integer, intent(IN) :: order, Ngrid
! real(8) beta(Ngrid)
!! real(8) suma,suma1,suma2,idi
! real(8), allocatable :: knot(:)
! integer i,j
!
! allocate(knot(order))
!
! knot(1) = 1.0d0
! knot(2) = 0.0d0
!
! do i=3,order
!   knot(i)=0.0d0
!   idi = 1.0d0/dble(i-1)
!   do j=i,2,-1
!      knot(j)=(dble(j-1)*knot(j)+dble(i-j+1)*knot(j-1))*idi
!   enddo
! enddo

!
! do i=0,Ngrid-1
!   auxi=(0.d0,0.d0)
!   do k=0,order-2
!     auxi=auxi+spline_knots(k+2)*ww_x(mod(i*k,k_max_xxx)+1)
!   enddo
!   spline(i+1)=ww(mod(i*(N_splines-1),k_max_xxx)+1)/auxi
! enddo
!
!
!
! dNgrid = dble(Ngrid)
! do i = 1, Ngrid
!   di = dble(i-1)*dNgrid
!   suma1=0.0d0 ; suma2=0.0d0
!   do j = 1, order
!     suma1 = suma1 + knot(j)*dcos(idi*(dble(j-1)))
!     suma2 = suma2 + knot(j)*dcos(idi*(dble(j-1)))
!   enddo
!   beta(i) = suma1*suma1+suma2*suma2
! enddo
!
! do i = 2, Ngrid-2
!   if (beta(i) < 1.0d-9) then
!   beta(i) = 0.5d0*(beta(i+1)+beta(i-1))
!   endif
! enddo
!
! deallocate(knot)
! end subroutine spline_coef_COMPLEX_2
!
! subroutine spline_coef_COMPLEX_Z_2(order,Ngrid,z_step,beta)
!!! N grid is the z_grid in this case
! integer, intent(IN) :: order,Ngrid
! real(8),intent(IN) :: z_step
! real(8) , intent(OUT) :: beta(Ngrid)
! real(8) suma,suma1,suma2,idi
! real(8), allocatable :: knot(:)
! allocate(knot(Ngrid))
! knot(1) = 1.0d0
! knot(2) = 0.0d0
! do i = 3, order
!   idi = 1.0d0/dble(i)
!   suma = 0.0d00
!   do j = 2, i-1
!     knot(i-j) = (dble(j)*knot(i-j-1)+dble(i-j+1)*knot(i-j))*idi
!   enddo
!   knot(1) = knot(1)*idi
! enddo
!
! do i = -Ngrid+1,Ngrid
!   suma1=0.0d0
!   suma2=0.0d0
!   di = dble(i-1)*dz
!   do j = 1, order
!     suma1=suma1+knot(j)*dcos((di*dble(j-1)))
!     suma2=suma2+knot(j)*dsin((di*dble(j-1)))
!   enddo
!   beta(i) = suma1*suma1+suma2*suma2
! enddo
!
! do i=-Ngrid+2,Ngrid-1
!  if (beta(i) < 1.0d-9) then
!    beta(i) = 0.5d0*(beta(i+1)+beta(i-1))
!  endif
! enddo
!deallocate(knot)
!end subroutine spline_coef_COMPLEX_Z_2

 end module spline_module
 
