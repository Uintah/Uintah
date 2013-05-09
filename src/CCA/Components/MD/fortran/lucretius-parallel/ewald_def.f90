module ewald_def_module
implicit none
public :: ewald_def
public :: get_ewald_parameters
public :: interpolate_charge_vectors
contains
subroutine ewald_def(key,i_type_Ewald)
character(*), intent(IN) :: key
integer, intent(OUT) :: i_type_Ewald

 select case (trim(key) )
 case ('SMPE')
   i_type_Ewald = 1
 case('SLOW')
   i_type_Ewald = 2
 case default
   i_type_Ewald = 0
 end select
 
end subroutine ewald_def

subroutine get_ewald_parameters
 use atom_type_data, only : N_STYLE_ATOMS,atom_STYLE_1GAUSS_charge_distrib
 use Ewald_data, only : ewald_beta,ewald_gamma,ewald_alpha, ewald_eta
 implicit none
 integer ii,jj
 real(8) beta,beta1,beta2

  do ii = 1, N_STYLE_ATOMS
    beta = atom_STYLE_1GAUSS_charge_distrib(ii)
    ewald_beta(ii) = 1.0d0/dsqrt(1.0d0/ewald_alpha**2 + 1.0d0/beta**2)
  enddo
   do ii = 1, N_STYLE_ATOMS
   do jj = 1, N_STYLE_ATOMS
    beta1 = atom_STYLE_1GAUSS_charge_distrib(ii)
    beta2 = atom_STYLE_1GAUSS_charge_distrib(jj)
!print*, ii,jj,beta1,beta2
    ewald_gamma(ii,jj) = 1.0d0/dsqrt(1.0d0/ewald_alpha**2 + 1.0d0/beta1**2+1.0d0/beta2**2)
    ewald_eta(ii,jj) = 1.0d0/dsqrt( 1.0d0/beta1**2+1.0d0/beta2**2)
   enddo
   enddo
!stop
!print*, 'ewald_eta=',ewald_eta
!print*, 'ewald_gamma=',ewald_gamma
!stop
end subroutine get_ewald_parameters

subroutine interpolate_charge_vectors
 use interpolate_data
 use Ewald_data
 use cut_off_data
 use atom_type_data,only : atom_type_1GAUSS_charge_distrib,atom_type_charge,q_reduced,&
                           atom_type_1GAUSS_charge,q_reduced_G, N_STYLE_ATOMS,&
                           is_Style_dipole_pol,atom_style_dipole_pol
 use thole_data

 implicit none
 integer i,j,k,N2,i1,ii,jj,MAX_grid_short_range, it
 real(8) beta,rrr,q1,q2,q1_G,q2_G,dlr_pot, pol,a

  MAX_grid_short_range = MX_interpol_points
  RDR = (cut_off+displacement)/dble(MAX_grid_short_range-4)
  irdr = 1.0d0/RDR
  dlr_pot = RDR
  N2 = ((N_STYLE_ATOMS+1)*N_STYLE_ATOMS)/2 
! the grid should be exaclty as for vdw.

  do i =  1, MAX_grid_short_range
     rrr = dble(i)*dlr_pot
     vele(i)=vv0(rrr,ewald_alpha )
     gele(i)=vv1(rrr,ewald_alpha )
     vele2(i) = vv2(rrr,ewald_alpha)
     vele3(i) = vv3(rrr,ewald_alpha)
!print*,i,vele(i),gele(i):w
!read(*,*)
  enddo
  i1 = 0
  do ii = 1, N_STYLE_ATOMS
  do jj = ii, N_STYLE_ATOMS
   i1 = i1 + 1
   beta = ewald_eta(ii,jj)
   do i =  1, MAX_grid_short_range
       rrr = dble(i)*dlr_pot
       vele_G(i,i1)=-vv0(rrr,beta) + vele(i)   
       a = -vv1(rrr,beta) + gele(i)
       if (rrr <= cut_off_short + displacement) then
           gele_G_short(i,i1) =  a * trunc_and_shift(rrr,cut_off_short,displacement)
       else
           gele_G_short(i,i1) = 0.0d0
       endif
       gele_G(i,i1)= a
       vele2_G(i,i1)=-vv2(rrr,beta) + vele2(i)
       vele3_G(i,i1)=-vv3(rrr,beta) + vele3(i)

       if ( is_Style_dipole_pol(ii).and.is_Style_dipole_pol(jj)) then
        it = i_type_THOLE_function_CTRL
        pol = (atom_style_dipole_pol(ii)*atom_style_dipole_pol(jj))**(1.0d0/6.0d0) ! atom_style_pol must be in A^3
        vele_THOLE(i,i1) = vv0_THOLE(rrr,it,pol,aa_thole)
        gele_THOLE(i,i1) = vv1_THOLE(rrr,it,pol,aa_thole)
        vele_THOLE_DERIV(i,i1) = vv0_THOLE_DERIV(rrr,it,pol,aa_thole)
        gele_THOLE_DERIV(i,i1) = vv1_THOLE_DERIV(rrr,it,pol,aa_thole)
       else 
        vele_THOLE(i,i1) = 0.0d0
        gele_THOLE(i,i1) = 0.0d0
        vele_THOLE_DERIV(i,i1) = 0.0d0
        gele_THOLE_DERIV(i,i1) = 0.0d0
       endif
    enddo
!if (ii==2.and.jj==10)then
!do i =1, MAX_grid_short_range
!rrr = dble(i)*dlr_pot
!write(21,*)rrr,vele_THOLE(i,i1), gele_THOLE(i,i1)
!enddo
!endif

  enddo
  enddo

!STOP


 contains
 real(8) function trunc_and_shift(r,cut,delta)
  implicit none
  real(8) r,cut,delta
  real(8) g
  if (cut < delta) then
!    print*, 'WARNING in def_ewald%trunc_and_shift cut < delta '
    trunc_and_shift=0.0d0
    RETURN
  endif

  if (r <= cut .and. r > cut-delta) then
    g = (r-(cut-delta))/delta
    trunc_and_shift = 1.0d0+(g*g*(2.0d0*g-3.0d0))
  else 
   if (r <= cut-delta) trunc_and_shift = 1.0d0
   if (r > cut) trunc_and_shift = 0.0d0
  endif

  
 end function trunc_and_shift

  real(8) function vv0(r,alpha)
     implicit none
     real(8) r,q1,q2,alpha
     vv0 = erfc(alpha*r)/r
  end function vv0
  real(8) function vv1(r,alpha)
     implicit none
     real(8), parameter :: Pi=3.14159265358979d0
     real(8) r,alpha
     real(8) B1, B0,EEE
     EEE = dexp(-(alpha*r)**2)
     B0 = erfc(alpha*r)/r
     B1 = 2.0d0/dsqrt(Pi)*alpha*EEE+B0
     vv1 = B1/r**2
  end function vv1

    real(8) function vv2(r,alpha)
     implicit none
     real(8), parameter :: Pi=3.14159265358979d0
     real(8) r,alpha,B0,B1,B2,t2, fct_UP,fct_DOWN,EEE
     real(8) f1,f2
     fct_UP = 2.0d0*alpha*alpha
     fct_DOWN = 1.0d0/alpha/dsqrt(Pi)
     EEE = dexp(-(alpha*r)**2)
     f1 = EEE * 2.0d0 * alpha * alpha
     f2 = f1 * 2.0d0 * alpha * alpha
     B0 = erfc(alpha*r)/r
     B1 = (B0 + f1*fct_DOWN) / r**2
     B2 = (3.0d0*B1 + f2*fct_DOWN) / r**2
     vv2 = B2
  end function vv2

   real(8) function vv3(r,alpha)
     implicit none
     real(8), parameter :: Pi=3.14159265358979d0
     real(8) r,alpha,B0,B1,B2,t2,EEE,fct_UP,fct_DOWN,B3
     real(8) f1,f2,f3
     fct_UP = 2.0d0*alpha*alpha
     fct_DOWN = 1.0d0/alpha/dsqrt(Pi)
     EEE = dexp(-(alpha*r)**2)
     f1 = EEE * 2.0d0 * alpha * alpha
     f2 = f1 * 2.0d0 * alpha * alpha
     f3 = f2 * 2.0d0 * alpha * alpha
     B0 = erfc(alpha*r)/r
     B1 = (B0 + f1*fct_DOWN) / r**2
     B2 = (3.0d0*B1 + f2*fct_DOWN) / r**2
     B3 = (5.0d0*B2 + f3*fct_DOWN) / r**2
     vv3 = B3
  end function vv3


  real(8) function vv0_THOLE(r,i_type_func,polarizability,a_thole)
    implicit none
    real(8) r,polarizability,a_thole
    integer i_type_func
    real(8) u,dist
    dist = 2**(1.0d0/6.0d0)*pol

     select case (i_type_func) 
     case(1)
        u = a_thole*(r/polarizability)**3
        vv0_THOLE = - (dexp(-u)/r) / r**2
     case default
       print*, 'NOT IMPLEMENTED CASE OF THOLE SWITCH FUNCTION'
       STOP
     end select
  end function vv0_THOLE

   real(8) function vv1_THOLE(r,i_type_func,polarizability,a_thole)
    implicit none
    real(8) r,polarizability,a_thole
    integer i_type_func
    real(8) u,vv0,dist
    dist = 2**(1.0d0/6.0d0)*pol

     select case (i_type_func)
     case(1)
        u = a_thole*(r/polarizability)**3
        vv0 = dexp(-u)/r**3
        vv1_THOLE = - vv0*(u + 1.0d0)  / r**2 * 3.0d0
     case default
       print*, 'NOT IMPLEMENTED CASE OF THOLE SWITCH FUNCTION'
       STOP
     end select
  end function vv1_THOLE


  real(8) function vv0_THOLE_DERIV(r,i_type_func,polarizability,a_thole)
    implicit none
    real(8) r,polarizability,a_thole
    integer i_type_func
    real(8) u,dist,vv0,uu
    dist = 2**(1.0d0/6.0d0)*pol

     select case (i_type_func)
     case(1)
        u = a_thole*(r/polarizability)**3
        vv0 = - (dexp(-u)/r) / r**2
        vv0_THOLE_DERIV = vv0 * 3.0d0 * (u + 1.0d0)/r /r   ! divide one more time by r
     case default
       print*, 'NOT IMPLEMENTED CASE OF THOLE SWITCH FUNCTION'
       STOP
     end select
  end function vv0_THOLE_DERIV


  real(8) function vv1_THOLE_DERIV(r,i_type_func,polarizability,a_thole)
    implicit none
    real(8) r,polarizability,a_thole
    integer i_type_func
    real(8) u,dist,vv0,uu
    dist = 2**(1.0d0/6.0d0)*pol

     select case (i_type_func)
     case(1)
        u = a_thole*(r/polarizability)**3
        vv0 = - (dexp(-u)/r) / r**2
        vv1_THOLE_DERIV = 3.0d0 * vv0 * (3.0d0*u*u + 5.0d0*(u+1.0d0)) /(r*r*r) / r
     case default
       print*, 'NOT IMPLEMENTED CASE OF THOLE SWITCH FUNCTION'
       STOP
     end select
  end function vv1_THOLE_DERIV















     real(8) function vv2_THOLE(r,i_type_func,polarizability,a_thole)
    implicit none
    real(8) r,polarizability,a_thole
    integer i_type_func
    real(8) u,vv0,BB, dist
    dist = 2**(1.0d0/6.0d0)*pol
     select case (i_type_func)
     case(1)
        u = a_thole*(r/polarizability)**3
        vv0 = dexp(-u)
        BB = 3.0d0*u**2+u+1.0d0
        vv2_THOLE = 3.0d0*vv0/r**4*BB
     case default
       print*, 'NOT IMPLEMENTED CASE OF THOLE SWITCH FUNCTION'
       STOP
     end select
  end function vv2_THOLE

    real(8) function vv3_THOLE(r,i_type_func,polarizability,a_thole)
    implicit none
    real(8) r,polarizability,a_thole
    integer i_type_func
    real(8) u,vv0,BB
     select case (i_type_func)
     case(1)
        u = a_thole*(r/polarizability)**3
        vv0 = dexp(-u)
        BB =  9.0d0*u**3+5.0d0*u+5.0d0
        vv3_THOLE = 3.0d0*vv0/r**6*BB
     case default
       print*, 'NOT IMPLEMENTED CASE OF THOLE SWITCH FUNCTION'
       STOP
     end select
  end function vv3_THOLE


end subroutine interpolate_charge_vectors
end module ewald_def_module
