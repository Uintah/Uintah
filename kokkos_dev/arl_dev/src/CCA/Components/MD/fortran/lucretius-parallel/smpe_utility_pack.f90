
  module  smpe_utility_pack_0
!basic stuff added here (get reduced coordinates ; set array vectors ; get splines
  implicit none
  private :: very_first_pass_1G

  contains
 
  include 'set_Q_2D.f90'
  include 'set_Q_2D_ENERGY.f90'
  include 'smpe_eval1_Q_2D.f90'
  include 'smpe_eval1_Q_2D_ENERGY.f90'
  include 'smpe_eval2_Q_2D.f90'
  include 'set_Q_3D.f90'
  include 'smpe_eval1_Q_3D.f90'
  include 'smpe_eval1_Q_3D_ENERGY.f90'
  include 'smpe_eval2_Q_3D.f90'


  subroutine get_reduced_coordinates
  use variables_smpe, only : nfftx,nffty,nfftz,tx,ty,tz
  use Ewald_data
  use sim_cel_data
  use sizes_data, only : Natoms
  use all_atoms_data, only : xx,yy,zz,xxx,yyy,zzz
  implicit none
  real(8) db(3), inv_box(3), bbox(3)
  integer k,i

   db(1) = dble(nfftx); db(2) = dble(nffty) ; db(3) = dble ( nfftz )
   bbox(1) = sim_cel(1) ; bbox(2) = sim_cel(5) ; bbox(3) = sim_cel(9)
   inv_box = 1.0d0 / bbox
! To be changed latter
  allocate(tx(Natoms), ty(Natoms),tz(Natoms))
  do i = 1, Natoms  
   tx(i) = xxx(i)  - (int(2.0d0*(xxx(i)*inv_box(1))) - int(xxx(i)*inv_box(1)) ) * bbox(1) 
   tx(i) = db(1) * ( inv_box(1) * tx(i) + 0.5d0) 
   ty(i) = yyy(i) - (int(2.0d0*(yyy(i)*inv_box(2))) - int(yyy(i)*inv_box(2)) ) * bbox(2)
   ty(i) = db(2) * ( inv_box(2) * ty(i) + 0.5d0)
   tz(i) = zzz(i) - (int(2.0d0*(zzz(i)*inv_box(3))) - int(zzz(i)*inv_box(3)) ) * bbox(3)
   tz(i) = db(3) * ( inv_box(3) * tz(i) + 0.5d0)
   enddo

  end subroutine get_reduced_coordinates

   subroutine get_ALL_spline_coef_REAL
! These are for real space (for Q and forces)
   use variables_smpe
   use Ewald_data
   use sim_cel_data
   use sizes_data, only : Natoms
   use spline_module
   use ALL_atoms_data, only : zz,zzz
   implicit none
   real(8) x,y,z
   integer ox,oy,oz
   integer i
    ox = order_spline_xx ; oy = order_spline_yy ; oz = order_spline_zz
    if (i_boundary_CTRL == 1) then  ! 
    do i = 1, Natoms
      x = tx(i)-int(tx(i)) ! tt are the reduced coordinates
      y = ty(i)-int(ty(i))
      z = zzz(i)-int(zzz(i))
      call beta_spline_REAL_coef_pp(ox,x,spline2_REAL_pp_x(i,1:ox))
      call beta_spline_REAL_coef_pp(oy,y,spline2_REAL_pp_y(i,1:oy))
      call beta_spline_REAL_coef_pp(oz,z,spline2_REAL_pp_z(i,1:oz))
    enddo
    else
    do i = 1, Natoms
      x = tx(i)-int(tx(i)) ! tt are the reduced coordinates
      y = ty(i)-int(ty(i))
      z = tz(i)-int(tz(i)) 
      call beta_spline_REAL_coef(ox,x,spline2_REAL_pp_x(i,1:ox),spline2_REAL_dd_x(i,1:ox))
      call beta_spline_REAL_coef(oy,y,spline2_REAL_pp_y(i,1:oy),spline2_REAL_dd_y(i,1:oy))
      call beta_spline_REAL_coef(oz,z,spline2_REAL_pp_z(i,1:oz),spline2_REAL_dd_z(i,1:oz))
    enddo
    endif
   end subroutine get_ALL_spline_coef_REAL

   subroutine get_pp_spline_coef_REAL
! These are for real space (for Q and forces)
   use variables_smpe
   use Ewald_data
   use sim_cel_data
   use sizes_data, only : Natoms
   use spline_module
   use ALL_atoms_data, only : zz,zzz
   implicit none
   real(8) x,y,z
   integer ox,oy,oz
   integer i
    ox = order_spline_xx ; oy = order_spline_yy ; oz = order_spline_zz
    if (i_boundary_CTRL == 1) then
    do i = 1, Natoms
      x = tx(i)-int(tx(i)) ! tt are the reduced coordinates
      y = ty(i)-int(ty(i))
      z = zzz(i)-int(zzz(i))
      call beta_spline_REAL_coef_pp(ox,x,spline2_REAL_pp_x(i,1:ox))
      call beta_spline_REAL_coef_pp(oy,y,spline2_REAL_pp_y(i,1:oy))
      call beta_spline_REAL_coef_pp(oz,z,spline2_REAL_pp_z(i,1:oz))
    enddo
    else
    do i = 1, Natoms
      x = tx(i)-int(tx(i)) ! tt are the reduced coordinates
      y = ty(i)-int(ty(i))
      z = tz(i)-int(tz(i))
      call beta_spline_REAL_coef_pp(ox,x,spline2_REAL_pp_x(i,1:ox))
      call beta_spline_REAL_coef_pp(oy,y,spline2_REAL_pp_y(i,1:oy))
      call beta_spline_REAL_coef_pp(oz,z,spline2_REAL_pp_z(i,1:oz))
    enddo
    endif
   end subroutine get_pp_spline_coef_REAL


   subroutine get_ALL_spline2_coef_REAL ! does it with respect to zzz as real coordinate rather than ....
! These are for real space (for Q and forces)
! zzz coordinate is treated differently and interpolated as real coordinate rather than reduced one
! that is one key point in doing FFT integral: treat zzz different 
   use variables_smpe
   use Ewald_data
   use sim_cel_data
   use sizes_data, only : Natoms
   use ALL_atoms_data, only : zz,zzz
   use spline2
   
   implicit none
   real(8) x,y,z
   integer ox,oy,oz
   integer i
      ox = order_spline_xx ; oy = order_spline_yy ; oz = order_spline_zz
      if (i_boundary_CTRL == 1) then
      do i = 1, Natoms
        x = tx(i)-int(tx(i)) ! tt are the reduced coordinates
        y = ty(i)-int(ty(i))
        z = zzz(i)-int(zzz(i))   !beta_spline_REAL_coef(order,x,nfft,spline,spline_DERIV)
        call real_spline2_and_deriv(ox,x,spline2_REAL_pp_x(i,1:ox),spline2_REAL_dd_x(i,1:ox))
        call real_spline2_and_deriv(oy,y,spline2_REAL_pp_y(i,1:oy),spline2_REAL_dd_y(i,1:oy))
        call real_spline2_and_deriv(oz,z,spline2_REAL_pp_z(i,1:oz),spline2_REAL_dd_z(i,1:oz))
      enddo
      else
      do i = 1, Natoms
        x = tx(i)-int(tx(i)) ! tt are the reduced coordinates
        y = ty(i)-int(ty(i))
        z = tz(i)-int(tz(i))   !beta_spline_REAL_coef(order,x,nfft,spline,spline_DERIV)
        call real_spline2_and_deriv(ox,x,spline2_REAL_pp_x(i,1:ox),spline2_REAL_dd_x(i,1:ox))
        call real_spline2_and_deriv(oy,y,spline2_REAL_pp_y(i,1:oy),spline2_REAL_dd_y(i,1:oy))
        call real_spline2_and_deriv(oz,z,spline2_REAL_pp_z(i,1:oz),spline2_REAL_dd_z(i,1:oz))
      enddo
      endif
   end subroutine get_ALL_spline2_coef_REAL

   subroutine get_pp_spline2_coef_REAL ! does it with respect to zzz as real coordinate rather than ....
! These are for real space (for Q and forces)
! zzz coordinate is treated differently and interpolated as real coordinate rather than reduced one
! that is one key point in doing FFT integral: treat zzz different
   use variables_smpe
   use Ewald_data
   use sim_cel_data
   use sizes_data, only : Natoms
   use ALL_atoms_data, only : zz,zzz
   use spline2
   implicit none
   real(8) x,y,z
   integer ox,oy,oz
   integer i
      ox = order_spline_xx ; oy = order_spline_yy ; oz = order_spline_zz
      if (i_boundary_CTRL == 1) then
      do i = 1, Natoms
        x = tx(i)-int(tx(i)) ! tt are the reduced coordinates
        y = ty(i)-int(ty(i))
        z = zzz(i)-int(zzz(i))   !beta_spline_REAL_coef(order,x,nfft,spline,spline_DERIV)
        call real_spline2_pp(ox,x,spline2_REAL_pp_x(i,1:ox))
        call real_spline2_pp(oy,y,spline2_REAL_pp_y(i,1:oy))
        call real_spline2_pp(oz,z,spline2_REAL_pp_z(i,1:oz))
      enddo
      else
       do i = 1, Natoms
        x = tx(i)-int(tx(i)) ! tt are the reduced coordinates
        y = ty(i)-int(ty(i))
        z = tz(i)-int(tz(i))   !beta_spline_REAL_coef(order,x,nfft,spline,spline_DERIV)
        call real_spline2_pp(ox,x,spline2_REAL_pp_x(i,1:ox))
        call real_spline2_pp(oy,y,spline2_REAL_pp_y(i,1:oy))
        call real_spline2_pp(oz,z,spline2_REAL_pp_z(i,1:oz))
      enddo
      endif
   end subroutine get_pp_spline2_coef_REAL


   subroutine get_CMPLX_splines ! they are evaluated once in the beggining of simulations.
   use spline2, only : complex_spline2_xy, complex_spline2_z
   use variables_smpe
    implicit none
      call complex_spline2_xy(order_spline_xx,nfftx,spline2_CMPLX_xx(1:nfftx))
      call complex_spline2_xy(order_spline_yy,nffty,spline2_CMPLX_yy(1:nffty))
      call complex_spline2_xy(order_spline_zz,nfftz,spline2_CMPLX_zz(1:nfftz))
!      call complex_spline2_z (order_spline_zz,h_cut_z,nfftz,spline2_CMPLX_zz(1:nfftz)) 
! using complex_spline2_xy and complex_spline2_z gives the same splines, but in different order
! (shifted by Pi/2) . The splines as comes from complex_spline2_z are not 
! in a wrap-arround sequence whiile the splines from complex_spline2_xy are in 
! the right order for fft. 
   end  subroutine get_CMPLX_splines

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

   subroutine get_CMPLX_splines_NEW ! they are evaluated once in the beggining of simulations.
   use spline_module, only : beta_spline_COMPLEX_coef
   use variables_smpe
    implicit none
      call beta_spline_COMPLEX_coef(order_spline_xx,nfftx,ww1,spline2_CMPLX_xx(1:nfftx))
      call beta_spline_COMPLEX_coef(order_spline_yy,nffty,ww2,spline2_CMPLX_yy(1:nffty))
      call beta_spline_COMPLEX_coef(order_spline_zz,nfftz,ww3,spline2_CMPLX_zz(1:nfftz))
!      call complex_spline2_z (order_spline_zz,h_cut_z,nfftz,spline2_CMPLX_zz(1:nfftz))
! using complex_spline2_xy and complex_spline2_z gives the same splines, but in different order
! (shifted by Pi/2) . The splines as comes from complex_spline2_z are not
! in a wrap-arround sequence whiile the splines from complex_spline2_xy are in
! the right order for fft.
   end  subroutine get_CMPLX_splines_NEW


   subroutine very_first_pass_1G(eta,err_msg)
   use atom_type_data, only : N_TYPE_ATOMS,atom_type_1GAUSS_charge_distrib
   implicit none
   real(8), intent(OUT) :: eta
   character(*), intent(IN):: err_msg
   integer ii,jj
   logical ll
   do ii = 1, N_TYPE_ATOMS
   do jj = 1, N_TYPE_ATOMS
   if (atom_type_1GAUSS_charge_distrib(ii) > 1.0d-9 .and. atom_type_1GAUSS_charge_distrib(jj)>1.0d-9) then
    ll = atom_type_1GAUSS_charge_distrib(ii)-atom_type_1GAUSS_charge_distrib(jj) > 1.0d-6
    if (ll) then
     write(6,*) 'ERROR in ',trim(err_msg),'&very_first_pass_1G'
     write(6,*) 'Not all Gaussian distributions are equal therefore you cannot use the 1G subroutines'
     STOP
    endif
   endif
   enddo
   enddo
   do ii = 1, N_TYPE_ATOMS
      if (atom_type_1GAUSS_charge_distrib(ii) > 1.0d-9) then
        eta = atom_type_1GAUSS_charge_distrib(ii)
        RETURN
      endif
   enddo
   end subroutine very_first_pass_1G

  end module smpe_utility_pack_0

