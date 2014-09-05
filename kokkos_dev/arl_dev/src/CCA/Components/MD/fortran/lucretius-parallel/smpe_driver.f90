
module smpe_driver

implicit none

private :: initialize
private :: finalize
public :: smpe_Q
public :: smpe_Q_ENERGY
public :: smpe_Q_DIP
public :: smpe_Q_DIP_ENERGY
 
contains

 subroutine smpe_Q  ! 
 use spline_module
 use Ewald_data
 use fft_3D_modified
 use fft_3D
 use variables_smpe
 use diverse, only : Re_Center_forces
 use allocate_them, only : smpe_alloc
 use ALL_atoms_data, only : Natoms
 use smpe_utility_pack_0
 use sim_cel_data, only : i_boundary_CTRL

 implicit none
 logical, save :: l_first_time = .true.
 real(8) , parameter :: Pi2 = 6.28318530717959d0
 integer i,j,k,kk

!    if (l_first_time) then
!      l_first_time = .false.
!      call very_first_initializations
!    endif

    call smpe_alloc
    call very_first_initializations

    call initialize
    call get_reduced_coordinates
    call get_ALL_spline2_coef_REAL
    if (i_boundary_CTRL == 1) then  ! SLAB (2D)
    call set_Q_2D
    call dlpfft3_MOD(0,1,nfftx,nffty,nfftz,key1,key2,key3,ww1,ww2,ww3,qqq1)
    call dlpfft3_MOD(0,1,nfftx,nffty,nfftz,key1,key2,key3,ww1,ww2,ww3,qqq2)
    call smpe_eval1_Q_2D   !  potential and stresses ; 
    call dlpfft3_MOD(0,-1,nfftx,nffty,nfftz,key1,key2,key3,ww1,ww2,ww3,qqq1)
    call smpe_eval2_Q_2D ! forces
    else
    call set_Q_3D
!print*,'4:'
    call dlpfft3_MOD(0,1,nfftx,nffty,nfftz,key1,key2,key3,ww1,ww2,ww3,qqq1)
!print*,'5:',shape(qqq1)
    call smpe_eval1_Q_3D   !  potential and stresses ; 
!print*,'6'
    call dlpfft3_MOD(0,-1,nfftx,nffty,nfftz,key1,key2,key3,ww1,ww2,ww3,qqq1)
    call smpe_eval2_Q_3D ! forces
    endif

    call finalize
 contains
   subroutine very_first_initializations
  real(8) , parameter :: Pi2 = 6.28318530717959d0
      call smpe_alloc
      if (i_boundary_CTRL == 1) then 
        reciprocal_zz = Pi2/dble(nfftz)
        inv_rec_zz = 1.0d0/reciprocal_zz
      call dlpfft3_MOD(1,1,nfftx,nffty,nfftz,key1,key2,key3,ww1,ww2,ww3,qqq1)
      call dlpfft3_MOD(1,1,nfftx,nffty,nfftz,key1,key2,key3,ww1,ww2,ww3,qqq2)
      else  !3D
      call dlpfft3_MOD(1,1,nfftx,nffty,nfftz,key1,key2,key3,ww1,ww2,ww3,qqq1)
      endif      
      call get_CMPLX_splines

  end subroutine very_first_initializations

 end subroutine smpe_Q

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 subroutine smpe_Q_ENERGY  !
 use spline_module
 use Ewald_data
 use fft_3D_modified
 use fft_3D
 use variables_smpe
 use diverse, only : Re_Center_forces
 use allocate_them, only : smpe_alloc
 use ALL_atoms_data, only : Natoms
 use smpe_utility_pack_0
 use sim_cel_data, only : i_boundary_CTRL

 implicit none
 logical, save :: l_first_time = .true.
 real(8) , parameter :: Pi2 = 6.28318530717959d0
 integer i,j,k,kk

!    if (l_first_time) then
!      l_first_time = .false.
!      call very_first_initializations
!    endif

    call smpe_alloc
    call very_first_initializations

    call initialize
    call get_reduced_coordinates
    call get_ALL_spline2_coef_REAL
    if (i_boundary_CTRL == 1) then  ! SLAB (2D)
     call set_Q_2D_ENERGY
     call dlpfft3_MOD(0,1,nfftx,nffty,nfftz,key1,key2,key3,ww1,ww2,ww3,qqq1)
     call smpe_eval1_Q_2D_ENERGY   !  potential and stresses ;
    else
     call set_Q_3D
     call dlpfft3_MOD(0,1,nfftx,nffty,nfftz,key1,key2,key3,ww1,ww2,ww3,qqq1)
     call smpe_eval1_Q_3D_ENERGY   !  potential and stresses ;
    endif
    call finalize
 contains
   subroutine very_first_initializations
  real(8) , parameter :: Pi2 = 6.28318530717959d0
      call smpe_alloc
      if (i_boundary_CTRL == 1) then
        reciprocal_zz = Pi2/dble(nfftz)
        inv_rec_zz = 1.0d0/reciprocal_zz
        call dlpfft3_MOD(1,1,nfftx,nffty,nfftz,key1,key2,key3,ww1,ww2,ww3,qqq1)
      else  !3D
        call dlpfft3_MOD(1,1,nfftx,nffty,nfftz,key1,key2,key3,ww1,ww2,ww3,qqq1)
      endif
      call get_CMPLX_splines

  end subroutine very_first_initializations

 end subroutine smpe_Q_ENERGY
!----------------------------------------
!----------------------------------------
!---------------------------------------
 subroutine smpe_Q_DIP  !
 use smpe_dipole_module
 use beta_spline_new
 use spline_module
 use Ewald_data
 use fft_3D_modified
 use fft_3D
 use variables_smpe
 use diverse, only : Re_Center_forces
 use allocate_them, only : smpe_alloc
 use ALL_atoms_data, only : Natoms
 use smpe_utility_pack_0, only : get_reduced_coordinates, get_ALL_spline_coef_REAL,&
     get_pp_spline_coef_REAL, get_ALL_spline2_coef_REAL, get_pp_spline2_coef_REAL,&
     get_CMPLX_splines,get_CMPLX_splines_NEW
 use sim_cel_data, only : i_boundary_CTRL

 implicit none
 logical, save :: l_first_time = .true.
 real(8) , parameter :: Pi2 = 6.28318530717959d0
 integer i,j,k,kk
!    if (l_first_time) then
!      l_first_time = .false.
!      call very_first_initializations
!    endif

    call smpe_alloc
    call very_first_initializations 

    call initialize ; 
    allocate (spline2_REAL_dd_2_x(Natoms,order_spline_xx),&
              spline2_REAL_dd_2_y(Natoms,order_spline_yy),&
              spline2_REAL_dd_2_z(Natoms,order_spline_zz))
    call get_reduced_coordinates
    call beta_spline_pp_dd_2
!call get_ALL_spline2_coef_REAL
!print*,'splineR=', spline2_REAL_dd_x(4000,:),':',sum(spline2_REAL_dd_x(4000,:))

    if (i_boundary_CTRL == 1) then  ! SLAB (2D)
      call set_Q_DIP_2D
      call dlpfft3_MOD(0,1,nfftx,nffty,nfftz,key1,key2,key3,ww1,ww2,ww3,qqq1)
      call dlpfft3_MOD(0,1,nfftx,nffty,nfftz,key1,key2,key3,ww1,ww2,ww3,qqq2)
      call smpe_eval1_Q_DIP_2D   !  potential and stresses ;
      call dlpfft3_MOD(0,-1,nfftx,nffty,nfftz,key1,key2,key3,ww1,ww2,ww3,qqq1)
      call smpe_eval2_Q_DIP_2D ! forces
    else
      call set_Q_DIP_3D
      call dlpfft3_MOD(0,1,nfftx,nffty,nfftz,key1,key2,key3,ww1,ww2,ww3,qqq1)
      call smpe_eval1_Q_DIP_3D   !  potential and stresses ;
      call dlpfft3_MOD(0,-1,nfftx,nffty,nfftz,key1,key2,key3,ww1,ww2,ww3,qqq1)
      call smpe_eval2_Q_DIP_3D ! forces
    endif

    call finalize ;
    deallocate(spline2_REAL_dd_2_x,spline2_REAL_dd_2_y,spline2_REAL_dd_2_z)
 contains
   subroutine very_first_initializations
  real(8) , parameter :: Pi2 = 6.28318530717959d0
      call smpe_alloc
      if (i_boundary_CTRL == 1) then
        reciprocal_zz = Pi2/dble(nfftz)
        inv_rec_zz = 1.0d0/reciprocal_zz
      endif
      call dlpfft3_MOD(1,1,nfftx,nffty,nfftz,key1,key2,key3,ww1,ww2,ww3,qqq1)
      call get_CMPLX_splines_NEW
  end subroutine very_first_initializations

 
 end subroutine smpe_Q_DIP
!---------------------------------------
!--------------------------------------
!----------------------------------------

 subroutine smpe_Q_DIP_ENERGY  !
 use smpe_dipole_module
 use beta_spline_new
 use spline_module
 use Ewald_data
 use fft_3D_modified
 use fft_3D
 use variables_smpe
 use diverse, only : Re_Center_forces
 use allocate_them, only : smpe_alloc
 use ALL_atoms_data, only : Natoms
 use smpe_utility_pack_0, only : get_reduced_coordinates, get_ALL_spline_coef_REAL,&
     get_pp_spline_coef_REAL, get_ALL_spline2_coef_REAL, get_pp_spline2_coef_REAL,&
     get_CMPLX_splines,get_CMPLX_splines_NEW
 use sim_cel_data, only : i_boundary_CTRL

 implicit none
 logical, save :: l_first_time = .true.
 real(8) , parameter :: Pi2 = 6.28318530717959d0
 integer i,j,k,kk
!    if (l_first_time) then
!      l_first_time = .false.
!      call very_first_initializations
!    endif
    call very_first_initializations
    call initialize ;
    allocate (spline2_REAL_dd_2_x(Natoms,order_spline_xx),&
              spline2_REAL_dd_2_y(Natoms,order_spline_yy),&
              spline2_REAL_dd_2_z(Natoms,order_spline_zz))
    call get_reduced_coordinates
    call beta_spline_pp_dd_2
!call get_ALL_spline2_coef_REAL
!print*,'splineR=', spline2_REAL_dd_x(4000,:),':',sum(spline2_REAL_dd_x(4000,:))

    if (i_boundary_CTRL == 1) then  ! SLAB (2D)
      call set_Q_DIP_2D_ENERGY
      call dlpfft3_MOD(0,1,nfftx,nffty,nfftz,key1,key2,key3,ww1,ww2,ww3,qqq1)
      call smpe_eval1_Q_DIP_2D_ENERGY   !  potential and stresses ;
    else
      call set_Q_DIP_3D
      call dlpfft3_MOD(0,1,nfftx,nffty,nfftz,key1,key2,key3,ww1,ww2,ww3,qqq1)
      call smpe_eval1_Q_DIP_3D_ENERGY   !  potential and stresses ;
    endif

    call finalize ;

 contains
   subroutine very_first_initializations
  real(8) , parameter :: Pi2 = 6.28318530717959d0
      call smpe_alloc
      if (i_boundary_CTRL == 1) then
        reciprocal_zz = Pi2/dble(nfftz)
        inv_rec_zz = 1.0d0/reciprocal_zz
      endif
      call dlpfft3_MOD(1,1,nfftx,nffty,nfftz,key1,key2,key3,ww1,ww2,ww3,qqq1)
      call get_CMPLX_splines_NEW
  end subroutine very_first_initializations


 end subroutine smpe_Q_DIP_ENERGY


!------------------------------
!-------------------------------------------
!---------------------------------
  subroutine finalize
   use variables_smpe
   deallocate(tx,ty,tz)
   deallocate(spline2_REAL_pp_x,spline2_REAL_pp_y,spline2_REAL_pp_z)
   deallocate(spline2_REAL_dd_x,spline2_REAL_dd_y,spline2_REAL_dd_z)
  end subroutine finalize


 subroutine initialize
  use variables_smpe
  use ALL_atoms_data, only : Natoms
    allocate (spline2_REAL_pp_x(natoms,order_spline_xx),&
              spline2_REAL_pp_y(Natoms,order_spline_yy),&
              spline2_REAL_pp_z(Natoms,order_spline_zz))
    allocate (spline2_REAL_dd_x(Natoms,order_spline_xx),&
              spline2_REAL_dd_y(Natoms,order_spline_yy),&
              spline2_REAL_dd_z(Natoms,order_spline_zz))
 end subroutine initialize

end module smpe_driver
