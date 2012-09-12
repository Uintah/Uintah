
module smpe_driver

implicit none

private :: initialize
private :: finalize
public :: smpe_Q
 
contains

 subroutine smpe_Q  ! The main smpe subroutine 
 use Ewald_data
 use fft_3D
 use variables_smpe
 use allocate_them, only : smpe_alloc
 use ALL_atoms_data, only : Natoms
 use smpe_utility_pack_0
 use sim_cel_data, only : i_boundary_CTRL

 implicit none
 logical, save :: l_first_time = .true.
 real(8) , parameter :: Pi2 = 6.28318530717959d0
 integer i,j,k,kk

    call smpe_alloc  ! allocate the required  arrays
    call very_first_initializations  ! initialize the variables in the initial pass

    call initialize                  ! initialize all other variables for all passing here
    call get_reduced_coordinates     ! convert cartezian coordinates to reduced coordinates
    call get_ALL_spline2_coef_REAL   ! get the real space spline coefficients require to fit the regular charge grid
    call set_Q_3D                    ! set/fit the ragular charge grid
    call dlpfft3_MOD(0,1,nfftx,nffty,nfftz,key1,key2,key3,ww1,ww2,ww3,qqq1)  ! do a directe FFT in 3D
    call smpe_eval1_Q_3D   !  evaluate the potential and stresses  in Fourier space
    call dlpfft3_MOD(0,-1,nfftx,nffty,nfftz,key1,key2,key3,ww1,ww2,ww3,qqq1) ! do a inverse FFT in 3D
    call smpe_eval2_Q_3D ! evaluate the forces back in real space

    call finalize  ! clean up the stuff before exiting the subroutine 

 contains
   subroutine very_first_initializations
  real(8) , parameter :: Pi2 = 6.28318530717959d0
      call smpe_alloc 
      call dlpfft3_MOD(1,1,nfftx,nffty,nfftz,key1,key2,key3,ww1,ww2,ww3,qqq1)  ! this will initialize the FFT local arrays in dlpfft3_MOD
      call get_CMPLX_splines   ! get the Fourier space splines

  end subroutine very_first_initializations

 end subroutine smpe_Q

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!----------------------------------------
!----------------------------------------
!---------------------------------------
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
