
! Eval the z-profile of rsmd . Usefull for discriminationg between liquid and solid.
! Also my prefered way of getting difussion coeficients

 module rsmd_module

 implicit none

 real(8), allocatable, save, private :: local_mol_xyz(:,:), local_mol_xyz_MOM2(:,:)
 real(8), allocatable, private :: rmsd_xyz_0(:,:), mol_xyz_0(:,:),molzz(:)
 integer , allocatable , save , private :: NB(:)
 integer, private, save :: i_time,  i_all_counter, i_counter_collected
  
 private :: finalize_and_print_in_rsmd
 private :: zero_in_counters
 private :: collect_rsmd_X0
 private :: bin_in_Z_rsmd_X0
 private :: get_z_grid
 public :: driver_rsmd

 contains
  subroutine driver_rsmd
    use rsmd_data
    use integrate_data, only : integration_step, time_step
    implicit none
    integer,save:: N_collect,N_eval,N_print
    logical , save :: l_very_first_run = .true.
    integer i,j,k
  if (.not.rsmd%any_request) RETURN
    if (l_very_first_run) then
      l_very_first_run=.false.
      call very_first_pass
    endif
    if (mod(integration_step,N_collect) == 0) then
!print*, integration_step,i_counter_collected+1,i_time+1,i_all_counter+1
       call get_z_grid
       i_counter_collected=i_counter_collected+1
       call collect_rsmd_X0
    if (mod(i_counter_collected,N_eval) ==0) then
       i_time = i_time + 1
       call bin_in_Z_rsmd_X0(N_eval,i_time)
    if (mod(i_time,N_print) ==0) then
       i_all_counter = i_all_counter + 1
       call finalize_and_print_in_rsmd(N_collect,N_eval,N_print)
    endif !  (mod(i_time,N_print).eq.0)  
   endif
  endif

  CONTAINS
    subroutine very_first_pass
     use ALL_mols_data, only : mol_xyz,Nmols
     use mol_type_data, only : N_type_molecules
     use ALL_rigid_mols_data, only : qn
     use integrate_data, only : l_do_QN_CTRL

     integer NZ,NT,i
     real(8) dt

       N_print = rsmd%N_print
       N_eval = rsmd%N_eval
       N_collect = rsmd%N_collect

       i_counter_collected = 0
       i_time = 0
       i_all_counter = 0
!!!????       MAX_TIME_RMSD= rsmd%N_print
       NZ = rsmd%N_Z_BINS
       NT = rsmd%N_print
       allocate (rsmd%time_axis(1:NT))
       allocate(NB(1:Nmols))
       rsmd%time_axis(:) = 0.0d0     !time_axis_rmsd=0.0d0
       dt = time_step * dble(rsmd%N_collect)*dble(rsmd%N_eval)
       do i = 1, rsmd%N_print ; rsmd%time_axis(i) = dble(i)*dt ; enddo;

       allocate(rmsd_xyz_0(1:Nmols,1:3))
       allocate(mol_xyz_0(1:Nmols,3))
       allocate(molzz(Nmols))
       allocate (zp_t_z_rmsd_trans_x0(1:NT,1:NZ,1:N_type_molecules,1:3))
       allocate(RA_zp_t_z_rmsd_trans_x0(1:NT,1:NZ,1:N_type_molecules,1:3))
       allocate (Diff_trans_X0(1:NZ,1:N_type_molecules,1:3))
       rmsd_xyz_0=0.0d0;
       mol_xyz_0(:,:)=mol_xyz(:,:)
       zp_t_z_rmsd_trans_x0=0.0d0 ;  Diff_trans_X0=0.0d0
       RA_zp_t_z_rmsd_trans_x0=0.0d0
    end subroutine very_first_pass

 end subroutine driver_rsmd


    subroutine zero_in_counters
    use rsmd_data, only : Diff_trans_X0, zp_t_z_rmsd_trans_x0
    use ALL_mols_data, only : mol_xyz,Nmols
    implicit none
        mol_xyz_0(:,:) = mol_xyz(:,:)
        rmsd_xyz_0 = 0.0d0
        Diff_trans_X0 = 0.0d0 ;
        zp_t_z_rmsd_trans_x0 = 0.0d0 ; 
    end subroutine zero_in_counters

    subroutine finalize_and_print_in_rsmd(N_collect,N_eval,N_print)
    use rsmd_data
    use file_names_data, only : continue_job, path_out
    use chars, only : char_intN_ch
    use printer_arrays
    use sim_cel_data, only : sim_cel
    use array_math, only : fit_a_line
    use mol_type_data, only : N_type_molecules
    implicit none
     integer, intent(IN) :: N_print,N_collect,N_eval
     integer i,j,k
     character(4) ch4
     real(8) Z_axis(rsmd%N_Z_BINS)
     real(8) f1,f2
     integer i1,i2
     real(8), allocatable :: reziduu_fit(:,:,:)
     real(8), allocatable :: tmp(:,:)
     
     f2 = 1.0d0/dble(i_all_counter)
     f1 = 1.0d0-f2
     RA_zp_t_z_rmsd_trans_x0 = RA_zp_t_z_rmsd_trans_x0*f1 + zp_t_z_rmsd_trans_x0*f2
     call char_intN_ch(4,i_all_counter,ch4)
     do i = 1, rsmd%N_Z_BINS ; Z_axis(i) = dble(i)/dble(rsmd%N_Z_BINS)*sim_cel(9); enddo

     if (rsmd%print_details) then
       call print_in_file( rsmd%time_axis,Z_axis, &
              zp_t_z_rmsd_trans_x0,trim(path_out)//'3D_rmsd_X0_trans_'//&
              trim(continue_job%field1%ch)//'_'//trim(ch4))
     endif
     
     i1 = rsmd%skip_times
     i2 = rsmd%N_print
    allocate(reziduu_fit(rsmd%N_Z_BINS,N_type_molecules,3))
    do i = 1, rsmd%N_Z_BINS
    do j = 1, N_type_molecules
    do k = 1, 3
    call fit_a_line(rsmd%time_axis(i1:i2) , RA_zp_t_z_rmsd_trans_x0(i1:i2,i,j,k) , Diff_trans_X0(i,j,k),reziduu_fit(i,j,k)) 
    enddo 
    enddo
    enddo
!    Diff_trans_X0 is in A^2 / ps  = 10.  * (m^2/s)*10^-9
    call print_in_file ( Z_axis, reshape(Diff_trans_X0, (/ rsmd%N_Z_BINS, N_type_molecules*3 /)  ) , &
    trim(path_out)//'M_diffusion_'//trim(continue_job%field1%ch)//'_'//trim(ch4))

    call print_in_file ( Z_axis, reshape(reziduu_fit, (/ rsmd%N_Z_BINS, N_type_molecules*3 /)  ) , &
    trim(path_out)//'trash.reziduu_fit_M_diffusion_'//trim(continue_job%field1%ch)//'_'//trim(ch4))


    call zero_in_counters
    i_time = 0
    i_counter_collected = 0 
    deallocate(reziduu_fit)
    end subroutine  finalize_and_print_in_rsmd


  subroutine collect_rsmd_X0
   use ALL_mols_data, only : mol_xyz,Nmols
     implicit none
     real(8) , allocatable :: zz2(:),xy2(:)
     allocate(xy2(Nmols),zz2(Nmols))
     zz2(:) = (mol_xyz(:,3)-mol_xyz_0(:,3))**2
     xy2(:) = (mol_xyz(:,1)-mol_xyz_0(:,1))**2+(mol_xyz(:,2)-mol_xyz_0(:,2))**2 
     rmsd_xyz_0(:,1) = rmsd_xyz_0(:,1) + zz2(:) + xy2(:)
     rmsd_xyz_0(:,2) = rmsd_xyz_0(:,2) + xy2(:)
     rmsd_xyz_0(:,3) = rmsd_xyz_0(:,3) + zz2(:)
     deallocate(xy2,zz2)
   end subroutine collect_rsmd_X0


  subroutine bin_in_Z_rsmd_X0(N_eval,i_time)
  use rsmd_data
  use ALL_mols_data, only : i_type_molecule,Nmols, l_WALL_MOL_CTRL
  use mol_type_data, only : N_type_molecules
  implicit none
  integer, intent(IN) :: N_eval,i_time
  integer i,j,l,k,Nzbins
  real(8) N1
  real(8) , allocatable :: local_counts(:,:)

  N1 = 1.0d0 /dble(N_eval*i_time)
  Nzbins = rsmd%N_Z_BINS
  allocate(local_counts(Nzbins,N_type_molecules))
  local_counts = 0.0d0
    do i = 1, Nmols
       j = i_type_molecule(i)
       zp_t_z_rmsd_trans_x0(i_time,NB(i),j,:) = zp_t_z_rmsd_trans_x0(i_time,NB(i),j,:) + rmsd_xyz_0(i,:)*N1
       if (.not.l_WALL_MOL_CTRL(i)) then   ! the frozen molecules don't count
       local_counts(NB(i),j) = local_counts(NB(i),j) + 1.0d0 
       endif
    enddo
     do k  = 1,3
     where (local_counts /= 0.0d0) 
        zp_t_z_rmsd_trans_x0(i_time,1:Nzbins,1:N_type_molecules,k) = &
        zp_t_z_rmsd_trans_x0(i_time,1:Nzbins,1:N_type_molecules,k) / local_counts(:,:)
     endwhere
     enddo
  deallocate(local_counts)
  end subroutine bin_in_Z_rsmd_X0

  subroutine get_z_grid
  use sim_cel_data
  use ALL_mols_data, only : mol_xyz, Nmols
  use rsmd_data
  implicit none
  integer Nzbins
  real(8) Inverse_Z,BIN_dZ_RMD,re_center
    integer i
    real(8) Z
    if (i_boundary_CTRL==0.or.i_boundary_CTRL==1) then
       re_center = re_center_ZZ
       molzz(:) = mol_xyz(:,3) - re_center * 0.5d0
    else if  (i_boundary_CTRL==2.or.i_boundary_CTRL==3) then
      molzz(:) = mol_xyz(:,3) - &
      sim_cel(9)*(dble(INT(2.0d0*(mol_xyz(:,3)/sim_cel(9)))) -dble(INT((mol_xyz(:,3)/sim_cel(9)))) )
    else
     print*, 'in statistic not implemented yet i_type_boundary_CTRL=',i_boundary_CTRL
     STOP
    endif

     Nzbins = rsmd%N_Z_BINS
     Inverse_Z = 1.0d0/sim_cel(9)
     BIN_dZ_RMD = dble(Nzbins) * Inverse_Z
     do i = 1, Nmols
        Z = molzz(i)
        NB(i) = INT((Z+0.5d0*sim_cel(9))*BIN_dZ_RMD) + 1 ; NB(i) = min(NB(i),Nzbins)
!      print* , ' i NB(i) =',i,NB(i)
     enddo    
     
  end subroutine get_z_grid

 end module rsmd_module

