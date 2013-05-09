 module force_driver_module

 implicit none

 private :: recenter_forces
 private :: xyz_init_before_force
 public :: force_driver_all_forces_1_step
 public :: force_driver

 contains

!************************************
!!!!!!!!!!!!!!!!!!!!!!!!!!

 subroutine force_driver
 use non_bonded_lists_builder
 use short_range_pair_forces
 use smpe_driver
 use ewald_2D_k0_Q_module
 use ewald_2D_k0_Q_DIP_module
 use exclude_intra
 use ALL_atoms_data, only : fxx,fyy,fzz, Natoms, l_WALL1, all_charges,all_p_charges,all_g_charges,&
                            xxx,yyy,zzz, all_dipoles_xx,all_dipoles_yy,all_dipoles_zz
 use long_range_corrections
 use CTRLs_data
 use Ewald_data
 use s_fiels_Q_energy_module
 use field_constrain_data
 use cg_buffer, only : var_history, l_DO_CG_CTRL
 use cg_Q_DIP_module
 use boundaries, only : adjust_box, cel_properties
 use non_bonded_lists_data, only : l_update_VERLET_LIST 
 use energies_data
 use sim_cel_data
 use profiles_data, only : atom_profile
 use water_Pt_ff_module
 use intramolecular_forces
 use kinetics, only : is_list_2BE_updated
! use connectivity_ALL_data, only : list_excluded_HALF, size_list_excluded_HALF

 implicit none
 logical l_update
 integer i,j,k
!validate 
   select case (i_boundary_CTRL)
   case(0)    ! just vacuum ; not implemented and not deeded
print*, 'in force selector; case 0 (vacuum) not implemented'
STOP
   case(1,2,3,4)
! Just do nothing
   case default
print*, 'CASE default in force_selector; option N/A: STOP'
STOP
   end select
! \\\ validate


 call adjust_box ! it is mandatory to start with this statement
! call intramolecular_forces_driver

! call is_list_2BE_updated(l_update) ! done in ensamble  need to be very carefull in ensamble when move the coordinates do the update...
! l_update_VERLET_LIST = l_update
 call driver_link_nonbonded_pairs(l_update_VERLET_LIST)

! call cg_Q_DIP
! call pair_short_forces_Q
! call exclude_intra_Q_DIP
!open(unit=14,file='fort.14',recl=1000)
!write(14,*) Natoms
!do i = 1, Natoms
!write(14,*) i,fxx(i)/418.0d0,fyy(i)/418.0d0,fzz(i)/418.4d0
!enddo
!close(14)
!STOP


 select case (system_force_CTRL)
 case(0) ! only vdw

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 call pair_short_forces_vdw
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

 case(1,2,6) !vdw + QP + QG  AND vdw + QG only


    if (l_DO_CG_CTRL) then
    select case (system_polarizabil_CTRL)

      case(0) ! no + no
      case(1) ! yes(sfield) + no(pol)
        if ( l_ANY_S_FIELD_QG_CONS_CTRL.or.l_ANY_S_FIELD_QP_CONS_CTRL) then 
             call cg_Q_DIP
        else
print*, 'IT SHOULD NOT HIT THIS BRANCH IN FORCE DRIVER; case2/sys_pol1 ; TOTAL MESS_UP OF FTAGS!!!!',system_polarizabil_CTRL ; STOP
        endif
      case(2) ! no(sfield) + yes(pol)
print*, 'force driver case2/sys_pol2 : NOT IMPLEMENTED'; STOP
      case(3) ! yes + yes
print*, 'force driver case2/sys_pol3 : NOT IMPLEMENTED'; STOP
    end select
    endif ! l_DO_CG_CTRL
    call pair_short_forces_Q
    if (i_type_EWALD_CTRL==2) then   ! 2D-Slow
       call Ew_Q_SLOW
       if (i_boundary_CTRL == 1) call Ew_2D_k0_Q_SLOW
       call exclude_intra_Q
    else if (i_type_EWALD_CTRL==1) then  ! 2D-SMPE
    call smpe_Q
       if (i_boundary_CTRL == 1) call driver_ewald_2D_k0_Q
    call exclude_intra_Q
   else
      write(6,*) 'Incompatible case for Ewald module in force_driver_2D'
      STOP
    endif ! (i_type_EWALD_CTRL==4)


 case(3,4,5) !vdw + QP + dipoles ! case 5 = dipoles only
  if(l_DO_CG_CTRL) then
    if (system_polarizabil_CTRL==1.or.system_polarizabil_CTRL==2.or.system_polarizabil_CTRL==3.or.system_polarizabil_CTRL==4) then
        call cg_Q_DIP
    elseif (system_polarizabil_CTRL==0) then ! DO NOTHING
    else
print*, 'IT SHOULD NOT HIT THIS BRANCH IN FORCE DRIVER; case2/sys_pol1 ; TOTAL MESS_UP OF FTAGS!!!!',system_polarizabil_CTRL; STOP
    endif
   endif
   call pair_short_forces_Q
   if (i_type_EWALD_CTRL==2) then   ! 2D-Slow
       call Ew_Q_SLOW
       if (i_boundary_CTRL == 1) call Ew_2D_k0_Q_SLOW 
       call exclude_intra_Q_DIP
    else if (i_type_EWALD_CTRL==1) then  ! 2D-SMPE
    call smpe_Q_DIP
       if (i_boundary_CTRL == 1) call driver_ewald_2D_k0_Q_DIP
       call exclude_intra_Q_DIP
    else
      write(6,*) 'Incompatible case for Ewald module in force_driver_2D'
      STOP
    endif ! (i_type_EWALD_CTRL==4)


 case default
print*, 'case default in force driver ; unknown method',i_type_EWALD_CTRL
STOP
 end select 

! finalize with 

  call tail_correct_vdw
  call s_fiels_Q_energy
  call driver_water_surface_ff  ! do it if necesary 
  call recenter_forces

  where(l_WALL1) 
   fxx=0.0d0;fyy=0.0d0;fzz=0.0d0
  end where

!  if (active_transport%do_it) call active_transport_forces
 
 end subroutine force_driver

 subroutine force_driver_all_forces_1_step
 use intramolecular_forces, only : intramolecular_forces_driver
 use sizes_data, only : Ndummies
 use dummy_module, only : Do_Dummy_Forces,Do_Dummy_Coords
 use ALL_atoms_data, only : fxx,fyy,fzz,Natoms
 use energies_data
 use integrate_data, only : l_do_QN_CTRL
 use ALL_mols_data, only : mol_force
 use ALL_rigid_mols_data, only : mol_torque
 integer i,j,k
 fxx = 0.0d0; fyy = 0.0d0 ; fzz = 0.0d0 
 if (l_do_QN_CTRL) then
  mol_force=0.0d0;
  mol_torque = 0.0d0
 endif 
! if (Ndummies>0) call Do_Dummy_Coords
   call force_driver
   call intramolecular_forces_driver
   if (Ndummies>0) call Do_Dummy_Forces(.true.,fxx,fyy,fzz)
!  open(unit=14,file='fort.14',recl=400)
!  do i=1,Natoms
!   write(14,*) i,fxx(i)/418.4,fyy(i)/418.4,fzz(i)/418.4
!  enddo
!  close(14)
!stop
 end subroutine force_driver_all_forces_1_step

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!*************************************************************************!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


 subroutine xyz_init_before_force
! use boundaries, only : adjust_box, cel_properties
!   call adjust_box
!   call cel_properties(.true.)
 end subroutine xyz_init_before_force

  subroutine recenter_forces
  use ALL_atoms_data, only : fxx,fyy,fzz,Natoms,l_WALL_CTRL,all_atoms_mass
   
  implicit none
  real(8) t(3)
  integer i
  integer, save :: iii111
  logical,allocatable,save::when(:)
  logical,save :: first_time=.true.

  if (first_time) then
    first_time=.false.
    iii111=0
    allocate(when(Natoms))
    do i=1,Natoms
      when(i)=.not.(l_WALL_CTRL(i).or.(all_atoms_mass(i)<1.0d-8))
    enddo
    do i = 1, Natoms
      if(when(i)) then
        iii111 = iii111 + 1
      endif
    enddo

  endif
  t(1) = sum(fxx(1:Natoms))
  t(2) = sum(fyy(1:Natoms))
  t(3) = sum(fzz(1:Natoms))
  t = t / dble(iii111)

  where (when)
   fxx = fxx - t(1)
   fyy = fyy - t(2)
   fzz = fzz - t(3)
  end where

  end subroutine recenter_forces
 end module force_driver_module
