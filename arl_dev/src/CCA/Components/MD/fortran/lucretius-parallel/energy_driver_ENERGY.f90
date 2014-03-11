 module energy_driver_module_ENERGY

 implicit none

 public :: energy_driver_ENERGY
 public :: ALL_energies_1_step
 public :: energy_difference_MC_move_1atom_1step

 contains

!************************************
!!!!!!!!!!!!!!!!!!!!!!!!!!


 subroutine energy_driver_ENERGY
 use non_bonded_lists_builder
 use pair_short_ENERGY
 use smpe_driver, only : smpe_Q_ENERGY, smpe_Q_DIP_ENERGY
 use ewald_2D_k0_Q_module_ENERGY
 use ewald_2D_k0_Q_DIP_module_ENERGY
 use exclude_intra_ENERGY
 use ALL_atoms_data, only : Natoms,  all_charges,all_p_charges,all_g_charges,&
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



 select case (system_force_CTRL)
 case(0) ! only vdw

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 call pair_short_vdw_ENERGY
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
    call pair_short_Q_ENERGY
    if (i_type_EWALD_CTRL==2) then   ! 2D-Slow
       call Ew_Q_SLOW_ENERGY
       if (i_boundary_CTRL == 1) call Ew_2D_k0_Q_SLOW_ENERGY
       call exclude_intra_Q_ENERGY
    else if (i_type_EWALD_CTRL==1) then  ! 2D-SMPE
    call smpe_Q_ENERGY
       if (i_boundary_CTRL == 1) call driver_ewald_2D_k0_Q_ENERGY
    call exclude_intra_Q_ENERGY
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
   call pair_short_Q_ENERGY
   if (i_type_EWALD_CTRL==2) then   ! 2D-Slow
       call Ew_Q_SLOW_ENERGY
       if (i_boundary_CTRL == 1) call Ew_2D_k0_Q_SLOW_ENERGY
       call exclude_intra_Q_DIP_ENERGY
    else if (i_type_EWALD_CTRL==1) then  ! 2D-SMPE
print*,'before smpe_Q_DIP_ENERGY'
    call smpe_Q_DIP_ENERGY
print*,'after smpe_Q_DIP_ENERGY'
       if (i_boundary_CTRL == 1) call driver_ewald_2D_k0_Q_DIP_ENERGY
       call exclude_intra_Q_DIP_ENERGY
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
  call driver_water_surface_ff_ENERGY  ! do it if necesary 

 
 end subroutine energy_driver_ENERGY

 subroutine ALL_energies_1_step
 use intramolecular_forces, only : intramolecular_driver_ENERGY
 use saved_energies, only : save_energies, print_instant_energies, zero_energies
 use kinetics, only : get_kinetic_energy_stress
 use profiles_data, only : l_need_2nd_profile

 implicit none
 logical l_save
 real(8) bla
! use energies_data

! call print_instant_energies

   call save_energies
   call zero_energies
   call energy_driver_ENERGY
   call intramolecular_driver_ENERGY
!  And the kinetic part
   l_save = l_need_2nd_profile
   l_need_2nd_profile = .false.
   call get_kinetic_energy_stress(bla)
   l_need_2nd_profile = l_save

! call print_instant_energies

   
 end subroutine ALL_energies_1_step


 subroutine energy_difference_MC_move_1atom_1step(iwhich)
 use saved_energies, only : d_zero_energies
 use intramolecular_forces, only : intramolecular_driver_ENERGY_MCmove_1atom
 use pair_short_MCmove_1atom

 implicit none
 integer, intent(IN) :: iwhich
  call d_zero_energies
  call intramolecular_driver_ENERGY_MCmove_1atom(iwhich)
  call pair_short_Q_MCmove_1atom(iwhich) ! do both Q and vdw; no flag for vdw only.
  
 end subroutine energy_difference_MC_move_1atom_1step

 end module energy_driver_module_ENERGY
