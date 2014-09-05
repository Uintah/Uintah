
 program main

 use main_loops

 use paralel_env_data
 use initialize_MD_module
 use initialize_MD_CYCLE_module
 use force_driver_module
 use integrate_data
 use ensamble_driver_module
 use ALL_atoms_data
 use energies_data
 use diverse, only : Re_Center_forces
 use statistics_eval
 use Lucretius_integrator_module, only : Lucretius_integrator
 use sys_preparation_data
 use temperature_anneal_data
 use energy_driver_module_ENERGY
 use CTRLs_data, only  : put_CM_molecules_in_box_CTRL
 
 use statistics_colect_write_module_drv

 use stresses_data , only : stress
 use variables_short_pairs
 use profiles_data, only : atom_profile
 use ensamble_data 
 use write_stuff
 use collect_data
 use sim_cel_data
 use physical_constants!, only : Red_Vacuum_EL_permitivity_4_Pi,Red_Boltzmann_constant
 use math_constants
 use rolling_averages_data
 use ALL_rigid_mols_data 
 use ALL_mols_data
 use thermostat_Lucretius_data, only : use_lucretius_integrator
 use dummy_module
 use mol_utils, only : put_CM_molecules_in_box
 use atom_type_data, only : atom_type_molindex


 implicit none
real(8), allocatable :: all_g_charges0(:)
 integer i,j,k,ii,Nel
 real(8) fq,sign_,factor,ddd

 fq = dsqrt(Red_Vacuum_EL_permitivity_4_Pi)
!print*, 'fq=',fq
!stop
 nprocs = 1 ; rank = 0  ! for serial
 call initialize_whole_MD
 
 if (put_CM_molecules_in_box_CTRL) call put_CM_molecules_in_box
 if (sys_prep%any_prep) call sys_prep_alloc_and_init
 call MD_loop_noSTAT(N_MD_STEPS_BLANK_FIRST)



   do i = 1, N_MD_STEPS
!print*,'integrate step=',i
     ddd=dble(i)/dble(N_MD_STEPS)
     integration_step = i  ! THAT IS MANDATORY
     if (sys_prep%any_prep) call sys_prep_act(ddd)
     if (anneal_T%any_Tanneal) call anneal_temperature(N_MD_STEPS)

     call initialize_MD_cycle
     call set_CG_flags
     if (use_Lucretius_integrator) then
        call Lucretius_integrator
     else
        call ensamble_driver(1)
        call force_driver_all_forces_1_step
        call Re_Center_forces
        call ensamble_driver(2)
     endif 

!call ALL_energies_1_step

     call statistic_collector_driver
!read(*,*)
 enddo


 end program main


