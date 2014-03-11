module main_loops

public :: basic_MD_main_loop
public :: MD_loop_noSTAT

contains
 subroutine basic_MD_main_loop(N_CYCLES)
 use sys_preparation_data
 use initialize_MD_module
 use initialize_MD_CYCLE_module
 use statistics_colect_write_module_drv, only : statistic_collector_driver
 use Lucretius_integrator_module, only : Lucretius_integrator
 use ensamble_driver_module
 use integrate_data, only : integration_step
 use diverse, only : Re_Center_forces
 use force_driver_module, only : force_driver_all_forces_1_step
 use thermostat_Lucretius_data, only : use_Lucretius_integrator

 implicit none
 integer, intent(IN) :: N_CYCLES
 integer i
 real(8) ddd
 do i = 1, N_CYCLES
     ddd=dble(i)/dble(N_CYCLES)
     integration_step = i  ! THAT IS MANDATORY
     if (sys_prep%any_prep) call sys_prep_act(ddd)

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
     call statistic_collector_driver
  enddo
 end subroutine basic_MD_main_loop

 subroutine MD_loop_noSTAT(N_CYCLES)
 use sys_preparation_data
 use initialize_MD_module
 use initialize_MD_CYCLE_module
 use statistics_colect_write_module_drv, only : statistic_collector_driver
 use Lucretius_integrator_module, only : Lucretius_integrator
 use ensamble_driver_module
 use integrate_data, only : integration_step
 use diverse, only : Re_Center_forces
 use force_driver_module, only : force_driver_all_forces_1_step
 use thermostat_Lucretius_data, only : use_Lucretius_integrator
 implicit none
 integer, intent(IN) :: N_CYCLES
 integer i
 do i = 1, N_CYCLES
     integration_step = i
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
  enddo
 end subroutine MD_loop_noSTAT

end module main_loops
