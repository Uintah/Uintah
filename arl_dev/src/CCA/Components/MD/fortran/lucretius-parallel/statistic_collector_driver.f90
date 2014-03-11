module statistics_colect_write_module_drv
implicit none
private :: restart_some
private :: profiles_driver
public :: statistic_collector_driver

contains
 subroutine  statistic_collector_driver
 use write_stuff, only : writting_driver, write_more_energies,quick_preview_statistics,&
                         write_history
 use collect_data
 use CTRLs_data 
 use integrate_data
 use scalar_statistics_eval
 use rsmd_module, only : driver_rsmd
 use rdfs_module, only : drivers_rdfs_eval
 use rdfs_data, only : rdfs
 use rsmd_data, only : rsmd
 use quick_preview_stats_data, only : quick_preview_stats
 use history_data, only : history
   implicit none
   integer i,j,k
     call update_scalar_statistics   ! scalar statistics done at each time step
     if (quick_preview_stats%any_request) then
       call quick_preview_statistics
     endif
     if (rsmd%any_request) then
       call driver_rsmd
     endif
     if (mod(integration_step,collect_skip)==0) then
       call profiles_driver  ! evaluate profiles if needed
     endif
     if (mod(integration_step,collect_length)==0) then
       call writting_driver    ! write data if needed
       call restart_some
     endif
     if (l_print_more_energies_CTRL.and.(mod(integration_step,N_PRINT_MORE_ENERGIES)==0)) then
       call write_more_energies  ! usefull for NVE
     endif
     if (rdfs%any_request.and.mod(integration_step,rdfs%N_collect) == 0) then
       call drivers_rdfs_eval
     endif
     if (history%any_request) then
       call write_history
     endif 
 end subroutine statistic_collector_driver


! private objects-------------------
   subroutine restart_some
   use collect_data, only : di_collections_short
   use rsmd_data, only : rmsd_qn_med, rmsd_qn_med_2,rmsd_xyz_med, rmsd_xyz_med_2,&
                         zp_rmsd_xyz_med_2, zp_rmsd_xyz_med
      di_collections_short = 0.0d0
      rmsd_qn_med=0.0d0; 
      rmsd_qn_med_2=0.0d0; 
      rmsd_xyz_med=0.0d0; 
      rmsd_xyz_med_2=0.0d0; 
      zp_rmsd_xyz_med_2=0.0d0; 
      zp_rmsd_xyz_med=0.0d0
   end subroutine restart_some   

    subroutine profiles_driver
     use collect_data
     use integrate_data, only : integration_step
     use profiles_data
     use statistics_eval
     implicit none
     logical l_collect,l_print

         di_collections = di_collections + 1.0d0
         di_collections_short = di_collections_short + 1.0d0
         call Z_profiles_evals

    end subroutine profiles_driver


end module statistics_colect_write_module_drv

