module initialize_MD_module

 implicit none

 private :: write_startinginfo_in_out_file
 private :: zero_scalars
 public :: initialize_whole_MD
 
 contains 
 subroutine initialize_whole_MD
 use paralel_env_data
 use char_constants
 use sys_data
 use math_constants
 use physical_constants
 use sim_cel_data
 use max_sizes_data
 use sizes_data
 use Ewald_data
 use connectivity_type_data
 use file_names_data
 use sysdef_module
 use boundaries, only : adjust_box,cel_properties
 use parser
 use poisson_1D, only : Poisson_Matrix, setup_poisson_MATRIX
 use timing_module, only : starting_date, measure_clock
 use non_bonded_lists_builder, only : driver_link_nonbonded_pairs
 use mol_utils, only : verify_zero_molar_mass
  
  integer i0,i1,iostat
  type (continue_job_type) attempt_continuing_jobs
    name_geom_file = trim(path_in) // trim(name_geom_file)

   call sysdef 

   call zero_rolling_averages ! zero the rolling averages
!   call setup_poisson_MATRIX

   call measure_clock(starting_date) ! start the clock
   call write_startinginfo_in_out_file  ! write some final info in output file
   call adjust_box
   call cel_properties(.true.)
   call driver_link_nonbonded_pairs(.true.)
   call zero_scalars
   call verify_zero_molar_mass

 end subroutine initialize_whole_MD

 subroutine zero_rolling_averages
   use zeroes_types_module
   use rolling_averages_data
   use integrate_data,  only  : l_do_QN_CTRL
   implicit none
   call zero_5_type(RA_energy)
   call zero_5_type(RA_stress)
   call zero_5_type(RA_Temperature)
   call zero_5_type(RA_MOM_0)
   call zero_5_type(RA_cg_iter)
   call zero_5_type(RA_shake_iter)
   call zero_5_type(RA_msd2)
   call zero_5_type(RA_diffusion)
   call zero_5_type(RA_sum_charge)
   if (l_do_QN_CTRL) then
    call zero_5_type(RA_Temperature_trans)
    call zero_5_type(RA_Temperature_rot)
   endif
   call zero_5_type(RA_sfc)
   call zero_5_type(RA_dip)

 end  subroutine zero_rolling_averages

 subroutine write_startinginfo_in_out_file
 use file_names_data, only : name_out_file
 use timing_module, only : starting_date
   open(unit=1234,file=trim(name_out_file),access='append',recl=200)
   write(1234,'(A15,A2,I5,2(A3,I3),A2,A1,2(I3,A3),I3)') &
   'JOB STARTED AT:','y:',starting_date%year,': m',starting_date%month,': d',starting_date%day,'||',&
   'h',starting_date%hour,': m',starting_date%min,': s',starting_date%sec
   close(1234)
 end subroutine write_startinginfo_in_out_file

 subroutine zero_scalars
 use RA1_stresses_data
 implicit none
  RA1_stress_kin(:)=0.0d0; 
  RA1_stress_shake(:)=0.0d0
  RA1_stress_bond(:)=0.0d0; 
  RA1_stress_angle(:)=0.0d0; 
  RA1_stress_dih(:)=0.0d0; 
  RA1_stress_deform(:)=0.0d0; 
  RA1_stress_dummy(:)=0.0d0;
  RA1_stress_vdw(:) = 0.0d0;
  RA1_stress_vdw(:) = 0.0d0
  RA1_stress_Qreal(:)=0.0d0;   
  RA1_stress_Qcmplx(:)=0.0d0; 
  RA1_stress_Qcmplx_k_eq_0(:) = 0.0d0
  RA1_stress_Qcmplx_as_in_3D(:) = 0.0d0
  RA1_stress_thetering=0.0d0
  RA1_stress_counts = 0.0d0
 end subroutine zero_scalars

end module initialize_MD_module
