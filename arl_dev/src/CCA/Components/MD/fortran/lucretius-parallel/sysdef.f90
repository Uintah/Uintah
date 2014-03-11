 module sysdef_module
 implicit none

public :: sysdef
public :: COMM_bcast_CTRLS
public :: get_MX_size_lists
public :: COMM_bcast_profiles
public :: COMM_bcast_InitTypes
public :: get_ALL_sizes
public :: set_ALL_masses
public :: set_ALL_dummies
public :: set_field_and_polarizabilities
public :: set_polarizabilities
public :: set_field_constrains
public :: set_ALL_charges
public :: set_WALL_flags
public :: locate_atoms_in_molecules
public :: profiles_setup
public :: get_connectivity
public :: validate_of_qn_dynamics
public :: rigid_dynamics_alloc
public :: scan_and_validate_in_config
public :: read_prelim_from_config
public :: read_from_config
public :: COMM_bcast_prelim_from_config
public :: bcast_config
public :: get_DOF
public :: set_force_flag
public :: get_mol_properties
public :: get_eigen
public :: rdfs_initialize
public :: rsmd_initialize
public :: assign_velocities
public :: write_prelimirary_in_out_file
public :: reset_file
 

 contains
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 subroutine sysdef
  use file_names_data
  use paralel_env_data
  use allocate_them, only : ALL_sizes_allocate,Ewald_data_alloc,profiles_alloc, preconditioner_alloc
  use cg_buffer, only : use_cg_preconditioner_CTRL
  use parser
  use vdw_def
  use ewald_def_module
  use atom_type_data
  use excuded_list_eval_module
  use non_bonded_lists_builder
  use integrate_data
  use collect_data
  use cut_off_data
  use profiles_data, only : N_COLLECT_LENGHT
  use Def_atomic_units_module
  use Ewald_data
  use sizes_data, only : Ndummies, Natoms
  use DoDummyInit_module
  use thermostat_Lucretius_data, only : use_lucretius_integrator
  use ALL_atoms_data, only : l_proceed_kin_atom,l_WALL_CTRL,all_atoms_mass,is_dummy

 use non_bonded_lists_data
 use code_generator
 
  implicit none
  logical l_update

  call Def_atomic_units
  name_ff_file = trim(path_in)//trim(name_ff_file)
  if (rank == 0) call read_force_field(trim(name_ff_file))
  name_input_file = trim(path_in)//trim(name_input_file)
  if (rank == 0) call read_input_file(trim(name_input_file))

  name_input_config_file = 'config.config'
  call validate_of_qn_dynamics
  call scan_and_validate_in_config
  if (rank == 0) call read_prelim_from_config ! cell etc
  call COMM_bcast_prelim_from_config

  call COMM_bcast_InitTypes
  call COMM_bcast_CTRLS ! from in.in file ... this is messy....
  cut_off_sq = cut_off*cut_off
  call COMM_bcast_profiles 
  call ewald_data_alloc
  call get_ewald_parameters

  call get_ALL_sizes
  call profiles_setup ! set up its size for allocation profiles
  call profiles_alloc
  call get_MX_size_lists
  set_2_nonbonded_lists_CTRL = use_lucretius_integrator.or.multistep_integrator_CTRL
  call ALL_sizes_allocate
  call locate_atoms_in_molecules

  call set_ALL_masses
  call set_ALL_dummies ! this will give me Ndummies
!  call get_connectivity
!  call get_excluded_lists
  call set_field_and_polarizabilities
  call set_WALL_flags
  call set_ALL_charges ! dipoles may be re-read from config.
print*, 'l_do_QN_CTRL=',l_do_QN_CTRL

  call get_connectivity
  call get_excluded_lists
  if(Ndummies>0) call Do_Dummy_Init

  call rigid_dynamics_alloc
  if (rank == 0) call read_from_config

  call get_DOF          ! Ndummies must be already computed
  call set_force_flag ! after bcast config get the force flags
  call get_mol_properties

  if (use_cg_preconditioner_CTRL) call preconditioner_alloc
 
  call bcast_config
   

  call set_up_vdw_interpol_interact
  call interpolate_charge_vectors
  N_COLLECT_LENGHT = collect_length / collect_skip 
  dfftx = dble(nfftx); dffty = dble(nffty); dfftz = dble(nfftz)
  dfftx2=dfftx**2; dffty2=dffty**2 ; dfftz2=dfftz**2
  dfftxy=dfftx*dffty ; dfftxz=dfftx*dfftz; dfftyz=dffty*dfftz

  call rdfs_initialize
  call rsmd_initialize
  call write_prelimirary_in_out_file
  call reset_file(nf_more_energies)
print*, 'EXIT SYSDEF'

  l_proceed_kin_atom(:) = .not.(is_dummy(:).or.l_WALL_CTRL(:).or.(all_atoms_mass(:)<1.0d-8))

 short_sysdef_f90_file_name=trim(path_out)//'short_sysdef_scr_'//trim(continue_job%field1%ch)//'.f90'
 short_sysdef_java_file_name = trim(path_out)//'short_sysdef_scr_'//trim(continue_job%field1%ch)//'.java'
print*,'name file=',trim(short_sysdef_f90_file_name)
 call generate_short_sysdef_in_f90

 end subroutine sysdef
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 subroutine COMM_bcast_CTRLS
  use ensamble_data
  use integrate_data
  use cut_off_data
  use Ewald_data
  use paralel_env_data 
  use comunications
  use profiles_data
  use collect_data
  use cg_buffer, only : CG_TOLERANCE,cg_predict_restart_CTRL,order_lsq_cg_predictor
  use rsmd_data, only : rsmd
  use rdfs_data, only : rdfs, l_details_rdf_CTRL
   real(8) v(100)
   integer iv(100)
    if (nprocs == 1) RETURN 

iv(1:20)=0
v(1:2) = 0
IF (rank == 0) then
   iv(1) = i_type_EWALD_CTRL
   iv(2) = i_type_integrator_CTRL
   iv(3) = i_type_ensamble_CTRL
   iv(4) = i_type_thermostat_CTRL
   iv(5) = i_type_barostat_CTRL
   select case (i_type_EWALD_CTRL)
   case(0)
   case(1) ! 3D-SMPE
    iv(6) = order_spline_xx
    iv(7) = order_spline_yy
    iv(8) = order_spline_zz
    iv(9) = nfftx
    iv(10) = nffty
    iv(11) = nfftz
   case(2) ! 2D-SMPE
    iv(6) = order_spline_xx
    iv(7) = order_spline_yy
    iv(8) = order_spline_zz
    iv(9) = nfftx
    iv(10) = nffty
    iv(11) = nfftz
    iv(12) = order_spline_zz_k0
    iv(13) = n_grid_zz_k0
   case(3) ! 3D-SLOW
    iv(6) = k_max_x
    iv(7) = k_max_y
    iv(8) = k_max_z
   case(4) ! 2D-SLOW
    iv(6) = k_max_x
    iv(7) = k_max_y
    iv(8) = k_max_z
    v(1) = h_step_ewald2D
   case default
    print*, 'NOT DEFINED EWALD METHOD in COMM_bcast_CTRLS /sysdef ',i_type_EWALD_CTRL
    STOP
   endselect
ENDIF
  call COMM_bcast(iv(1:20))
  call COMM_bcast(v(1:2))
IF (rank /= 0) then
   i_type_EWALD_CTRL         = iv(1)
   i_type_integrator_CTRL    = iv(2)
   i_type_ensamble_CTRL      = iv(3)
   i_type_thermostat_CTRL    = iv(4)
   i_type_barostat_CTRL      = iv(5)
   select case (i_type_EWALD_CTRL)
   case(0)
   case(1) ! 3D-SMPE
    order_spline_xx   = iv(6)
    order_spline_yy   = iv(7)
    order_spline_zz   = iv(8)
    nfftx             = iv(9)
    nffty             = iv(10)
    nfftz             = iv(11)
   case(2) ! 2D-SMPE
    order_spline_xx   = iv(6)
    order_spline_yy   = iv(7)
    order_spline_zz   = iv(8)
    nfftx             = iv(9)
    nffty             = iv(10)
    nfftz             = iv(11)
    order_spline_zz_k0 = iv(12)
    n_grid_zz_k0       = iv(13)
   case(3) ! 3D-SLOW
    k_max_x = iv(6)
    k_max_y = iv(7)
    k_max_z = iv(8)
   case(4) ! 2D-SLOW
    k_max_x = iv(6)
    k_max_y = iv(7)
    k_max_z = iv(8)
    h_step_ewald2D = v(1)
   case default
    print*, 'NOT DEFINED EWALD METHOD in COMM_bcast_CTRLS /sysdef ',i_type_EWALD_CTRL
    STOP
   endselect
ENDIF

v(1:10) = 0.0d0   
IF(rank==0) then
   v(1) = cut_off
   v(2) = displacement
   v(3) = preconditioner_cut_off
   if (i_type_EWALD_CTRL /= 0) then
     v(4) = ewald_alpha
   endif 
   if (i_type_ensamble_CTRL /= 0) then
     v(5) = temperature
   endif
   v(6) = pressure_xx ; v(7) = pressure_yy ; v(8) = pressure_zz ; v(9) = pressure_ALL !they are defaulted by zero.
   v(10) = dens_var
ENDIF
 call  COMM_bcast(v(1:10))
IF(rank/=0) then
   cut_off = v(1)
   displacement = v(2)
   preconditioner_cut_off = v(3)
   if (i_type_EWALD_CTRL /= 0) then
     ewald_alpha = v(4)
   endif
   if (i_type_ensamble_CTRL /= 0) then
     temperature = v(5)
   endif
   pressure_xx=v(6) ; pressure_yy=v(7) ; pressure_zz=v(8) ; pressure_ALL=v(9) !they are defaulted by zero.
   dens_var = v(10)
ENDIF
 call COMM_bcast(collect_skip) ; 
 call COMM_bcast(collect_length)
 call COMM_bcast(time_step)
 call COMM_bcast(N_MD_STEPS)
 call COMM_bcast(CG_TOLERANCE)
 call COMM_bcast(cg_predict_restart_CTRL)
 call COMM_bcast(order_lsq_cg_predictor) 
 call COMM_bcast(rsmd%any_request)
 if (rsmd%any_request) then
   call COMM_bcast(rsmd%N_print)
   call COMM_bcast(rsmd%N_collect)
   call COMM_bcast(rsmd%N_eval)
   call COMM_bcast(rsmd%N_Z_BINS)
 endif
 call COMM_bcast(rdfs%any_request)
 if (rdfs%any_request) then
   call COMM_bcast(rdfs%N_print)
   call COMM_bcast(rdfs%N_collect)
   call COMM_bcast(rdfs%N_PAIRS)
   call COMM_bcast(rdfs%N_Z_BINS)
   call COMM_bcast(l_details_rdf_CTRL)
   IF (rank /= 0) then
     allocate(rdfs%what_input_pair(rdfs%N_PAIRS))
   ENDIF
   call COMM_bcast(rdfs%what_input_pair)
 endif
 end subroutine COMM_bcast_CTRLS

 subroutine get_MX_size_lists
 use connectivity_ALL_data
 use mol_type_data
 use sim_cel_data
 use cut_off_data
 use max_sizes_data 
 use sim_cel_data
 use boundaries
 implicit none
 real(8) local_dens,V1
 call cel_properties(.false.)
 local_dens = dble(Natoms)/Volume
 V1 = 4.0d0/3.0d0*3.14d0*(cut_off+displacement)**3

 MX_list_nonbonded = INT(local_dens*(V1+1.1d0)*dens_var)+1
 MX_list_nonbonded_short = max(MX_list_nonbonded * (INT ( (cut_off_short/cut_off)**3+1.0d0 )),1)
 if (cut_off_short==0.0d0)MX_list_nonbonded_short=0.0d0
 print*, 'cut_off_short cut_off  =',cut_off_short, cut_off
 print*, 'MX_list_nonbonded & short =',MX_list_nonbonded , MX_list_nonbonded_short
 if (MX_list_nonbonded > 2000) then
   print*, 'Make the dens_var smaller ; MX_list_nonbonded=',MX_list_nonbonded,' which is a bit too large'
   STOP
 endif
! list excluded 
 
 end subroutine get_MX_size_lists

 subroutine COMM_bcast_profiles
 use profiles_data
 use comunications, only : COMM_bcast

 implicit none
! do some initializations as well
    call COMM_bcast(l_1st_profile_CTRL); call COMM_bcast(l_2nd_profile_CTRL)
 l_ANY_profile_CTRL = l_1st_profile_CTRL.or.l_2nd_profile_CTRL
 if (l_ANY_profile_CTRL) then
   call COMM_bcast(N_BINS_XX); call COMM_bcast(N_BINS_YY); call COMM_bcast(N_BINS_ZZ)
 endif
 
 end subroutine COMM_bcast_profiles
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 subroutine COMM_bcast_InitTypes
 use paralel_env_data
 use vdw_type_data
 use mol_type_data
 use atom_type_data
 use connectivity_type_data
 use allocate_them, only : mol_type_alloc,vdw_type_alloc,atom_type_alloc,&
     connectivity_type_alloc, interpolate_alloc, pairs_14_alloc
 use comunications
 use init_bcasting
 
 implicit none
! Comunicate the initial sizes of the system
  integer iv(100)

  if (nprocs == 1) RETURN   ! if it is serial then return
IF (rank == 0) then
   iv(1) = N_TYPE_MOLECULES
   iv(2) = N_type_bonds
   iv(3) = N_type_constrains
   iv(4) = N_type_angles
   iv(5) = N_type_dihedrals
   iv(6) = N_TYPES_VDW
   iv(7) = N_TYPE_ATOMS
ENDIF
   call COMM_bcast(iv(1:7)) 
IF (rank /= 0) then
   N_TYPE_MOLECULES     = iv(1)
   N_type_bonds         = iv(2)
   N_type_constrains    = iv(3)
   N_type_angles        = iv(4)
   N_type_dihedrals     = iv(5)
   N_TYPES_VDW          = iv(6)
   N_TYPE_ATOMS         = iv(7)
   call mol_type_alloc 
   call atom_type_alloc
   call vdw_type_alloc 
   call connectivity_type_alloc
   call interpolate_alloc
   call pairs_14_alloc
ENDIF
   call COMM_bcast_mol_type
   call COMM_bcast_atom_type
   call COMM_bcast_vdw_type
   call COMM_bcast_connectivity_type

   
 end subroutine COMM_bcast_InitTypes

 subroutine get_ALL_sizes ! Nmols and Natoms
   use atom_type_data
   use mol_type_data
   use ALL_atoms_data
   use ALL_mols_data
   use connectivity_type_data
   use sizes_data

   implicit none
   integer i,j,k,isum,i1,i2, imoltype,i_type_bond
   Nmols = sum (N_mols_of_type)

   i1 = 0
   do i = 1, N_TYPE_MOLECULES
   do j = 1, N_mols_of_type(i)
   do k = 1, N_type_atoms_per_mol_type(i)
    i1 = i1 + 1
   enddo
   enddo
   enddo
   Natoms = i1

   i1 = 0
   i_type_bond=0
   do i = 1, N_TYPE_MOLECULES
   do j = 1, Nm_type_bonds(i)
   do k = 1, N_mols_of_type(i)
      i1 = i1 + 1
   enddo
   enddo
   enddo
   Nbonds = i1


   i1 = 0
   do i = 1, N_TYPE_MOLECULES
   do j = 1, Nm_type_angles(i)
   do k = 1, N_mols_of_type(i)
      i1 = i1 + 1
   enddo
   enddo
   enddo
   Nangles = i1
  
   i1 = 0
   do i = 1, N_TYPE_MOLECULES
   do j = 1, Nm_type_dihedrals(i)
   do k = 1, N_mols_of_type(i)
      i1 = i1 + 1
   enddo
   enddo
   enddo
   Ndihedrals = i1
  
   i1 = 0
   do i = 1, N_TYPE_MOLECULES
   do j = 1, Nm_type_deforms(i)
   do k = 1, N_mols_of_type(i)
      i1 = i1 + 1
   enddo
   enddo
   enddo
   Ndeforms = i1
 
   i1 = 0
   do i = 1, N_TYPE_MOLECULES
   do j = 1, Nm_type_constrains(i)
   do k = 1, N_mols_of_type(i)
      i1 = i1 + 1
   enddo
   enddo
   enddo
   Nconstrains = i1

   i1 = 0
   do i = 1, N_TYPE_MOLECULES
   do j = 1, Nm_type_14(i)
   do k = 1, N_mols_of_type(i)
      i1 = i1 + 1
   enddo
   enddo
   enddo
   N_pairs_14 = i1


   write (6,*) 'nbonds = ',Nbonds
   write (6,*) 'nangless = ',Nangles
   write (6,*) 'ndihedrals = ',Ndihedrals
   write (6,*) 'nbondconstrains=',Nconstrains

 end subroutine get_ALL_sizes

 subroutine set_ALL_masses
  use ALL_atoms_data
  use atom_type_data
  use mol_type_data
  use ALL_mols_data
  implicit none
  integer i,j,k,i1,jj
  logical l_1

   i1 = 0 
   do i = 1, N_TYPE_MOLECULES
   do j = 1, N_mols_of_type(i)
   do k = 1, N_type_atoms_per_mol_type(i)
    i1 = i1 + 1
    jj = i_type_atom(i1)
    all_atoms_mass(i1) = atom_type_mass(jj)
    l_1 = all_atoms_mass(i1) < 1.0d-6
    if (l_1) then
      all_atoms_massinv(i1) = 1.0d90
    else
      all_atoms_massinv(i1) = 1.0d0/all_atoms_mass(i1)
    endif
   enddo
   enddo
   enddo
 end subroutine set_ALL_masses

 subroutine set_ALL_dummies
  use ALL_atoms_data
  use atom_type_data
  use mol_type_data
  use ALL_mols_data
  use sizes_data, only : Ndummies
  use allocate_them, only : ALL_dummies_alloc, ALL_dummies_DEalloc
  use ALL_dummies_data
  implicit none
  integer i,j,k,i1,jj,i_dummy

   i1 = 0
   i_dummy = 0
   do i = 1, N_TYPE_MOLECULES
   do j = 1, N_mols_of_type(i)
   do k = 1, N_type_atoms_per_mol_type(i)
    i1 = i1 + 1
    jj = i_type_atom(i1)
    is_dummy(i1) = atom_type_isDummy(jj) 
    if (is_dummy(i1)) then 
      i_dummy = i_dummy + 1
    endif
   enddo
   enddo
   enddo

  Ndummies = i_dummy

 call ALL_dummies_alloc


   i1 = 0
   i_dummy = 0
   do i = 1, N_TYPE_MOLECULES
   do j = 1, N_mols_of_type(i)
   do k = 1, N_type_atoms_per_mol_type(i)
    i1 = i1 + 1
    jj = i_type_atom(i1)
    if (is_dummy(i1)) then
      i_dummy = i_dummy + 1
      all_dummy_params(i_dummy,1:3) = atom_type_DummyInfo(jj)%r(1:3)
      i_Style_dummy(i_dummy) = atom_type_DummyInfo(jj)%i
      map_dummy_to_atom(i_dummy) = i1
      map_atom_to_dummy(i1) = i_dummy
    endif
   enddo
   enddo
   enddo 



 print*, 'Ndummies=',Ndummies
 deallocate(atom_type_DummyInfo)
 if (Ndummies==0) then
    call ALL_dummies_DEalloc
 endif
 end subroutine set_ALL_dummies
  subroutine set_field_and_polarizabilities
  use field_constrain_data
  use atom_type_data
  use sizes_data
  use ALL_atoms_data
  implicit none
  integer i,j,k,i1

  call set_field_constrains
  call set_polarizabilities

   N_dipol_polarizable = N_dipol_g_pol + N_dipol_p_pol
   N_charge_polarizable = N_Qg_pol + N_Qp_pol
   N_variables_field_constrained = N_atoms_field_constrained + 3*N_dipol_polarizable  + N_charge_polarizable
   N_atoms_variables  = N_atoms_field_constrained + N_dipol_polarizable + N_charge_polarizable

if (N_variables_field_constrained > 0) then

   allocate(ndx_remap%s_field(Natoms))
   allocate(rec_ndx_remap%s_field(Natoms))
   allocate(ndx_remap%pol_dipole(Natoms))
   allocate(rec_ndx_remap%pol_dipole(Natoms))
   allocate(ndx_remap%pol_Q(Natoms))
   allocate(rec_ndx_remap%pol_Q(Natoms))
   allocate(ndx_remap%var(Natoms))
   allocate(rec_ndx_remap%var(Natoms))
   allocate(ndx_remap%type_var(Natoms))
   allocate(rec_ndx_remap%type_var(Natoms))


! s-field constrained
 i1 = 0
 do i = 1, Natoms
   if (is_sfield_constrained(i)) then
     i1 = i1 + 1
     ndx_remap%s_field(i1) = i
     rec_ndx_remap%s_field(i) = i1
   endif
 enddo
 do i = 1, Natoms
   if (.not.is_sfield_constrained(i)) then
     i1 = i1 + 1
     ndx_remap%s_field(i1) = i
     rec_ndx_remap%s_field(i) = i1
   endif
 enddo
! - DIPOLE POLARIZABLE
 i1 = 0
  do i = 1, Natoms
   if (is_dipole_polarizable(i)) then
     i1 = i1 + 1
     ndx_remap%pol_dipole(i1) = i
     rec_ndx_remap%pol_dipole(i) = i1
   endif
 enddo
 do i = 1, Natoms
   if (.not.is_dipole_polarizable(i)) then
     i1 = i1 + 1
     ndx_remap%pol_dipole(i1) = i
     rec_ndx_remap%pol_dipole(i) = i1
   endif
 enddo
! charge - polarizable
 i1 = 0
  do i = 1, Natoms
   if (is_charge_polarizable(i)) then
     i1 = i1 + 1
     ndx_remap%pol_Q(i1) = i
     rec_ndx_remap%pol_Q(i) = i1
   endif
 enddo
 do i = 1, Natoms
   if (.not.is_charge_polarizable(i)) then
     i1 = i1 + 1
     ndx_remap%pol_Q(i1) = i
     rec_ndx_remap%pol_Q(i) = i1
   endif
 enddo

 
endif ! N_variables_field_constrained > 0

print*,'vars=',N_atoms_field_constrained,N_dipol_polarizable,N_charge_polarizable,N_variables_field_constrained

 end subroutine set_field_and_polarizabilities
 
 subroutine set_polarizabilities
   use physical_constants
   use atom_type_data
   use mol_type_data
   use ALL_atoms_data
   use ALL_mols_data
   use connectivity_type_data
   use sizes_data
   use field_constrain_data
 implicit none
 integer i,j,k,i1,jj
 real(8) fq

   fq = 1.0d0 / Red_Vacuum_EL_permitivity_4_Pi

   i1 = 0
   N_qp_pol = 0
   N_qg_pol = 0
   do i = 1, N_TYPE_MOLECULES
   do j = 1, N_mols_of_type(i)
   do k = 1, N_type_atoms_per_mol_type(i)
    i1 = i1 + 1
    jj = i_type_atom(i1)
    is_charge_polarizable(i1) = is_type_charge_pol(jj)
    is_dipole_polarizable(i1) = is_type_dipole_pol(jj)
    all_Q_pol(i1) = atom_type_Q_pol(jj) 
    if (dabs(atom_type_DIPOLE_pol(jj))>1.0d-10) then
      all_DIPOLE_pol(i1) = 1.0d0/atom_type_DIPOLE_pol(jj) ! * type_dipole_pol came and stay in Amstrom^3  
! while all_DIPOLE_pol is in Amstrom^-3
    else
      all_DIPOLE_pol(i1) = 1.0d40
    endif
    if (is_charge_polarizable(i1)) then
       if (is_charge_distributed(i1)) then
          N_qg_pol = N_qg_pol + 1
       else
          N_qp_pol = N_qp_pol + 1
       endif
    endif
    if (is_dipole_polarizable(i1)) then
       if (is_charge_distributed(i1)) then
          N_dipol_g_pol = N_dipol_g_pol + 1
       else
          N_dipol_p_pol = N_dipol_p_pol + 1
       endif
    endif
   if (is_charge_polarizable(i1).and.is_dipole_polarizable(i1)) then
    print*,'ERROR: both charges and dipoles are polarizable for the same atom; The code ',&
    'atom index = ',i1, 'atom type index = ',i_type_atom(i1),'molecule type =',i,&
    ' cannot handle this case; it should be either the charge or the dipol but NOT BOTH polarizable.; ',&
    ' The program will STOP'
    STOP
   endif
! kind of polarizabilities.  
   enddo
   enddo
   enddo

!print*, 'atom_type_sfield_CONSTR=',atom_type_sfield_CONSTR
!print*, 'is_type_atom_field_constrained=',is_type_atom_field_constrained
 end subroutine set_polarizabilities

 subroutine set_field_constrains
   use physical_constants
   use atom_type_data
   use mol_type_data
   use ALL_atoms_data
   use ALL_mols_data
   use connectivity_type_data
   use sizes_data
   use field_constrain_data
 implicit none
 integer i,j,k,i1,jj
 real(8) fq

   N_atoms_field_constrained = 0

   fq = 1.0d0 / Red_Vacuum_EL_permitivity_4_Pi
   i1 = 0
   do i = 1, N_TYPE_MOLECULES
   do j = 1, N_mols_of_type(i)
   do k = 1, N_type_atoms_per_mol_type(i)
    i1 = i1 + 1
    jj = i_type_atom(i1)
    is_sfield_constrained(i1) = is_type_atom_field_constrained(jj) ! field constrained leads to some
    external_sfield(i1) = atom_type_sfield_ext(jj) * Volt_to_internal_field  ! input in V
    external_sfield_CONSTR(i1) = atom_type_sfield_CONSTR(jj) * Volt_to_internal_field
! kind of polarizabilities.
    if (is_sfield_constrained(i1)) N_atoms_field_constrained = N_atoms_field_constrained + 1
   enddo
   enddo
   enddo

 end subroutine set_field_constrains

 subroutine set_ALL_charges
   use atom_type_data
   use mol_type_data
   use ALL_atoms_data
   use ALL_mols_data
   use connectivity_type_data
   use sizes_data
   use physical_constants
   use sys_data
   use field_constrain_data
 implicit none
 integer i,j,k,i1,jj
 real(8) fq,d,dir(3),ps


  fq = 1.0d0 / Red_Vacuum_EL_permitivity_4_Pi
! The charge enter in e- units; it is converted in internal units.
! The dipol moment p = q*r (q=charge separation ; r = distance)
! The dipole comes in units of (electron-charge*Amstrom)
! I will assign (arbitrarily) the pozition of dipole along OZ
! The dipole may need to be read from config file
   i1 = 0
   do i = 1, N_TYPE_MOLECULES
   do j = 1, N_mols_of_type(i)
   do k = 1, N_type_atoms_per_mol_type(i)

    i1 = i1 + 1
    jj = i_type_atom(i1)
    dir(:) = atom_type_DIR_dipol(jj,:)
    ps = dsqrt(dot_product(dir,dir))
    dir = dir / ps
    all_p_charges(i1)=0.0d0 ; all_g_charges(i1) = 0.0d0
    if (is_type_charge_distributed(jj)) then
        all_G_charges(i1) = atom_type_charge(jj) * dsqrt(fq)
    else
        all_p_charges(i1) = atom_type_charge(jj) * dsqrt(fq)
    endif
    all_charges(i1) = all_p_charges(i1)+all_g_charges(i1)
    is_charge_distributed(i1) = is_type_charge_distributed(jj)
    d = atom_type_dipol(jj) * dsqrt(fq)
    all_dipoles_xx(i1) = dir(1) * d !0.0d0  ! may need change here
    all_dipoles_yy(i1) = dir(2) * d !0.0d0  ! may need change here
    all_dipoles_zz(i1) = dir(3) * d !atom_type_dipol(jj) * dsqrt(fq) ! may need change here 
    all_dipoles(i1) = d
   enddo
   enddo
   enddo

 end subroutine set_ALL_charges

 subroutine set_WALL_flags
      use atom_type_data
   use mol_type_data
   use ALL_atoms_data
   use ALL_mols_data
   use connectivity_type_data
   use sizes_data
   use CTRLs_data, only : l_ANY_WALL_CTRL
 implicit none
 integer i,j,k,i1,jj
 logical l
   i1 = 0
   N_frozen_atoms = 0
   do i = 1, N_TYPE_MOLECULES
   do j = 1, N_mols_of_type(i)
   do k = 1, N_type_atoms_per_mol_type(i)
    i1 = i1 + 1
    jj = i_type_atom(i1)
    l_WALL(i1) = l_TYPE_ATOM_WALL(jj)
    l_WALL1(i1) = l_TYPE_ATOM_WALL_1(jj)
    l_WALL_CTRL(i1) = l_WALL(i1).or.l_WALL1(i1) ! or any other
    if (l_WALL(i1)) then
       N_frozen_atoms = N_frozen_atoms + 1
    endif
    if (l_WALL1(i1)) then
       N_frozen_atoms = N_frozen_atoms + 1
    endif
    if (l_WALL(i1).and.l_WALL1(i1)) then
      print*, 'ERROR : cannot have an atom both l_WALL and l_WALL1'
      STOP
    endif
   enddo
   enddo
   enddo

   l_ANY_WALL_CTRL = N_frozen_atoms > 0 

   i1 = 0
   do i = 1, Nmols
    l = .false.
    do j = start_group(i),end_group(i)
      l=l.or.l_WALL_CTRL(j)
    enddo
    l_WALL_MOL_CTRL(i)=l
   enddo
 end subroutine set_WALL_flags 

 subroutine locate_atoms_in_molecules
 use atom_type_data
 use mol_type_data
 use ALL_atoms_data, only : Natoms, atom_in_which_molecule, atom_in_which_type_molecule, i_type_atom,i_style_atom
 use ALL_mols_data, only : Nmols, i_type_molecule, start_group, end_group, N_atoms_per_mol
 implicit none
 integer i,j,k,isum,i1,i2
  i1 = 0; isum = 0; i2 = 0


 i1=0
 do i = 1, N_TYPE_MOLECULES
 do j = 1, N_type_atoms_per_mol_type(i)
  i1 = i1 + 1
  atom_type_in_which_mol_type(i1) = i
 enddo
 enddo

 i1 = 0
 do i = 1, N_TYPE_MOLECULES
   do j = 1, N_mols_of_type(i)
   isum = isum + 1
   i_type_molecule(isum) = i
    do k = 1, N_type_atoms_per_mol_type(i)
      i2 = i2 + 1
      i1 = i1 + 1
      atom_in_which_molecule(i1) = isum
      atom_in_which_type_molecule(i1) = i
      if (k == 1) then 
           start_group(isum) = i1
      endif
      if (k==N_type_atoms_per_mol_type(i)) then 
       end_group(isum) = i1
      endif
      i_type_atom(i1) = i2
      i_style_atom(i1) = map_atom_type_to_style(i2)
    enddo
    i2 = i2 -  N_type_atoms_per_mol_type(i)
   enddo
   i2 = i2 + N_type_atoms_per_mol_type(i)
 enddo

!do i = 1, Natoms
! print*, i,atom_type_name(i_type_atom(i)),atom_in_which_molecule(i),atom_in_which_type_molecule(i)
!enddo
 end subroutine locate_atoms_in_molecules

 subroutine profiles_setup
 use atom_type_data, only : N_type_atoms, l_TYPEatom_do_stat_on_type,statistics_AtomPair_type
 use mol_type_data, only : N_type_molecules,l_TYPEmol_do_stat_on_type,statistics_MolPair_type
 use sizes_data, only :  N_type_atoms_for_statistics,N_type_mols_for_statistics

 implicit none
 integer i,i1
  i1 = 0
 do i = 1, N_type_atoms
   if (l_TYPEatom_do_stat_on_type(i)) then
      i1 = i1 + 1
      statistics_AtomPair_type(i) = i1
   else
      statistics_AtomPair_type(i) = -999
   endif
 enddo
 N_type_atoms_for_statistics = i1

 i1 = 0
 do i = 1, N_type_molecules
   if (l_TYPEmol_do_stat_on_type(i)) then
      i1 = i1 + 1
      statistics_molPair_type(i) = i1
   else
      statistics_molPair_type(i) = -999
   endif
 enddo
 N_type_mols_for_statistics = i1


 end subroutine profiles_setup

  subroutine get_connectivity
  use connectivity_type_data
  use mol_type_data
  use ALL_mols_data   ! that will inherit ALL_mol_data
  use connectivity_ALL_data
  use ALL_atoms_data, only : contrains_per_atom, is_dummy, &
                            any_intramol_constrain_per_atom,&
                            map_from_intramol_constrain_to_atom,&
                            l_WALL_CTRL,i_type_atom
  use sizes_data, only : Natoms_with_intramol_constrains
  use atom_type_data, only : atom_type_isDummy, is_type_bond_constrained
                         

  integer i,j,k,na,nb,nc,imoltype,it,i1, itbond, ibond,i_type_bond,iii,iat,jat,iconstrain,i11
  integer, allocatable :: ibuf2(:,:)
  real(8), allocatable :: buf2(:,:)

  na = 0; nb=0; nc=0
!  call ALL_mol_alloc   ???? DO I need it here? I don t think so....

  i1 = 0
  do i = 1, N_TYPE_MOLECULES
  do j = 1, Nm_type_bonds(i)
    i1 = i1 + 1
    bond_types(4,i1) = i
  enddo
  enddo

! exclude dummies that are 
  print*, 'update Nm_type_constr from:',Nm_type_constrains
  i1 = 0
  i11 = 0
  iconstrain=0
  it = sum(Nm_type_constrains)
  do i = 1, N_TYPE_MOLECULES
   do j = 1, Nm_type_bonds(i)
   i1 = i1 + 1
   iat = sum(N_type_atoms_per_mol_type(1:i)) - N_type_atoms_per_mol_type(i)
   if (is_type_bond_constrained(i1)) then
    iconstrain = iconstrain + 1
    i11 = i11 + 1
!print*,'i j i1 =',i,j,i1
!print*, 'atoms=',bond_types(2:3,i1)+iat
!print*,'is dummy?',atom_type_isDummy(bond_types(2:3,i1)+iat)
!print*, 'Nm old =',Nm_type_constrains(i)
!read(*,*)
   if (atom_type_isDummy(bond_types(2,i1)+iat).or.atom_type_isDummy(bond_types(3,i1)+iat)) then
         Nm_type_constrains(i) =  Nm_type_constrains(i)  -  1  ! take out the dummy bond as a constrain
         is_type_bond_constrained(i1) = .false.
         i11 = i11 - 1
         if (i11 < 0 ) then
           print*, 'CLUSTERF..CK ERROR in sysdef%get_connectivity when extract out dummies from constrains'
           STOP
         endif
   else
   constrain_types(:,i11)     = constrain_types(:,iconstrain) ! i11 <= i1
   prm_constrain_types(:,i11) = prm_constrain_types(:,iconstrain)
   endif

   endif
   enddo
  enddo
   allocate(ibuf2(lbound(constrain_types,dim=1):ubound(constrain_types,dim=1), i11))
   ibuf2(:,1:i11) = constrain_types(:,1:i11)
   deallocate(constrain_types) 
   allocate(constrain_types(lbound(ibuf2,dim=1):ubound(ibuf2,dim=1), i11))
   constrain_types = ibuf2
   deallocate(ibuf2)
   allocate(buf2(lbound(prm_constrain_types,dim=1):ubound(prm_constrain_types,dim=1), i11))
   buf2(:,1:i11) = prm_constrain_types(:,1:i11)
   deallocate(prm_constrain_types)
   allocate(prm_constrain_types(lbound(buf2,dim=1):ubound(buf2,dim=1), i11))
   prm_constrain_types=buf2
   deallocate(buf2)
   print*, 'to : Nm_type_constr=       ',Nm_type_constrains
   print*, 'update Nconstrains due to dummmies special treatment from',it, ' to ', sum(Nm_type_constrains)
   print*, ' i11 Nconstrains',i11,  sum(Nm_type_constrains)
   if (i11 .ne. sum(Nm_type_constrains)) then
    print*, 'CLUSTERF..K ERROR in sysdef%get_conectivyty i11 is not equal to sum(Nm_type_constrains)'
    STOP
   endif

! done the updates of constrains due to dummies
   i1 = 0
   it = Nconstrains
   do i = 1, N_TYPE_MOLECULES
   do j = 1, Nm_type_constrains(i)
   do k = 1, N_mols_of_type(i)
      i1 = i1 + 1
   enddo
   enddo
   enddo
   Nconstrains = i1
   print*, 'Total number of constrained updated from : ', it , ' to Nconstrains=',Nconstrains, ' by considering the special treatment of dummies'

  do i = 1, Nmols
   imoltype = i_type_molecule(i)
   N_atoms_per_mol(i) = N_type_atoms_per_mol_type(imoltype)
   N_bonds_per_mol(i) = Nm_type_bonds(imoltype)
   N_angles_per_mol(i) = Nm_type_angles(imoltype)
   N_dihedrals_per_mol(i) = Nm_type_dihedrals(imoltype)
   N_deforms_per_mol(i) = Nm_type_deforms(imoltype)
   N_constrains_per_mol(i) = Nm_type_constrains(imoltype)
   l_RIGID_GROUP(i)        = l_RIGID_GROUP_TYPE(imoltype)
  enddo
   ! sum(N_bonds_per_mol(1:Nmols))    !sum(nbondsch(1:Nmols))
   ! sum(N_angles_per_mol(1:Nmols)) !um(nbendsch(1:Nmols))
   ! sum(N_dihedrals_per_mol(1:Nmols)) !sum(nconstrainsch(1:Nmols))
   ! sum(N_constrains_per_mol(1:Nmols)) !sum(ntortsch(1:Nmols))

!  start with bonds

   ibond = 0
   iconstrain=0
   is_bond_constrained=.false.
   do i = 1, Nmols
     imoltype = i_type_molecule(i)
     i_type_bond = sum(Nm_type_bonds(1:imoltype)) - Nm_type_bonds(imoltype)
     iat = sum(N_atoms_per_mol(1:i)) - N_atoms_per_mol(i)
     do j = 1, Nm_type_bonds(imoltype)
       i_type_bond = i_type_bond + 1
       ibond = ibond + 1
       list_bonds(0,ibond) = i_type_bond
       list_bonds(1:2,ibond) = bond_types(2:3,i_type_bond) + iat
       list_bonds(3,ibond) = i
       list_bonds(4,ibond) = bond_types(1,i_type_bond)
       is_bond_dummy(ibond) = is_dummy(list_bonds(1,ibond)) .or. is_dummy(list_bonds(2,ibond))
       is_bond_constrained(ibond) = is_type_bond_constrained(i_type_bond)
     enddo
   enddo

   do i = 1, Nbonds
    if (list_bonds(1,i) < 1 .or. list_bonds(1,i) > Natoms) then
      print*,'CLUSTERF**K ERROR in sysdef%get_connectivity: list_bonds(1,i) outside 1..Natoms',list_bonds(1,i)
      STOP
    endif
    if (list_bonds(2,i) < 1 .or. list_bonds(2,i) > Natoms) then
      print*,'CLUSTERF**K ERROR in sysdef%get_connectivity: list_bonds(2,i) outside 1..Natoms',list_bonds(2,i)
      STOP
    endif
   enddo


! do i = 1, Nbonds
!  print*, list_bonds(:,i), ' > ',i_type_molecule(list_bonds(3,i))
!  if (mod(i,20)==0) read(*,*)
! enddo

   ibond = 0
   do i = 1, Nmols
     imoltype = i_type_molecule(i)
     iii = sum(Nm_type_angles(1:imoltype)) - Nm_type_angles(imoltype)
     iat = sum(N_atoms_per_mol(1:i)) - N_atoms_per_mol(i)
     do j = 1, Nm_type_angles(imoltype)
       iii = iii + 1
       ibond = ibond + 1
       list_angles(0,ibond) = iii
       list_angles(1:3,ibond) = angle_types(2:4,iii) + iat
       list_angles(4,ibond) = i
       list_angles(5,ibond) = angle_types(1,iii)
     enddo
   enddo
   do i = 1, Nangles
    if (list_angles(1,i) < 1 .or. list_angles(1,i) > Natoms) then
      print*,'CLUSTERF**K ERROR in sysdef%get_connectivity: list_angles(1,i) outside 1..Natoms',list_angles(1,i)
      STOP
    endif
    if (list_angles(2,i) < 1 .or. list_angles(2,i) > Natoms) then
      print*,'CLUSTERF**K ERROR in sysdef%get_connectivity: list_angles(2,i) outside 1..Natoms',list_angles(2,i)
      STOP
    endif
    if (list_angles(3,i) < 1 .or. list_angles(3,i) > Natoms) then
      print*,'CLUSTERF**K ERROR in sysdef%get_connectivity: list_angles(3,i) outside 1..Natoms',list_angles(3,i) 
      STOP
    endif
   enddo

! print*,'iangle Nangles=',ibond,Nangles
! do i = 1, Nangles
!  print*, list_angles(:,i), ' > ',i_type_molecule(list_angles(4,i))
!  if (mod(i,20)==0) read(*,*)
! enddo


   ibond = 0
    do i = 1, Nmols
     imoltype = i_type_molecule(i)
     iii = sum(Nm_type_dihedrals(1:imoltype)) - Nm_type_dihedrals(imoltype)
     iat = sum(N_atoms_per_mol(1:i)) - N_atoms_per_mol(i)
     do j = 1, Nm_type_dihedrals(imoltype)
       iii = iii + 1
       ibond = ibond + 1
       list_dihedrals(0,ibond) = iii
       list_dihedrals(1:4,ibond) = dihedral_types(2:5,iii) + iat
       list_dihedrals(5,ibond) = i
       list_dihedrals(6,ibond) = dihedral_types(1,iii)
     enddo
   enddo
   do i = 1, Ndihedrals
    if (list_dihedrals(1,i) < 1 .or. list_dihedrals(1,i) > Natoms) then
      print*,'CLUSTERF**K ERROR in sysdef%get_connectivity: list_dihedrals(1,i) outside 1..Natoms',list_dihedrals(1,i)
      STOP
    endif
    if (list_dihedrals(2,i) < 1 .or. list_dihedrals(2,i) > Natoms) then
      print*,'CLUSTERF**K ERROR in sysdef%get_connectivity: list_dihedrals(2,i) outside 1..Natoms',list_dihedrals(2,i)
      STOP
    endif
    if (list_dihedrals(3,i) < 1 .or. list_dihedrals(3,i) > Natoms) then
      print*,'CLUSTERF**K ERROR in sysdef%get_connectivity: list_dihedrals(3,i) outside 1..Natoms',list_dihedrals(3,i)
      STOP
    endif
    if (list_dihedrals(4,i) < 1 .or. list_dihedrals(4,i) > Natoms) then
      print*,'CLUSTERF**K ERROR in sysdef%get_connectivity: list_dihedrals(4,i) outside 1..Natoms',list_dihedrals(4,i)
      STOP
    endif
   enddo

! print*,'iangle Ndihedrals=',ibond,Ndihedrals
! do i = 1, Ndihedrals
!  print*, list_dihedrals(:,i), ' > ',i_type_molecule(list_dihedrals(5,i))
!  if (mod(i,20)==0) read(*,*)
! enddo

   ibond = 0
    do i = 1, Nmols
     imoltype = i_type_molecule(i)
     iii = sum(Nm_type_deforms(1:imoltype)) - Nm_type_deforms(imoltype)
     iat = sum(N_atoms_per_mol(1:i)) - N_atoms_per_mol(i)
     do j = 1, Nm_type_deforms(imoltype)
       iii = iii + 1
       ibond = ibond + 1
       list_deforms(0,ibond) = iii
       list_deforms(1:4,ibond) = deform_types(2:5,iii) + iat
       list_deforms(5,ibond) = i
       list_deforms(6,ibond) = deform_types(1,iii)
     enddo
   enddo
! done with deforms

   do i = 1, Ndeforms
    if (list_deforms(1,i) < 1 .or. list_deforms(1,i) > Natoms) then
      print*,'CLUSTERF**K ERROR in sysdef%get_connectivity: list_deforms(1,i) outside 1..Natoms',list_deforms(1,i)
      STOP
    endif
    if (list_deforms(2,i) < 1 .or. list_deforms(2,i) > Natoms) then
      print*,'CLUSTERF**K ERROR in sysdef%get_connectivity: list_deforms(2,i) outside 1..Natoms',list_deforms(2,i)
      STOP
    endif
    if (list_deforms(3,i) < 1 .or. list_deforms(3,i) > Natoms) then
      print*,'CLUSTERF**K ERROR in sysdef%get_connectivity: list_deforms(3,i) outside 1..Natoms',list_deforms(3,i)
      STOP
    endif
    if (list_deforms(4,i) < 1 .or. list_deforms(4,i) > Natoms) then
      print*,'CLUSTERF**K ERROR in sysdef%get_connectivity: list_deforms(4,i) outside 1..Natoms',list_deforms(4,i)
      STOP
    endif
   enddo


  ibond = 0
    do i = 1, Nmols
     imoltype = i_type_molecule(i)
     iii = sum(Nm_type_constrains(1:imoltype)) - Nm_type_constrains(imoltype)
     iat = sum(N_atoms_per_mol(1:i)) - N_atoms_per_mol(i)
     do j = 1, Nm_type_constrains(imoltype)
       iii = iii + 1
       ibond = ibond + 1
       list_constrains(0,ibond) = iii
       list_constrains(1:2,ibond) = constrain_types(1:2,iii) + iat  ! no style here
       list_constrains(3,ibond) = i
     enddo
   enddo

   do i = 1, Nconstrains
    if (list_constrains(1,i) < 1 .or. list_constrains(1,i) > Natoms) then
      print*,'CLUSTERF**K ERROR in sysdef%get_connectivity: list_constrains(1,i) outside 1..Natoms',list_constrains(1,i)
      STOP
    endif
    if (list_constrains(2,i) < 1 .or. list_constrains(2,i) > Natoms) then
      print*,'CLUSTERF**K ERROR in sysdef%get_connectivity: list_constrains(2,i) outside 1..Natoms',list_constrains(2,i)
      STOP
    endif
   enddo


!print*,'iconstr Nconst=',ibond, Nconstrains
!print*,'shapes=',shape(bond_types), shape(constrain_types)
!do i = 1, ubound(constrain_types,dim=2)
!print*,constrain_types(1:2,i), ': ',bond_types(2:3,i), '>',constrain_types(1:2,i)-bond_types(2:3,i)
!read(*,*)
!enddo
!print*,'c=',constrain_types(1:2,:)
!print*,'b=',bond_types(1:2,:)
!stop
   contrains_per_atom(1:Natoms) = 0
   do i = 1, Nconstrains
!print*,i,iat,jat
      iat = list_constrains(1,i)
      jat = list_constrains(2,i)
      if (.not.(atom_type_isDummy(i_type_atom(iat)).or.l_WALL_CTRL(iat)) )& 
      contrains_per_atom(iat) = contrains_per_atom(iat) + 1
      if (.not.(atom_type_isDummy(i_type_atom(jat)).or.l_WALL_CTRL(jat)) )&
      contrains_per_atom(jat) = contrains_per_atom(jat) + 1
   enddo 
!stop
   Natoms_with_intramol_constrains=0
   do i = 1, Natoms
   if (contrains_per_atom(i)>0) then 
      Natoms_with_intramol_constrains=Natoms_with_intramol_constrains+1
      any_intramol_constrain_per_atom(i) = .true.
   endif
   enddo
   allocate(map_from_intramol_constrain_to_atom(Natoms_with_intramol_constrains))
   i1 = 0
   do i = 1, Natoms
    if (contrains_per_atom(i)>0) then
       i1 = i1 + 1
       map_from_intramol_constrain_to_atom(i1) = i
    endif
   enddo

 end subroutine get_connectivity

 subroutine validate_of_qn_dynamics
 use integrate_data
 use mol_type_data, only : l_RIGID_GROUP_TYPE, N_TYPE_MOLECULES 
 use allocate_them, only : ALL_rigid_mols_alloc,qn_and_low_deriv_alloc,qn_and_hi_deriv_alloc
 use ALL_mols_data, only : Nmols
 use ALL_atoms_data, only : Natoms
   implicit none
   logical l
   integer i,j,k,i1

   if (.not.l_do_QN_CTRL) RETURN ! do flexible
   do i = 1, N_TYPE_MOLECULES
     if (.not.l_RIGID_GROUP_TYPE(i)) then
       write(6,*) 'ERROR: Molecule type',i,'was not declared as a rigid one; The code cannot do qn (rigid) dynamics'
       write(6,*) 'Either request flexible dynamics in input file or specify in force field file that molecules are rigid'
       STOP
     endif
   enddo
   if (i_type_integrator_CTRL /= 1) then
     print*, 'ERROR in sysdef%validate_of_qn_dynamics', ' qn (rigid) dynamics can only be performed with GEAR_4 ',&
             'integrator for now: PS: update the message if add more integrators to qn dynamics'
     STOP
   endif

 end subroutine validate_of_qn_dynamics

 subroutine rigid_dynamics_alloc
 use integrate_data
 use mol_type_data, only : l_RIGID_GROUP_TYPE, N_TYPE_MOLECULES
 use allocate_them, only : ALL_rigid_mols_alloc,qn_and_low_deriv_alloc,qn_and_hi_deriv_alloc
 use ALL_mols_data, only : Nmols
 use ALL_atoms_data, only : Natoms
   implicit none

   if (.not.l_do_QN_CTRL) return

   call ALL_rigid_mols_alloc
   if (i_type_integrator_CTRL == 1) then
      call qn_and_low_deriv_alloc
      call qn_and_hi_deriv_alloc
   else
print*,'NOT implemented integrator in sysdef&rigid_dynamics_alloc'
STOP
   endif
 
 end subroutine rigid_dynamics_alloc

 subroutine scan_and_validate_in_config
! verify if the config file exist and if it is the right version
 use types_module
 use file_names_data, only : MAX_CH_size,continue_job_type, continue_job, path_out, &
     FILE_continuing_jobs_indexing, name_input_config_file , name_out_file,nf_more_energies
 use chars, only : char_intN_ch_NOBLANK,char_intN_ch, search_file_for_starting_word
 implicit none
 integer i,j,k,iostat,i_index, isave1,isave2,isave3
 logical l_found,ll_star
 character(MAX_CH_size) nf
 type(continue_job_type) attempt1_continuing_jobs
 character(4) ch4,ch4_first,ch_append
 integer  NN0


   nf = ' '
   open(unit=33,file=trim(trim(path_out)//trim(FILE_continuing_jobs_indexing)),status='old',iostat=iostat)
   if (iostat == 0 ) then  
     ll_star=.false.
     read(33,*) continue_job%field1%i, continue_job%field2%i,continue_job%keep_indexing
     call char_intN_ch(4,continue_job%field1%i,ch4)
     if (ch4(1:1).eq.'0') then
      ch4(1:1)=' '
      NN0=2
     endif
     if (ch4(1:1).eq.' '.and.ch4(2:2).eq.'0') then
      ch4(1:1) = ' '; ch4(2:2) =' ' ; NN0=3
     endif
     if (ch4(1:1).eq.' '.and.ch4(2:2).eq.' '.and.ch4(3:3).eq.'0') then
       ch4(1:1) = ' '; ch4(2:2) =' '; ch4(3:3)=' ' ; NN0=4
     endif
     continue_job%field1%ch(1:4)=' '; continue_job%field1%ch(1:4-NN0+1) = ch4(NN0:4)


     call char_intN_ch(4,continue_job%field2%i,continue_job%field2%ch)
     nf = trim(path_out)//'config'//'_'//trim(continue_job%field1%ch)//'_'//trim(continue_job%field2%ch)
     isave1 = continue_job%field1%i ;
     continue_job%field1%i = continue_job%field1%i + 1
     isave2 = continue_job%field2%i
     continue_job%field2%i = 0
     isave3 = continue_job%keep_indexing
   else 
     ll_star = .true.
     nf = trim(path_out)//'config'
   endif
   name_input_config_file = ' '; name_input_config_file=trim(nf)
   print*, 'NAME config file=',trim(name_input_config_file) 
   open(unit=33,file=trim(name_input_config_file),status='old',iostat=iostat)
   if (iostat /= 0) then
     write(6,*) 'The attempted config file "',trim(name_input_config_file),'" does not exist in directory "',trim(path_out),'"',&
     'generate a config file and restart the program'
     STOP
   endif
   close(33)

   call search_file_for_starting_word('INDEX_CONTINUE_JOB', trim(name_input_config_file),i_index,l_found)
   if (.not.l_found) then
         print*, 'The attepmted config file "',trim(name_input_config_file),&  
                  '"does not have the keyword INDEX_CONTINUE_JOB '
         print*, 'Make sure you use the correct version of config file ; add INDEX_CONTINUE_JOB to there '&
                 ' and restart the job'
         STOP
   else
        open(unit=33,file=trim(name_input_config_file))
        do i = 1, i_index;  read(33,*);   enddo
        read(33,*) attempt1_continuing_jobs%field1%i,attempt1_continuing_jobs%field2%i,attempt1_continuing_jobs%keep_indexing
        if (ll_star) then
          isave1 = attempt1_continuing_jobs%field1%i
          isave2 = attempt1_continuing_jobs%field2%i
          isave3 = attempt1_continuing_jobs%keep_indexing
          continue_job%field1%i = attempt1_continuing_jobs%field1%i + 1
          continue_job%field2%i = 0
          continue_job%keep_indexing = attempt1_continuing_jobs%keep_indexing
        endif
        close(33)
        if (attempt1_continuing_jobs%field1%i.ne.isave1) then
          print*, 'Inconsistent readings from config and indexing files for field_1',attempt1_continuing_jobs%field1%i,&
                   continue_job%field1%i
          STOP
        endif
        if (attempt1_continuing_jobs%field2%i.ne.isave2) then
          print*, 'Inconsistent readings from config and indexing files for field_2',attempt1_continuing_jobs%field2%i,&
                   continue_job%field2%i
          STOP
        endif
        if (attempt1_continuing_jobs%keep_indexing.ne.isave3) then
          print*, 'Inconsistent readings from config and indexing files for field keep_indexing',&
                   attempt1_continuing_jobs%keep_indexing,&
                   continue_job%keep_indexing
          STOP
        endif


   endif

  call char_intN_ch(4,continue_job%field1%i ,ch4_first)
    NN0=1
   ch4 = ch4_first
   if (ch4(1:1).eq.'0') then
      ch4(1:1)=' '
      NN0=2
   endif
   if (ch4(1:1).eq.' '.and.ch4(2:2).eq.'0') then
      ch4(1:1) = ' '; ch4(2:2) =' ' ; NN0=3
   endif
   if (ch4(1:1).eq.' '.and.ch4(2:2).eq.' '.and.ch4(3:3).eq.'0') then
       ch4(1:1) = ' '; ch4(2:2) =' '; ch4(3:3)=' ' ; NN0=4
   endif
   continue_job%field1%ch(1:4)=' '; continue_job%field1%ch(1:4-NN0+1) = ch4(NN0:4)

   name_out_file = trim(trim(path_out)//'out_'//trim(continue_job%field1%ch(1:4)))
   nf_more_energies = trim(trim(path_out)//trim('more_energies_'//trim(continue_job%field1%ch(1:4))))
   open(unit=233,file=trim(nf_more_energies))
   write(233,*)''
   close(233)
end subroutine scan_and_validate_in_config

 subroutine read_prelim_from_config
 use file_names_data, only : MAX_CH_size, continue_job, path_out, &
     FILE_continuing_jobs_indexing, name_input_config_file
 use chars, only : char_intN_ch_NOBLANK,char_intN_ch, search_file_for_starting_word
 use comunications
 use sim_cel_data, only : i_boundary_CTRL, sim_cel
 use ensamble_data
 use ALL_atoms_data
 use sizes_data
 use max_sizes_data, only : Max_Mol_Name_Len, Max_Atom_Name_Len
 use atom_type_data
 use in_config_data

 implicit none
 character(Max_Atom_Name_Len) a_ch
 integer iostat, i, i_index,i1,i_type
 logical l_found
 logical, allocatable :: logical_work(:)

print*, 'name input config file=',trim(name_input_config_file)
   call search_file_for_starting_word('SIM_BOX', trim(name_input_config_file),i_index,l_found)
   open(unit = 33,file=trim(name_input_config_file))
   if (.not.l_found) then
       print*, 'ERROR in config file : SIM_BOX not found in config file',trim(name_input_config_file)
       STOP
   else
       do i = 1, i_index;  read(33,*);   enddo
       read(33,*)  sim_cel(1:3) ;
       read(33,*)  sim_cel(4:6) ;
       read(33,*)  sim_cel(7:9) ;
       read(33,*)  i_boundary_CTRL
   endif
   close(33)
   call search_file_for_starting_word('ENSAMBLE', trim(name_input_config_file),i_index,l_found)
   open(unit = 33,file=trim(name_input_config_file))
   if (.not.l_found) then
    print*, 'WARNING : No ensamble defined in config; the value defined in input file will be used'
   else
       do i = 1, i_index;  read(33,*);   enddo
       read(33,*)  i_type
       if (i_type /= i_type_ensamble_CTRL) then
        print*, 'WARNING : CONFIG FILE INCOMPATIBLE WITH INPUT FILE: different kind of ensambles'
        print*, 'i_type_ensamble_CTRL from config =',i_type
        print*, 'i_type_ensamble_CTRL from input =',i_type_ensamble_CTRL
!        STOP
       endif
   endif
   close(33)

 print*, 'preliminaries were read'
 
 end subroutine read_prelim_from_config

 subroutine read_from_config
 use file_names_data, only : MAX_CH_size, continue_job, path_out, &
     FILE_continuing_jobs_indexing, name_input_config_file
 use chars, only : char_intN_ch_NOBLANK,char_intN_ch, search_file_for_starting_word
 use comunications
 use sim_cel_data, only : i_boundary_CTRL, sim_cel
 use ensamble_data
 use ALL_atoms_data
 use sizes_data
 use max_sizes_data, only : Max_Mol_Name_Len, Max_Atom_Name_Len
 use atom_type_data
 use in_config_data
 use field_constrain_data
 use integrate_data, only : l_do_QN_CTRL
 use ALL_rigid_mols_data, only: qn,mol_ANG,mol_MOM
 use ALL_mols_data, only : mol_xyz
 use rotor_utils
 use mol_utils , only : get_all_mol_CM
 use boundaries, only : adjust_box
 use ALL_mols_data, only : N_atoms_per_mol
 use thetering_data
 use allocate_them, only : thetering_alloc
 
 implicit none
 character(Max_Atom_Name_Len) a_ch
 integer iostat, i, i_index,i1,i_type
 logical l_found
 logical, allocatable :: logical_work(:)

   call search_file_for_starting_word('XYZ', trim(name_input_config_file),i_index,l_found)
   open(unit = 33,file=trim(name_input_config_file))
   l_found_in_config%xyz=l_found
   if (.not.l_found) then
      print*, 'WARNING : Atomic position not defined in config file ',trim(name_input_config_file)
      if (.not.l_do_QN_CTRL) then 
       print*, 'ERROR : Atomic position not defined in config file ',trim(name_input_config_file)
       STOP
      endif
   else
       do i = 1, i_index;  read(33,*);   enddo
       do i = 1, Natoms
         read(33,*) xxx(i),yyy(i),zzz(i)
       enddo
   print*, 'XYZ read from config'
   endif
  close(33)

  call search_file_for_starting_word('VXYZ', trim(name_input_config_file),i_index,l_found)
  l_found_in_config%vxyz = l_found
  open(unit = 33,file=trim(name_input_config_file))   
   if (.not.l_found) then
   print*, 'WARNING : initial velocities not defined in config file ',trim(name_input_config_file)
   vxx=0.0d0;vyy=0.0d0;vzz=0.0d0;
   else
       do i = 1, i_index;  read(33,*);   enddo
       do i = 1, Natoms
         read(33,*) vxx(i),vyy(i),vzz(i)
       enddo
    print*, 'vxyz read'   
   endif
   close(33)

   call search_file_for_starting_word('CHARGES', trim(name_input_config_file),i_index,l_found)
   l_found_in_config%qp = l_found
   open(unit = 33,file=trim(name_input_config_file))
   if (l_found) then
   print*, 'sysdef MESSAGE : the charges from force field file will be replaced by those from config file ',trim(name_input_config_file)
       do i = 1, i_index;  read(33,*);   enddo
       i1 = 0
       do i = 1, Natoms
         read(33,*) all_charges(i)
         if (is_charge_distributed(i)) then
            all_G_charges(i) = all_charges(i)
            all_p_charges(i) = 0.0d0
         else
             all_p_charges(i) = all_charges(i)
             all_G_charges(i) = 0.0d0
         endif 
       enddo
   endif
   close(33)
   
   call search_file_for_starting_word('DIPOLES', trim(name_input_config_file),i_index,l_found)
   l_found_in_config%dipole = l_found
   open(unit = 33,file=trim(name_input_config_file))
   if (l_found) then
   allocate(logical_work(Natoms))
   print*, 'sysdef MESSAGE : the dipoles from force field file will be replaced by those from config file ',trim(name_input_config_file)
   i1 = 0
       do i = 1, i_index;  read(33,*);   enddo
       do i = 1, Natoms
         read(33,*) all_dipoles_xx(i),all_dipoles_yy(i),all_dipoles_zz(i),&
                    logical_work(i)
         if (logical_work(i) /= is_dipole_polarizable(i)) then
            print*,'ERROR: In config file the polarizable flag do not correspond to that of input file'
            print*, 'Make the polarizability NON-zero in input file for the atom: ',i_type_atom(i),&
                    ' : ',atom_type_name(i_type_atom(i)), ' or removes DIPOLES entries from config'
            STOP
         endif
         if (logical_work(i)) i1 = i1 + 1
       enddo
!      N_diople_pol = i1
    print*, 'dipoles read from config'
   deallocate(logical_work)
   else
    print*, 'DIPOLES NOT found in config'
   endif
   close(33)


 if (l_do_QN_CTRL) then

   call search_file_for_starting_word('MOL_XYZ',trim(name_input_config_file),i_index,l_found)
   l_found_in_config%mol_xyz = l_found
   if (.not.l_found) then
      print*,'ERROR: In config file MOL_XYZ not defined'
      if (.not.l_found_in_config%xyz) then 
         print*,'ERROR: In config file nor MOL_XYZ neither XYZ not defined'
         STOP
      endif
!      STOP
   endif
   if (l_found) then
     open(unit = 33,file=trim(name_input_config_file))
     do i = 1, i_index;  read(33,*);   enddo
     do i = 1, Nmols
       read(33,*) mol_xyz(i,:)
     enddo
     close(33)
   endif

   call search_file_for_starting_word('QN',trim(name_input_config_file),i_index,l_found)
   l_found_in_config%qn = l_found
   if (.not.l_found) then
      if (l_found_in_config%xyz) then
        call get_quaternions
        open(unit=67,file=trim(path_out)//'trash.QN_'//trim(continue_job%field1%ch(1:4)),recl=200) 
         write(67,*) 'MOL_XYZ'
         do i = 1, Nmols
           write(67,*) mol_xyz(i,:)
         enddo
         write(67,*)
         write(67,*) 'QN'
         do i = 1, Nmols
           write(67,*) qn(i,:)
         enddo
        close(67)
      else
        print*, 'ERROR: Quaternions cannot be generated ; xyz missing ; STOP'
        STOP
      endif
   endif
   if (l_found) then
     open(unit = 33,file=trim(name_input_config_file))
     do i = 1, i_index;  read(33,*);   enddo
     do i = 1, Nmols
       read(33,*) qn(i,:)
     enddo
     close(33)
   endif

   call search_file_for_starting_word('MOL_MOM',trim(name_input_config_file),i_index,l_found)
   l_found_in_config%mol_MOM = l_found
   if (.not.l_found) then
      print*,'sysdef MESSAGE: In config file MOL_MOM not defined'
!      call set_translational_temperature
   else
     open(unit = 33,file=trim(name_input_config_file))
     do i = 1, i_index;  read(33,*);   enddo
     do i = 1, Nmols
       read(33,*) mol_MOM(i,:)
     enddo
     close(33)
   endif

   call search_file_for_starting_word('MOL_ANG',trim(name_input_config_file),i_index,l_found)
   l_found_in_config%mol_ANG = l_found
   if (.not.l_found) then
      print*,'sysdef MESSAGE: In config file MOL_ANG not defined'
   else
     open(unit = 33,file=trim(name_input_config_file))
     do i = 1, i_index;  read(33,*);   enddo
     do i = 1, Nmols
       read(33,*) mol_ANG(i,:)
!     call set_rotational_temperature
     enddo
     close(33)
   endif

 endif

  if (l_do_QN_CTRL) then
  do i = 1, Nmols
    if (N_atoms_per_mol(i)==1) then
     qn(i,:) = 0.0d0 ; qn(i,4) = 1.0d0
     mol_ANG(i,:) = 0.0d0
    endif
  enddo
  endif

   call search_file_for_starting_word('THETERING', trim(name_input_config_file),i_index,l_found)
   open(unit = 33,file=trim(name_input_config_file))
   l_found_in_config%thetering=l_found
   if (.not.l_found) then
   else
       print*, 'THETERING ATOMS IN  config file ',trim(name_input_config_file)
       do i = 1, i_index;  read(33,*);   enddo
       read(33,*) thetering%N
       call thetering_alloc
       do i = 1, thetering%N
         read(33,*) thetering%to_atom(i) , thetering%x0(i),thetering%y0(i),thetering%z0(i), &
                    thetering%kx(i),thetering%ky(i),thetering%kz(i)
         if (thetering%to_atom(i)<1.or.thetering%to_atom(i)>Natoms) then
             print*,'ERROR: INVALID THETERING ATOM :',i,thetering%to_atom(i)
             STOP
         endif
         is_thetering(thetering%to_atom(i))=.true.
       enddo
   print*, 'THETERING read from config'
   endif
  close(33)


 ! DO the re-centering aliong OZ if use slab:
 if (i_boundary_CTRL == 1) then 
    if (l_do_QN_CTRL) then 
       call get_mol_orient_atom_xyz ! atoms in body frame
       call atom_in_lab_frame
    endif
    call adjust_box 
    zzz = zz
 endif

 if (l_do_QN_CTRL) then
    if ((.not.l_found_in_config%mol_xyz).and.l_found_in_config%xyz) then 
     call get_all_mol_CM
    endif
 endif
 
 print*,'l_found_in_config%xyz%molxyz%QN%vxyz%mol_MOM%mol_ANG=',&
 l_found_in_config%xyz, l_found_in_config%mol_xyz,l_found_in_config%qn,&
 l_found_in_config%vxyz,l_found_in_config%mol_MOM,&
 l_found_in_config%mol_ANG
 print*,'l_found_in_config%q%dipoles=',l_found_in_config%qp,l_found_in_config%dipole
 end subroutine read_from_config

 subroutine COMM_bcast_prelim_from_config
 use comunications
 use sim_cel_data, only : i_boundary_CTRL, sim_cel
 use ensamble_data
implicit none
   call COMM_bcast(sim_cel(1:9))
   call COMM_bcast(i_boundary_CTRL)
   call COMM_bcast(i_type_ensamble_CTRL)

 end subroutine COMM_bcast_prelim_from_config

 subroutine bcast_config
 use comunications
 use sim_cel_data, only : i_boundary_CTRL, sim_cel
 use ensamble_data
 use ALL_atoms_data
 use sizes_data, only : Natoms
 use in_config_data
 use field_constrain_data
 implicit none
!   call  COMM_bcast(i_type_thermostat_CTRL)
!   select case(i_type_thermostat_CTRL)
!         case(1)   ! BERDENSEN
!         case(2)   ! NOSE HOOVER
!            call COMM_bcast(thermo_position)
!            call COMM_bcast(thermo_velocity)
!            call COMM_bcast(thermo_force)
!         case(3)   ! NOSE HOOVER CHAIN
!   end select
!   call  COMM_bcast(i_type_ensamble_CTRL)
!   call  COMM_bcast(i_type_barostat_CTRL)
!   select case(i_type_barostat_CTRL)
!          case(1)   ! BERDENSEN anisotropic
!          case(2)   ! BERDENSEN ISOTROPIC
!          case(3)   ! NOSE HOOVER STYLE
!             call  COMM_bcast( baro_position(1:3))
!             call  COMM_bcast( baro_velocity(1:3))
!             call  COMM_bcast(  baro_force(1:3) )
!   end select
   call COMM_bcast(xxx(1:Natoms)) ; call COMM_bcast(yyy(1:Natoms)) ; call COMM_bcast(zzz(1:Natoms))
   if (l_found_in_config%vxyz) then
   call COMM_bcast(vxx(1:Natoms)) ; call COMM_bcast(vyy(1:Natoms)) ; call COMM_bcast(vzz(1:Natoms))
   else
    ! eval them ; for now set them to zero
     vxx=0.0d0 ; vyy=0.0d0 ; vzz=0.0d0
    ! call assign_velocities(temperature)
   endif

     call COMM_bcast(all_p_charges) ;
     call COMM_bcast(all_g_charges) ; 
     call COMM_bcast(all_charges)
     call COMM_bcast(all_dipoles) ; 
     call COMM_bcast(all_dipoles_xx); call COMM_bcast(all_dipoles_yy); call COMM_bcast(all_dipoles_zz)
   call COMM_bcast(external_sfield) ! defaulted at zero in read_config
   call COMM_bcast(l_WALL) ! defaulted at false in read_config
   call COMM_bcast(N_atoms_field_constrained)
   call COMM_bcast(is_sfield_constrained)

      

 end subroutine bcast_config

 subroutine get_DOF
  use CTRLs_data
  use DOF_data
  use ALL_atoms_data, only : Natoms,atom_dof,l_WALL_CTRL,is_dummy,contrains_per_atom
  use sizes_data, only : Nconstrains,N_frozen_atoms, Ndummies
  use sim_cel_data, only : i_boundary_CTRL
  use ALL_mols_data, only : Nmols, l_WALL_MOL_CTRL,mol_dof,start_group,end_group
  use mol_type_data, only : N_TYPE_MOLECULES, N_mols_of_type,N_type_atoms_per_mol_type
! set_WALL_flags must be called prior to this subroutine
  implicit none
  real(8) starting_dof  
  integer i,j, i1

  starting_dof = 3.0d0 * dble(Natoms) - 3.0d0  ! 3 dof are lost because we remove any center of mass motion.
  if(i_boundary_CTRL==0) starting_dof = starting_dof - 3.0d0
  starting_dof = starting_dof - 3.0d0*dble(N_frozen_atoms)
  DOF_total = starting_dof  - dble(Nconstrains)   - 3.0d0*dble(Ndummies)

  if (DOF_total < 1) then
    print*, 'ERROR: in get_DOF: the system is completely constrained and has no degree of freedom'
    print*, 'DOF = ',DOF_total
!    STOP
  endif 

     i1 = 0
   DOF_MOL_trans=0.0d0
   DOF_MOL_rot=0.0d0
   do i = 1, N_TYPE_MOLECULES
   do j = 1, N_mols_of_type(i)
   i1  =  i1 + 1
   if (.not.l_WALL_MOL_CTRL(i1)) then
       DOF_MOL_trans = DOF_MOL_trans + 3.0d0
     if (N_type_atoms_per_mol_type(i) == 1) then
     ! nothing left
     else if (N_type_atoms_per_mol_type(i) == 2) then 
       DOF_MOL_rot = DOF_MOL_rot + 2.0d0
     else if (N_type_atoms_per_mol_type(i) > 2 ) then
       DOF_MOL_rot = DOF_MOL_rot + 3.0d0 
     endif 
   endif
   enddo
   enddo

 do i = 1, Natoms
 if (l_WALL_CTRL(i).or.is_dummy(i)) then
     atom_dof(i) = 0.0d0
   else
     atom_dof(i) = 3.0d0 - contrains_per_atom(i)/2.0d0
   endif
 enddo
 print*,'sum atom dof = ',sum(atom_dof), 'DOF TOTAL = ',DOF_total

 do i = 1, Nmols
 mol_dof(i) = 0.0d0
  do j = start_group(i),end_group(i)
     mol_dof(i) = mol_dof(i) + atom_dof(j)
  enddo 
 print*,i,'mol_dof=',mol_dof(i)
 enddo

 end subroutine get_DOF

 subroutine set_force_flag
    use CTRLs_data
    use ALL_atoms_data
    implicit none
    real(8), parameter :: SMALL = 1.0d-9
    integer i
    l_ANY_QP_CTRL = .false.
    l_ANY_QP_pol_CTRL=.false.
    l_ANY_S_FIELD_QP_CONS_CTRL=.false.
    l_ANY_QG_CTRL = .false.
    l_ANY_QG_pol_CTRL=.false.
    l_ANY_S_FIELD_QG_CONS_CTRL=.false.
    l_ANY_DIPOLE_CTRL=.false.
    l_ANY_DIPOLE_POL_CTRL=.false.
    
    do i = 1, Natoms   
       if (dabs(all_p_charges(i)) > SMALL) then
         l_ANY_QP_CTRL = .true.
       endif
       if (is_charge_polarizable(i)) then
         l_ANY_Q_pol_CTRL = .true.
          if (is_charge_distributed(i)) then
              l_ANY_QG_pol_CTRL =.true.
          else
              l_ANY_QP_pol_CTRL=.true.
          endif
       endif
       if (dabs(all_G_charges(i)) > SMALL) then
         l_ANY_QG_CTRL = .true.
       endif
       if (dabs(all_charges(i)) > SMALL) then
         l_ANY_Q_CTRL = .true.
       endif

       if (is_sfield_constrained(i)) then
         l_ANY_S_FIELD_CONS_CTRL = .true.
         if (is_charge_distributed(i)) then
            l_ANY_S_FIELD_QG_CONS_CTRL = .true.
         else
            l_ANY_S_FIELD_QP_CONS_CTRL = .true.
         endif
       endif

       if (all_dipoles_xx(i)**2 + all_dipoles_yy(i)**2+all_dipoles_zz(i)**2 > SMALL*1.0d-4) then
          l_ANY_DIPOLE_CTRL = .true.
       endif
       if (is_dipole_polarizable(i)) then
          l_ANY_DIPOLE_POL_CTRL = .true.
       endif

    enddo


    l_ANY_SFIELD_CTRL = l_ANY_S_FIELD_QG_CONS_CTRL.or.l_ANY_S_FIELD_QP_CONS_CTRL
    L_ANY_POL_CTRL = l_ANY_QG_pol_CTRL.or.l_ANY_QP_pol_CTRL.or.l_ANY_DIPOLE_POL_CTRL


    l_QP_CTRL = l_ANY_QP_CTRL.or.l_ANY_QP_pol_CTRL.or.l_ANY_S_FIELD_QP_CONS_CTRL
    l_QG_CTRL = l_ANY_QG_CTRL.or.l_ANY_QG_pol_CTRL.or.l_ANY_S_FIELD_QG_CONS_CTRL
    l_DIP_CTRL = l_ANY_DIPOLE_CTRL.or.l_ANY_DIPOLE_POL_CTRL

   print*, 'l_QP_CTRL l_QG_CTRL l_DIP_CTRL =',l_QP_CTRL,l_QG_CTRL,l_DIP_CTRL
   system_force_CTRL = -999 ! default to something negative nonsense


    if ((.not.l_QP_CTRL).and.(.not.l_QG_CTRL).and.(.not.l_DIP_CTRL)) then
        system_force_CTRL = 0  ! no charges of any kind, only dipoles
    endif 
    if ((l_QP_CTRL).and.(.not.l_QG_CTRL).and.(.not.l_DIP_CTRL)) then
        system_force_CTRL = 1   ! point Only charges
    endif
    if ((l_QP_CTRL).and.(l_QG_CTRL).and.(.not.l_DIP_CTRL)) then
        system_force_CTRL = 2   ! POINT + GAUSS AND no_DIPOLE
    endif
    if ((l_QP_CTRL).and.(l_QG_CTRL).and.(l_DIP_CTRL)) then
        system_force_CTRL = 3   ! POINT + GAUSS + DIPOLE
    endif
    if ((l_QP_CTRL).and.(.not.l_QG_CTRL).and.(l_DIP_CTRL)) then
        system_force_CTRL = 4   ! POINT + NO_GAUSS + DIPOLE
    endif
    if ((.not.l_QP_CTRL).and.(l_QG_CTRL).and.(l_DIP_CTRL)) then
        system_force_CTRL = 5   ! NO_POINT + GAUSS  + DIPOLE
    endif
    if ((.not.l_QP_CTRL).and.(l_QG_CTRL).and.(.not.l_DIP_CTRL)) then
        system_force_CTRL = 6   ! GAUSS only
    endif


    if ((.not.l_ANY_SFIELD_CTRL).and.(.not.L_ANY_POL_CTRL)) then
        system_polarizabil_CTRL = 0  ! no charges of any kind, only dipoles
    endif
    if ((l_ANY_SFIELD_CTRL).and.(.not.L_ANY_POL_CTRL)) then
        system_polarizabil_CTRL = 1  ! no charges of any kind, only dipoles
    endif
    if ((.not.l_ANY_SFIELD_CTRL).and.(L_ANY_POL_CTRL)) then
        system_polarizabil_CTRL = 2  ! no charges of any kind, only dipoles
    endif
    if ((l_ANY_SFIELD_CTRL).and.(L_ANY_POL_CTRL)) then
        system_polarizabil_CTRL = 3  ! no charges of any kind, only dipoles
    endif

 print*, '! system_force_CTRL=', system_force_CTRL
 print*, '! system_polarizabil_CTRL=',system_polarizabil_CTRL
 end subroutine set_force_flag

 subroutine get_mol_properties
 use mol_utils, only : get_all_mol_mass, get_all_mol_CM
 use ALL_rigid_mols_data
 use ALL_mols_data, only : Nmols, mol_mass, i_type_molecule, N_atoms_per_mol, start_group,end_group,&
                           l_WALL_MOL_CTRL, l_RIGID_GROUP
 use integrate_data, only : l_do_QN_CTRL
 use mol_type_data, only: N_type_molecules
 use ALL_atoms_data, only : l_WALL_CTRL
 implicit none
 integer i,j,k

   call get_all_mol_mass
   if (.not.l_do_qn_CTRL) call get_all_mol_CM

   if (l_do_QN_CTRL) then 
      Inverse_Molar_mass = 1.0d0/mol_mass
      l_non_linear_rotor=.true.
      do i = 1, Nmols
        if (N_atoms_per_mol(i)==1) then
           l_non_linear_rotor(i) = .false.
        endif 
      enddo
      call  get_eigen(Inertia_MAIN_TYPE(1:N_type_molecules, 1:3)) 
      do i =1, N_TYPE_Molecules
      do j = 1,3
      if (dabs(Inertia_MAIN_TYPE(i,j)).gt.1.0d-10) then 
        Inverse_Inertia_MAIN_TYPE(i,j)=1.0d0/Inertia_MAIN_TYPE(i,j)
      else
        Inverse_Inertia_MAIN_TYPE(i,j) = 0.0d0
      endif 
      enddo
      enddo
      Inertia_SEC_TYPE(:,1)=Inertia_MAIN_TYPE(:,1)-Inertia_MAIN_TYPE(:,2) !XY
      Inertia_SEC_TYPE(:,2)=Inertia_MAIN_TYPE(:,3)-Inertia_MAIN_TYPE(:,1) !ZX
      Inertia_SEC_TYPE(:,3)=Inertia_MAIN_TYPE(:,2)-Inertia_MAIN_TYPE(:,3) !YZ
      do j=1,3
        Inertia_SEC(:,j)=Inertia_SEC_TYPE(i_type_molecule(:),j)
        Inertia_MAIN(:,j)=Inertia_MAIN_TYPE(i_type_molecule(:),j)
        Inverse_inertia_MAIN(:,j) =Inverse_Inertia_MAIN_TYPE(i_type_molecule(:),j)
        inverse_Inertia_SEC(:,j)= 1.0d0/Inertia_SEC(:,j)
      enddo

   endif

! Validate if all atoms into a rigid molecule are constrained
  do i = 1, Nmols
  if (l_WALL_MOL_CTRL(i)) then
  if (l_RIGID_GROUP(i)) then
   do j = start_group(i),end_group(i)
     if (.not.l_WALL_CTRL(j)) then
print*, 'ERROR: a rigid molecule must have all atoms fixed and not just one'
STOP
     endif
   enddo
  endif
  endif
  enddo
 end subroutine get_mol_properties


  subroutine get_eigen(AA_eigen)
  use eigen, only : cg
  use mol_type_data, only : N_type_molecules, mol_type_xyz0, N_type_atoms_per_mol_type, &
                            N_mols_of_type
  use atom_type_data, only : atom_type_mass
  implicit none
  real(8),intent(OUT):: AA_eigen(N_Type_Molecules,3)
  integer, parameter ::  nm=3, n=nm
  integer i1,i2,i,j,jj
  real(8) Ar(3,3), Ai(3,3)
  integer matz, ierr
  real(8) wr(n),wi(n),zr(nm,n),zi(nm,n), t(3),mass, &
              fv1(n),fv2(n),fv3(n), vvv(n,n)
  real(8) x,y,z,massa
! mol_type_xyz0 is centered around mass-centra
  matz=1 !ask only for eigenvalues and not eigenvectors

  i2 = 0
  do i=1,N_Type_Molecules
   Ai(:,:)=0.0d0
   Ar(:,:) = 0.0d0
   do j=1,N_type_atoms_per_mol_type(i)
     i2=i2+1
     massa = atom_type_mass(i2)
     x = mol_type_xyz0(i,j,1) ; y = mol_type_xyz0(i,j,2) ; z = mol_type_xyz0(i,j,3)
     Ar(1,1)=Ar(1,1)+ (y**2+z**2)*massa
     Ar(2,2)=Ar(2,2)+ (x**2+z**2)*massa
     Ar(3,3)=Ar(3,3)+ (x**2+y**2)*massa
     Ar(1,2)=Ar(1,2)- x*y*massa 
     Ar(1,3)=Ar(1,3)- x*z*massa
     Ar(2,3)=Ar(2,3)- y*z*massa
   enddo
    Ar(2,1)=Ar(1,2)
    Ar(3,1)=Ar(1,3)
    Ar(3,2)=Ar(2,3)
    call cg(nm,n,ar,ai,wr,wi,matz,zr,zi,fv1,fv2,fv3,ierr)
    if (ierr.ne.0) then
      write(6,*) 'ERROR in get_eigen: The alghorithm failed to provide the eigenvalues.'
       write(6,*) ' Provide principial axis from input using a different method'
       write(6,*) ' The program will now stop'
       STOP
    endif
! Eigen  values are in wr (the reals) and wi (the imag)
!   print*, 'molecule i=',i
!   print*, 'eigenvalues reals=',wr(:)
   do jj=1,3
     if (dabs(wi(jj)).gt.1.0d-5) then
       write(6,*) 'ERROR in get_eigen:You have imaginary eigenvalues. The alghorithm failed to provide them '
       write(6,*) ' Provide principial axis from input using a different method'
       write(6,*) ' The program will now stop'
       STOP
     endif 
   enddo
   AA_eigen(i,:)=wr(:)
  enddo
  end subroutine get_eigen

 subroutine rdfs_initialize
 use rdfs_data
 use cut_off_data, only : cut_off
 use allocate_them, only : rdfs_alloc
 use atom_type_data, only : N_type_atoms
 integer i,j,k,i_pair, i_counter

   if (.not.rdfs%any_request) RETURN

   if (rdfs%N_pairs==1.and.rdfs%what_input_pair(1)==-9999) then
! they were not present in input file and all pairs will be considered
   i_pair = 0
   deallocate(rdfs%what_input_pair)
   allocate(rdfs%what_input_pair(N_type_atoms*(N_type_atoms+1)/2))
   do i = 1, N_type_atoms
     do j = i, N_type_atoms
       i_pair = i_pair + 1
       rdfs%what_input_pair(i_pair)  =  i_pair
     enddo
   enddo
   rdfs%N_pairs = N_type_atoms*(N_type_atoms+1)/2
   endif

   call rdfs_alloc
   BIN_rdf = Cut_off/dble(N_BIN_rdf)
   BIN_rdf_inv =  1.0d0 / BIN_rdf



  if (rdfs%N_pairs > N_type_atoms*(N_type_atoms+1)/2) then
     print*, 'ERROR: Too many pairs defined in input file when defining "rdfs" pairs; delete a few'
     STOP
  endif
  do i = 1, rdfs%N_pairs
    if (rdfs%what_input_pair(i) > N_type_atoms*(N_type_atoms+1)/2 ) then
      print*, 'ERROR: when definding "rdfs" pairs; one pairs is out of range; redefine in input file the pair with the ',&
      'value=',rdfs%what_input_pair(i),'to something smaller that',N_type_atoms*(N_type_atoms+1)/2 + 1
       STOP
    endif
  enddo 

  i_pair = 0
  i_counter = 0
  allocate(counter_to_pair_rdfs(rdfs%N_pairs))
  do i = 1, N_type_atoms
     do j = i, N_type_atoms
       i_pair = i_pair + 1
       if (is_in_array(i_pair,rdfs%what_input_pair)) then
          l_rdf_pair_eval(i,j) = .true.
          l_rdf_pair_eval(j,i) = l_rdf_pair_eval(i,j)
          i_counter = i_counter + 1
          which_pair_rdf(i,j) = i_counter
          which_pair_rdf(j,i) = i_counter
          counter_to_pair_rdfs(i_counter) = i_pair  
if (i_counter>rdfs%N_pairs) then
print*,'!!!!!!!!!!!in sysdef%rdfs_initialize i_counter>which%N_pairs ; this cannot be... that is a dramatic error...'
STOP
endif
       endif
       enddo
     enddo
  

if (i_counter /= rdfs%N_pairs) then
print*,'!!!!!!!!!!!in sysdef%rdfs_initialize i_counter/=which%N_pairs ; this cannot be... that is a dramatic error...'
STOP
endif
  
do i = 1, N_type_atoms
print*, l_rdf_pair_eval(i,:)
enddo
do i = 1, N_type_atoms
print*, which_pair_rdf(i,:)
enddo
print*,'counter_to_pair_rdfs=',counter_to_pair_rdfs
 contains 
  logical function is_in_array(i,v)
     implicit none
     integer i,v(:)
     integer k
     do k = lbound(v,dim=1),ubound(v,dim=1)
        if (v(k) == i ) then
             is_in_array = .true.
             RETURN
        endif
     enddo
     is_in_array=.false.
   end function is_in_array
 end subroutine rdfs_initialize

 subroutine rsmd_initialize
    use rsmd_data, only : rmsd_qn_med, rmsd_qn_med_2, rmsd_xyz_med,rmsd_xyz_med_2, zp_rmsd_xyz_med_2, zp_rmsd_xyz_med,&
                          zp_translate_cryterion
    use ALL_mols_data, only : Nmols
    use profiles_data, only : N_BINS_ZZs
    use mol_type_data, only : N_type_molecules
    use sizes_data, only : N_type_atoms_for_statistics, N_type_mols_for_statistics


    allocate(rmsd_qn_med(Nmols,4), rmsd_qn_med_2(Nmols), rmsd_xyz_med(Nmols,3),rmsd_xyz_med_2(Nmols))
    allocate(zp_rmsd_xyz_med_2(N_BINS_ZZs, N_type_molecules) , zp_rmsd_xyz_med(N_BINS_ZZs, 1:3,N_type_molecules))
    rmsd_qn_med=0.0d0; rmsd_qn_med_2=0.0d0; rmsd_xyz_med=0.0d0; rmsd_xyz_med_2=0.0d0
    zp_rmsd_xyz_med_2=0.0d0; zp_rmsd_xyz_med=0.0d0

    allocate(zp_translate_cryterion(N_BINS_ZZs,N_type_mols_for_statistics)) ; zp_translate_cryterion=0.0d0
 end subroutine rsmd_initialize
 

 subroutine assign_velocities(Temp)
 use ALL_atoms_data, only : xxx,yyy,zzz,vxx,vyy,vzz,Natoms, all_atoms_mass,l_WALL,l_WALL1, l_WALL_CTRL, is_dummy
 use random_generator_module, only : GAUSS_DISTRIB, RANF
 use physical_constants, only : Red_Boltzmann_constant
 use DOF_data
 implicit none
 real(8), intent(IN) :: Temp
 integer i,j,k,i_WALL
 real(8) dummy,t(3), TT
 logical no_wall
 real(8) mass
 call get_DOF
 TT = Red_Boltzmann_constant * Temp
 dummy = 1.20d0

 
 t(:) = 0 ! t is the extra-momentum
 i_WALL = 0
 do i = 1, Natoms
  no_wall = .not.l_WALL_CTRL(i)
  mass = all_atoms_mass(i)
  if (mass > 1.0d-9.and.no_wall.and.(.not.is_dummy(i))) then
  vxx(i) = (dsqrt(TT/mass)) * GAUSS_DISTRIB(dummy)
  vyy(i) = (dsqrt(TT/mass)) * GAUSS_DISTRIB(dummy)
  vzz(i) = (dsqrt(TT/mass)) * GAUSS_DISTRIB(dummy)
  else
  vxx(i) = 0.0d0 ; vyy(i) = 0.0d0 ; vzz(i) = 0.0d0
  i_wall = i_wall + 1
  endif
  t(1) = t(1) + vxx(i)*mass
  t(2) = t(2) + vyy(i)*mass
  t(3) = t(3) + vzz(i)*mass
 enddo
 t = t / dble(Natoms-i_Wall)

 do i = 1, Natoms
     no_wall = .not.l_WALL_CTRL(i)
     mass = all_atoms_mass(i)
     if (mass > 1.0d-9.and.no_wall.and.(.not.is_dummy(i))) then
     vxx(i) = vxx(i) - t(1)/mass
     vyy(i) = vyy(i) - t(2)/mass
     vzz(i) = vzz(i) - t(3)/mass
   endif
 enddo

 t(1) = dot_product(vxx,all_atoms_mass)
 t(2) = dot_product(vyy,all_atoms_mass)
 t(3) = dot_product(vzz,all_atoms_mass)

 print*, 'Starting Temp=',dot_product(all_atoms_mass,(vxx*vxx+vyy*vyy+vzz*vzz))/(DOF_total*Red_Boltzmann_constant)

 if (maxval(t)>1.0d-10) then
print*, 'ERRROR in assign_velocities: MOMENTUM is not conserved pxx pyy pzz=',t
 endif
 
 
   
 end subroutine assign_velocities 

 subroutine write_prelimirary_in_out_file
 use file_names_data, only : name_out_file
 use collect_data
 use ensamble_data
 use ALL_atoms_data, only : all_atoms_mass,vxx,vyy,vzz,Natoms
 use physical_constants
 use DOF_data
 use atom_type_data
 use connectivity_type_data
 use mol_type_data
 use Ewald_data
 use integrate_data
 use pairs_14_data
 use CTRLs_data
 use rdfs_data, only : l_rdf_pair_eval, rdfs,counter_to_pair_rdfs,which_pair_rdf
 implicit none
 integer i,j,k
  open(unit=1234,file=trim(name_out_file),recl=200)
  write(1234,*)'----STARTING THE SIMULATIONS-------'
  write(1234,*)'N_MD_STEPS time_step =',N_MD_STEPS, time_step
  write(1234,*)' skip ; collect =',collect_skip, collect_length
  write(1234,*)'ENSAMBLE /thermos/barr=', i_type_ensamble_CTRL,i_type_thermostat_CTRL,i_type_barostat_CTRL
  write(1234,*) 'Degrees of freedom ; total atoms = ',DOF_total, Natoms
  write(1234,*) 'Starting Temperature =',dot_product(all_atoms_mass,(vxx*vxx+vyy*vyy+vzz*vzz))/(DOF_total*Red_Boltzmann_constant)  
  write(1234,*) 

   write(1234,*) 'N_TYPE_ATOMS=',N_TYPE_ATOMS
   write(1234,*) 'q  dipol mass name isDummy? which_mol'
   do i = 1, N_TYPE_ATOMS
     write(1234,'(2(F8.4,1X), F8.3,1X, A4,1X, L2,1X, I3)'  ) atom_type_charge(i),atom_type_dipol(i),&
                                                      atom_type_mass(i),atom_type_name(i)
   enddo
   write(1234,*) 'Intramolecular types 1) mol  : bond: constrain: angle: dihedral: 14'
   do i = 1, N_TYPE_MOLECULES
     write(1234,*)i, Nm_type_bonds(i),Nm_type_constrains(i),Nm_type_angles(i),Nm_type_dihedrals(i),Nm_type_14(i)
   enddo

   do i = 1, N_TYPE_MOLECULES
     write(1234,*) ' MOLNAME=',trim(mol_type_name(i)), '      :NumMols=',N_mols_of_type(i),&
                '  : Atoms-type=',N_type_atoms_per_mol_type(i)
   enddo
!   do i = 1, N_TYPES_VDW
!     write(6,*) i,atom_type_name(vdw_type_atom_1_type(i,1:size_vdw_type_atom_1_type(i))),&
!     atom_type_name(vdw_type_atom_2_type(i,1:size_vdw_type_atom_1_type(i))),&
!     'prm_vdw=', prm_vdw_type(1:vdw_type_Nparams(i),i)
!   enddo
   if (i_type_unit_vdw_input_CTRL==0) then
   write(1234,*) 'units of input vdw : STANDARD input units'
   else if (i_type_unit_vdw_input_CTRL==1) then
   write(1234,*) 'units of input vdw : ATOMIC UNITS'
   else
     print*,'ERROR in sysdef%write_prelimirary_in_out_file; unknown input units for vdw'
     print*,'i_type_unit_vdw_input_CTRL=',i_type_unit_vdw_input_CTRL
     STOP
   endif 
   write(1234,*) 'pair  | Style | N_vdwPrm | vdwPrm '
   do i = 1, N_STYLE_ATOMS*(N_STYLE_ATOMS+1)/2
     write(1234,'(2(A4,1X),I4,1X,A3,I3,A3)',advance='NO')&
     atom_style_name(pair_which_style(i)%i), atom_style_name(pair_which_style(i)%j),&
     atom_style2_vdwStyle(i),' | ', atom_style2_N_vdwPrm(i),' | '
     do j = 1, atom_style2_N_vdwPrm(i)
       write(1234,'(1X,F14.5)',advance='no') atom_style2_vdwPrm(j,i)
     enddo
     write(1234,*)
   enddo
 if (rdfs%any_request) then
    write(1234,*)'rdfs matrix:'
    do i = 1, N_type_atoms
      write(1234,*)l_rdf_pair_eval(i,:)
    enddo
    do i = 1, N_type_atoms
      write(1234,*)which_pair_rdf(i,:)
    enddo
    write(1234,*)'counter_to_pair_rdfs=',counter_to_pair_rdfs
 endif 
  write(1234,*) '-----------------------\\\\\\  END PRELIMINARIES --------------------'


 close(1234)
 end subroutine write_prelimirary_in_out_file
 subroutine reset_file(nf)
 character(*), intent(IN):: nf
 open(unit=767,file=trim(nf),status='replace')
 close(767)
 end subroutine reset_file
 end module sysdef_module
