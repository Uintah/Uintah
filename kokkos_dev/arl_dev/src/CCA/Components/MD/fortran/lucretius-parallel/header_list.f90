module types_module
   implicit none
   type word_type
     character(256):: ch
     integer :: length
  end type word_type
   type two_I_one_L_type
      integer :: line
      integer :: word
      logical :: find
   end type two_I_one_L_type
   type two_I_type
      integer :: i
      integer :: j
   end type two_I_type
   type one_I_one_L_type
      integer :: i
      logical :: l
   end type one_I_one_L_type
   type zp1_profile_type
      real(8), allocatable :: p_charge(:)
      real(8), allocatable :: g_charge(:)
      real(8), allocatable :: p_dipole(:)
      real(8), allocatable :: g_dipole(:)
      real(8), allocatable :: density(:)
      real(8), allocatable :: OP(:,:)
      real(8) ::  kin
      real(8) :: DOF
   end type zp1_profile_type
   type zp2_profile_type
     real(8), allocatable :: pot(:)
     real(8), allocatable :: Qpot(:)
     real(8), allocatable :: stress(:,:)
     real(8), allocatable :: fi(:)
     real(8), allocatable :: force(:,:)
   end type zp2_profile_type
   type statistics_3_type
      real(8) :: val ! the actual value
      real(8) :: counts  ! how many times this value is counted
      real(8) :: val_sq ! for msd2
   end type statistics_3_type
   type statistics_5_type
      real(8) :: val ! the actual value
      real(8) :: counts  ! how many times this value is counted
      real(8) :: val_sq ! for msd2
      real(8) :: MIN
      real(8) :: MAX 
   end type statistics_5_type

   type atom_profile_type
      real(8) :: pot
      real(8) :: Qpot
      real(8) :: kin ! kinetic energy
      real(8) :: sxx
      real(8) :: sxy
      real(8) :: sxz
      real(8) :: syx
      real(8) :: syy
      real(8) :: syz
      real(8) :: szx
      real(8) :: szy
      real(8) :: szz
      real(8) :: fi !(electric scalar field)
      real(8) :: EE_xx ! vectorial field xx
      real(8) :: EE_yy ! vectorial field xx
      real(8) :: EE_zz ! vectorial field xx
      real(8) :: buffer3(3)
   end type atom_profile_type
   type bond_profile_type
      real(8) :: pot
      real(8) :: sxx
      real(8) :: sxy
      real(8) :: sxz
      real(8) :: syx
      real(8) :: syy
      real(8) :: syz
      real(8) :: szx
      real(8) :: szy
      real(8) :: szz
      real(8) :: fx
      real(8) :: fy
      real(8) :: fz
      real(8) :: fff
   end type bond_profile_type

   type units_type
      real(8) :: length
      real(8) :: time
      real(8) :: mass
      real(8) :: temperature
      real(8) :: charge
      real(8) :: velocity
      real(8) :: acceleration
      real(8) :: energy
      real(8) :: force
   end type units_type

   type rsmd_var_type
     logical :: any_request
     logical :: print_details
     integer :: N_print
     integer :: N_eval
     integer :: N_collect
     integer :: N_Z_BINS
     integer :: skip_times   ! it will be the starting time
     real(8), allocatable :: time_axis(:)
   end type rsmd_var_type

   type rdfs_var_type
     logical :: any_request
     integer :: N_print
     integer :: N_PAIRS
     integer :: N_collect
     integer :: N_Z_BINS
     integer, allocatable :: what_input_pair(:)
   end type rdfs_var_type  
   type mol_tag_type
     integer :: N
     character(150), allocatable :: tag(:)
   end type mol_tag_type
end module types_module

module extra_line_module
 use types_module, only : word_type
 type(word_type) , allocatable :: extra_line(:)
 integer extra_line_length
 integer extra_line_line
end module extra_line_module

module Def_atomic_units_module
  use types_module, only : units_type
  type (units_type) atomic_units
  contains
   subroutine Def_atomic_units
     atomic_units%length = 5.291772108d-11 ! m
     atomic_units%mass = 9.1093826d-31    ! kg
     atomic_units%charge = 1.60217653-19  ! Coulumbs
     atomic_units%time = 2.418884326505d-17 !s
     atomic_units%velocity = atomic_units%length/atomic_units%time
     atomic_units%acceleration = atomic_units%velocity/atomic_units%time
     atomic_units%force = atomic_units%acceleration * atomic_units%mass
     atomic_units%energy = atomic_units%force * atomic_units%length !4.35974417d-18 ! J
   end subroutine Def_atomic_units
end module Def_atomic_units_module
module paralel_env_data
     integer nprocs ! number of threads (CPUs)
     integer rank   ! the actual thread (CPU)
end module paralel_env_data

module char_constants
   character(1), parameter :: comment_character_1 ='!'
   character(1), parameter :: comment_character_2 ='#'
end module char_constants

module sys_data
     real(8), parameter :: SYS_ZERO = 5.0d-10
end module sys_data

module math_constants
 real(8), parameter :: Pi=3.14159265358979d0, Pi2=6.28318530717959d0, &
                       Pi4=12.5663706143592d0, &
                       Pi_sq=9.86960440108936d0, &
                       Pi_V = 4.0d0/3.0d0*3.14159265358979d0, &
                       rootpi = 1.77245385090552d0, &
                       sqrt_Pi = 1.77245385090552d0,&
                       i_Pi2 = 0.159154943091895d0, &
                       two_per_sqrt_Pi = 1.12837916709551d0
end module math_constants

module physical_constants
      real(8), parameter :: bohr_radius = 5.291772108d-11
      real(8), parameter :: gaskinkc = 1.98709d-3
      real(8), parameter :: Gas_constant=8.314510d0 !J/mol/K
      real(8), parameter :: Avogadro_number=6.0221367d23 !1/mol
      real(8), parameter :: Boltzmann_constant=1.380658d-23 !J/K
      real(8), parameter :: Vacuum_EL_permitivity=8.854187817d-12 !F/m
      real(8), parameter :: Vacuum_EL_permitivity_4_Pi=1.112650056d-10 !F/m
      real(8), parameter :: Vacuum_MAG_perm=12.5663706143592d-7  !H/m
      real(8), parameter :: electron_charge=1.60217733d-19 !C
      real(8), parameter :: Planck_constant=6.6260755d-34 !Js
      real(8), parameter :: light_speed=299792458 !ms
      real(8), parameter :: Faraday_constant=9.64853093d4 !C/mol
      real(8), parameter :: unit_mass=1.6605402d-27 !(u.m.a/Kg)
      real(8), parameter :: unit_time=1.0d-12 !1 picosecond
      real(8), parameter :: unit_length=1.0d-10  !1 Amstrom
      real(8), parameter :: unit_charge=electron_charge !
      real(8), parameter :: unit_temperature=1.0d0 !1 Kelvin
      real(8), parameter :: unit_velocity = unit_length/unit_time !100m/s
      real(8), parameter :: unit_acceleration=unit_velocity/unit_time !10^14 m/s
      real(8), parameter :: unit_force=unit_mass*unit_acceleration !1.6605403d-14 N
      real(8), parameter :: unit_torque=unit_force*unit_length
      real(8), parameter :: unit_Energy=unit_force*unit_length !1.6605402d-23J=10J/mol
      real(8), parameter :: unit_pressure=unit_force/unit_length**2  !1.6605402d7 Pa
      real(8), parameter :: unit_MOM=unit_mass*unit_velocity
      real(8), parameter :: unit_Inertia=unit_mass*unit_length**2
      real(8), parameter :: unit_ANG = unit_Inertia/unit_time
      real(8), parameter :: Red_Planck_constant=Planck_constant/unit_time/unit_energy !!E0*T0
      real(8), parameter :: Red_Boltzmann_constant=Boltzmann_constant/unit_Energy !E0/K
      real(8), parameter :: Red_Gas_constant=Gas_constant/unit_Energy !E0/mol/K
      real(8), parameter :: Red_Vacuum_EL_permitivity=Vacuum_EL_permitivity/unit_charge/unit_charge &
                                                  *unit_force*unit_length**2 !q0**2/(F0*l0**2)
      real(8), parameter :: Red_Vacuum_EL_permitivity_4_Pi=Vacuum_EL_permitivity_4_Pi &
                                                 /unit_charge/unit_charge &
                                                  *unit_force*unit_length**2 !4*Pi*q0**2*F0/l0**2
      real(8), parameter :: Red_light_speed=light_speed*unit_time/unit_length !l0/T0
      real(8), parameter :: Red_Faraday_constant=Faraday_constant/unit_charge !q0/mol
      real(8), parameter :: Red_Vacuum_MAG_perm=1.0d0/Red_Vacuum_EL_permitivity/Red_light_speed**2
      real(8), parameter :: internal_field_to_volt = 3.863184055054315d-2!unit_force*unit_length/electron_charge/dsqrt(Red_Vacuum_EL_permitivity_4_Pi)
      real(8), parameter :: Volt_to_internal_field = 25.8853832939094d0   !1.0d0/internal_field_to_volt
      real(8), parameter :: LJ_epsilon_convert=1000.0d0/Avogadro_number/unit_Energy !E0;eps in Kj/mol
      real(8), parameter :: convert_press_atms_to_Nm2= 101324.999662841d0
      real(8), parameter :: convert_press_Nm2_to_atms = 1.0d0/convert_press_atms_to_Nm2
      real(8), parameter :: temp_cvt_en_ALL = 0.01d0
      real(8), parameter :: factor_pressure = unit_energy/( unit_length**3 )/1.0d5/1.0d3 ! convert pressure from internal units in exit unite
      real(8), parameter :: electron_per_Amstrom_to_Volt = electron_charge/Vacuum_EL_permitivity_4_Pi/unit_length
! that is 14.39965173 V)
      real(8) temp_cvt_en   ! = temp_cvt_en_ALL / dble(Nmols) in kJ/mol/1molecule
      real(8), parameter :: water_compresibility =   0.76365d-2
      real(8), parameter :: calory_to_joule = 4.1840d0
      real(8), parameter :: joule_to_calorie = 1.0d0/calory_to_joule
      real(8), parameter :: eV_to_kJ_per_mol = electron_charge*Avogadro_number/1000.0d0
      real(8), parameter :: Kelvin_to_kJ_per_mol = Gas_constant /1000.0d0

end module physical_constants

MODULE file_names_data
 implicit none

  integer, parameter :: MAX_CH_size=250

 type int_char_type
  character(4)  :: ch
  integer :: i
 end type int_char_type
 type continue_job_type
     type(int_char_type) :: field1
     type(int_char_type) :: field2
     integer :: keep_indexing
 end type continue_job_type

 type (continue_job_type) continue_job

 character (len=MAX_CH_size) :: path_in = './'
 character (len=MAX_CH_size) :: path_out ='./runs/'
 character (len=MAX_CH_size) :: z_A_path ='./runs/z_atoms/'
 character (len=MAX_CH_size) :: z_M_path ='./runs/z_mols/'
 character (len=MAX_CH_size) :: A_path   ='./runs/atoms/'
 character (len=MAX_CH_size) :: M_path   ='./runs/mols/'

 character (len=MAX_CH_size) :: FILE_continuing_jobs_indexing = 'continuing_job_last_index'
 character (len=MAX_CH_size) :: name_input_file = 'in.in'
 character (len=MAX_CH_size) name_input_config_file
 character (len=MAX_CH_size) name_out_config_file
 character (len=MAX_CH_size) :: name_config_file = 'config'
 character (len=MAX_CH_size)  name_out_file !redirect the output
 character (len=MAX_CH_size) :: name_geom_file ='geometry'
 character (len=MAX_CH_size) :: name_ff_file = 'ff.dat'
 character (len=MAX_CH_size) name_xyz_all_atoms_file

 character (len=MAX_CH_size) nf_atom_density
 character (len=MAX_CH_size) nf_mol_density
 character (len=MAX_CH_size) nf_atom_energy
 character (len=MAX_CH_size) nf_mol_energy
 character (len=MAX_CH_size) nf_atom_temperature
 character (len=MAX_CH_size) nf_mol_charges
 character (len=MAX_CH_size) nf_atom_charges
 character (len=MAX_CH_size) nf_atom_dipoles
 character (len=MAX_CH_size) nf_atom_pot
 character (len=MAX_CH_size) nf_atom_field
 character (len=MAX_CH_size) nf_atom_pot_Q
 character (len=MAX_CH_size) nf_mol_pot
 character (len=MAX_CH_size) nf_mol_pot_Q
 character (len=MAX_CH_size) nf_mol_force
 character (len=MAX_CH_size) nf_mol_stress
 character (len=MAX_CH_size) nf_more_energies
 character (len=MAX_CH_size) nf_mol_OP1

 character (len=MAX_CH_size) nf_poisson_field
 character (len=MAX_CH_size)  nf_A_poisson_field
 character (len=MAX_CH_size) name_rdf_CM_CM_file
 character (len=MAX_CH_size) name_rdf_par_CM_CM_file
 character (len=MAX_CH_size) name_rdf_perp_CM_CM_file
 character (len=MAX_CH_size) name_rdf_A_A_file
 character (len=MAX_CH_size) name_rdf_par_A_A_file
 character (len=MAX_CH_size) name_rdf_perp_A_A_file

 character(len=MAX_CH_size) name_autocorr_file
 character(len=MAX_ch_size) name_diffusion_file


 character (len=MAX_CH_size) name_kin_eng_prof_file
 character (len=MAX_CH_size) name_xyz_file_just_mol

 character (len=MAX_CH_size) nf_xyz_av_file
 character (len=MAX_CH_size) nf_zp_MOL_rsmd
 character (len=MAX_CH_size) nf_mol_MOL_rsmd 
 character (len=MAX_CH_size) nf_atom_fi

 integer :: i_out = 6  !=6  !if i_out = 6 then is printed  on dispay
 character (len=MAX_CH_size) nf_quick_preview_stat
 character (len=MAX_CH_size) nf_history, nf_history_Q
 character (len=MAX_CH_size) short_sysdef_f90_file_name
 character (len=MAX_CH_size) short_sysdef_java_file_name

END MODULE file_names_data 

module temperature_anneal_data
type anneal_T_type
 real(8) :: Tstart
 real(8) :: Tend
 real(8) :: dT 
 logical :: any_Tanneal
end type anneal_T_type
type (anneal_T_type) :: anneal_T
contains
subroutine defauts_temperature_anneal_data
 anneal_T%Tstart = 0.0d0 
 anneal_T%Tend = 0.0d0
 anneal_T%dt = anneal_T%Tend - anneal_T%Tstart
 anneal_T%any_Tanneal = .false.
end subroutine defauts_temperature_anneal_data
end module temperature_anneal_data

module sim_cel_data
      integer i_boundary_CTRL ! 
      real(8) sim_cel(9) 
      real(8) Inverse_cel(9) ! 1/sim_cel
      real(8) Volume , inv_Volume
      real(8) Area_xy
      real(8) re_center_ZZ ! for slab only
      real(8) cel_a2, cel_b2, cel_c2    ! |a|^2 ; |b|^2 ; |c|^2
      real(8) cel_cos_ab, cel_cos_ac, cel_cos_bc
      real(8) cel_cos_a_bxc, cel_cos_b_axc, cel_cos_c_axb

      real(8) Reciprocal_cel(9) ! 2Pi/sim_cel
      real(8) Reciprocal_Volume ! 4Pi/V
      real(8) Reciprocal_perp(3)

      real(8) inv_cel_a2, inv_cel_b2, inv_cel_c2    ! |a|^2 ; |b|^2 ; |c|^2
      real(8) inv_cel_cos_ab, inv_cel_cos_ac, inv_cel_cos_bc
      real(8) inv_cel_cos_a_bxc, inv_cel_cos_b_axc, inv_cel_cos_c_axb
end module sim_cel_data

module DOF_data  ! degrees of fredom
      real(8) DOF_total
      real(8) DOF_MOL_rot
      real(8) DOF_MOL_trans
end module DOF_data

module max_sizes_data
      implicit none

      integer, parameter :: MX_BOND_PRMS = 5
      integer, parameter :: MX_CONSTRAINT_PRMS = 2
      integer, parameter :: MX_ANGLE_PRMS = 6
      integer, parameter :: MX_DIH_PRMS = 8
      integer, parameter :: MX_DEFORM_PRMS = 3
      integer, parameter :: MX_VDW_PRMS = 7
      integer :: MX_in_list_14 = 50

      integer MX_excluded ! maxval(N_type_atoms_per_mol_type)

      integer :: MX_interpol_points = 1001 ! number of point of spline interpolation for coffeee()..
      integer, parameter :: MX_box = 25 ! how many boxes
      integer, parameter :: MX_dim = MX_box ! Related to boxes (linked cell) algorithm is maxdim the same with maxbox ?
      integer, parameter :: MX_dim3= MX_dim * MX_dim * MX_dim
      integer, parameter :: Max_Mol_Name_Len = 10
      integer, parameter :: Max_Atom_Name_Len = 5
      integer, parameter :: MAX_Bond_Name_Len = 40
      integer, parameter :: MAX_Angle_Name_Len = 40
      integer, parameter :: MAX_Dih_Name_Len = 40
      integer, parameter :: Max_DEFORM_Name_Len = MAX_Dih_Name_Len
      integer, parameter :: Max_DummyStyle_Name_Len = 40

      integer, parameter :: Max_BondStyle_Name_Len = 25
      integer, parameter :: Max_vdwStyle_Name_Len=25 
      integer, parameter :: Max_AngleStyle_Name_Len = 25
      integer, parameter :: Max_DihStyle_Name_Len = 25
      integer, parameter :: Max_DeformStyle_Name_Len=Max_DihStyle_Name_Len
      

      integer :: MX_cells = 77

      integer MX_GAUSS_charges_per_atom ! maximum number gaussian charges assigned per atoms

      integer MX_list_nonbonded, MX_list_nonbonded_short
   

      integer  maxdummy ! ???
      integer iChainSurf ! ???

end module max_sizes_data

module cut_off_data
 real(8) cut_off, cut_off_sq
 real(8) :: cut_off_short = 0.0d0
 real(8) cut_off_short_sq
 real(8) displacement
 real(8) :: preconditioner_cut_off = 6.0d0 ! for preconditiong in CG minimization 
 real(8) :: preconditioner_cut_off_sq = 6.0d0*6.0d0
 real(8) :: dens_var=1.0d0 ! multiply MAX size of lists by a factor 
 real(8) reciprocal_cut, reciprocal_cut_sq

end module cut_off_data

module sizes_data
     integer N_type_bonds,N_type_constrains,N_type_angles,N_type_dihedrals, N_type_deforms
     integer N_TYPE_ATOMS , N_TYPES_VDW, N_STYLE_ATOMS
     integer Nbonds, Nconstrains, Nangles, Ndihedrals, Ndeforms
     integer Natoms, Nmols, Ndummies 
     integer :: N_frozen_atoms = 0
     integer N_pairs_14, N_type_pairs_14
     integer N_TYPES_DISTRIB_CHARGES   ! how many TYPES Gauss distrib charges are in the system
     integer N_DISTRIB_CHARGES   ! 
     integer Ncells ! number of cells in linked-cell alghorithm
     integer N_qp_pol ! NUmber of point_polarizable charges
     integer N_qG_pol ! Number of Gaussian polarizable charges
     integer N_dipol_p_pol ! number of polarizable dipoles
     integer N_dipol_g_pol

     integer N_type_atoms_for_statistics, N_type_mols_for_statistics
     integer Natoms_with_intramol_constrains

     integer :: N_pairs_sfc_sfc_123 = 0

end module sizes_data

module integrate_data
  real(8) time_step
  integer integration_step
  integer :: i_type_integrator_CTRL = 0   ! 0 for VV and 1 for Gear_4
  logical :: l_do_QN_CTRL = .false.       ! if true will do quaternions dynamics
  integer N_MD_STEPS ! simulation length
  integer :: N_MD_STEPS_BLANK_FIRST=0
  logical :: multistep_integrator_CTRL = .false.
  logical :: lucretius_integrator_more_speed_doit = .false. 
  integer :: lucretius_integrator_more_speed_skip = 1
end module integrate_data

module collect_data
 integer  collect_skip, collect_length
 real(8) :: di_collections = 0.0d0
 real(8) :: di_collections_short = 0.0d0
end module collect_data

module ensamble_data
   integer i_type_ensamble_CTRL
   integer i_type_thermostat_CTRL
   integer i_type_barostat_CTRL
   integer :: N_NH_CHAINS = 1 ! NH by default
   real(8) thermo_coupling
   real(8) barostat_coupling(3)
   real(8) :: thermo_position = 0.0d0 , thermo_velocity = 0.0d0 , thermo_force=0.0d0
   real(8) baro_position(3), baro_velocity(3), baro_force(3)

   real(8) :: pressure_xx = 0.0d0 , pressure_yy = 0.0d0 , pressure_zz = 0.0d0, pressure_ALL = 0.0d0
   real(8) temperature   ! the imposed one
   
   real(8) T_eval ! the evaluated one
   real(8) Temperature_trans_Calc, Temperature_rot_Calc ! for rigid dynamics
  
end module ensamble_data


module CTRLs_data
 implicit none
 logical l_ANY_QG_CTRL    ! any nonzero Gauss charge
 logical l_ANY_QP_CTRL    ! any nonzero Point charge
 logical l_ANY_DIPOLE_CTRL   ! any nonzero dipole

 logical l_ANY_S_FIELD_QP_CONS_CTRL   ! any scalarfield constrained point charge
 logical l_ANY_S_FIELD_QG_CONS_CTRL   ! any scalarfield constrained gauss charge
 logical l_ANY_S_FIELD_DIPOLE_CONS_CTRL ! any scalarfield constrained dipole
 logical l_ANY_S_FIELD_CONS_CTRL        ! any scalar field constrained total charge on an atom

 logical l_ANY_QP_POL_CTRL   ! any polarizable point charge
 logical l_ANY_QG_POL_CTRL   ! any polarizable Gauss charge
 logical l_ANY_DIPOLE_POL_CTRL  ! any polarizable dipole
 logical l_ANY_Q_POL_CTRL     ! any polarizable something on an atom

 logical l_ANY_SFIELD_CTRL
 logical l_ANY_POL_CTRL  

 logical l_ANY_Q_CTRL

 logical l_QP_CTRL !l_ANY_QP_CTRL.or.l_ANY_QP_pol_CTRL.or.l_ANY_S_FIELD_QP_CONS_CTRL
 logical l_QG_CTRL !l_ANY_QG_CTRL.or.l_ANY_QG_pol_CTRL.or.l_ANY_S_FIELD_QG_CONS_CTR
 logical l_DIP_CTRL !l_ANY_DIPOLE_CTRL.or.l_ANY_DIPOLE_POL_CTRL

 logical l_skip_cg_in_first_step_CTRL  ! skip cg calculation in the very first iteration step (usefull to tag things without updating charges)

 logical l_ANY_WALL_CTRL
 logical l_1Gwidth_CTRL     ! is true if all gaussian charges has the same width
 integer system_force_CTRL
 integer system_polarizabil_CTRL

 logical :: l_print_more_energies_CTRL = .false.
 integer N_PRINT_MORE_ENERGIES

 integer :: i_type_unit_vdw_input_CTRL = 0 ! 1 for atomic units 

 logical :: put_CM_molecules_in_box_CTRL = .true. ! sure I want to start with them centered back in the box
end module CTRLs_data

module in_config_data
  type l_found_in_config_type
  logical :: xyz
  logical :: vxyz ! are initial velocities in config?
  logical :: Qp ! are the poin charges in config?
  logical :: Qg ! are the gaussian charges in config?
  logical :: dipole ! are dipoles in config?
  logical :: q_field ! external field acting on each atom
  logical :: field_constrain
  logical :: wall ! ANY frozen atoms?
  logical :: mol_xyz
  logical :: qn
  logical :: mol_mom
  logical :: mol_ang
  logical :: thetering
  end type l_found_in_config_type 
  type(l_found_in_config_type) l_found_in_config

end module in_config_data

module field_constrain_data
  type ndx_remap_type
     integer,allocatable :: s_field(:)
     integer,allocatable :: pol_dipole(:)
     integer,allocatable :: pol_Q(:)
     integer,allocatable :: var(:)
     integer,allocatable :: type_var(:)
  end type ndx_remap_type
  type rec_ndx_remap_type
     integer,allocatable :: s_field(:)
     integer,allocatable :: pol_dipole(:)
     integer,allocatable :: pol_Q(:)
     integer,allocatable :: var(:)
     integer,allocatable :: type_var(:)
  end type rec_ndx_remap_type
  type delta_pol_type
     real(8),allocatable :: s_field(:)
     real(8),allocatable :: pol_dipole(:)
     real(8),allocatable :: pol_Q(:)
  end type delta_pol_type

 integer  N_atoms_field_constrained
 integer  N_variables_field_constrained ! 
 integer  N_type_atoms_field_constrained
 integer N_atoms_variables
 integer N_dipol_polarizable
 integer N_charge_polarizable
 logical, allocatable :: is_type_atom_field_constrained(:)
 integer, allocatable :: ndx_remap_constrained(:), inv_ndx_remap(:)
  
 type(ndx_remap_type)  ndx_remap
 type(rec_ndx_remap_type)  rec_ndx_remap
 type(delta_pol_type) delta_pol
 
end module field_constrain_data


module connectivity_type_data
   use max_sizes_data, only : MX_BOND_PRMS,MX_CONSTRAINT_PRMS,MX_ANGLE_PRMS,MX_DIH_PRMS, MX_DEFORM_PRMS
   use sizes_data, only : N_type_bonds,N_type_constrains,N_type_angles,N_type_dihedrals,N_type_deforms,N_type_pairs_14
   implicit none

   integer N_types_bonded(25)

   integer, allocatable :: bond_types(:,:)
   integer, allocatable :: constrain_types(:,:)
   logical, allocatable :: is_type_bond_constrained(:)
   integer, allocatable :: angle_types(:,:)
   integer, allocatable :: dihedral_types(:,:)
   integer, allocatable :: deform_types(:,:)
   integer, allocatable :: Nfolds_dihedral_types(:) ! an additional integer 
   integer, allocatable :: pair14_types(:,:)
   real(8), allocatable :: prm_bond_types(:,:)
   real(8), allocatable :: prm_constrain_types(:,:)
   real(8), allocatable :: prm_angle_types(:,:)
   real(8), allocatable :: prm_dihedral_types(:,:)
   real(8), allocatable :: prm_deform_types(:,:)
   real(8), allocatable :: prm_pair14_types(:,:)


! bond_type(0,:) = Number of parameters
! bond_type(1,:) = the Style
! bond_type(2,:) = type_atom_1
! bond_type(3,:) = type_atom_2
! bond_type(4,:) = which_type_molecule


end module connectivity_type_data
    
module connectivity_ALL_data
   use sizes_data, only :  Natoms,Nbonds, Nconstrains, Nangles, Ndihedrals,Ndeforms,N_pairs_14,&
                           N_pairs_sfc_sfc_123
   use max_sizes_data, only : MX_excluded, MX_in_list_14
   integer, allocatable :: list_bonds(:,:)
   integer, allocatable :: list_angles(:,:)
   integer, allocatable :: list_dihedrals(:,:)
   integer, allocatable :: list_deforms(:,:)
   integer, allocatable :: list_constrains(:,:)
   integer, allocatable :: list_14(:,:)
   integer, allocatable :: size_list_14(:)
!list_bonds(0,:) = i_type_bond
!list_bonds(1,:) = atom 1 
!list_bonds(2,:) = atom 2
!list_bonds(3,:) = which_molecule
   integer, allocatable :: list_excluded(:,:)
   integer, allocatable :: size_list_excluded(:)
   integer, allocatable :: list_excluded_HALF(:,:)
   integer, allocatable :: size_list_excluded_HALF(:)
   integer, allocatable :: list_excluded_HALF_no_SFC(:,:)
   integer, allocatable :: size_list_excluded_HALF_no_SFC(:)
   integer, allocatable :: list_excluded_sfc_iANDj_HALF(:,:),size_list_excluded_sfc_iANDj_HALF(:)

   logical, allocatable :: is_bond_constrained(:)
   logical, allocatable :: is_bond_dummy(:)

   logical :: l_exclude_14_dih_CTRL = .false.  ! if false the 14 are not included in the excluded list
   logical :: l_exclude_14_all_CTRL = .false.
   logical :: l_build_14_from_dih_CTRL = .false.  ! if true build 14 pairs from the conectivity due do dihedrals
   logical :: l_build_14_from_angle_CTRL = .true. ! if true build 14 pairs from the conectivity due do angles+next bond.
   logical :: l_red_14_vdw_CTRL = .true.
   logical :: l_red_14_Q_CTRL   = .true.
   logical :: l_red_14_Q_mu_CTRL= .true.
   logical :: l_red_14_mu_mu_CTRL = .false.
   real(8) :: red_14_vdw = 0.84d0
   real(8) :: red_14_Q   = 0.84d0
   real(8) :: red_14_Q_mu= 0.84d0
   real(8) :: red_14_mu_mu = 0.0d0
   
end module connectivity_ALL_data 

module Ewald_data
   integer i_type_EWALD_CTRL ! 0 - non-Ewald ; 1 - FAST 2-SLOW
   integer k_max_x,k_max_y,k_max_z,h_cut_z
   integer :: nfftx = 16 
   integer :: nffty = 16 
   integer :: nfftz = 16 
   integer :: NFFT = 16*16*16
   integer :: order_spline_xx = 4
   integer :: order_spline_yy = 4
   integer :: order_spline_zz = 4
   integer :: order_spline_zz_k0 = 6
   integer :: n_grid_zz_k0 = 50
   real(8) h_step_ewald2D 
   real(8) :: h_cut_off2D = 15.0d0 ! for slow 2D Ewald. It is in units of (2Pi/box_zz)
   real(8) :: ewald_alpha = 0.25d0 ! default value
   real(8),allocatable :: ewald_beta(:), ewald_gamma(:,:), ewald_eta(:,:)
   real(8)  q_distrib_eta  ! for distributed charges
   real(8) ewald_error
   integer N_K_VECTORS ! Number of K vectors within cut-off which are actually used.
   real(8) dfftx,dffty,dfftz,dfftx2,dffty2,dfftz2,dfftxy,dfftxz,dfftyz
end module Ewald_data

module pairs_14_data
  use sizes_data, only : N_pairs_14, N_type_pairs_14
  real(8), allocatable :: prm_14_types(:) ! is the reduction factor of each type of 14 pair
end module pairs_14_data

module basic_MC_data
 integer :: i_seed = -86680
 real(8) :: mc_translate_displ_xx = 0.10d0 ! mc attempted translational displacement in Amstroms 
 real(8) :: mc_translate_displ_yy = 0.10d0 
 real(8) :: mc_translate_displ_zz = 0.10d0
 
end module basic_MC_data

module ALL_atoms_data
     use sizes_data, only :  Natoms
     integer, allocatable :: atom_in_which_molecule(:)
     integer, allocatable :: atom_in_which_type_molecule(:)
     integer, allocatable :: i_type_atom(:)
     integer, allocatable :: i_style_atom(:)
     integer, allocatable :: contrains_per_atom(:)
     real(8), allocatable :: xxx(:), yyy(:), zzz(:)  ! atom positions
     real(8), allocatable :: base_dx(:),base_dy(:), base_dz(:)
     real(8), allocatable :: ttx(:), tty(:), ttz(:)  ! reduced coordinates of atoms
     real(8), allocatable :: xx(:), yy(:), zz(:)  ! atom positions in BOX
     real(8), allocatable :: vxx(:), vyy(:), vzz(:)  ! atom speeds
     real(8), allocatable :: axx(:), ayy(:), azz(:)  ! atom accelerations
     real(8), allocatable :: fxx(:), fyy(:), fzz(:)  ! atom forces
     real(8), allocatable :: fshort_xx(:),fshort_yy(:),fshort_zz(:)
     real(8), allocatable :: all_atoms_mass(:), all_atoms_massinv(:)
     logical, allocatable :: l_WALL(:)
     logical, allocatable :: l_WALL1(:)
     logical, allocatable :: l_WALL_CTRL(:)
     real(8), allocatable :: all_p_charges(:)
     real(8), allocatable :: all_g_charges(:)
     real(8), allocatable :: all_charges(:)
     real(8), allocatable :: all_dipoles_xx(:),all_dipoles_yy(:),all_dipoles_zz(:),all_dipoles(:)
     real(8), allocatable :: external_sfield(:)
     real(8), allocatable :: external_sfield_CONSTR(:)
     logical, allocatable :: is_dipole_polarizable(:)
     logical, allocatable :: is_charge_polarizable(:)
     logical, allocatable :: is_sfield_constrained(:)
     logical, allocatable :: is_charge_distributed(:)
     logical, allocatable :: is_dummy(:)
     logical, allocatable :: is_thetering(:)
     real(8), allocatable :: all_Q_pol(:)
     real(8), allocatable :: all_DIPOLE_pol(:)
     logical, allocatable :: l_proceed_kin_atom(:)
     integer, allocatable :: map_from_intramol_constrain_to_atom(:)
     logical, allocatable :: any_intramol_constrain_per_atom(:)
     real(8), allocatable :: atom_dof(:)
end module ALL_atoms_data

module ALL_dummies_data
use sizes_data, only : Ndummies
    
  integer, allocatable :: i_Style_dummy(:)
  integer, allocatable :: map_dummy_to_atom(:)
  integer, allocatable :: map_atom_to_dummy(:)
  real(8), allocatable :: all_dummy_params(:,:)
  integer, allocatable :: all_dummy_connect_info(:,:) 
end module ALL_dummies_data

module thetering_data
 type thetering_type
   integer :: N
   integer, allocatable :: to_atom(:)
   real(8), allocatable :: kx(:)
   real(8), allocatable :: ky(:)
   real(8), allocatable :: kz(:)
   real(8), allocatable :: x0(:)
   real(8), allocatable :: y0(:)
   real(8), allocatable :: z0(:)
 end type thetering_type
 type (thetering_type) thetering
end module thetering_data

module non_bonded_lists_data
     use max_sizes_data, only : MX_list_nonbonded, MX_list_nonbonded_short, MX_cells
     use sizes_data, only : Ncells
     integer, allocatable :: list_nonbonded(:,:), size_list_nonbonded(:)
     integer, allocatable :: list_nonbonded_short(:,:), size_list_nonbonded_short(:)
     integer, allocatable ::  listsubbox(:)
     integer, allocatable :: link_atom_to_cell(:)
     integer, allocatable :: link_cell_to_atom(:)
logical :: l_update_VERLET_LIST = .true.
     logical :: set_2_nonbonded_lists_CTRL = .false. ! if do two lists
end module non_bonded_lists_data

module LR_corrections_data
 real(8), allocatable :: EN0_LR_vdw(:,:)
 real(8), allocatable :: STRESS0_LR_vdw(:,:)
 
end module LR_corrections_data

module atom_type_data
     use max_sizes_data, only : Max_Atom_Name_Len, MX_GAUSS_charges_per_atom
     use sizes_data, only : N_TYPE_ATOMS, N_STYLE_ATOMS
     use types_module, only : two_I_type
     type dmy_type
       integer :: i
       real(8) :: r(3)
     end type dmy_type
     integer , allocatable :: atom_type_molindex(:)
     character(Max_Atom_Name_Len) , allocatable :: atom_type_name(:)
     character(Max_Atom_Name_Len) , allocatable :: atom_style_name(:)
     real(8), allocatable :: atom_type_mass(:)
     real(8), allocatable :: atom_type_charge(:), q_reduced(:), q_reduced_G(:)
     real(8), allocatable :: atom_type_dipol(:)
     real(8), allocatable :: atom_type_DIR_dipol(:,:)
     logical, allocatable :: atom_type_isDummy(:)
     type(dmy_type), allocatable :: atom_type_DummyInfo(:)
     integer, allocatable :: atom_type_in_which_mol_type(:)

     integer, allocatable :: map_atom_type_to_predef_atom_ff(:)
     integer, allocatable :: map_atom_type_to_style(:)

     integer, allocatable :: atom_type2_N_vdwPrm(:)  ! double loop
     real(8), allocatable :: atom_type2_vdwPrm(:,:) ! double loop
     integer, allocatable :: atom_type2_vdwStyle(:)

     real(8), allocatable :: atom_Style_charge(:)
     real(8), allocatable :: atom_Style_dipole(:,:)
     real(8), allocatable :: atom_Style_dipole_pol(:)
     logical, allocatable :: is_Style_dipole_pol(:)
     integer, allocatable :: atom_Style2_N_vdwPrm(:)  ! double loop
     real(8), allocatable :: atom_Style2_vdwPrm(:,:) ! double loop
     integer, allocatable :: atom_Style2_vdwStyle(:)

     integer, allocatable :: atom_Style_N_GAUSS_charges(:)
     real(8), allocatable :: atom_Style_1GAUSS_charge(:)
     real(8), allocatable :: atom_Style_1GAUSS_charge_distrib(:)

     integer, allocatable :: which_atom_pair(:,:)
     integer, allocatable :: which_atomStyle_pair(:,:)
     integer, allocatable :: which_vdw_style(:,:)
     type(two_I_type), allocatable :: pair_which_style(:)
     type(two_I_type), allocatable :: pair_which_atom(:)  ! map_inverse_innb(:)
     logical, allocatable :: l_TYPE_ATOM_WALL(:)
     logical, allocatable :: l_TYPE_ATOM_WALL_1(:)
     logical, allocatable :: l_TYPEatom_do_stat_on_type(:) ! if do type 1 and 2 statistics on type atom
     integer, allocatable :: atom_type_N_GAUSS_charges(:)
     real(8), allocatable :: atom_type_1GAUSS_charge(:)
     real(8), allocatable :: atom_type_1GAUSS_charge_distrib(:)
     logical, allocatable :: is_type_dipole_pol(:)
     logical, allocatable :: is_type_charge_pol(:)
!     real(8), allocatable :: polarizability(:)
     real(8), allocatable :: atom_type_Q_pol(:)
     real(8), allocatable :: atom_type_DIPOLE_pol(:) 
     real(8), allocatable :: atom_type_sfield_ext(:)
     real(8), allocatable :: atom_type_sfield_CONSTR(:)
     logical, allocatable :: is_type_charge_distributed(:)
     logical, allocatable :: sfc_already_defined(:)
     integer , allocatable :: statistics_AtomPair_type(:)


end module atom_type_data

module interpolate_data
  use max_sizes_data, only : MX_interpol_points
  use sizes_data, only : N_STYLE_ATOMS
  use cut_off_data
  real(8) RDR , iRDR
  real(8), allocatable :: vvdw(:,:), gvdw(:,:) ! potential and force vdw
  real(8), allocatable :: vele(:), gele(:) ! potential and force Q
  real(8), allocatable :: vele2(:), vele3(:)
  real(8), allocatable :: vele_G(:,:),gele_G(:,:) ! for gaussian distribution
  real(8), allocatable :: vele2_G(:,:),vele3_G(:,:)
  real(8), allocatable :: v_B0(:,:),v_B1(:,:),v_B2(:,:)
  real(8), allocatable :: vele_THOLE(:,:), gele_THOLE(:,:)
  real(8), allocatable :: vele_THOLE_DERIV(:,:), gele_THOLE_DERIV(:,:)


  real(8), allocatable :: gele_G_short(:,:) ! do not contain any thole in it  ; it is the same as vele_G if no thole is used
end module interpolate_data

module thole_data
 real(8) :: aa_thole = 0.2d0
 integer :: i_type_THOLE_function_CTRL = 1 ! 1 for exp(-ar^3/(pol(i)pol(j))^(1/6))
end module thole_data

module mol_type_data
    use max_sizes_data, only : Max_Mol_Name_Len
    use types_module, only : mol_tag_type
    integer N_type_molecules
    character(Max_Mol_Name_Len) , allocatable :: mol_type_name(:)
    integer, allocatable :: N_mols_of_type(:)
    integer, allocatable :: N_type_atoms_per_mol_type(:)
    integer, allocatable :: Nm_type_bonds(:)
    integer, allocatable :: Nm_type_constrains(:)
    integer, allocatable :: Nm_type_angles(:)
    integer, allocatable :: Nm_type_dihedrals(:)
    integer, allocatable :: Nm_type_deforms(:)
    integer, allocatable :: Nm_type_14(:)
    logical,allocatable :: l_RIGID_GROUP_TYPE(:)
    logical,allocatable :: l_FLEXIBLE_GROUP_TYPE(:)
    logical,allocatable :: l_TYPEmol_do_stat_on_type(:) ! if do stat12 on this type of molecule
    real(8), allocatable :: mol_type_xyz0(:,:,:) ! xyz coordinates of atoms ; it is also the standard orientation
    logical, allocatable :: is_mol_type_sfc(:)  ! if the whole molecule is sfc ! the values will be overwritten at atom_type
    real(8), allocatable :: param_mol_type_sfc(:)  ! the actual parameter (Voltage) of mol type sfc
! param_mol_type_sfc will be overwrtitten by atom_type-sfc

    integer , allocatable :: statistics_MolPair_type(:)
    type (mol_tag_type) , allocatable :: mol_tag(:)

! needs to be re-centered to mass-centra
end module mol_type_data

module vdw_type_data
    use max_sizes_data, only : MX_VDW_PRMS
    use sizes_data, only : N_TYPES_VDW
    integer N_PREDEF_VDW ! predefined vdw in input file
    real(8), allocatable :: prm_vdw_type(:,:)
    integer, allocatable :: vdw_type_atom_1_type(:,:)
    integer, allocatable :: vdw_type_atom_2_type(:,:)
    integer, allocatable :: size_vdw_type_atom_1_type(:)
    integer, allocatable :: size_vdw_type_atom_2_type(:)
    integer, allocatable :: vdw_type_style(:)
    integer, allocatable :: vdw_type_Nparams(:)
    logical, allocatable :: is_self_vdw(:) ! is the predef vdw in input filel self-vdw?
end module vdw_type_data

module ALL_mols_data
    use sizes_data, only : Nmols
    integer, allocatable :: i_type_molecule(:) ! what is the type of a molecule of a certain molecule
    integer, allocatable :: start_group(:), end_group(:)
    integer, allocatable :: N_atoms_per_mol(:)
    integer, allocatable :: N_bonds_per_mol(:)
    integer, allocatable :: N_angles_per_mol(:)
    integer, allocatable :: N_dihedrals_per_mol(:)
    integer, allocatable :: N_deforms_per_mol(:)
    integer, allocatable :: N_constrains_per_mol(:)
    logical, allocatable :: l_RIGID_GROUP(:)
!    real(8), allocatable :: mol_xxx(:),mol_yyy(:),mol_zzz(:)
    real(8), allocatable :: mol_xyz(:,:) ! the same as mol_xxx,mol_yyy,mol_zzz
    real(8), allocatable :: mol_dipole(:,:)
    real(8), allocatable :: mol_mass(:)
    real(8), allocatable :: mol_dof(:)
    real(8), allocatable :: all_mol_p_charges(:)
    real(8), allocatable :: all_mol_g_charges(:)
    real(8), allocatable :: mol_potential(:), mol_potential_Q(:)
    real(8), allocatable :: mol_pressure(:,:)
    real(8), allocatable :: mol_force(:,:)
    logical, allocatable :: l_WALL_MOL_CTRL(:)  ! if all atoms are frozen then the molecule is frozen
end module ALL_mols_data

module ALL_rigid_mols_data
  real(8), allocatable :: qn(:,:)
  real(8), allocatable :: mol_MOM(:,:)
  real(8), allocatable :: mol_ANG(:,:)
  real(8), allocatable :: xyz_body(:,:) ! with respect to body frame
 logical, allocatable:: l_non_linear_rotor(:)
 real(8) , allocatable:: mol_orient(:,:) 
 real(8) , allocatable:: Inverse_Molar_mass(:)
 real(8) , allocatable:: Inertia_MAIN(:,:),Inverse_Inertia_MAIN(:,:)
 real(8) , allocatable:: Inertia_SEC(:,:),Inverse_Inertia_SEC(:,:)
 real(8) , allocatable:: Inertia_MAIN_TYPE(:,:), Inertia_SEC_TYPE(:,:)
 real(8) , allocatable:: Inverse_Inertia_MAIN_TYPE(:,:)
 real(8) , allocatable:: mol_ANG_body(:,:)
 real(8) , allocatable:: mol_torque(:,:)
 real(8) , allocatable :: mol_torque_body(:,:)
end module ALL_rigid_mols_data

MODULE qn_and_hi_deriv_data ! for rigid dynamics integrate with GEAR4
implicit none
 real(8) , allocatable:: mol_xyz_3_deriv(:,:), mol_xyz_4_deriv(:,:)
 real(8) , allocatable:: mol_MOM_3_deriv(:,:), mol_MOM_4_deriv(:,:)
 real(8) , allocatable:: mol_ANG_3_deriv(:,:), mol_ANG_4_deriv(:,:)
 real(8) , allocatable:: qn_3_deriv(:,:)
 real(8) , allocatable:: qn_4_deriv(:,:)
end MODULE qn_and_hi_deriv_data

module qn_and_low_deriv_data
 implicit none
 real(8) , allocatable:: mol_xyz_1_deriv(:,:), mol_xyz_2_deriv(:,:) !Mass centra all are MAX_MOLS
 real(8) , allocatable:: mol_MOM_1_deriv(:,:), mol_MOM_2_deriv(:,:) !momentum
 real(8) , allocatable:: mol_ANG_1_deriv(:,:), mol_ANG_2_deriv(:,:) !angular momentum
 real(8) , allocatable:: qn_1_deriv(:,:)
 real(8) , allocatable:: qn_2_deriv(:,:)
end module qn_and_low_deriv_data


module shake_data
    integer :: MX_SHAKE_ITERATIONS = 10000
    real(8) :: SHAKE_TOLERANCE = 1.0d-5
    real(8), allocatable :: dx_in(:),dy_in(:),dz_in(:),dr_sq_in(:) ! a buffer in which I store the bonds
    real(8) shake_iterations
end module shake_data

module stresses_data
  
 real(8) stress(10)
 real(8) pressure(10)
!details on stress
 real(8) stress_kin(10), stress_shake(10)
 real(8) stress_bond(10), stress_angle(10), stress_dih(10), stress_deform(10), stress_dummy(10)
 real(8) stress_vdw(10)
 real(8) stress_Qcmplx_as_in_3D(10)
 real(8) stress_Qreal(10), stress_Qcmplx(10),stress_Qcmplx_k_eq_0(10)
 real(8) stress_excluded(10)
 real(8) stress_thetering(10)
 end module stresses_data

 module RA1_stresses_data ! rolling averages details on stress
   real(8) RA1_stress_kin(10), RA1_stress_shake(10)
   real(8) RA1_stress_bond(10), RA1_stress_angle(10),RA1_stress_dih(10),RA1_stress_deform(10),RA1_stress_dummy(10)
   real(8) RA1_stress_vdw(10)
   real(8) RA1_stress_Qcmplx_as_in_3D(10)
   real(8) RA1_stress_Qreal(10), RA1_stress_Qcmplx(10), RA1_stress_Qcmplx_k_eq_0(10)
   real(8) RA1_stress_excluded(10)
   real(8) RA1_stress_thetering(10)
   real(8) :: RA1_stress_counts = 0.0d0
 end module RA1_stresses_data

 module general_statistics_data
   real(8) :: RESCALE_DENSITY = 1.0d0 ! if use vacuum make it smaller than  1
 end module general_statistics_data

 module rolling_averages_data

  use types_module, only : statistics_5_type
   type(statistics_5_type) RA_energy(100) ! 1: vdw ; 2: Q ; 3 : kin ;4 pot; 5 tot
   type(statistics_5_type) RA_stress(10) !: xx yy zz ; p ; xy xz yz | yx zx zy
   type(statistics_5_type) RA_pressure(10)
   type(statistics_5_type) RA_Temperature
   type(statistics_5_type) RA_Temperature_trans, RA_Temperature_rot
   type(statistics_5_type) RA_MOM_0(3)  ! xx yy zz (conserve linear momentum to 0)
   type(statistics_5_type) RA_cg_iter   ! statistics on the number of iterations required if iterative methods are needed
   type(statistics_5_type) RA_shake_iter
   type(statistics_5_type) RA_msd2(4)      ! mean square displacemets xx, yy, zz r
   type(statistics_5_type) RA_diffusion(3) !x^2+y^2, z^2, r^2  ! based on atom positions.
   type(statistics_5_type) RA_sum_charge(3)  ! charge (if variable) : is total charge zero  1: point ; 2: gauss; 3: point+gauss
   type(statistics_5_type) RA_dip(4)
   type(statistics_5_type) RA_sfc(4)
   
 end module rolling_averages_data

 module energies_data
      real(8) energy(100)
      real(8) en_Qreal, En_Q_cmplx, En_Q123, En_14, En_Q, En_vdw, en_tot, en_pot
      real(8) EN_Qreal_sfc_sfc_intra
      real(8) En_Q_cmplx_k0_CORR, En_Q_k0_cmplx
      real(8) En_Q_intra_corr
      real(8) en_bond, en_angle, en_dih, en_deform
      real(8) en_intra_mol
      real(8) ew_self , En_Q_Gausian_self
      real(8) en_kin ! kinetic energy
      real(8) en_sfield ! the component due to s-field
      real(8) K_energy_translation(3),K_energy_rotation(3) ! for rotors
      real(8) En_kin_rotation , En_kin_translation
      real(8) :: en_water_surface_extra = 0.0d0
      real(8) Hamilton
      real(8) en_induced_dip
      real(8) en_thetering
 end module energies_data 

 module d_energies_data
! for 1 particle MC move
      real(8) d_energy(100)
      real(8) d_en_Qreal, d_En_Q_cmplx, d_En_Q123, d_En_14, d_En_Q, d_En_vdw, d_en_tot, d_en_pot
      real(8) d_EN_Qreal_sfc_sfc_intra
      real(8) d_En_Q_cmplx_k0_CORR, d_En_Q_k0_cmplx
      real(8) d_En_Q_intra_corr
      real(8) d_en_bond, d_en_angle, d_en_dih, d_en_deform
      real(8) d_en_intra_mol
      real(8) d_ew_self , d_En_Q_Gausian_self
      real(8) d_en_kin ! kinetic energy
      real(8) d_en_sfield ! the component due to s-field
      real(8) d_K_energy_translation(3),d_K_energy_rotation(3) ! for rotors
      real(8) d_En_kin_rotation , d_En_kin_translation
      real(8) :: d_en_water_surface_extra = 0.0d0
      real(8) d_Hamilton
      real(8) d_en_induced_dip
      real(8) d_en_thetering
 end module d_energies_data


 module saved_energies
      real(8) saved_energy(100)
      real(8) saved_en_Qreal, saved_En_Q_cmplx, saved_En_Q123, saved_En_14, saved_En_Q,&
              saved_En_vdw, saved_en_tot, saved_en_pot
      real(8) saved_EN_Qreal_sfc_sfc_intra
      real(8) saved_En_Q_cmplx_k0_CORR, saved_En_Q_k0_cmplx
      real(8) saved_En_Q_intra_corr
      real(8) saved_en_bond, saved_en_angle, saved_en_dih, saved_en_deform
      real(8) saved_en_intra_mol
      real(8) saved_ew_self , saved_En_Q_Gausian_self
      real(8) saved_en_kin ! kinetic energy
      real(8) saved_en_sfield ! the component due to s-field
      real(8) saved_K_energy_translation(3),saved_K_energy_rotation(3) ! for rotors
      real(8) saved_En_kin_rotation , saved_En_kin_translation
      real(8) saved_en_water_surface_extra
      real(8) saved_Hamilton
      real(8) saved_en_induced_dip
      real(8) saved_en_thetering

  public :: save_energies
  public :: restore_energies
  public :: zero_energies
  public :: d_zero_energies
  public :: update_other_energies
  public :: print_instant_energies

 contains
 subroutine save_energies
 use energies_data
 implicit none
      saved_energy                 =  energy
      saved_en_Qreal               =  en_Qreal  
      saved_En_Q_cmplx             =  En_Q_cmplx 
      saved_En_Q123                =  En_Q123 
      saved_En_14                  =  En_14
      saved_En_Q                   =  En_Q
      saved_En_vdw                 =  En_vdw
      saved_en_tot                 =  en_tot
      saved_en_pot                 =  en_pot
      saved_EN_Qreal_sfc_sfc_intra =  EN_Qreal_sfc_sfc_intra
      saved_En_Q_cmplx_k0_CORR     =  En_Q_cmplx_k0_CORR
      saved_En_Q_k0_cmplx          =  En_Q_k0_cmplx
      saved_En_Q_intra_corr        =  En_Q_intra_corr
      saved_en_bond                =  en_bond
      saved_en_angle               =  en_angle
      saved_en_dih                 =  en_dih
      saved_en_deform              =  en_deform
      saved_en_intra_mol           =  en_intra_mol
      saved_ew_self                =  ew_self
      saved_En_Q_Gausian_self      =  En_Q_Gausian_self
      saved_en_sfield              =  en_sfield
      saved_en_kin                 =  en_kin
      saved_K_energy_translation   =  K_energy_translation
      saved_K_energy_rotation      =  K_energy_rotation
      saved_En_kin_rotation        =  En_kin_rotation
      saved_En_kin_translation     =  En_kin_translation
      saved_en_water_surface_extra =  en_water_surface_extra
      saved_Hamilton               =  Hamilton
      saved_en_induced_dip         =  en_induced_dip
      saved_en_thetering           =  en_thetering
 end subroutine save_energies
 subroutine restore_energies
 use energies_data
 implicit none
      energy                      = saved_energy
      en_Qreal                    = saved_en_Qreal 
      En_Q_cmplx                  = saved_En_Q_cmplx 
      En_Q123                     = saved_En_Q123 
      En_14                       = saved_En_14 
      En_Q                        = saved_En_Q 
      En_vdw                      = saved_En_vdw 
      en_tot                      = saved_en_tot 
      en_pot                      = saved_en_pot 
      EN_Qreal_sfc_sfc_intra      = saved_EN_Qreal_sfc_sfc_intra 
      En_Q_cmplx_k0_CORR          = saved_En_Q_cmplx_k0_CORR 
      En_Q_k0_cmplx               = saved_En_Q_k0_cmplx 
      En_Q_intra_corr             = saved_En_Q_intra_corr 
      en_bond                     = saved_en_bond 
      en_angle                    = saved_en_angle 
      en_dih                      = saved_en_dih 
      en_deform                   = saved_en_deform 
      en_intra_mol                = saved_en_intra_mol 
      ew_self                     = saved_ew_self 
      En_Q_Gausian_self           = saved_En_Q_Gausian_self 
      en_sfield                   = saved_en_sfield
      en_kin                      = saved_en_kin 
      K_energy_translation        = saved_K_energy_translation 
      K_energy_rotation           = saved_K_energy_rotation 
      En_kin_rotation             = saved_En_kin_rotation 
      En_kin_translation          = saved_En_kin_translation 
      en_water_surface_extra      = saved_en_water_surface_extra 
      Hamilton                    = saved_Hamilton 
      en_induced_dip              = saved_en_induced_dip
      en_thetering                = saved_en_thetering
 end subroutine restore_energies
 subroutine zero_energies
 use energies_data
 implicit none
      energy                      = 0.0d0
      en_Qreal                    = 0.0d0
      En_Q_cmplx                  = 0.0d0
      En_Q123                     = 0.0d0
      En_14                       = 0.0d0
      En_Q                        = 0.0d0
      En_vdw                      = 0.0d0
      en_tot                      = 0.0d0
      en_pot                      = 0.0d0
      EN_Qreal_sfc_sfc_intra      = 0.0d0
      En_Q_cmplx_k0_CORR          = 0.0d0
      En_Q_k0_cmplx               = 0.0d0
      En_Q_intra_corr             = 0.0d0
      en_bond                     = 0.0d0
      en_angle                    = 0.0d0
      en_dih                      = 0.0d0
      en_deform                   = 0.0d0
      en_intra_mol                = 0.0d0
      ew_self                     = 0.0d0
      En_Q_Gausian_self           = 0.0d0
      en_sfield                   = 0.0d0
      en_kin                      = 0.0d0
      K_energy_translation        = 0.0d0
      K_energy_rotation           = 0.0d0
      En_kin_rotation             = 0.0d0
      En_kin_translation          = 0.0d0
      en_water_surface_extra      = 0.0d0
      Hamilton                    = 0.0d0
      en_induced_dip              = 0.0d0
      en_thetering                = 0.0d0
 end subroutine zero_energies

 subroutine d_zero_energies
 use d_energies_data
 implicit none
      d_energy                      = 0.0d0
      d_en_Qreal                    = 0.0d0
      d_En_Q_cmplx                  = 0.0d0
      d_En_Q123                     = 0.0d0
      d_En_14                       = 0.0d0
      d_En_Q                        = 0.0d0
      d_En_vdw                      = 0.0d0
      d_en_tot                      = 0.0d0
      d_en_pot                      = 0.0d0
      d_EN_Qreal_sfc_sfc_intra      = 0.0d0
      d_En_Q_cmplx_k0_CORR          = 0.0d0
      d_En_Q_k0_cmplx               = 0.0d0
      d_En_Q_intra_corr             = 0.0d0
      d_en_bond                     = 0.0d0
      d_en_angle                    = 0.0d0
      d_en_dih                      = 0.0d0
      d_en_deform                   = 0.0d0
      d_en_intra_mol                = 0.0d0
      d_ew_self                     = 0.0d0
      d_En_Q_Gausian_self           = 0.0d0
      d_en_sfield                   = 0.0d0
      d_en_kin                      = 0.0d0
      d_K_energy_translation        = 0.0d0
      d_K_energy_rotation           = 0.0d0
      d_En_kin_rotation             = 0.0d0
      d_En_kin_translation          = 0.0d0
      d_en_water_surface_extra      = 0.0d0
      d_Hamilton                    = 0.0d0
      d_en_induced_dip              = 0.0d0
      d_en_thetering                = 0.0d0
 end subroutine d_zero_energies


 subroutine update_other_energies
 use energies_data
 implicit none
   en_intra_mol = en_bond + en_angle + en_dih+en_deform
   en_pot=en_Q+en_vdw+en_sfield+en_water_surface_extra+en_intra_mol+en_thetering
   en_tot = en_pot + en_kin
   energy(1) = en_vdw; energy(2) = en_Q; energy(3) = en_pot; energy(4) = en_kin; energy(5) = en_tot
   energy(6) = en_bond; energy(7) = en_angle; energy(8) = en_dih; energy(9) = en_deform; energy(10) = en_intra_mol
   energy(11) = En_Qreal
   energy(12) = En_Q_cmplx
   energy(13) = En_Q_k0_cmplx
   energy(14) = En_Q_intra_corr
   energy(15) = En_Q_Gausian_self+ew_self
   energy(16) = en_induced_dip
   energy(17) = en_thetering
 end subroutine update_other_energies


 subroutine print_instant_energies
 use energies_data
 use integrate_data, only : l_do_QN_CTRL
  implicit none
  real(8) :: f = 1.0d-5
  print*,'--------------------------------'
  print*,' Start energies in kJ/mol-------'
!      print*,'energy                 =',energy(1:20)*f  
      print*,'en_Qreal               =',en_Qreal*f
      print*,'En_Q_cmplx             =',En_Q_cmplx*f
      print*,'En_Q123                =',En_Q123*f
      print*,'En_14                  =',En_14*f
      print*,'En_Q                   =',En_Q*f
      print*,'En_vdw                 =',En_vdw*f
      print*,'en_tot                 =',en_tot*f
      print*,'en_pot                 =',en_pot*f   
      print*,'EN_Qreal_sfc_sfc_intra =',EN_Qreal_sfc_sfc_intra*f
      print*,'En_Q_cmplx_k0_CORR     =',En_Q_cmplx_k0_CORR*f
      print*,'En_Q_k0_cmplx          =',En_Q_k0_cmplx*f
      print*,'En_Q_intra_corr        =',En_Q_intra_corr*f
      print*,'en_bond                =',en_bond*f
      print*,'en_angle               =',en_angle*f
      print*,'en_dih                 =',en_dih*f
      print*,'en_deform              =',en_deform*f
      print*,'en_intra_mol           =',en_intra_mol*f
      print*,'ew_self                =',ew_self*f
      print*,'En_Q_Gausian_self      =',En_Q_Gausian_self*f
      print*,'en_sfield              =',en_sfield*f
      print*,'en_kin                 =',en_kin*f
      print*,'en_thetering           =',en_thetering*f
      if (l_do_QN_CTRL) then
      print*,'K_energy_translation   =',K_energy_translation*f
      print*,'K_energy_rotation      =',K_energy_rotation*f
      print*,'En_kin_rotation        =',En_kin_rotation*f
      print*,'En_kin_translation     =',En_kin_translation*f
      endif
      print*,'en_water_surface_extra =',en_water_surface_extra*f
      print*,'Hamilton               =',Hamilton*f
      print*,'En induced dip         =',en_induced_dip*f
   print*,'  \End energies in kJ/mol-------'
   print*,'--------------------------------'
 end subroutine print_instant_energies
 end module saved_energies


 module preconditioner_data
  integer, allocatable :: size_preconditioner(:), list_preconditioner(:,:)
  real(8), allocatable :: preconditioner_rr(:,:), preconditioner_xx(:,:),preconditioner_yy(:,:),&
                          preconditioner_zz(:,:)
  integer, parameter :: MX_preconditioner_size = 350
 end module preconditioner_data

 module cg_buffer
! for conjugate gradient buffer
   use types_module , only : statistics_3_type
   implicit none

   logical :: l_DO_CG_CTRL = .true.
   logical :: l_DO_CG_CTRL_Q = .true.
   logical :: l_DO_CG_CTRL_DIP = .true.
   logical :: l_do_FFT_in_inner_CG = .true.
   logical :: use_cg_preconditioner_CTRL = .true.!.false.

   type cg_skip_type 
     integer :: Q
     integer :: dip
   end type cg_skip_type
   type(cg_skip_type) cg_skip_MAIN

   integer :: cg_predict_restart_CTRL  = 1  ! How predict to next iteration? 
! cg_predict_restart_CTRL = 0; take the result from the last iter
! cg_predict_restart_CTRL = 1; polynomial order 5
! cg_predict_restart_CTRL = 2; least square prediction of the last order_lsq_cg_predictor (+1) iterations
   real(8) :: CG_TOLERANCE = 1.0d-20
   integer, parameter :: REDUCED_INTEGER = 4
   real(8), allocatable ::  q(:),BB0(:),GG(:,:), BB(:),BB0_Fourier(:)
   real(8), allocatable :: GG_0(:,:),GG_1(:,:),GG_2(:,:)
   real(8), allocatable :: GG_0_14(:,:),GG_1_14(:,:),GG_2_14(:,:)
   real(8), allocatable :: GG_0_excluded(:,:),GG_1_excluded(:,:),GG_2_excluded(:,:)
   real(8), allocatable :: GG_0_THOLE(:,:),GG_1_THOLE(:,:)
   real(8), allocatable :: sns(:,:),css(:,:),pre(:),pre1(:),pre2(:)
   real(8), allocatable :: GG_Fourier_k0(:,:), GG_Fourier(:,:)
   integer(REDUCED_INTEGER),allocatable :: list1(:,:),size_list1(:), list2(:,:), size_list2(:)
   integer(REDUCED_INTEGER),allocatable :: list1_14(:,:),size_list1_14(:), list2_14(:,:), size_list2_14(:)
   integer(REDUCED_INTEGER),allocatable :: list1_ex(:,:),size_list1_ex(:), list2_ex(:,:), size_list2_ex(:)
   integer(REDUCED_INTEGER),allocatable :: list1_sfc_sfc(:,:),size_list1_sfc_sfc(:), &
                                           list2_sfc_sfc(:,:),size_list2_sfc_sfc(:)
   integer, allocatable :: NNX(:),NNY(:),NNZ(:)
   real(8), allocatable :: Ih_on_grid(:,:),Ih_on_grid_dx(:,:),Ih_on_grid_dy(:,:),Ih_on_grid_dz(:,:)
   real(8), allocatable :: Ih_on_grid_FREE(:,:)
   real(8), allocatable :: var_history(:,:)
!   type(statistics_3_type) cg_iter_counts
   real(8) cg_iterations
   real(8), allocatable :: MAT_lsq_cp_predictor(:,:) ! work matrix to predict next step via least square displacements
   integer :: order_lsq_cg_predictor = 10  ! the order of least square displacements
   
   logical :: safe_mode_get_AX = .true. ! a bit slower but HECK it is safer.

   real(8) , allocatable :: mask_qi(:)
   real(8) , allocatable :: mask_di_xx(:)
   real(8) , allocatable :: mask_di_yy(:)
   real(8) , allocatable :: mask_di_zz(:)

   type cg_skip_Fourier_type
     logical :: lskip
     integer :: how_often
   end type cg_skip_Fourier_type
   
   type(cg_skip_Fourier_type) cg_skip_fourier

   integer :: aspc_update_4full_iters = 1    !  when to do the full update
   integer :: aspc_coars_Niters = 200        !  how many iterations to perform on coarse update
   real(8) :: aspc_omega = 0.03d0

   logical :: l_try_picard_CTRL=.false.
   real(8) :: picard_dumping=1.0d0  ! p_old*(1-piccard_dumpling) + p_new*piccard_dumpling
 end module cg_buffer

 module sys_preparation_data
   integer N_actions_sys_prep
   type sys_prep_type
      logical :: any_prep ! If do any prep ; false by default
      integer :: Nel ! NUmber of atoms per electrode
      integer :: type_prep ! 0 = adjust boxes ; 1 = modify electrodes
      integer :: where_in_file
      real(8) :: zsfc_by
      real(8) :: box_to(3)
   end type sys_prep_type
   type(sys_prep_type) sys_prep
  real(8), allocatable :: prep_zzz0(:)
  real(8), allocatable :: prep_thetering_zzz0(:)
  real(8) prep_cel0(9)
  public :: sys_prep_default
  public :: sys_prep_act
  public :: sys_prep_alloc
  real(8), allocatable :: move_Zmol_type_sys_prep(:)
  contains
  subroutine sys_prep_default
    implicit none
    sys_prep%any_prep=.false.
    sys_prep%Nel=0
    sys_prep%type_prep=-999
    sys_prep%zsfc_by=0.0d0
    sys_prep%box_to(:)=0.0d0
    sys_prep%where_in_file=-999
  end subroutine sys_prep_default
  subroutine sys_prep_act(rate)
  use sim_cel_data, only : sim_cel
  use ALL_atoms_data, only : zzz, is_sfield_constrained,atom_in_which_molecule,is_thetering
  use ALL_mols_data, only : i_type_molecule
  use sizes_data, only : Natoms
  use field_constrain_data, only : N_atoms_field_constrained
  use thetering_data, only : thetering
  implicit none
  real(8), intent(IN) :: rate
  integer i,j,k,i1,imol
  integer, save :: Nel
  logical, save :: first_time = .true.
       if (sys_prep%type_prep==0) then
           sim_cel(1) = prep_cel0(1) - rate*(prep_cel0(1)-sys_prep%box_to(1))
           sim_cel(5) = prep_cel0(5) - rate*(prep_cel0(5)-sys_prep%box_to(2))
           sim_cel(9) = prep_cel0(9) - rate*(prep_cel0(9)-sys_prep%box_to(3))
       elseif (sys_prep%type_prep==1) then
           i1=0;
           if (first_time) then
            Nel = N_atoms_field_constrained/2
            first_time=.false. 
           endif
           do i = 1, Natoms
              if(is_sfield_constrained(i))then
               i1 = i1 + 1
               if (i1 < Nel+1) then
                 zzz(i) = prep_zzz0(i) + rate*sys_prep%zsfc_by
                 if (is_thetering(i)) then
                   thetering%z0(i) = prep_thetering_zzz0(i) + rate*sys_prep%zsfc_by
                 endif
               else
                 zzz(i) = prep_zzz0(i) - rate*sys_prep%zsfc_by
                 if (is_thetering(i)) then
                   thetering%z0(i) = prep_thetering_zzz0(i) - rate*sys_prep%zsfc_by
                 endif
               endif
              endif !is_field_constrained
           enddo
       elseif (sys_prep%type_prep==2) then
! If moelculea RE NOT wall T NOT ALWAYS WORK
         do i = 1, Natoms
            imol=atom_in_which_molecule(i)
            j=i_type_molecule(imol)
            if (move_Zmol_type_sys_prep(j)*rate /= 0.0d0) then
            zzz(i) = prep_zzz0(i) + rate * move_Zmol_type_sys_prep(j)
            if (is_thetering(i)) then
             thetering%z0(i) = prep_thetering_zzz0(i) + rate * move_Zmol_type_sys_prep(j)
            endif
            endif
         enddo
       else!do nothing
       endif
  end subroutine sys_prep_act
  subroutine sys_prep_alloc_and_init
  use sizes_data, only : Natoms
  use sim_cel_data, only : sim_cel
  use ALL_atoms_data, only : zzz, is_sfield_constrained
  use thetering_data,only : thetering
  implicit none
   if (sys_prep%type_prep==1.or.sys_prep%type_prep==2) then 
      allocate(prep_zzz0(Natoms))
      prep_zzz0 = zzz
      if (thetering%N>0) then
        allocate(prep_thetering_zzz0(thetering%N))
        prep_thetering_zzz0=thetering%z0
      endif
   endif
   prep_cel0 = sim_cel
  end subroutine sys_prep_alloc_and_init
 end module sys_preparation_data
 
 module profiles_data
    use types_module, only : atom_profile_type, bond_profile_type, zp2_profile_type,zp1_profile_type
    logical l_need_2nd_profile, l_need_1st_profile, l_1st_profile_CTRL,l_2nd_profile_CTRL 
    logical l_ANY_profile_CTRL
    logical l_need_ANY_profile
    integer N_COLLECT_LENGHT
    integer :: N_BINS_ZZ=80
    integer :: N_BINS_ZZs = 40
    integer N_BINS_XX,N_BINS_YY
    real(8) BIN_dZ
    real(8), allocatable :: z_scale(:)
    type(zp1_profile_type), allocatable :: zp1_atom(:),zp1_atom_x(:),zp1_atom_y(:)
    type(zp2_profile_type), allocatable :: zp2_atom(:)
    type(zp1_profile_type), allocatable :: zp1_mol(:)
    type(zp2_profile_type), allocatable :: zp2_mol(:)

    type(atom_profile_type), allocatable :: atom_profile(:)
    type(bond_profile_type), allocatable :: bond_profile(:), angle_profile(:),dihedral_profile(:)
    type(atom_profile_type), allocatable :: xyz_atom_profile(:)
    type(bond_profile_type), allocatable :: xyz_bond_profile(:),& 
                                           xyz_angle_profile(:),xyz_dihedral_profile(:)

    real(8), allocatable ::&
 QQ_PP_a_pot(:),&
 QQ_PG_a_pot(:),&
 QQ_GP_a_pot(:), & 
 QQ_GG_a_pot(:),&

 QD_PP_a_pot(:),&
 DQ_PG_a_pot(:),&
 QD_GP_a_pot(:),&
 DQ_PP_a_pot(:),&
 DD_PP_a_pot(:)

 real(8), allocatable :: P_a_fi(:),G_a_fi(:),D_a_fi(:)
 real(8), allocatable :: P_a_EE_xx(:),G_a_EE_xx(:),D_a_EE_xx(:)
 real(8), allocatable :: P_a_EE_yy(:),G_a_EE_yy(:),D_a_EE_yy(:)
 real(8), allocatable :: P_a_EE_zz(:),G_a_EE_zz(:),D_a_EE_zz(:)

 real(8), allocatable :: counter_MOLS_global(:,:), counter_ATOMS_global(:,:)
 real(8), allocatable :: counter_ATOMS_global_x(:,:), counter_ATOMS_global_y(:,:)

 real(8), allocatable :: RA_fi(:) ! rolling average of atomic field 
 real(8) :: RA_fi_counts=0.0d0

 end module profiles_data

 module rsmd_data
   use types_module, only : rsmd_var_type
   type (rsmd_var_type) rsmd
   real(8), allocatable :: zp_t_z_rmsd_trans_x0(:,:,:,:)
   real(8), allocatable :: RA_zp_t_z_rmsd_trans_x0(:,:,:,:)
   real(8), allocatable :: Diff_trans_X0(:,:,:)
   real(8), allocatable :: rmsd_qn_med(:,:), rmsd_qn_med_2(:)
   real(8), allocatable :: rmsd_xyz_med(:,:), rmsd_xyz_med_2(:)
   real(8), allocatable :: zp_rmsd_xyz_med_2(:,:), zp_rmsd_xyz_med(:,:,:)
   real(8), allocatable :: zp_translate_cryterion(:,:)
 end module rsmd_data

 module rdfs_data
   use types_module, only : rdfs_var_type
   type(rdfs_var_type) rdfs
   logical :: l_details_rdf_CTRL = .false.
   real(8),allocatable :: gr_counters(:,:,:), ALL_gr_counters(:,:),&
                          gr_counters_perp(:,:,:),ALL_gr_counters_perp(:,:),&
                          gr_counters_par(:,:,:),ALL_gr_counters_par(:,:),&
                          RA_gr_counters(:,:,:),RA_gr_counters_perp(:,:,:),RA_gr_counters_par(:,:,:)

   real(8) BIN_rdf,BIN_rdf_inv
   integer, allocatable :: which_pair_rdf(:,:)
   logical , allocatable :: l_rdf_pair_eval(:,:)
   integer :: N_BIN_rdf = 40 ! How I bin the distance between atoms
   integer, allocatable :: counter_to_pair_rdfs(:) !   
 end module rdfs_data

 module zeroes_types_module
 use types_module, only : statistics_3_type,statistics_5_type
 interface zero_5_type
   module procedure zero_5_type_scalar
   module procedure zero_5_type_vector
   module procedure zero_5_type_matrix
 end interface zero_5_type
 interface zero_3_type
   module procedure zero_3_type_scalar
   module procedure zero_3_type_vector
   module procedure zero_3_type_matrix
 end interface zero_3_type
 
 contains 
  subroutine zero_5_type_scalar(din)
    implicit none
    type(statistics_5_type) din
    din%val = 0.0d0 
    din%counts = 0.0d0
    din%val_sq = 0.0d0
    din%MIN = 0.0d0
    din%max = 0.0d0
  end subroutine zero_5_type_scalar
  subroutine zero_5_type_vector(din)
    implicit none
    type(statistics_5_type) din(:)
      din(:)%val = 0.0d0
      din(:)%counts = 0.0d0
      din(:)%val_sq = 0.0d0
      din(:)%MIN = 0.0d0
      din(:)%max = 0.0d0
  end subroutine zero_5_type_vector
  subroutine zero_5_type_matrix(din)
    implicit none
    type(statistics_5_type) din(:,:)
      din(:,:)%val = 0.0d0
      din(:,:)%counts = 0.0d0
      din(:,:)%val_sq = 0.0d0
      din(:,:)%MIN = 0.0d0
      din(:,:)%max = 0.0d0
  end subroutine zero_5_type_matrix

  subroutine zero_3_type_scalar(din)
    implicit none
    type(statistics_3_type) din
    din%val = 0.0d0
    din%counts = 0.0d0
    din%val_sq = 0.0d0
  end subroutine zero_3_type_scalar
   subroutine zero_3_type_vector(din)
    implicit none
    type(statistics_3_type) din(:)
      din(:)%val = 0.0d0
      din(:)%counts = 0.0d0
      din(:)%val_sq = 0.0d0
  end subroutine zero_3_type_vector
  subroutine zero_3_type_matrix(din)
    implicit none
    type(statistics_3_type) din(:,:)
      din(:,:)%val = 0.0d0
      din(:,:)%counts = 0.0d0
      din(:,:)%val_sq = 0.0d0
  end subroutine zero_3_type_matrix
 
 end module zeroes_types_module


!--------------------------------



module force_field_data
use max_sizes_data, only : Max_Atom_Name_Len,Max_vdwStyle_Name_Len, &
                    MX_VDW_PRMS, MX_BOND_PRMS, MX_ANGLE_PRMS, MX_DIH_PRMS, MX_DEFORM_PRMS,&
                    MAX_Bond_Name_Len, MAX_Angle_Name_Len, MAX_Dih_Name_Len, MAX_Deform_Name_Len,&
                    Max_BondStyle_Name_Len, Max_AngleStyle_Name_Len, Max_DihStyle_Name_Len,&
                    Max_DeformStyle_Name_Len, Max_DummyStyle_Name_Len
                    
implicit none

type dummy_ff_type
   character(Max_DummyStyle_Name_Len) ::Name
   integer :: style
   real(8) :: the_params(3)
   integer :: GeomType
end type dummy_ff_type

type vdw_ff_type
  character(Max_vdwStyle_Name_Len) :: StyleName
  integer :: style
  character(Max_Atom_Name_Len) :: atom_1_name
  character(Max_Atom_Name_Len) :: atom_2_name
  integer :: atom_1_style
  integer :: atom_2_style
  integer :: N_params
  real(8) :: the_params(MX_VDW_PRMS)
  integer :: units
end type vdw_ff_type


type atom_ff_type
 character(Max_Atom_Name_Len) :: name
 real(8) :: mass
 real(8) :: Q
 logical :: isQdistributed
 logical :: isQsfc
 real(8) :: sfc
 real(8) :: QGaussWidth
 real(8) :: Qpol
 real(8) :: Dip
 real(8) :: DipPol
 logical :: isQpol
 logical :: isDipPol
 real(8) :: DipDir(3)
 logical :: isWALL_1
 logical :: isWALL
 type (vdw_ff_type) :: self_vdw
 logical :: is_self_vdw_Def
 logical :: is_dummy
 type(dummy_ff_type) :: dummy
 logical :: more_logic(5)
end type atom_ff_type

type bond_ff_type
  character(MAX_Bond_Name_Len) :: name
  character(Max_Atom_Name_Len) :: atom_1_name
  character(Max_Atom_Name_Len) :: atom_2_name
  integer :: atom_1_style
  integer :: atom_2_style
  character(Max_BondStyle_Name_Len) :: StyleName
  integer :: Style
  integer :: N_params
  real(8) :: the_params(MX_BOND_PRMS)
  logical :: is_constrain
  real(8) :: constrained_bond
  integer :: units
end type bond_ff_type

type angle_ff_type
  character(MAX_Angle_Name_Len) :: name
  character(Max_Atom_Name_Len) :: atom_1_name
  character(Max_Atom_Name_Len) :: atom_2_name
  character(Max_Atom_Name_Len) :: atom_3_name
  integer :: atom_1_style
  integer :: atom_2_style
  integer :: atom_3_style
  character(Max_AngleStyle_Name_Len) :: StyleName
  integer :: Style
  integer :: N_params
  real(8) :: the_params(MX_ANGLE_PRMS)
  integer :: units
end type angle_ff_type

type dih_ff_type
  character(MAX_Dih_Name_Len) :: name
  character(Max_Atom_Name_Len) :: atom_1_name
  character(Max_Atom_Name_Len) :: atom_2_name
  character(Max_Atom_Name_Len) :: atom_3_name
  character(Max_Atom_Name_Len) :: atom_4_name
  integer :: atom_1_style
  integer :: atom_2_style
  integer :: atom_3_style
  integer :: atom_4_style
  character(Max_DihStyle_Name_Len) :: StyleName
  integer :: Style
  integer :: N_params
  real(8) :: the_params(MX_DIH_PRMS)
  logical :: is_improper
  integer :: units
end type dih_ff_type

type deform_ff_type
  character(MAX_Deform_Name_Len) :: name
  character(Max_Atom_Name_Len) :: atom_1_name
  character(Max_Atom_Name_Len) :: atom_2_name
  character(Max_Atom_Name_Len) :: atom_3_name
  character(Max_Atom_Name_Len) :: atom_4_name
  integer :: atom_1_style
  integer :: atom_2_style
  integer :: atom_3_style
  integer :: atom_4_style
  character(Max_DeformStyle_Name_Len) :: StyleName
  integer :: Style
  integer :: N_params
  real(8) :: the_params(MX_DEFORM_PRMS)
  integer :: units
end type deform_ff_type

type(dummy_ff_type),allocatable :: predef_ff_dummy(:)
type(atom_ff_type), allocatable :: predef_ff_atom(:), Def_ff_atom(:)
type(vdw_ff_type), allocatable :: predef_ff_vdw(:), Def_ff_vdw(:)
type(bond_ff_type), allocatable :: predef_ff_bond(:)
type(angle_ff_type), allocatable :: predef_ff_angle(:)
type(dih_ff_type), allocatable :: predef_ff_dih(:)
type(deform_ff_type), allocatable :: predef_ff_deform(:)

integer N_predef_ff_dummies
integer N_predef_ff_atoms
integer N_predef_ff_vdw
integer N_predef_ff_bonds
integer N_predef_ff_angles
integer N_predef_ff_dihs
integer N_predef_ff_deforms

interface initialize_dummy_ff
  module procedure initialize_dummy_ff_scalar
  module procedure initialize_dummy_ff_vector
end interface initialize_dummy_ff

interface initialize_atom_ff
   module procedure initialize_atom_ff_scalar
   module procedure initialize_atom_ff_vector
end interface initialize_atom_ff

interface initialize_bond_ff
   module procedure initialize_bond_ff_scalar
   module procedure initialize_bond_ff_vector
end interface initialize_bond_ff

interface initialize_angle_ff
   module procedure initialize_angle_ff_scalar
   module procedure initialize_angle_ff_vector
end interface initialize_angle_ff

interface initialize_dih_ff
   module procedure initialize_dih_ff_scalar
   module procedure initialize_dih_ff_vector
end interface initialize_dih_ff

interface initialize_deform_ff
   module procedure initialize_deform_ff_scalar
   module procedure initialize_deform_ff_vector
end interface initialize_deform_ff


interface overwrite_2ffvdw
    module procedure overwrite_2ffvdw_scalar
    module procedure overwrite_2ffvdw_vector
end interface overwrite_2ffvdw

interface overwrite_2ffatom
    module procedure overwrite_2ffatom_scalar
    module procedure overwrite_2ffatom_vector
end interface overwrite_2ffatom
contains

subroutine initialize_dummy_ff_scalar(a)
implicit none
type(dummy_ff_type) a
a%Name(:) = ' '
a%Style = 0
a%the_params=0.0d0
a%GeomType=0
end subroutine initialize_dummy_ff_scalar
subroutine initialize_dummy_ff_vector(a)
implicit none
type(dummy_ff_type) a(:)
integer i
do i = lbound(a,dim=1),ubound(a,dim=1)
 call initialize_dummy_ff_scalar(a(i))
enddo
end subroutine  initialize_dummy_ff_vector

subroutine initialize_atom_ff_scalar(a)
implicit none
type(atom_ff_type) a
 a%mass=0.0d0
 a%Q   =0.0d0
 a%isQdistributed = .false.
 a%isQsfc         = .false.
 a%sfc            = 0.0d0
 a%QGaussWidth    = 1.0d90
 a%Qpol           = 0.0d0
 a%Dip            = 0.0d0
 a%DipPol         = 0.0d0
 a%isQpol         = .false.
 a%isDipPol       = .false.
 a%DipDir(1:3)      = (/ 0.0d0,0.0d0,1.0d0 /)
 a%isWALL_1       = .false.
 a%isWALL         = .false.
 a%self_vdw%StyleName='LJ6-12'
 a%self_vdw%Style=2
 a%self_vdw%atom_1_name(:) = ' '
 a%self_vdw%atom_2_name(:) = ' '
 a%self_vdw%atom_1_style = 0
 a%self_vdw%atom_2_style = 0
 a%self_vdw%N_params=2
 a%self_vdw%the_params(:)  = 0.0d0
 a%is_self_vdw_Def = .false.
 a%self_vdw%units = 1 ! 1=kJ/mol 2=kcal/mol 3=a.u. 4=eV 
 a%is_dummy = .false.
 a%dummy%Name(:) = ' '
 a%dummy%Style = 0
 a%dummy%the_params=0.0d0
 a%dummy%GeomType=0
 a%more_logic(:) = .false.
end subroutine initialize_atom_ff_scalar
subroutine initialize_atom_ff_vector(a)
implicit none
type(atom_ff_type) a(:)
integer i
do i = lbound(a,dim=1),ubound(a,dim=1)
 call initialize_atom_ff_scalar(a(i))
enddo
end subroutine initialize_atom_ff_vector

subroutine initialize_bond_ff_scalar(a)
 type (bond_ff_type) a
  a%name(1:MAX_Bond_Name_Len) = ' ' 
  a%atom_1_name(1:Max_Atom_Name_Len) = ' '
  a%atom_2_name(1:Max_Atom_Name_Len) = ' '
  a%atom_1_style=0
  a%atom_2_style=0
  a%StyleName(1:Max_BondStyle_Name_Len) = ' ' 
  a%Style = 0 ! unassigned
  a%N_params=1
  a%the_params(1:MX_BOND_PRMS)= 0.0d0 ! unassigned
  a%is_constrain = .false. ! it is not constrained by default
  a%constrained_bond = 0.0d0
  a%units = 1 ! kJ/mol
end subroutine initialize_bond_ff_scalar
subroutine initialize_bond_ff_vector(a)
type(bond_ff_type) a(:)
integer i
do i = lbound(a,dim=1),ubound(a,dim=1)
 call initialize_bond_ff_scalar(a(i))
enddo
end subroutine initialize_bond_ff_vector

subroutine initialize_angle_ff_scalar(a)
 type (angle_ff_type) a
  a%name(1:MAX_Angle_Name_Len) = ' '
  a%atom_1_name(1:Max_Atom_Name_Len) = ' '
  a%atom_2_name(1:Max_Atom_Name_Len) = ' '
  a%atom_3_name(1:Max_Atom_Name_Len) = ' '
  a%atom_1_style=0
  a%atom_2_style=0
  a%atom_3_style=0
  a%StyleName(1:Max_AngleStyle_Name_Len) = ' '
  a%Style = 0 ! unassigned
  a%N_params=1
  a%the_params(1:MX_ANGLE_PRMS)= 0.0d0 ! unassigned
  a%units = 1 ! kJ/mol
end subroutine initialize_angle_ff_scalar
subroutine initialize_angle_ff_vector(a)
   type (angle_ff_type) a(:)
   integer i
   do i = lbound(a,dim=1),ubound(a,dim=1)
     call initialize_angle_ff_scalar(a(i))
   enddo
end subroutine initialize_angle_ff_vector

subroutine initialize_dih_ff_scalar(a)
 type (dih_ff_type) a
  a%name(1:MAX_Dih_Name_Len) = ' '
  a%atom_1_name(1:Max_Atom_Name_Len) = ' '
  a%atom_2_name(1:Max_Atom_Name_Len) = ' '
  a%atom_3_name(1:Max_Atom_Name_Len) = ' '
  a%atom_4_name(1:Max_Atom_Name_Len) = ' '
  a%atom_1_style=0
  a%atom_2_style=0
  a%atom_3_style=0
  a%atom_4_style=0
  a%StyleName(1:Max_DihStyle_Name_Len) = ' '
  a%Style = 0 ! unassigned
  a%N_params=1
  a%the_params(1:MX_DIH_PRMS)= 0.0d0 ! unassigned
  a%units = 1 ! kJ/mol
  a%is_improper=.false.
end subroutine initialize_dih_ff_scalar
subroutine initialize_dih_ff_vector(a)
   type (dih_ff_type) a(:)
   integer i
   do i = lbound(a,dim=1),ubound(a,dim=1)
     call initialize_dih_ff_scalar(a(i))
   enddo
end subroutine initialize_dih_ff_vector

subroutine initialize_deform_ff_scalar(a)
 type (deform_ff_type) a
  a%name(1:MAX_Deform_Name_Len) = ' '
  a%atom_1_name(1:Max_Atom_Name_Len) = ' '
  a%atom_2_name(1:Max_Atom_Name_Len) = ' '
  a%atom_3_name(1:Max_Atom_Name_Len) = ' '
  a%atom_4_name(1:Max_Atom_Name_Len) = ' '
  a%atom_1_style=0
  a%atom_2_style=0
  a%atom_3_style=0
  a%atom_4_style=0
  a%StyleName(1:Max_DeformStyle_Name_Len) = ' '
  a%Style = 0 ! unassigned
  a%N_params=1
  a%the_params(1:MX_DEFORM_PRMS)= 0.0d0 ! unassigned
  a%units = 1 ! kJ/mol  character(MAX_Deform_Name_Len) :: name
end subroutine initialize_deform_ff_scalar
subroutine initialize_deform_ff_vector(a)
   type (deform_ff_type) a(:)
   integer i
   do i = lbound(a,dim=1),ubound(a,dim=1)
     call initialize_deform_ff_scalar(a(i))
   enddo
end subroutine initialize_deform_ff_vector


subroutine overwrite_2ffvdw_scalar(A,B)
implicit none 
type(vdw_ff_type), intent(IN) :: B
type(vdw_ff_type), intent(INOUT) :: A

 a%StyleName=b%StyleName
 a%Style=    b%Style
 a%atom_1_name(:) = b%atom_1_name
 a%atom_2_name(:) = b%atom_2_name
 a%atom_1_style = b%atom_1_style
 a%atom_2_style = b%atom_2_style
 a%N_params=b%N_params
 a%the_params(:)  = b%the_params(:)
 a%units = b%units 
 
end subroutine overwrite_2ffvdw_scalar

subroutine overwrite_2ffvdw_vector(A,B)
implicit none
type(vdw_ff_type), intent(IN) :: B(:)
type(vdw_ff_type), intent(INOUT) :: A(:)
integer i

do i = lbound(A,dim=1),ubound(A,dim=1)
  call overwrite_2ffvdw_scalar(A(i),B(i))
enddo
end  subroutine overwrite_2ffvdw_vector


 
subroutine overwrite_2ffatom_scalar(a,b)
implicit none
type(atom_ff_type),intent(INOUT) :: a
type(atom_ff_type),intent(INOUT) :: b
 a%name = b%name
 a%mass=b%mass
 a%Q   =b%Q
 a%isQdistributed = b%isQdistributed
 a%isQsfc         = b%isQsfc
 a%sfc            = b%sfc 
 a%QGaussWidth    = b%QGaussWidth
 a%Qpol           = b%Qpol
 a%Dip            = b%Dip
 a%DipPol         = b%DipPol 
 a%isQpol         = b%isQpol 
 a%isDipPol       = b%isDipPol   
 a%DipDir(:)      = b%DipDir(:) 
 a%isWALL_1       = b%isWALL_1
 a%isWALL       = b%isWALL
 a%self_vdw%StyleName=b%self_vdw%StyleName
 a%self_vdw%Style=b%self_vdw%Style
 a%self_vdw%atom_1_name(:) = b%self_vdw%atom_1_name(:)
 a%self_vdw%atom_2_name(:) = b%self_vdw%atom_2_name(:)
 a%self_vdw%atom_1_style =  b%self_vdw%atom_1_style
 a%self_vdw%atom_2_style =  b%self_vdw%atom_2_style
 a%self_vdw%N_params= b%self_vdw%N_params
 a%self_vdw%the_params(:)  =b%self_vdw%the_params(:)
 a%is_self_vdw_Def =  b%is_self_vdw_Def
 a%self_vdw%units =  b%self_vdw%units ! 1=kJ/mol 2=kcal/mol 3=a.u. 4=eV 
 a%is_dummy        = b%is_dummy
 a%dummy%Name(:)   = b%dummy%Name(:)
 a%dummy%Style     = b%dummy%Style
 a%dummy%the_params= b%dummy%the_params
 a%dummy%GeomType  = b%dummy%GeomType
 a%more_logic(:)   = b%more_logic(:)
end subroutine overwrite_2ffatom_scalar

subroutine overwrite_2ffatom_vector(a,b)
implicit none
type(atom_ff_type),intent(INOUT) :: a(:)
type(atom_ff_type),intent(INOUT) :: b(:)
integer i
do i = lbound(A,dim=1),ubound(A,dim=1)
  call overwrite_2ffatom_scalar(A(i),B(i))
enddo
end subroutine overwrite_2ffatom_vector

end module force_field_data









 module generic_statistics_module

 interface statistics_5_eval
   module procedure update_5_scalar
   module procedure update_5_vector
   module procedure update_5_matrix
  end interface statistics_5_eval

  interface min_max_stat_1st_pass
   module procedure min_max_scalar
   module procedure min_max_vector
   module procedure min_max_matrix
  end interface min_max_stat_1st_pass

  contains
  subroutine update_5_scalar(quantity,value)
  use types_module, only : statistics_5_type
  implicit none
  type(statistics_5_type) quantity
  real(8) , intent(IN) :: value
  integer i
    quantity%val    = quantity%val + value
    quantity%val_sq = quantity%val_sq + value*value
    quantity%max = max(quantity%max,value)
    quantity%min = min(quantity%min,value)
    quantity%counts = quantity%counts + 1.0d0
  end subroutine update_5_scalar

  subroutine update_5_vector(quantity,value)
  use types_module, only : statistics_5_type
  implicit none
  type(statistics_5_type) quantity(:)
  real(8) value(:)
  integer i
    quantity(:)%val    = quantity(:)%val + value(:)
    quantity(:)%val_sq = quantity(:)%val_sq + value(:)*value(:)
    quantity(:)%max = max(quantity(:)%max,value(:))
    quantity(:)%min = min(quantity(:)%min,value(:))
    quantity(:)%counts = quantity(:)%counts + 1.0d0
  end subroutine update_5_vector

  subroutine update_5_matrix(quantity,value)
  use types_module, only : statistics_5_type
  implicit none
  type(statistics_5_type) quantity(:,:)
  real(8) value(:,:)
  integer i,j
    quantity(:,:)%val    = quantity(:,:)%val + value(:,:)
    quantity(:,:)%val_sq = quantity(:,:)%val_sq + value(:,:)*value(:,:)
    quantity(:,:)%max = max(quantity(:,:)%max,value(:,:))
    quantity(:,:)%min = min(quantity(:,:)%min,value(:,:))
    quantity(:,:)%counts = quantity(:,:)%counts + 1.0d0
  end subroutine update_5_matrix

  subroutine min_max_scalar(quantity,value)
  use types_module, only : statistics_5_type
  implicit none
  type(statistics_5_type) quantity
  real(8) value
   quantity%max = value
   quantity%min = value
  end subroutine min_max_scalar

  subroutine min_max_vector(quantity,value)
  use types_module, only : statistics_5_type
  implicit none
  type(statistics_5_type) quantity(:)
  real(8) value(:)
   quantity(:)%max = value(:)
   quantity(:)%min = value(:)
  end subroutine min_max_vector

   subroutine min_max_matrix(quantity,value)
  use types_module, only : statistics_5_type
  implicit none
  type(statistics_5_type) quantity(:,:)
  real(8) value(:,:)
   quantity(:,:)%max = value(:,:)
   quantity(:,:)%min = value(:,:)
  end subroutine min_max_matrix

 end module generic_statistics_module

 module quick_preview_stats_data
 type quick_preview_stats_data_type
  integer :: how_often
  logical :: any_request
 end type quick_preview_stats_data_type
 type(quick_preview_stats_data_type) quick_preview_stats
 contains
 subroutine quick_preview_stats_default
    quick_preview_stats%how_often = 0
    quick_preview_stats%any_request = .false.
 end subroutine quick_preview_stats_default
 end module quick_preview_stats_data


 module history_data
  type history_type
    logical :: any_request
    integer :: how_often
    integer :: cel
    integer :: x
    integer :: v
    integer :: f ! 0 is absent 1 is presentS
    integer :: en
  end type history_type
  type(history_type) history
  contains
  subroutine default_history
   history%any_request=.false.
   history%how_often=-999
   history%cel=0
   history%x=0
   history%v=0
   history%f=0
   history%en=0
  end subroutine default_history
! HISTORY FILE WILL CONTAIN:
!header
!Natoms, Nmols, N_type_atoms, N_type_mols, N_type_atoms_per_mol_type
!cel 0 (initial cell)
!info flags: history_type%cel,history_type%xyz,history_type%vxyz,history_type%fxyz
!N_MD_RUNS/history_type%how_often
!\end header
!\repeat
!integratins_step, integration_time(ps)
!if(history_type%cel) history_type%cel
!if(history_type%xyz) history_type%xyz
!if(history_type%vxyz) history_type%vxyz
!if(history_type%fxyz) history_type%fxyz
!\end repeat
 end module history_data

