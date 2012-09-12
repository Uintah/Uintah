
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


