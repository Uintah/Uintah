 module allocate_them

 public :: ALL_atoms_alloc
 public :: ALL_dummies_alloc
 public :: ALL_dummies_DEalloc
 public :: ALL_mols_alloc
 public :: vdw_type_alloc
 public :: non_bonded_lists_alloc
 public :: preconditioner_alloc
 public :: pairs_14_alloc
 public :: atom_type_alloc
 public :: atom_style_alloc
 public :: interpolate_alloc
 public :: Ewald_data_alloc
 public :: mol_type_alloc
 public :: ALL_rigid_mols_alloc
 public :: qn_and_hi_deriv_alloc
 public :: qn_and_low_deriv_alloc
 public :: connectivity_type_alloc
 public :: profiles_alloc
 public :: ALL_sizes_allocate
 public :: smpe_alloc
 public :: smpe_DEalloc
 public :: field_constrain_alloc
 public :: rdfs_alloc

  contains

 subroutine ALL_atoms_alloc
   use ALL_atoms_data
   implicit none
   integer Na
   Na = Natoms
   allocate(atom_in_which_molecule(Na))
   allocate(atom_in_which_type_molecule(Na))
   allocate(i_type_atom(Na))
   allocate(i_Style_atom(Na))
   allocate(contrains_per_atom(Na))
   allocate(ttx(Na),tty(Na),ttz(Na)) 
   allocate(xxx(Na),yyy(Na),zzz(Na))
   allocate(base_dx(Na),base_dy(Na),base_dz(Na));base_dx=0.0d0;base_dy=0.0d0;base_dz=0.0d0;
   allocate(xx(Na),yy(Na),zz(Na))
   allocate(vxx(Na),vyy(Na),vzz(Na)) ; vxx=0.0d0; vyy=0.0d0; vzz=0.0d0
   allocate(axx(Na),ayy(Na),azz(Na)) ; axx=0.0d0; ayy=0.0d0; azz=0.0d0
   allocate(fxx(Na),fyy(Na),fzz(Na)) ; fxx=0.0d0; fyy=0.0d0; fzz=0.0d0
   allocate(fshort_xx(Na),fshort_yy(Na),fshort_zz(Na)); fshort_xx=0.0d0;fshort_yy=0.0d0;fshort_zz=0.0d0
   allocate(all_atoms_mass(Na), all_atoms_massinv(Na))
   allocate(l_WALL(Na))   ;   l_WALL = .false.
   allocate(l_WALL1(Na)); l_WALL1 = .false.
   allocate(l_WALL_CTRL(Na)) ; l_WALL_CTRL = .false.
   allocate(all_p_charges(Na)) ; all_p_charges=0.0d0
   allocate(all_g_charges(Na)) ; all_g_charges=0.0d0
   allocate(all_charges(Na)) ; all_charges=0.0d0
   allocate(all_dipoles(Na)) ; all_dipoles=0.0d0
   allocate(all_dipoles_xx(Na),all_dipoles_yy(Na),all_dipoles_zz(Na))
   all_dipoles_xx=0.0d0 ; all_dipoles_yy=0.0d0 ; all_dipoles_zz=0.0d0
   allocate(is_dipole_polarizable(Na)); is_dipole_polarizable=.false.
   allocate(is_charge_polarizable(Na)); is_charge_polarizable=.false.
   allocate(external_sfield(Na)); external_sfield=0.0d0
   allocate(external_sfield_CONSTR(Na)); external_sfield_CONSTR=0.0d0
   allocate(is_charge_distributed(Na)) ; is_charge_distributed=.false. ! assume point charge
   allocate(is_sfield_constrained(Na)); is_sfield_constrained=.false.
   allocate(all_Q_pol(Na)) ; all_Q_pol=0.0d0
   allocate(all_DIPOLE_pol(Na)) ; all_DIPOLE_pol=0.0d0
   allocate(is_dummy(Na)) ; is_dummy=.false.
   allocate(l_proceed_kin_atom(Na)); l_proceed_kin_atom=.true.
   allocate(any_intramol_constrain_per_atom(Na));any_intramol_constrain_per_atom=.false.
   allocate(atom_dof(Na));atom_dof=0.00d0
   allocate(is_thetering(Na)); is_thetering=.false.
 end subroutine ALL_atoms_alloc


 subroutine ALL_dummies_alloc
  use ALL_dummies_data
  use sizes_data, only : Natoms

  allocate(i_Style_dummy(Ndummies)) ! the type of geometry
  allocate(map_dummy_to_atom(Ndummies))
  allocate(map_atom_to_dummy(Natoms))
  allocate(ALL_dummy_params(Ndummies,3))
  allocate(ALL_dummy_connect_info(Ndummies,6))
  i_Style_dummy=0
  map_dummy_to_atom=0
  ALL_dummy_params=0
  ALL_dummy_connect_info=0
  map_atom_dummy=0
end subroutine ALL_dummies_alloc

 subroutine ALL_dummies_DEalloc
   use ALL_dummies_data
   deallocate(i_Style_dummy)
   deallocate(map_dummy_to_atom)
   deallocate(map_atom_to_dummy)
   deallocate(ALL_dummy_params)
   deallocate(ALL_dummy_connect_info)
 end subroutine ALL_dummies_DEalloc

subroutine thetering_alloc
 use thetering_data, only : thetering
 integer N
 N=thetering%N
 if (N<1) RETURN
 allocate(thetering%kx(N),thetering%ky(N),thetering%kz(N))
 allocate(thetering%x0(N),thetering%y0(N),thetering%z0(N))
 allocate(thetering%to_atom(N))
end subroutine thetering_alloc


 subroutine ALL_mols_alloc
  use ALL_mols_data
  implicit none
  integer Nm
  Nm = Nmols
  allocate(i_type_molecule(Nm))
  allocate(start_group(Nm),end_group(Nm))
  allocate(N_atoms_per_mol(Nm))
  allocate(N_bonds_per_mol(Nm))
  allocate(N_constrains_per_mol(Nm))
  allocate(N_angles_per_mol(Nm))
  allocate(N_dihedrals_per_mol(Nm))
  allocate(N_deforms_per_mol(Nm))
  allocate(l_RIGID_GROUP(Nm)) ; l_RIGID_GROUP = .false.
!  allocate( mol_xxx(Nm),mol_yyy(Nm),mol_zzz(Nm))
  allocate ( mol_xyz(Nm,3) )
  allocate ( mol_dipole(Nm,3) ) ; mol_dipole = 0.0d0
  allocate(mol_mass(Nm))
  allocate(all_mol_p_charges(Nm),all_mol_G_charges(Nm)) ; 
  all_mol_p_charges=0.0d0; all_mol_G_charges=0.0d0
  allocate( mol_potential(Nm),mol_potential_Q(Nm))
  allocate( mol_pressure(Nm,10)) ! 1 2 3 xx yy zz 4 (xx+yy+zz)/3 5-10 xy xz yz yx zx zy
  allocate(mol_force(Nm,3))
  allocate(l_WALL_MOL_CTRL(Nm)); l_WALL_MOL_CTRL=.false. ! not fixed by default
  allocate(mol_dof(Nm)) ; mol_dof=0.0d0
 end subroutine ALL_mols_alloc

subroutine vdw_type_alloc
  use vdw_type_data
  use sizes_data, only : N_TYPE_ATOMS
  implicit none
!   integer N_TYPES_VDW
!   allocate(prm_vdw_type(MX_VDW_PRMS,N_PREDEF_VDW))
!   allocate(size_vdw_type_atom_1_type(N_PREDEF_VDW))
!   allocate(size_vdw_type_atom_2_type(N_PREDEF_VDW))
!   allocate(vdw_type_atom_1_type(N_PREDEF_VDW,N_TYPE_ATOMS))
!   allocate(vdw_type_atom_2_type(N_PREDEF_VDW,N_TYPE_ATOMS))
!   allocate(vdw_type_style(N_PREDEF_VDW))
!   allocate(vdw_type_Nparams(N_PREDEF_VDW)) 
!   if(.not.allocated(is_self_vdw)) allocate(is_self_vdw(N_PREDEF_VDW))
end subroutine vdw_type_alloc

 subroutine non_bonded_lists_alloc
     use non_bonded_lists_data
     !use max_sizes_data, only : maxdim3
     use ALL_atoms_data, only : Natoms
     implicit none
     integer N
     N=Natoms
     allocate(list_nonbonded(N,MX_list_nonbonded),size_list_nonbonded(N))
   if (set_2_nonbonded_lists_CTRL) &
     allocate(list_nonbonded_short(N,MX_list_nonbonded_short),size_list_nonbonded_short(N))
!     allocate(listsubbox(maxdim3))
!     allocate(link_cell_to_atom(MX_cells))
     allocate(link_atom_to_cell(N))

 end subroutine non_bonded_lists_alloc

 subroutine preconditioner_alloc
 use sizes_data, only : Natoms
 use preconditioner_data
 allocate(size_preconditioner(Natoms)); size_preconditioner=0
 allocate(preconditioner_rr(Natoms, MX_preconditioner_size),&
          preconditioner_xx(Natoms, MX_preconditioner_size), &
          preconditioner_yy(Natoms, MX_preconditioner_size), &
          preconditioner_zz(Natoms, MX_preconditioner_size)) 
    preconditioner_rr=0.0d0; preconditioner_xx=0.0d0; preconditioner_yy= 0.0d0; preconditioner_zz=0.0d0
 end subroutine preconditioner_alloc

 subroutine pairs_14_alloc
   use pairs_14_data
   allocate (prm_14_types(N_type_pairs_14))
 end subroutine pairs_14_alloc

 subroutine  atom_type_alloc
   use atom_type_data
   use max_sizes_data, only : MX_VDW_PRMS
   use field_constrain_data, only : is_type_atom_field_constrained
   implicit none
   integer Na,N2,Ns2,i
   Na = N_type_atoms
   N2 = Na*(Na+1)/2
!   Ns2 = N_STYLE_ATOMS*(N_STYLE_ATOMS+1)/2
   allocate(atom_type_name(Na))
   allocate(atom_type_mass(Na)) 
   allocate(atom_type_charge(Na),q_reduced(Na),q_reduced_G(Na))
   allocate(atom_type_dipol(Na)) ; atom_type_dipol=0.0d0
   allocate(atom_type_DIR_dipol(Na,3)) ; atom_type_DIR_dipol=0.0d0 ; atom_type_DIR_dipol(:,3) = 1.0d0 
   allocate(atom_type_isDummy(Na)) ; atom_type_isDummy=.false.
   allocate(atom_type_DummyInfo(Na)) ; atom_type_DummyInfo(:)%i = 0; 
   do i = 1, 3 ; atom_type_DummyInfo(:)%r(i) = 0.0d0  ; enddo
   allocate(map_atom_type_to_predef_atom_ff(Na))
   allocate(map_atom_type_to_style(Na))
   allocate(atom_type_molindex(Na))
   allocate(atom_type_in_which_mol_type(Na))


   allocate(atom_type2_vdwStyle(N2)) ; atom_type2_vdwStyle = 0
   allocate(which_atom_pair(Na,Na))
   allocate(pair_which_atom(N2))
   allocate(l_TYPE_ATOM_WALL(Na)) ; l_TYPE_ATOM_WALL = .false.
   allocate(l_TYPE_ATOM_WALL_1(Na)) ; l_TYPE_ATOM_WALL_1 = .false.
   allocate(l_TYPEatom_do_stat_on_type(Na)) ;  l_TYPEatom_do_stat_on_type=.false.
!   allocate(atom_type_N_GAUSS_charges(Na)) ; atom_type_N_GAUSS_charges=0
!   allocate(atom_type_1GAUSS_charge(Na)) ; atom_type_1GAUSS_charge = 0.0d0
!   allocate(atom_type_1GAUSS_charge_distrib(Na)) ; atom_type_1GAUSS_charge_distrib = 1.0d93

   allocate(is_type_dipole_pol(N_type_atoms))  ;   is_type_dipole_pol = .false.
   allocate(is_type_charge_pol(N_type_atoms)) ;  is_type_charge_pol=.false.
!   allocate(polarizability(N_type_atoms)) ; polarizability =0.0d0
   allocate(atom_type_Q_pol(N_type_atoms))  ; atom_type_Q_pol=0.0d0
   allocate(atom_type_DIPOLE_pol(N_type_atoms))   ;  atom_type_DIPOLE_pol =0.0d0
   allocate(atom_type_sfield_ext(N_type_atoms)) ; atom_type_sfield_ext=0.0d0
   allocate(atom_type_sfield_CONSTR(N_type_atoms)) ; atom_type_sfield_CONSTR=0.0d0
   allocate(IS_TYPE_ATOM_FIELD_CONSTRAINED(Na))  ; IS_TYPE_ATOM_FIELD_CONSTRAINED=.false.
   allocate(is_type_charge_distributed(N_type_atoms)) ; is_type_charge_distributed=.false.
   allocate(sfc_already_defined(N_type_atoms)); sfc_already_defined=.false.
   allocate(statistics_AtomPair_type(N_type_atoms)); statistics_AtomPair_type=-999

end subroutine atom_type_alloc

subroutine atom_style_alloc
use atom_type_data
use LR_corrections_data ! long range vdw corr
   use max_sizes_data, only : MX_VDW_PRMS
   use field_constrain_data, only : is_type_atom_field_constrained
   implicit none
   integer Ns2
   Ns2 = N_STYLE_ATOMS*(N_STYLE_ATOMS+1)/2

   
   allocate ( atom_Style2_N_vdwPrm(Ns2) ) ! double loop
   allocate ( atom_Style2_vdwPrm(0:MX_VDW_PRMS,Ns2) ) ! double loop
   allocate ( atom_Style2_vdwStyle(Ns2)) ; atom_Style2_vdwStyle = 0

   allocate ( atom_Style_1GAUSS_charge_distrib(N_STYLE_ATOMS) ) ; atom_Style_1GAUSS_charge_distrib = 1.0d93
   allocate ( which_atomStyle_pair (N_STYLE_ATOMS,N_STYLE_ATOMS), which_vdw_style(N_STYLE_ATOMS,N_STYLE_ATOMS))
   allocate ( pair_which_style(Ns2))
   allocate ( atom_style_name(N_STYLE_ATOMS))

   allocate ( EN0_LR_vdw(N_STYLE_ATOMS,N_STYLE_ATOMS)) ! long range vdw corr
   allocate ( STRESS0_LR_vdw(N_STYLE_ATOMS,N_STYLE_ATOMS))
   EN0_LR_vdw=0.0d0 
   STRESS0_LR_vdw=0.0d0

   allocate ( is_Style_dipole_pol(N_STYLE_ATOMS))
   allocate ( atom_Style_dipole_pol(N_STYLE_ATOMS))
   is_Style_dipole_pol=.false.
   atom_Style_dipole_pol=0.0d0

end subroutine atom_style_alloc


subroutine interpolate_alloc
 use interpolate_data
 implicit none
 integer N2, N
 N= MX_interpol_points
 N2 =  N_style_atoms*( N_style_atoms+1) / 2
print*, 'MX_interpol_points=',MX_interpol_points
print*,'N_style_atoms=',N_style_atoms, N2
 allocate (vvdw(0:N,N2), gvdw(0:N,N2))  ! potential and force vdw  
 vvdw=0.0d0 ; gvdw=0.0d0
 allocate (vele(0:N), gele(0:N)) ! potential and force Q
 vele=0.0d0 ; gele=0.0d0
 allocate (vele_G(0:N,N2),gele_G(0:N,N2))
 vele_G=0.0d0 ; gele_G=0.0d0
 allocate(v_B0(0:N,N_style_atoms),v_B1(0:N,N_style_atoms),v_B2(0:N,N_style_atoms)) 
 v_B0=0.0d0 ; v_B1=0.0d0 ; v_B2=0.0d0
 allocate(vele3(0:N), vele2(0:N))
 vele3 = 0.0d0 ; vele2 = 0.0d0
 allocate(vele2_G(0:N,N2))
 vele2_G = 0.0d0
 allocate(vele3_G(0:N,N2))
 vele3_G = 0.0d0

 allocate(vele_THOLE(0:N,N2),gele_THOLE(0:N,N2),vele_THOLE_DERIV(0:N,N2),&
                     gele_THOLE_DERIV(0:N,N2))
 vele_THOLE=0.0d0
 gele_THOLE=0.0d0
 vele_THOLE_DERIV=0.0d0
 gele_THOLE_DERIV=0.0d0

 allocate(gele_G_short(0:N,N2))
 gele_G_short=0.0d0

end subroutine interpolate_alloc

 subroutine Ewald_data_alloc
  use sizes_data, only : N_STYLE_ATOMS
  use Ewald_data, only : ewald_beta,ewald_gamma, ewald_eta
  
  allocate(ewald_beta(N_STYLE_ATOMS))
  allocate(ewald_gamma(N_STYLE_ATOMS,N_STYLE_ATOMS))
  allocate(ewald_eta(N_STYLE_ATOMS,N_STYLE_ATOMS))
 end subroutine Ewald_data_alloc
 
 subroutine mol_type_alloc
 use mol_type_data
 implicit none
 integer Nm,i
  Nm=N_type_molecules
  allocate(mol_type_name(Nm)) ; mol_type_name = ' '
  allocate(N_mols_of_type(Nm)) ; N_mols_of_type = 0
  allocate(N_type_atoms_per_mol_type(Nm))  ; N_type_atoms_per_mol_type = 0
  allocate(Nm_type_bonds(Nm))         ; Nm_type_bonds=0
  allocate(Nm_type_constrains(Nm))    ; Nm_type_constrains=0
  allocate(Nm_type_angles(Nm))        ; Nm_type_angles=0
  allocate(Nm_type_dihedrals(Nm))     ; Nm_type_dihedrals=0
  allocate(Nm_type_deforms(Nm))       ; Nm_type_deforms=0
  allocate(Nm_type_14(Nm))            ; Nm_type_14=0
  allocate(l_RIGID_GROUP_TYPE(Nm)) ; l_RIGID_GROUP_TYPE = .false.
  allocate(l_FLEXIBLE_GROUP_TYPE(Nm)); l_FLEXIBLE_GROUP_TYPE=.true. ! that is not required
  allocate(l_TYPEmol_do_stat_on_type(Nm)) ; l_TYPEmol_do_stat_on_type = .true.
  allocate(statistics_MolPair_type(Nm)); 
  do i = 1, Nm
   statistics_MolPair_type(i)=i
  enddo
  allocate(is_mol_type_sfc(Nm),param_mol_type_sfc(Nm))
  is_mol_type_sfc=.false.;param_mol_type_sfc=0.0d0; 
  allocate(mol_tag(Nm)); do i = 1, Nm ; mol_tag(i)%N = 0; enddo;
end subroutine mol_type_alloc

subroutine  ALL_rigid_mols_alloc
  use ALL_rigid_mols_data
  use sizes_data, only : Nmols,Natoms
  use mol_type_data, only : N_type_molecules
  implicit none
  integer N
  N=Nmols
   allocate ( qn(N,4) )
   allocate ( mol_MOM(N,3) ) ; mol_MOM=0.0d0
   allocate ( mol_ANG(N,3) ) ; mol_ANG=0.0d0
   allocate ( xyz_body(Natoms,3) ) 
   allocate (l_non_linear_rotor(N))
   allocate (mol_orient(N,9))
   allocate (Inverse_Molar_mass(N))
   allocate (Inertia_MAIN(N,3),Inverse_Inertia_MAIN(N,3))
   allocate (Inertia_SEC(N,3),Inverse_Inertia_SEC(N,3)) 
   allocate (mol_ANG_body(N,3)) ; mol_ANG_body=0.0d0
   allocate (mol_torque(N,3),mol_torque_body(N,3)) ; mol_torque=0.0d0;mol_torque_body=0.0d0

  allocate( Inertia_MAIN_TYPE(N_type_molecules,3), Inertia_SEC_TYPE(N_type_molecules,3) )
  allocate( Inverse_Inertia_MAIN_TYPE(N_type_molecules,3) )

end subroutine ALL_rigid_mols_alloc

subroutine qn_and_hi_deriv_alloc
use qn_and_hi_deriv_data ! for rigid dynamics integrate with GEAR4
use sizes_data, only : Nmols
implicit none
  integer N
  N=Nmols
  allocate (mol_xyz_3_deriv(N,3), mol_xyz_4_deriv(N,3))
  allocate (mol_MOM_3_deriv(N,3), mol_MOM_4_deriv(N,3))
  allocate (mol_ANG_3_deriv(N,3), mol_ANG_4_deriv(N,3))
  allocate (qn_3_deriv(N,4)) ; qn_3_deriv = 0.0d0
  allocate (qn_4_deriv(N,4)) ; qn_4_deriv = 0.0d0
  mol_xyz_3_deriv = 0.0d0 ; mol_xyz_4_deriv = 0.0d0
  mol_MOM_3_deriv = 0.0d0 ; mol_MOM_4_deriv = 0.0d0
  mol_ANG_3_deriv = 0.0d0 ; mol_ANG_4_deriv = 0.0d0

end subroutine qn_and_hi_deriv_alloc

subroutine qn_and_low_deriv_alloc
use qn_and_low_deriv_data
use sizes_data, only : Nmols
 implicit none
  integer N
  N=Nmols
  allocate (mol_xyz_1_deriv(N,3), mol_xyz_2_deriv(N,3)) 
  allocate (mol_MOM_1_deriv(N,3), mol_MOM_2_deriv(N,3))
  allocate (mol_ANG_1_deriv(N,3), mol_ANG_2_deriv(N,3)) 
  allocate (qn_1_deriv(N,4))
  allocate (qn_2_deriv(N,4))
  mol_xyz_1_deriv=0.0d0; mol_xyz_2_deriv=0.0d0
  mol_MOM_1_deriv=0.0d0; mol_MOM_2_deriv=0.0d0
  mol_ANG_1_deriv=0.0d0; mol_ANG_2_deriv=0.0d0
  qn_1_deriv=0.0d0
  qn_2_deriv=0.0d0
end subroutine qn_and_low_deriv_alloc

subroutine connectivity_type_alloc
  use connectivity_type_data
  implicit none
   allocate(bond_types(0:4,1:N_type_bonds)) ; bond_types=-999
   allocate(is_type_bond_constrained(N_type_bonds)) ; is_type_bond_constrained=.false.
   allocate(constrain_types(0:3,1:N_type_constrains)) ; constrain_types(0,:) = 1 ; constrain_types(1:3,:) = -999
   allocate(angle_types(0:5,1:N_type_angles)) ; angle_types = -999
   allocate(dihedral_types(0:6,1:N_type_dihedrals)) ; dihedral_types = -999
   allocate(deform_types(0:6,1:N_type_deforms))     ; deform_types=-999
   allocate(Nfolds_dihedral_types(1:N_type_dihedrals))  ; Nfolds_dihedral_types=0
   allocate(prm_bond_types(0:MX_BOND_PRMS,1:N_type_bonds))
   allocate(prm_constrain_types(0:MX_CONSTRAINT_PRMS,1:N_type_constrains)) ;prm_constrain_types=0.0d0
   allocate(prm_angle_types(0:MX_ANGLE_PRMS,1:N_type_angles))   ;prm_angle_types=0.0d0
   allocate(prm_dihedral_types(0:MX_DIH_PRMS,1:N_type_dihedrals)) ;prm_dihedral_types=0.0d0
   allocate(prm_deform_types(0:MX_DEFORM_PRMS,1:N_type_deforms)) ;prm_deform_types=0.0d0
   allocate(pair14_types(0:3,1:N_type_pairs_14))  ! 
   allocate(prm_pair14_types(1:2,1:N_type_pairs_14))
   
end subroutine connectivity_type_alloc

subroutine connectivity_ALL_alloc
 use connectivity_ALL_data
  allocate(list_bonds(0:4,Nbonds)     )        ; list_bonds=0
  allocate(list_angles(0:5,Nangles)    )       ; list_angles=0
  allocate(list_dihedrals(0:6,Ndihedrals) )    ; list_dihedrals=0
  allocate(list_deforms(0:6,Ndeforms) )        ; list_deforms=0
  allocate(list_constrains(0:3,Nconstrains))   ; list_constrains=0
  allocate(list_14(Natoms,MX_in_list_14), size_list_14(Natoms)  )   ; list_14=0 ; size_list_14=0
  allocate(list_excluded(Natoms,MX_excluded))  ; list_excluded=0
  allocate(size_list_excluded(Natoms))         ; size_list_excluded=0
  allocate(list_excluded_HALF(Natoms,MX_excluded)) ; list_excluded_HALF=0
  allocate(size_list_excluded_HALF(Natoms))    ; size_list_excluded_HALF=0
  allocate(list_excluded_HALF_no_SFC(Natoms,MX_excluded)) ; list_excluded_HALF_no_SFC=0
  allocate(size_list_excluded_HALF_no_SFC(Natoms))    ; size_list_excluded_HALF_no_SFC=0

  allocate(list_excluded_sfc_iANDj_HALF(Natoms,MX_excluded),size_list_excluded_sfc_iANDj_HALF(Natoms))
  list_excluded_sfc_iANDj_HALF=0; size_list_excluded_sfc_iANDj_HALF=0
 allocate(is_bond_constrained(Nbonds)) ; is_bond_constrained=.false.
 allocate(is_bond_dummy(Nbonds)) ; is_bond_dummy=.false.
end subroutine connectivity_ALL_alloc
 
 subroutine profiles_alloc
 use profiles_data
 use ALL_atoms_data, only : Natoms
 use connectivity_ALL_data, only :  Nbonds,Nangles,Ndihedrals
 use mol_type_data,only: N_type_molecules
 use atom_type_data, only : N_type_atoms
 use sizes_data, only : N_type_atoms_for_statistics,N_type_mols_for_statistics
 implicit none
 integer N
 integer i

  allocate(atom_profile(Natoms))
  allocate(bond_profile(Nbonds))
  allocate(angle_profile(Nangles))
  allocate(dihedral_profile(Ndihedrals))
  allocate(xyz_atom_profile(N_BINS_ZZ)) 
  allocate(xyz_bond_profile(N_BINS_ZZ))
  allocate(xyz_angle_profile(N_BINS_ZZ))
  allocate(xyz_dihedral_profile(N_BINS_ZZ))
  allocate(z_scale(N_BINS_ZZ))
  do i = 1, N_BINS_ZZ; z_scale(i) = dble(i)/dble(N_BINS_ZZ) ; enddo

 allocate(QQ_PP_a_pot(Natoms)) ; QQ_PP_a_pot=0.0d0
 allocate(QQ_PG_a_pot(Natoms)) ; QQ_PG_a_pot=0.0d0
 allocate(QQ_GP_a_pot(Natoms)) ; QQ_GP_a_pot=0.0d0
 allocate(QQ_GG_a_pot(Natoms)) ; QQ_GG_a_pot=0.0d0
 allocate(QD_PP_a_pot(Natoms)) ; QD_PP_a_pot=0.0d0
 allocate(DQ_PG_a_pot(Natoms)) ; DQ_PG_a_pot=0.0d0
 allocate(QD_GP_a_pot(Natoms)) ; QD_GP_a_pot=0.0d0
 allocate(DQ_PP_a_pot(Natoms)) ; DQ_PP_a_pot=0.0d0
 allocate(DD_PP_a_pot(Natoms)) ; DD_PP_a_pot=0.0d0

 allocate(RA_fi(Natoms)) ; RA_fi=0.0d0

 N=Natoms
 allocate (P_a_fi(N),G_a_fi(N),D_a_fi(N))
 allocate (P_a_EE_xx(N),G_a_EE_xx(N),D_a_EE_xx(N))
 allocate (P_a_EE_yy(N),G_a_EE_yy(N),D_a_EE_yy(N))
 allocate (P_a_EE_zz(N),G_a_EE_zz(N),D_a_EE_zz(N))

 P_a_fi=0.0d0 ; G_a_fi=0.0d0 ; D_a_fi=0.0d0;
 P_a_EE_xx=0.0d0 ; G_a_EE_xx=0.0d0 ; D_a_EE_xx=0.0d0
 P_a_EE_yy=0.0d0 ; G_a_EE_yy=0.0d0 ; D_a_EE_yy=0.0d0
 P_a_EE_zz=0.0d0 ; G_a_EE_zz=0.0d0 ; D_a_EE_zz=0.0d0
 
 allocate(counter_MOLS_global(N_BINS_ZZ,N_type_mols_for_statistics)) ; counter_MOLS_global=0.0d0
 allocate(counter_ATOMS_global(N_BINS_ZZ,N_type_atoms_for_statistics)) ; counter_ATOMS_global=0.0d0
 allocate(counter_ATOMS_global_x(N_BINS_xx,N_type_atoms_for_statistics)) ; counter_ATOMS_global_x=0.0d0
 allocate(counter_ATOMS_global_y(N_BINS_yy,N_type_atoms_for_statistics)) ; counter_ATOMS_global_y=0.0d0
!N_BINS_XX=N_BINS_ZZ;N_BINS_YY=N_BINS_ZZ; !do it when prepare the systems
if (N_BINS_XX==0)N_BINS_XX=N_BINS_ZZ
if (N_BINS_YY==0)N_BINS_YY=N_BINS_ZZ
 if (l_1st_profile_CTRL) then
 allocate( zp1_atom(N_BINS_ZZ),zp1_atom_x(N_BINS_XX),zp1_atom_y(N_BINS_YY))
 allocate( zp1_mol(N_BINS_ZZ))
 zp1_atom(:)%DOF = 0.0d0 ; zp1_atom(:)%kin = 0.0d0 ; 
 zp1_atom_x(:)%DOF = 0.0d0 ; zp1_atom_x(:)%kin = 0.0d0
 zp1_atom_y(:)%DOF = 0.0d0 ; zp1_atom_y(:)%kin = 0.0d0
 endif
 if (l_2nd_profile_CTRL) then
 allocate( zp2_atom(N_BINS_ZZ))
 allocate( zp2_mol(N_BINS_ZZ))
 endif

 do i = 1, N_BINS_ZZ
  if (l_1st_profile_CTRL) then
   allocate(zp1_atom(i)%density(N_type_atoms_for_statistics)); zp1_atom(i)%density(:) = 0.0d0
   allocate(zp1_atom_x(i)%density(N_type_atoms_for_statistics)); zp1_atom_x(i)%density(:) = 0.0d0
   allocate(zp1_atom_y(i)%density(N_type_atoms_for_statistics)); zp1_atom_y(i)%density(:) = 0.0d0
   allocate(zp1_mol(i)%density(N_type_mols_for_statistics)); zp1_mol(i)%density(:) = 0.0d0
   allocate(zp1_mol(i)%p_charge(4))  ;     zp1_mol(i)%p_charge(:)=0.0d0
   allocate(zp1_mol(i)%g_charge(4))  ;     zp1_mol(i)%g_charge(:)=0.0d0
   allocate(zp1_mol(i)%p_dipole(4))  ;     zp1_mol(i)%p_dipole(:)=0.0d0
   allocate(zp1_mol(i)%g_dipole(4))  ;     zp1_mol(i)%g_dipole(:)=0.0d0
   allocate(zp1_mol(i)%OP(N_type_mols_for_statistics,6)); zp1_mol(i)%OP(:,:) = 0.0d0
  endif
  if (l_2nd_profile_CTRL) then
   allocate(zp2_atom(i)%pot(N_type_atoms_for_statistics)) ;  zp2_atom(i)%pot(:) = 0.0d0
   allocate(zp2_atom(i)%Qpot(N_type_atoms_for_statistics)) ; zp2_atom(i)%Qpot(:) = 0.0d0
   allocate(zp2_atom(i)%fi(N_type_atoms_for_statistics)) ;   zp2_atom(i)%fi(:) = 0.0d0
   allocate(zp1_atom(i)%p_charge(4))  ;     zp1_atom(i)%p_charge(:)=0.0d0
   allocate(zp1_atom(i)%g_charge(4))  ;     zp1_atom(i)%g_charge(:)=0.0d0
   allocate(zp1_atom(i)%p_dipole(4))  ;     zp1_atom(i)%p_dipole(:)=0.0d0
   allocate(zp1_atom(i)%g_dipole(4))  ;     zp1_atom(i)%g_dipole(:)=0.0d0
   allocate(zp1_atom_x(i)%p_charge(4))  ;     zp1_atom_x(i)%p_charge(:)=0.0d0
   allocate(zp1_atom_x(i)%g_charge(4))  ;     zp1_atom_x(i)%g_charge(:)=0.0d0
   allocate(zp1_atom_x(i)%p_dipole(4))  ;     zp1_atom_x(i)%p_dipole(:)=0.0d0
   allocate(zp1_atom_x(i)%g_dipole(4))  ;     zp1_atom_x(i)%g_dipole(:)=0.0d0
   allocate(zp1_atom_y(i)%p_charge(4))  ;     zp1_atom_y(i)%p_charge(:)=0.0d0
   allocate(zp1_atom_y(i)%g_charge(4))  ;     zp1_atom_y(i)%g_charge(:)=0.0d0
   allocate(zp1_atom_y(i)%p_dipole(4))  ;     zp1_atom_y(i)%p_dipole(:)=0.0d0
   allocate(zp1_atom_y(i)%g_dipole(4))  ;     zp1_atom_y(i)%g_dipole(:)=0.0d0

   allocate(zp2_mol(i)%pot(N_type_mols_for_statistics)) ;         zp2_mol(i)%pot(:) = 0.0d0
   allocate(zp2_mol(i)%Qpot(N_type_mols_for_statistics)) ;  zp2_mol(i)%Qpot(:) = 0.0d0
!   allocate(zp2_mol(i)%stress(N_type_mols_for_statistics,1:10)) ; zp2_mol(i)%stress(:,:) = 0.0d0
   allocate(zp2_mol(i)%force(N_type_mols_for_statistics+1,1:4)) ;   zp2_mol(i)%force(:,:) = 0.0d0
  endif
 enddo
 
 end subroutine profiles_alloc

  subroutine ALL_sizes_allocate
    call ALL_atoms_alloc
    call ALL_mols_alloc
    call non_bonded_lists_alloc
    call connectivity_ALL_alloc
  end subroutine ALL_sizes_allocate

  subroutine smpe_alloc
     use variables_smpe
     use sim_cel_data, only : i_boundary_CTRL
     implicit none
     NFFT = nfftx * nffty * nfftz
     h_cut_z = nfftz/2
     if (.not.allocated(key1)) then 
       allocate(key1(nfftx))
     endif
     key1=0.0d0
     if (.not.allocated(key2)) then
       allocate(key2(nffty))
     endif
     key2=0.0d0;
     if (.not.allocated(key3)) then
       allocate(key3(nfftz))
     endif
     key3=0.0d0; 
     if (.not.(allocated(ww1_Re))) then 
        allocate(ww1_Re(nfftx))
     endif
     ww1_Re=0.0d0
     if (.not.(allocated(ww1_Im))) then
        allocate(ww1_Im(nfftx))
     endif
     ww1_Im=0.0d0
     if (.not.(allocated(ww2_Re))) then
         allocate(ww2_Re(nffty)) 
     endif
     ww2_Re = 0.0d0
    if (.not.(allocated(ww2_Im))) then
        allocate(ww2_Im(nffty))
     endif
     ww2_Im=0.0d0
     if (.not.(allocated(ww3_Re))) then
        allocate(ww3_Re(nfftz))
     endif
     ww3_Re=0.0d0
     if (.not.(allocated(ww3_Im))) then
        allocate(ww3_Im(nfftz))
     endif
     ww3_Im=0.0d0
    if (.not.(allocated(ww1))) then
        allocate(ww1(nfftx))
     endif
     ww1=(0.0d0,0.0d0)
     if (.not.(allocated(ww2))) then
        allocate(ww2(nffty))
     endif
     ww2=(0.0d0,0.0d0)
     if (.not.(allocated(ww3))) then
         allocate(ww3(nfftz))
     endif
     ww3 = (0.0d0,0.0d0)
     if (.not.(allocated(qqq1))) then
        allocate(qqq1(nfftx*nffty*nfftz))
     endif
     qqq1=(0.0d0,0.0d0)



     if (.not.(allocated(spline2_CMPLX_xx))) then 
         allocate(spline2_CMPLX_xx(nfftx))
     endif
     spline2_CMPLX_xx=0.0d0
     if (.not.(allocated(spline2_CMPLX_yy))) then
         allocate(spline2_CMPLX_yy(nffty))
     endif
     spline2_CMPLX_yy=0.0d0
     if (.not.(allocated(spline2_CMPLX_zz))) then
         allocate(spline2_CMPLX_zz(nfftz))
     endif
     spline2_CMPLX_zz=0.0d0 
     if (.not.allocated(qqq1_Re)) then 
           allocate(qqq1_Re(NFFT)) ;
     endif
      qqq1_Re = 0.0d0
!     if (.not.allocated(qqq1_Im)) then
!          allocate(qqq1_Im(NFFT)) ; 
!     endif
!     qqq1_Im = 0.0d0

  if(i_boundary_CTRL==1) then
     if (.not.(allocated(qqq2))) then
        allocate(qqq2(nfftx*nffty*nfftz))
     endif
      qqq2=(0.0d0,0.0d0)
     if  (.not.allocated(qqq2_Re)) then
          allocate(qqq2_Re(NFFT)) ; 
     endif 
     qqq2_Re = 0.0d0
!     if  (.not.allocated(qqq2_Im)) then
!          allocate(qqq2_Im(NFFT)) ;  
!     endif
!      qqq2_Im = 0.0d0
   endif !i_boundary_CTRL==1
  end subroutine smpe_alloc

  subroutine smpe_DEalloc
  use variables_smpe

    if (allocated(key1)) deallocate(key1)
    if (allocated(key2)) deallocate(key2)
    if (allocated(key3)) deallocate(key3)

    if (allocated(ww1_Re)) deallocate(ww1_Re)
    if (allocated(ww1_Im)) deallocate(ww1_Im)
    if (allocated(ww2_Re)) deallocate(ww2_Re)
    if (allocated(ww2_Im)) deallocate(ww2_Im)
    if (allocated(ww3_Re)) deallocate(ww3_Re)
    if (allocated(ww3_Im)) deallocate(ww3_Im)  
    if (allocated(ww1)) deallocate(ww1)
    if (allocated(ww2)) deallocate(ww2)  
    if (allocated(ww3)) deallocate(ww3)  
    if (allocated(qqq1)) deallocate(qqq1) 
    if (allocated(qqq2)) deallocate(qqq2)   
    if (allocated(qqq1_Re)) deallocate(qqq1_Re) 
    if (allocated(qqq1_Im)) deallocate(qqq1_Im)
    if (allocated(qqq2_Re)) deallocate(qqq2_Re) 
    if (allocated(qqq2_Im)) deallocate(qqq2_Im)
       
    if (allocated(spline2_CMPLX_xx)) deallocate(spline2_CMPLX_xx)
    if (allocated(spline2_CMPLX_yy)) deallocate(spline2_CMPLX_yy)
    if (allocated(spline2_CMPLX_zz)) deallocate(spline2_CMPLX_zz)


  end subroutine smpe_DEalloc

subroutine field_constrain_alloc
use field_constrain_data
use sizes_data, only : Natoms
use atom_type_data, only : N_TYPE_ATOMS
 allocate( is_type_atom_field_constrained(N_TYPE_ATOMS) ) ; is_type_atom_field_constrained=.false.
 allocate( ndx_remap_constrained(Natoms) )
end subroutine field_constrain_alloc


subroutine rdfs_alloc
use rdfs_data
use atom_type_data , only : N_type_atoms
implicit none
integer NZ,types
NZ = rdfs%N_Z_BINS
types = rdfs%N_PAIRS
allocate(gr_counters(0:N_BIN_rdf,1:NZ,1:types)) ; gr_counters=0.0d0
allocate(ALL_gr_counters(1:NZ,1:types))         ; ALL_gr_counters=0.0d0
allocate(RA_gr_counters(0:N_BIN_rdf,1:NZ,1:types)) ; RA_gr_counters=0.0d0
allocate(l_rdf_pair_eval(N_type_atoms,N_type_atoms))    ; l_rdf_pair_eval=.false.
allocate(which_pair_rdf(N_type_atoms,N_type_atoms))    
if (l_details_rdf_CTRL) then
  allocate(gr_counters_par(0:N_BIN_rdf,1:NZ,1:types));    gr_counters_par=0.0d0
  allocate(ALL_gr_counters_par(1:NZ,1:types))         ;   ALL_gr_counters_par=0.0d0
  allocate(RA_gr_counters_par(0:N_BIN_rdf,1:NZ,1:types)) ; RA_gr_counters_par=0.0d0
  allocate(gr_counters_perp(0:N_BIN_rdf,1:NZ,1:types))   ; gr_counters_perp=0.0d0
  allocate(ALL_gr_counters_perp(1:NZ,1:types))           ;  ALL_gr_counters_perp=0.0d0
  allocate(RA_gr_counters_perp(0:N_BIN_rdf,1:NZ,1:types))  ;  RA_gr_counters_perp=0.0d0
endif
end subroutine rdfs_alloc
 end module allocate_them
