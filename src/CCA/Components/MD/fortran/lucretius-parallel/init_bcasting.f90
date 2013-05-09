module init_bcasting
! bcast initial data types
contains
subroutine COMM_bcast_vdw_type
  use vdw_type_data
  use comunications
  implicit none
!   integer N_TYPES_VDW
   call COMM_bcast(prm_vdw_type)
   call COMM_bcast(vdw_type_atom_1_type)
   call COMM_bcast(vdw_type_atom_2_type)
   call COMM_bcast(vdw_type_style)
   call COMM_bcast(vdw_type_Nparams)
end subroutine COMM_bcast_vdw_type

subroutine COMM_bcast_atom_type
   use atom_type_data
   use comunications
   use field_constrain_data
   implicit none
!  call COMM_bcast(atom_type_name)  ! MPI cannot bcast characters; nor no I need the names bcasted
   call COMM_bcast(atom_type_mass)
   call COMM_bcast(atom_type_charge)
   call COMM_bcast(atom_type_dipol)
   call COMM_bcast(atom_type_DIR_dipol)
   call COMM_bcast(atom_type_isDummy)
   call COMM_bcast(atom_type_molindex)
   call COMM_bcast(atom_type_in_which_mol_type)
   call COMM_bcast(atom_type2_N_vdwPrm )
   call COMM_bcast(atom_type2_vdwPrm ) ! allocate it latter after get the MX_VDW_PRMS
   call COMM_bcast(atom_type2_vdwStyle)
   call COMM_bcast(which_atom_pair)
   call COMM_bcast(pair_which_atom%i)
   call COMM_bcast(pair_which_atom%j)
   call COMM_bcast(l_TYPE_ATOM_WALL)
   call COMM_bcast(atom_type_N_GAUSS_charges)
   call COMM_bcast(atom_type_1GAUSS_charge)
   call COMM_bcast(atom_type_1GAUSS_charge_distrib)
   call COMM_bcast(is_type_dipole_pol)
   call COMM_bcast(is_type_charge_pol)
!   call COMM_bcast(polarizability)
   call COMM_bcast(atom_type_Q_pol)  
   call COMM_bcast(atom_type_DIPOLE_pol)
   call COMM_bcast(atom_type_sfield_ext)
   call COMM_bcast(is_type_atom_field_constrained)
   call COMM_bcast(N_type_atoms_field_constrained)
end subroutine COMM_bcast_atom_type  

subroutine COMM_bcast_mol_type
   use mol_type_data
   use comunications
   use max_sizes_data, only : MX_excluded
   implicit none
!  call COMM_bcast(mol_type_name) ! MPI cannot bcast characters; nor no I need the names bcasted
   call COMM_bcast(MX_excluded)
   call COMM_bcast(N_mols_of_type) 
   call COMM_bcast(N_type_atoms_per_mol_type)  
   call COMM_bcast(Nm_type_bonds)
   call COMM_bcast(Nm_type_constrains)
   call COMM_bcast(Nm_type_angles)
   call COMM_bcast(Nm_type_dihedrals)
end subroutine COMM_bcast_mol_type

subroutine COMM_bcast_connectivity_type
   use connectivity_type_data
   use comunications
   implicit none
   call COMM_bcast(bond_types)
   call COMM_bcast(constrain_types) 
   call COMM_bcast(angle_types)
   call COMM_bcast(dihedral_types)
   call COMM_bcast(prm_bond_types)
   call COMM_bcast(prm_constrain_types)
   call COMM_bcast(prm_angle_types)
   call COMM_bcast(prm_dihedral_types)
end subroutine COMM_bcast_connectivity_type

end module init_bcasting
