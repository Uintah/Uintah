module code_generator
implicit none
public generate_short_sysdef_in_f90
public generate_short_sysdef_in_java

contains
subroutine generate_short_sysdef_in_f90
use file_names_data, only : short_sysdef_f90_file_name
use sim_cel_data
use atom_type_data
use mol_type_data
use connectivity_type_data
use field_constrain_data, only : is_type_atom_field_constrained
use max_sizes_data, only : MX_excluded
implicit none
integer ifile,i,j,k
ifile=763
open(unit=ifile,file=trim(short_sysdef_f90_file_name),recl=1000)
write(ifile,*) 'subroutine sysdef0'
write(ifile,*)  'use max_sizes_data'
write(ifile,*)  'use sizes_data'
write(ifile,*)  'use ALL_mols_data'
write(ifile,*)  'use ALL_atoms_data'
write(ifile,*)  'use mol_type_data'
write(ifile,*)  'use atom_type_data'
write(ifile,*)  'use allocate_them'
write(ifile,*)  'use sim_cel_data'
write(ifile,*)  'use connectivity_type_data, only : bond_types, is_type_bond_constrained'
write(ifile,*)  'use field_constrain_data'
write(ifile,*)  'use sysdef_module'
write(ifile,*)  'use excuded_list_eval_module'

write(ifile,*)  'integer t'
write(ifile,*) 'i_boundary_CTRL =',i_boundary_CTRL 
write(ifile,*) 'sim_cel = 0.0d0'
write(ifile,*) 'MX_excluded=',MX_excluded

do i = 1,9
write(ifile,*)'sim_cel(',i,')=',sim_cel(i)
enddo

write(ifile,*) '   N_type_molecules = ', N_type_molecules
write(ifile,*) '      call mol_type_alloc'
do i = 1, ubound(N_mols_of_type,dim=1)
 write(ifile,*) ' N_mols_of_type(',i,')=',N_mols_of_type(i)
enddo
do i = 1, ubound(N_type_atoms_per_mol_type,dim=1)
 write(ifile,*) ' N_type_atoms_per_mol_type(',i,')=',N_type_atoms_per_mol_type(i)
enddo
do i = 1, ubound(Nm_type_bonds,dim=1)
 write(ifile,*) ' Nm_type_bonds(',i,')=',Nm_type_bonds(i)
enddo
do i = 1, ubound(Nm_type_constrains,dim=1)
 write(ifile,*) ' Nm_type_constrains(',i,')=',Nm_type_constrains(i)
enddo


write(ifile,*) '  N_type_atoms = ',sum(N_type_atoms_per_mol_type)
write(ifile,*) '  N_type_bonds = ',sum(Nm_type_bonds(1:N_TYPE_MOLECULES))
write(ifile,*) '  N_type_constrains = ',sum(Nm_type_constrains(1:N_TYPE_MOLECULES))
write(ifile,*) '  N_type_angles = ',sum(Nm_type_angles(1:N_TYPE_MOLECULES))
write(ifile,*) '  N_type_dihedrals = ',sum(Nm_type_dihedrals(1:N_TYPE_MOLECULES))
write(ifile,*) '  N_type_deforms = ',sum(Nm_type_deforms(1:N_TYPE_MOLECULES))

write(ifile,*)  '  call get_ALL_sizes '
write(ifile,*)  '  call connectivity_type_alloc '
write(ifile,*)  '  call atom_type_alloc '
write(ifile,*)  '  call ALL_mols_alloc '
write(ifile,*)  '  call ALL_atoms_alloc '
write(ifile,*)  '  call connectivity_ALL_alloc '
write(ifile,*)  '  call locate_atoms_in_molecules '

write(ifile,*) '   atom_type_name=" " '
do i = 1, ubound(atom_type_name,dim=1)
   write(ifile,*) '   atom_type_name(',i,')=','"', trim(atom_type_name(i)) ,'"'
enddo

do i = 1, ubound(atom_type_mass,dim=1)
   write(ifile,*) '   atom_type_mass(',i,')=',atom_type_mass(i)
enddo

do i = 1, ubound(atom_type_charge,dim=1)
   write(ifile,*) '   atom_type_charge(',i,')=',atom_type_charge(i)
enddo

write(ifile,*) 'if (.not.allocated(l_TYPEatom_do_stat_on_type)) allocate(l_TYPEatom_do_stat_on_type(N_TYPE_ATOMS))'
do i = 1, ubound(l_TYPEatom_do_stat_on_type,dim=1)
   if (l_TYPEatom_do_stat_on_type(i)) then
   write(ifile,*) '   l_TYPEatom_do_stat_on_type(',i,')= .true.'
   else
   write(ifile,*) '   l_TYPEatom_do_stat_on_type(',i,')= .false.'
   endif
enddo

 write(ifile,*) '! is type atom WALL?'
do i = 1, ubound(l_TYPE_ATOM_WALL,dim=1)
  if (l_TYPE_ATOM_WALL(i)) then
  write(ifile,*) '   l_TYPE_ATOM_WALL(',i,')= .true.'
  else
  write(ifile,*) '   l_TYPE_ATOM_WALL(',i,')= .false.'
  endif
enddo
write(ifile,*) '! is type atom WALL_1?'
do i = 1, ubound(l_TYPE_ATOM_WALL_1,dim=1)
  if (l_TYPE_ATOM_WALL_1(i)) then
  write(ifile,*) '   l_TYPE_ATOM_WALL_1(',i,')= .true.'
  else
  write(ifile,*) '   l_TYPE_ATOM_WALL_1(',i,')= .false.'
  endif
enddo

write(ifile,*) '! is type atom field constrained (sfc)?'
write(ifile,*) ' if (.not.(allocated(is_type_atom_field_constrained))) allocate(is_type_atom_field_constrained(N_TYPE_ATOMS))'
do i = 1, ubound(is_type_atom_field_constrained,dim=1)
  if (is_type_atom_field_constrained(i)) then
  write(ifile,*) '   is_type_atom_field_constrained(',i,')= .true.'
  else
  write(ifile,*) '   is_type_atom_field_constrained(',i,')= .false.'
  endif
enddo
do i = 1, ubound(atom_type_sfield_CONSTR,dim=1)
  if (is_type_atom_field_constrained(i)) then
  write(ifile,*) '   atom_type_sfield_CONSTR(',i,')= .true.'
  else
  write(ifile,*) '   atom_type_sfield_CONSTR(',i,')= .false.'
  endif
enddo




do i = 1, ubound(bond_types,dim=2)
write(ifile,*)  '   bond_types(1:4',",",i,")= (/ ", bond_types(1,i), ",", &
                                                    bond_types(2,i), ",", &
                                                    bond_types(3,i), ",", &
                                                    bond_types(4,i), '  /);'
enddo

write(ifile,*) '!  constrained type bonds:'
write(ifile,*) 'if (.not.allocated(is_type_bond_constrained)) allocate(is_type_bond_constrained(N_type_bonds))'
do i = 1, ubound(is_type_bond_constrained,dim=1)
  if (is_type_bond_constrained(i)) then
  write(ifile,*)  '   is_type_bond_constrained(',i,')= .true.'
  else
  write(ifile,*)  '   is_type_bond_constrained(',i,')= .false.'
  endif
enddo

write(ifile,*) '   mol_type_name=" " '
do i = 1, ubound(mol_type_name,dim=1)
   write(ifile,*) '   mol_type_name(',i,')=','"', trim(mol_type_name(i)) ,'"'
enddo

write(ifile,*)  '  call get_connectivity '
write(ifile,*)  '  call set_ALL_masses '
write(ifile,*)  '  call get_excluded_lists '
write(ifile,*)  '  call set_ALL_charges '
write(ifile,*)  'end subroutine sysdef0'

close(ifile)

end subroutine generate_short_sysdef_in_f90

subroutine generate_short_sysdef_in_java
print*, 'generate_short_sysdef_in_java NOT IMPLEMENTED YET'
end subroutine generate_short_sysdef_in_java

end module code_generator
