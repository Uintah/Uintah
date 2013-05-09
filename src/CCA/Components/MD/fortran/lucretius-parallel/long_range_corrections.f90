
! This module does the long range correction  of forces and potential and stresses
 
module long_range_corrections
implicit none
public :: tail_correct_vdw
 contains

   subroutine tail_correct_vdw
   use LR_corrections_data
   use energies_data
   use stresses_data
   use atom_type_data
   use mol_type_data
   use vdw_type_data
   use ALL_atoms_data, only : atom_in_which_molecule, atom_in_which_type_molecule,&
                              i_type_atom, i_style_atom,natoms
   use ALL_mols_data, only : i_type_molecule, Nmols
   use profiles_data, only : l_need_2nd_profile, atom_profile
   use cut_off_data
   use physical_constants, only : LJ_epsilon_convert
   use sim_cel_data, only : Volume

   implicit none
    real(8), parameter :: Pi = 3.14159265358979d0
    real(8) dinp,d,v,dni, tail_pressure_LJ, sp, En, tail_energy_LJ
    integer i2,j2,i_Class,j_Class,i1,j1,j,i,k,i_type, ii, jj, istyle,jstyle
    logical, save :: l_very_first_pass = .true.
    real(8) p1,p2,p3,p4,p5,p6
    real(8), allocatable :: a_pot(:),lstr1(:)
    real(8) vir,t_i,p_i
    real(8), save :: tail_energy_LJ_0, tail_pressure_LJ_0
    real(8), allocatable , save :: tail_en_vct_0(:),tail_vir_vct_0(:)

   
   if (l_very_first_pass) then
    l_very_first_pass=.false.
    allocate(tail_en_vct_0(N_type_atoms),tail_vir_vct_0(N_type_atoms))
    tail_energy_LJ=0.0d0; 
    tail_pressure_LJ=0.0d0   
    do i = 1, N_type_atoms
    istyle = map_atom_type_to_style(i)
    ii = atom_type_in_which_mol_type(i)
    t_i = 0.0d0
    p_i = 0.0d0
    do j = 1, N_type_atoms
     jstyle = map_atom_type_to_style(j)
     jj = atom_type_in_which_mol_type(j)
     d=dble(N_mols_of_type(ii))*dble(N_mols_of_type(jj))
     En = EN0_LR_vdw(istyle,jstyle) * d 
     Vir = STRESS0_LR_vdw(istyle,jstyle) * d 
     tail_energy_LJ = tail_energy_LJ + En
     tail_pressure_LJ = tail_pressure_LJ + Vir
     t_i = t_i + EN0_LR_vdw(istyle,jstyle) * dble(N_mols_of_type(jj))
     p_i = p_i + STRESS0_LR_vdw(istyle,jstyle) * dble(N_mols_of_type(jj))
    enddo
    tail_en_vct_0(i)  = t_i
    tail_vir_vct_0(i) = p_i
    enddo
    tail_energy_LJ_0 = tail_energy_LJ
    tail_pressure_LJ_0 = tail_pressure_LJ
    endif


    tail_energy_LJ = tail_energy_LJ_0/Volume
    tail_pressure_LJ=tail_pressure_LJ_0/Volume/3.0d0
    stress(1:4) = stress(1:4) + tail_pressure_LJ
    stress_vdw(1:4) = stress_vdw(1:4) + tail_pressure_LJ
    en_vdw = en_vdw + tail_energy_LJ


 if (l_need_2nd_profile) then
   allocate(a_pot(Natoms),lstr1(Natoms))
   do i = 1, Natoms
     i_type = i_type_atom(i)
!     dni = 1.0d0/dble(N_mols_of_type(atom_type_in_which_mol_type(i_type)))
     En = tail_en_vct_0(i_type) * (2.0d0 / Volume) !* dni
     a_pot(i) =  En
     sp = tail_vir_vct_0(i_type) * (3.0d0/Volume * 2.0d0) !* dni 
     lstr1(i) =  sp
   enddo
   atom_profile%pot = atom_profile%pot + a_pot
   atom_profile%sxx = atom_profile%sxx + lstr1
   atom_profile%syy = atom_profile%syy + lstr1
   atom_profile%szz = atom_profile%szz + lstr1
   deallocate(a_pot,lstr1)
 endif
  end subroutine tail_correct_vdw


end module long_range_corrections


