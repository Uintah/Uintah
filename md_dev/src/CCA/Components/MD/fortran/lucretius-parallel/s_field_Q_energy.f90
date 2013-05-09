 module s_fiels_Q_energy_module
 implicit none
 public :: s_fiels_Q_energy
 contains
 subroutine s_fiels_Q_energy
  use ALL_atoms_data, only : Natoms,is_sfield_constrained,all_p_charges,all_g_charges,&
                             external_sfield_CONSTR,external_sfield
  use profiles_data, only : l_need_2nd_profile, atom_profile
  use energies_data, only : en_sfield
  integer i,j,k
  real(8), allocatable :: a_pot(:)
  real(8) local_energy

  allocate(a_pot(Natoms))
  local_energy=0.0d0
   do i = 1, Natoms
      if (is_sfield_constrained(i)) then
          a_pot(i) = - (all_p_charges(i) + all_g_charges(i)) * (external_sfield_CONSTR(i)+external_sfield(i))
          local_energy = local_energy + a_pot(i)
!if (i==1.or.i==Natoms) then
!print*, i, 'q=',all_g_charges(i),all_p_charges(i),'ef=', external_sfield_CONSTR(i),external_sfield(i), 'pot=',a_pot(i)
!read(*,*)
!endif
      else
          a_pot(i) = 0.0d0
      endif
   enddo
   if (l_need_2nd_profile) then
     atom_profile%pot = atom_profile%pot + a_pot*2.0d0  
   endif   
   en_sfield = en_sfield + local_energy
 deallocate(a_pot)
 end subroutine s_fiels_Q_energy 
 end module s_fiels_Q_energy_module
