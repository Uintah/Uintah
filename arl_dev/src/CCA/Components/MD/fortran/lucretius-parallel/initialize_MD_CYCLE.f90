module initialize_MD_CYCLE_module
 implicit none
 public :: initialize_MD_CYCLE
 contains
 subroutine initialize_MD_CYCLE
   use integrate_data
   use collect_data
   use ensamble_data
   use saved_energies, only : zero_energies, d_zero_energies
   use stresses_data
   use profiles_data
   use energies_data
   use ALL_atoms_data
   implicit none 
    l_need_1st_profile = l_1st_profile_CTRL.and.(mod(integration_step,collect_skip)==0)
    l_need_2nd_profile = l_2nd_profile_CTRL.and.(mod(integration_step,collect_skip)==0)
    l_need_ANY_profile = l_ANY_profile_CTRL.and.(l_need_1st_profile.or.l_need_2nd_profile)
    call zero_energies
    call d_zero_energies
!    call zero_velocities
    call zero_stresses
!    call zero_forces
    if (l_need_ANY_profile) then
      call zero_atom_profiles
      call zero_xyz_atom_profile
    endif
    contains

     subroutine zero_stresses
     use stresses_data
       stress = 0.0d0
       pressure =0.0d0
       stress_kin=0.0d0 ; stress_shake=0.0d0;
       stress_bond=0.0d0;stress_angle=0.0d0;stress_dih=0.0d0;stress_deform=0.0d0;stress_dummy=0.0d0
       stress_vdw=0.0d0;
       stress_Qcmplx_as_in_3D=0.0d0; 
       stress_Qreal=0.0d0;stress_Qcmplx=0.0d0;stress_Qcmplx_k_eq_0=0.0d0;
       stress_excluded=0.0d0;
       stress_thetering=0.0d0
     end subroutine zero_stresses

     subroutine zero_forces
      use ALL_atoms_data, only : fxx,fyy,fzz
      use integrate_data, only : l_do_QN_CTRL
      use ALL_mols_data, only : mol_force
      use ALL_rigid_mols_data, only : mol_torque
       fxx = 0.0d0
       fyy = 0.0d0
       fzz = 0.0d0
       if (l_do_QN_CTRL) then
         mol_force=0.0d0
         mol_torque=0.0d0
       endif
     end subroutine zero_forces

     subroutine zero_velocities
        use ALL_atoms_data, only : vxx, vyy, vzz
       vxx = 0.0d0 ; vyy = 0.0d0 ; vzz = 0.0d0
     end subroutine zero_velocities
     
     subroutine zero_atom_profiles 
      use profiles_data
      use sizes_data, only : Natoms
       implicit none
       integer i
       if (l_ANY_profile_CTRL) then
       atom_profile%pot=0.0d0 
       atom_profile%Qpot = 0.0d0
       atom_profile%sxx=0.0d0
       atom_profile%sxy=0.0d0
       atom_profile%sxz=0.0d0
       atom_profile%syx=0.0d0
       atom_profile%syy=0.0d0
       atom_profile%syz=0.0d0
       atom_profile%szx=0.0d0
       atom_profile%szy=0.0d0
       atom_profile%szz=0.0d0
       atom_profile%fi=0.0d0
       atom_profile%EE_xx=0.0d0
       atom_profile%EE_yy=0.0d0
       atom_profile%EE_zz=0.0d0 
       do i = 1, Natoms ; atom_profile(i)%buffer3(:) = 0.0d0; enddo
       endif
     end subroutine zero_atom_profiles
     subroutine zero_xyz_atom_profile
     use profiles_data
     if (l_ANY_profile_CTRL) then
       xyz_atom_profile%pot=0.0d0
       xyz_atom_profile%sxx=0.0d0
       xyz_atom_profile%sxy=0.0d0
       xyz_atom_profile%sxz=0.0d0
       xyz_atom_profile%syx=0.0d0
       xyz_atom_profile%syy=0.0d0
       xyz_atom_profile%syz=0.0d0
       xyz_atom_profile%szx=0.0d0
       xyz_atom_profile%szy=0.0d0
       xyz_atom_profile%szz=0.0d0
       xyz_atom_profile%fi=0.0d0
       xyz_atom_profile%EE_xx=0.0d0
       xyz_atom_profile%EE_yy=0.0d0
       xyz_atom_profile%EE_zz=0.0d0
     endif
     end subroutine zero_xyz_atom_profile
     
    end subroutine initialize_MD_CYCLE

end module initialize_MD_CYCLE_module
