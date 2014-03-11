
! Converting some atomic properties to molecular ones 

module convert_mol_to_atom

public :: from_atom_to_mol

 contains
  subroutine from_atom_to_mol ! when call this subroutine atom_xyz has to be in local frame
   use stresses_data, only : stress
   use ALL_atoms_data, only : fxx,fyy,fzz, Natoms,atom_in_which_molecule
   use ALL_mols_data, only : mol_force
   use ALL_rigid_mols_data, only : mol_torque, xyz_body
   
    implicit none
    integer i,j
    real(8) lsxx,lsxy,lsxz,lsyx,lsyy,lsyz,lszx,lszy,lszz      

!convert forces on atoms to forces on whole molecules. Same with torques.
      mol_force=0.0d0
      mol_torque=0.0d0

      lsxx = 0.0d0; lsxy=0.0d0; lsxz=0.0d0
      lsyx = 0.0d0; lsyy=0.0d0; lsyz=0.0d0
      lszx = 0.0d0; lszy=0.0d0; lszz=0.0d0
      do i=1,Natoms
         j=atom_in_which_molecule(i)
         mol_force(j,1)=mol_force(j,1)+fxx(i)
         mol_force(j,2)=mol_force(j,2)+fyy(i)
         mol_force(j,3)=mol_force(j,3)+fzz(i)
         mol_torque(j,1)=mol_torque(j,1)+xyz_body(i,2)*fzz(i)-xyz_body(i,3)*fyy(i)
         mol_torque(j,2)=mol_torque(j,2)+xyz_body(i,3)*fxx(i)-xyz_body(i,1)*fzz(i)
         mol_torque(j,3)=mol_torque(j,3)+xyz_body(i,1)*fyy(i)-xyz_body(i,2)*fxx(i)
         lsxx = lsxx +  xyz_body(i,1)*fxx(i)
         lsxy = lsxy +  xyz_body(i,1)*fyy(i)
         lsxz = lsxz +  xyz_body(i,1)*fzz(i)
         lsyx = lsyx +  xyz_body(i,2)*fxx(i)
         lsyy = lsyy +  xyz_body(i,2)*fyy(i)
         lsyz = lsyz +  xyz_body(i,2)*fzz(i)
         lszx = lszx +  xyz_body(i,3)*fxx(i)
         lszy = lszy +  xyz_body(i,3)*fyy(i)
         lszz = lszz +  xyz_body(i,3)*fzz(i)
      enddo

     stress(1) = stress(1) - lsxx
     stress(2) = stress(2) - lsyy
     stress(3) = stress(3) - lszz
     stress(5) = stress(5) - lsxy
     stress(6) = stress(6) - lsxz
     stress(7) = stress(7) - lsyz
     stress(8) = stress(8) - lsyx
     stress(9) = stress(9) - lszx
     stress(10) = stress(10) - lszy
     stress(4) = stress(4) - (lsxx+lsyy+lszz)/3.0d0

  end subroutine from_atom_to_mol


end module convert_mol_to_atom

