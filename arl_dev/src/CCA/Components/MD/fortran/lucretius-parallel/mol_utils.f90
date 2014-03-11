 module mol_utils
 ! subroutines uusefull for actions on molecules
 public :: put_CM_molecules_in_box
 public :: get_all_mol_mass
 public :: get_all_mol_CM
 public :: get_all_mol_charge
 public :: get_all_mol_dipoles
 public :: get_all_mol_potential
 public :: get_all_mol_forces
 public :: get_all_mol_stress 
 public :: get_all_mol_properties

 contains
subroutine verify_zero_molar_mass
   call get_all_mol_mass
end subroutine verify_zero_molar_mass
subroutine put_CM_molecules_in_box
! but keep xxx yyy zzz unwrapped.
use ALL_atoms_data, only : Natoms, xxx,yyy,zzz,atom_in_which_molecule,&
                           base_dx,base_dy,base_dz
use ALL_mols_data, only : Nmols, mol_xyz
use boundaries, only : periodic_images
implicit none
real(8) dx(Natoms),dy(Natoms),dz(Natoms)
real(8), allocatable :: xxx0(:),yyy0(:),zzz0(:)
integer i,j,imol

allocate(xxx0(Natoms),yyy0(Natoms),zzz0(Natoms))
xxx0=xxx;yyy0=yyy;zzz0=zzz

call get_all_mol_CM
do i = 1, Natoms
imol = atom_in_which_molecule(i)
dx(i) = xxx(i) - mol_xyz(imol,1)
dy(i) = yyy(i) - mol_xyz(imol,2)
dz(i) = zzz(i) - mol_xyz(imol,3)
enddo
call periodic_images(mol_xyz(1:Nmols,1),mol_xyz(1:Nmols,2),mol_xyz(1:Nmols,3))
do i = 1, Natoms
imol = atom_in_which_molecule(i)
xxx(i) = dx(i) + mol_xyz(imol,1)
yyy(i) = dy(i) + mol_xyz(imol,2)
zzz(i) = dz(i) + mol_xyz(imol,3)
enddo
base_dx=xxx0-xxx; base_dy=yyy0-yyy;base_dz=zzz0-zzz;

deallocate(xxx0,yyy0,zzz0)
end subroutine put_CM_molecules_in_box


 subroutine get_all_mol_mass
  use ALL_mols_data, only : Nmols, start_group, end_group,mol_mass
  use ALL_atoms_data, only : all_atoms_mass
  implicit none
  integer i,j
  real(8) sm
  do i = 1, Nmols
     sm = 0.0d0
     do j = start_group(i) , end_group(i)
        sm = sm + all_atoms_mass(j)
     enddo
     mol_mass(i) = sm
     if (mol_mass(i)==0.0d0) then
       print*,'ERROR molar mass of species ',i,' is zero: any molecule should have non zero mass'
       print*, 'If the molecule is WALL it still should have some mass while the flagg WALL will take care of issues'
       print*,' Open ff.dat and assigm atomic masses such that molecules has total nonzero mass'
       STOP
     endif
   enddo
 end subroutine get_all_mol_mass
 
 subroutine get_all_mol_CM
 use ALL_mols_data, only : Nmols, mol_xyz, start_group,end_group, mol_mass
 use ALL_atoms_data, only : xxx,yyy,zzz,Natoms,all_atoms_mass
 implicit none
 integer i,j,si,sf
 real(8) tx,ty,tz, sm
   do i = 1, Nmols
     tx = 0.0d0 ; ty = 0.0d0 ; tz = 0.0d0
     si = start_group(i) ; sf =  end_group(i)
     tx = dot_product(all_atoms_mass(si:sf),xxx(si:sf))
     ty = dot_product(all_atoms_mass(si:sf),yyy(si:sf))
     tz = dot_product(all_atoms_mass(si:sf),zzz(si:sf))
     sm = sum(all_atoms_mass(si:sf)) ! mol_mass(i) was evaluated prior to this.
     mol_xyz(i,1) = tx/sm ; mol_xyz(i,2) = ty/sm ; mol_xyz(i,3) = tz/sm
   enddo
 end subroutine get_all_mol_CM

 subroutine get_all_mol_charge
    use ALL_mols_data, only : all_mol_G_charges,all_mol_P_charges
    use ALL_atoms_data, only : all_p_charges, all_g_charges, is_charge_distributed, Natoms, &
                               atom_in_which_molecule
    implicit none
    integer i,j, imol
    all_mol_p_charges=0.0d0; all_mol_g_charges=0.0d0
    do i = 1, Natoms
    imol = atom_in_which_molecule(i)
        if (is_charge_distributed(i)) then
          all_mol_G_charges(imol) = all_mol_G_charges(imol) + all_G_charges(i)
        else
         all_mol_P_charges(imol) = all_mol_P_charges(imol) + all_P_charges(i)
        endif
     enddo
 end subroutine get_all_mol_charge 

 subroutine get_all_mol_dipoles
    use ALL_mols_data, only : all_mol_G_charges,all_mol_P_charges, mol_dipole, mol_xyz
    use ALL_atoms_data, only : all_p_charges, all_g_charges, is_charge_distributed, Natoms, &
                               atom_in_which_molecule, zzz,yyy,xxx
     implicit none
    integer i,j, imol
    real(8) dip(3),t(3),q

    mol_dipole=0.0d0
    do i = 1, Natoms
        imol = atom_in_which_molecule(i)
        dip(:)=0.0d0
        t(1) = xxx(i); t(2) = yyy(i); t(3) = zzz(i)
        if (is_charge_distributed(i)) then
          q =  all_G_charges(i)
        else
          q =  all_P_charges(i)
        endif
        mol_dipole(imol,:) = mol_dipole(imol,:) + q*(t(:)-mol_xyz(imol,:))
     enddo

 end subroutine get_all_mol_dipoles

 subroutine get_all_mol_potential
     use ALL_mols_data, only : Nmols, mol_potential,mol_potential_Q, start_group,end_group, mol_mass
     use profiles_data, only : atom_profile
     implicit none
     integer i,j 
     do i = 1, Nmols
     mol_potential(i) = 0.0d0
     mol_potential_Q(i) = 0.0d0
     do j = start_group(i) , end_group(i)
       mol_potential(i) = mol_potential(i) + atom_profile(j)%pot
       mol_potential_Q(i) = mol_potential_Q(i) + atom_profile(j)%Qpot
     enddo
     enddo
 end subroutine get_all_mol_potential
 subroutine get_all_mol_forces
     use ALL_atoms_data, only : fxx,fyy,fzz
     use ALL_mols_data, only : Nmols, mol_force, start_group,end_group, mol_mass
     use profiles_data, only : atom_profile
     implicit none
     integer i,j
     do i = 1, Nmols
     mol_force(i,:) = 0.0d0
     do j = start_group(i) , end_group(i)
       mol_force(i,1) = mol_force(i,1) + fxx(j)
       mol_force(i,2) = mol_force(i,2) + fyy(j)
       mol_force(i,3) = mol_force(i,3) + fzz(j)
     enddo
     enddo
 end subroutine get_all_mol_forces
 subroutine get_all_mol_stress
!    to be implemented latter 
         
 end subroutine get_all_mol_stress
 subroutine get_all_mol_properties
     implicit none
    call get_all_mol_mass
    call get_all_mol_CM
    call get_all_mol_charge
    call get_all_mol_potential
    call get_all_mol_forces
    call get_all_mol_stress
    call get_all_mol_dipoles
 end subroutine get_all_mol_properties 
 end module mol_utils
