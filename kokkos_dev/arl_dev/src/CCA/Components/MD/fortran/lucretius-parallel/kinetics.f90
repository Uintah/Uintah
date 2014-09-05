module kinetics
  implicit none

  public :: is_list_2BE_updated
  public :: get_instant_temperature
  public :: get_instant_MOL_temperature
  public :: get_mol_stresses
  public :: add_kinetic_pressure
  public :: get_kinetic_energy_stress
  public :: get_kinetic_energy 
  public :: mol_kinetics
  public :: get_kinetic_stress

  contains

 subroutine is_list_2BE_updated(l_update)
  use ALL_atoms_data, only : xxx,yyy,zzz , Natoms
  use cut_off_data
  use all_mols_data, only : mol_xyz
  use boundaries, only : cel_properties, adjust_box
  implicit none
  logical l_update
  real(8), allocatable , save :: dx2(:),dy2(:),dz2(:), disp2(:)
  logical, save :: l_first_pass = .true.
  integer i
  if (l_first_pass) then
    allocate(dx2(Natoms),dy2(Natoms),dz2(Natoms),disp2(Natoms))
    dx2=xxx; dy2=yyy; dz2=zzz; disp2=0.0d0
    l_first_pass=.false.
    l_update = .true.
     call adjust_box
     call cel_properties(.true.)
    RETURN
  endif
  l_update=.false.

    do i = 1, Natoms
       disp2(i)  = disp2(i) + (xxx(i)-dx2(i))**2+(yyy(i)-dy2(i))**2+(zzz(i)-dz2(i))**2
    enddo
    if (ANY(disp2(1:Natoms) > displacement**2)) then
       l_update = .true.
       disp2=0.0d0
       dx2=xxx; dy2=yyy; dz2=zzz;
       call adjust_box
       call cel_properties(.true.)
    endif
!  is_list_2BE_updated=l_update
  end subroutine is_list_2BE_updated


  subroutine get_instant_temperature(T_eval)
  use ALL_atoms_data, only : Natoms, vxx,vyy,vzz,all_atoms_mass, l_proceed_kin_atom
  use DOF_data
  use physical_constants, only : Red_Boltzmann_constant 
  use energies_data, only : en_kin
  implicit none
  real(8), intent(OUT) :: T_eval
  real(8)  En,sxx,syy,szz
  integer i

  en = 0.0d0
  do i = 1, Natoms
  if (l_proceed_kin_atom(i)) then 
    sxx = all_atoms_mass(i) * vxx(i)*vxx(i)
    syy = all_atoms_mass(i) * vyy(i)*vyy(i)
    szz = all_atoms_mass(i) * vzz(i)*vzz(i)
    en = en + sxx+syy+szz
  endif
  enddo
  en_kin = en * 0.5d0
  T_eval = en/(Red_Boltzmann_constant*DOF_total)
  end subroutine get_instant_temperature

  subroutine get_instant_MOL_temperature(T_trans,T_rot)
  use DOF_data
  use physical_constants, only : Red_Boltzmann_constant
  use ALL_rigid_mols_data, only : mol_MOM,mol_ANG, Inverse_Inertia_MAIN, Inverse_Molar_mass
  use ALL_mols_data, only : l_WALL_MOL_CTRL, Nmols
  real(8), intent(OUT) :: T_trans,T_rot
  integer i,j,k
  real(8) a_sq,t_sq
   t_sq = 0.0d0
   a_sq = 0.0d0
   do i = 1, Nmols
    if (.not.l_WALL_MOL_CTRL(i)) then
      t_sq = t_sq + dot_product(mol_MOM(i,:),mol_MOM(i,:))*Inverse_Molar_mass(i) ! twice kinetic energy
      a_sq = a_sq + dot_product(mol_ANG(i,:),mol_ANG(i,:)*Inverse_Inertia_MAIN(i,:) )
    endif
   enddo
!print*,'kinetic energy in get_instant_MOL_temperatur=',t_sq/2.0d0,a_sq/2.0d0
   T_trans = t_sq /(Red_Boltzmann_constant*DOF_MOL_trans)
   if (DOF_MOL_rot > 0 ) T_rot = a_sq / (Red_Boltzmann_constant*DOF_MOL_rot) 
  end subroutine get_instant_MOL_temperature

  subroutine get_mol_stresses
  use ALL_rigid_mols_data, only : mol_MOM,mol_ANG, inverse_Inertia_MAIN, Inverse_Molar_mass
  use ALL_mols_data, only : Nmols, l_WALL_MOL_CTRL
  use stresses_data, only : stress_kin,stress,pressure
  implicit none
  integer i
  real(8) sxx,sxy,sxz,syx,syy,syz,szx,szy,szz
     stress_kin=0.0d0
     do i = 1, Nmols
   if (.not.l_WALL_MOL_CTRL(i)) then
     sxx =  mol_MOM(i,1)*mol_MOM(i,1)*Inverse_Molar_mass(i)
     syy =  mol_MOM(i,2)*mol_MOM(i,2)*Inverse_Molar_mass(i)
     szz =  mol_MOM(i,3)*mol_MOM(i,3)*Inverse_Molar_mass(i)
     sxy =  mol_MOM(i,1)*mol_MOM(i,2)*Inverse_Molar_mass(i)
     sxz =  mol_MOM(i,1)*mol_MOM(i,3)*Inverse_Molar_mass(i)
     syz =  mol_MOM(i,2)*mol_MOM(i,3)*Inverse_Molar_mass(i)
     stress_kin(1) = stress_kin(1) + sxx
     stress_kin(2) = stress_kin(2) + syy
     stress_kin(3) = stress_kin(3) + szz
     stress_kin(5) = stress_kin(5) + sxy
     stress_kin(6) = stress_kin(6) + sxz
     stress_kin(7) = stress_kin(7) + syz
    endif
    enddo
     stress_kin(4) = sum(stress_kin(1:3))/3.0d0
     stress_kin(8) = stress_kin(5)
     stress_kin(9) = stress_kin(6)
     stress_kin(10) = stress_kin(7)

   end subroutine get_mol_stresses

  subroutine add_kinetic_pressure
    use stresses_data
     pressure = stress + stress_kin
  end subroutine add_kinetic_pressure

  subroutine get_kinetic_energy_stress(T_eval)
  use stresses_data, only : stress_kin
  use energies_data, only : en_kin
  use profiles_data, only : atom_profile, l_need_2nd_profile
  use ALL_atoms_data, only : Natoms, vxx,vyy,vzz,all_atoms_mass,l_proceed_kin_atom
  use physical_constants, only : Red_Boltzmann_constant
  use DOF_data

  implicit none
  real(8), intent(OUT) :: T_eval
  integer i,j,k
  real(8)  sxx, sxy, sxz, syy, syz, szz, En, mass
  

    stress_kin=0.0d0
    en_kin = 0.0d0
    if (l_need_2nd_profile) then
     do i = 1, Natoms
     if (l_proceed_kin_atom(i))then
       mass = all_atoms_mass(i)
       sxx = mass*vxx(i)*vxx(i)
       syy = mass*vyy(i)*vyy(i)
       szz = mass*vzz(i)*vzz(i)
       sxy = mass*vxx(i)*vyy(i)
       sxz = mass*vxx(i)*vzz(i)
       syz = mass*vyy(i)*vzz(i)
       stress_kin(1) = stress_kin(1) + sxx
       stress_kin(2) = stress_kin(2) + syy
       stress_kin(3) = stress_kin(3) + szz

       stress_kin(5) = stress_kin(5) + sxy
       stress_kin(6) = stress_kin(6) + sxz
       stress_kin(7) = stress_kin(7) + syz
       En = 0.5d0*(sxx+syy+szz)
       en_kin = en_kin + En

       atom_profile(i)%kin = En
     endif ! l_proceed_kin_atom
     enddo 
     else
     do i = 1, Natoms
     if (l_proceed_kin_atom(i)) then 
       mass = all_atoms_mass(i)
       sxx = mass*vxx(i)*vxx(i)
       syy = mass*vyy(i)*vyy(i)
       szz = mass*vzz(i)*vzz(i)
       sxy = mass*vxx(i)*vyy(i)
       sxz = mass*vxx(i)*vzz(i)
       syz = mass*vyy(i)*vzz(i)
       stress_kin(1) = stress_kin(1) + sxx
       stress_kin(2) = stress_kin(2) + syy
       stress_kin(3) = stress_kin(3) + szz

       stress_kin(5) = stress_kin(5) + sxy
       stress_kin(6) = stress_kin(6) + sxz
       stress_kin(7) = stress_kin(7) + syz
       
       en_kin = en_kin + 0.5d0*(sxx+syy+szz)
     endif
     enddo
     endif
     stress_kin(4) = sum(stress_kin(1:3))/3.0d0
     stress_kin(8) = stress_kin(5)
     stress_kin(9) = stress_kin(6)
     stress_kin(10) = stress_kin(7)

     T_eval = 2.0d0*en_kin/(Red_Boltzmann_constant*DOF_total)

  end subroutine get_kinetic_energy_stress

  subroutine get_kinetic_energy
  use stresses_data, only : stress_kin
  use energies_data, only : en_kin
  use profiles_data, only : atom_profile, l_need_2nd_profile
  use ALL_atoms_data, only : Natoms, vxx,vyy,vzz,all_atoms_mass,l_proceed_kin_atom
  implicit none
  integer i,j,k
  real(8)  En,mass

    en_kin = 0.0d0
     if (l_need_2nd_profile) then
     do i = 1, Natoms
     if (l_proceed_kin_atom(i))then
       mass = all_atoms_mass(i)
       En = mass * (vxx(i)*vxx(i)+vyy(i)*vyy(i)+vzz(i)*vzz(i))*0.5d0
       en_kin = en_kin + En
       atom_profile(i)%kin = En
     endif
     enddo
     else
       do i = 1, Natoms
       if(l_proceed_kin_atom(i)) then
       mass = all_atoms_mass(i)
       en_kin = en_kin + mass * (vxx(i)*vxx(i)+vyy(i)*vyy(i)+vzz(i)*vzz(i))*0.5d0
       endif
       enddo
     endif
  
  end subroutine get_kinetic_energy

   subroutine mol_kinetics
   use energies_data
   use ALL_rigid_mols_data, only : mol_MOM,mol_ANG, inverse_Inertia_MAIN, Inverse_Molar_mass
   use stresses_data, only : stress_kin
   use ALL_mols_data, only : Nmols, l_WALL_MOL_CTRL
   implicit none
   integer i,j,k
   real(8) sxx,sxy,sxz,syx,syy,syz,szx,szy,szz
   K_energy_translation = 0.0d0; 
   K_energy_rotation    = 0.0d0
   do i = 1, Nmols
   if (.not.l_WALL_MOL_CTRL(i)) then
     K_energy_translation(:)= K_energy_translation(:) + mol_MOM(i,:)*mol_MOM(i,:)*Inverse_Molar_mass(i)
     K_energy_rotation(:)=K_energy_rotation(:) + mol_ANG(i,:)*mol_ANG(i,:)*inverse_Inertia_MAIN(i,:)
   else
    mol_MOM(i,:) = 0.0d0
    mol_ANG(i,:) = 0.0d0
   endif
   enddo
   En_kin_translation=0.5d0*(SUM(K_energy_translation(1:3)))
   En_kin_rotation=0.5d0*(SUM(K_energy_rotation(1:3)))
   en_kin = En_kin_translation + En_kin_rotation
! Now do the stress_kin
   stress_kin=0.0d0
   do i = 1, Nmols
   if (.not.l_WALL_MOL_CTRL(i)) then
     sxx =  mol_MOM(i,1)*mol_MOM(i,1)*Inverse_Molar_mass(i)
     syy =  mol_MOM(i,2)*mol_MOM(i,2)*Inverse_Molar_mass(i)
     szz =  mol_MOM(i,3)*mol_MOM(i,3)*Inverse_Molar_mass(i)
     sxy =  mol_MOM(i,1)*mol_MOM(i,2)*Inverse_Molar_mass(i)
     sxz =  mol_MOM(i,1)*mol_MOM(i,3)*Inverse_Molar_mass(i)
     syz =  mol_MOM(i,2)*mol_MOM(i,3)*Inverse_Molar_mass(i)
     stress_kin(1) = stress_kin(1) + sxx
     stress_kin(2) = stress_kin(2) + syy
     stress_kin(3) = stress_kin(3) + szz
     stress_kin(5) = stress_kin(5) + sxy
     stress_kin(6) = stress_kin(6) + sxz
     stress_kin(7) = stress_kin(7) + syz
    endif
    enddo

     stress_kin(4) = sum(stress_kin(1:3))/3.0d0
     stress_kin(8) = stress_kin(5)
     stress_kin(9) = stress_kin(6)
     stress_kin(10) = stress_kin(7)
 
   end subroutine mol_kinetics


  subroutine get_kinetic_stress
  use stresses_data, only : stress_kin
  use energies_data, only : en_kin
  use profiles_data, only : atom_profile, l_need_2nd_profile
  use ALL_atoms_data, only : Natoms, vxx,vyy,vzz,all_atoms_mass,l_proceed_kin_atom
  implicit none
  integer i,j,k
  real(8)  sxx, sxy, sxz, syy, syz, szz,mass

    stress_kin=0.0d0
     if (l_need_2nd_profile) then
     do i = 1, Natoms
!       mass = all_atoms_mass(i)
!       sxx = mass*vxx(i)*vxx(i)
!       syy = mass*vyy(i)*vyy(i)
!       szz = mass*vzz(i)*vzz(i)
!       sxy = mass*vxx(i)*vyy(i)
!       sxz = mass*vxx(i)*vzz(i)
!       syz = mass*vyy(i)*vzz(i)
 !      stress_kin(1) = stress_kin(1) + sxx
 !      stress_kin(2) = stress_kin(2) + sxy
!       stress_kin(3) = stress_kin(3) + sxz
!       stress_kin(5) = stress_kin(5) + syy
!       stress_kin(6) = stress_kin(6) + syz
!       stress_kin(9) = stress_kin(9) + szz
!       atom_profile(i)%sxx = atom_profile(i)%sxx + sxx
!       atom_profile(i)%syy = atom_profile(i)%syy + syy
!       atom_profile(i)%szz = atom_profile(i)%szz + szz
!       atom_profile(i)%sxy = atom_profile(i)%sxy + sxy
!       atom_profile(i)%sxz = atom_profile(i)%sxz + sxz
!       atom_profile(i)%syz = atom_profile(i)%syz + sxz
!       atom_profile(i)%syx = atom_profile(i)%syx + sxy
!       atom_profile(i)%szy = atom_profile(i)%szy + syz
!       atom_profile(i)%szx = atom_profile(i)%szx + sxz
     enddo
     else
     do i = 1, Natoms
     if(l_proceed_kin_atom(i))then
       mass = all_atoms_mass(i)
       stress_kin(1) = stress_kin(1) + mass*vxx(i)*vxx(i)
       stress_kin(2) = stress_kin(2) + mass*vxx(i)*vyy(i)
       stress_kin(3) = stress_kin(3) + mass*vxx(i)*vzz(i)
       stress_kin(5) = stress_kin(5) + mass*vyy(i)*vyy(i)
       stress_kin(6) = stress_kin(6) + mass*vyy(i)*vzz(i)
       stress_kin(9) = stress_kin(9) + mass*vzz(i)*vzz(i)
     endif
     enddo
     endif
     stress_kin(4) = stress_kin(2)
     stress_kin(7) = stress_kin(3)
     stress_kin(8) = stress_kin(6)

  end subroutine get_kinetic_stress
 
end module kinetics
