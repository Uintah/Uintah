
  module z_profiles
  implicit none

  public :: driver_profiles
  public :: first_profile_collect
  public :: eval_profiles
  public :: write_profiles
    
  contains

    subroutine driver_profiles
     use collect_data
     use integrate_data, only : integration_step
     use profiles_data
     implicit none
     logical l_collect,l_print

      l_collect = mod(integration_step,collect_skip)==0
      l_print = mod(integration_step, collect_length)
      if (l_collect) then
         di_collections = di_collections + 1.0d0
         if (l_need_1st_profile) call first_profile_collect
!         if (l_need_2nd_profile) call second_profile_collect   to be implemented
      endif
      if (l_print) then
         call write_profiles
      endif

    end subroutine driver_profiles

    subroutine first_profile_collect
    use mol_utils
    use profiles_data
    use ALL_atoms_data, only : zz
    use ALL_mols_data, only :  all_mol_G_charges,all_mol_P_charges,mol_zzz,Nmols,i_type_molecule
    use sim_cel_data
    
 
    implicit none
    integer i,j,k,NB,itype
    real(8) z, inv_vol,q
 
     call get_all_mol_CM      
     call get_all_mol_charge
     BIN_dZ = dble(N_BINS_ZZ) / sim_cel(9)
     inv_vol = BIN_dZ / Area_xy
     do i  = 1, Nmols
       z = mol_zzz(i)
       itype = i_type_molecule(i)
       NB = INT((Z+0.5d0*sim_cel(9))*BIN_dZ) + 1
       counter_MOLS_1_global(NB,itype) =  counter_MOLS_1_global(NB,itype) + 1.0d0
       zp_mol_density(NB,itype) =  zp_mol_density(NB,itype) +   inv_vol
       q = (all_mol_G_charges(i)+all_mol_P_charges(i))
       zp_charge(NB) = zp_charge(NB) +  (all_mol_G_charges(i)+all_mol_P_charges(i))
       if (q > 0.0d0) then
         zp_charge_plus(NB) = zp_charge_plus(NB) + q
       else if (q < 0.0d0) then
         zp_charge_minus(NB) = zp_charge_minus(NB) + q
       endif
     enddo

    end subroutine first_profile_collect
  
    subroutine eval_profiles
     use profiles_data
     use collect_data
       implicit none
       real(8) idi
       idi = 1.0d0 / di_collections
       RA_zp_mol_density = zp_mol_density * idi
       RA_zp_charge = zp_charge * idi
       RA_zp_charge_plus = zp_charge_plus * idi
       RA_zp_charge_minus = zp_charge_minus * idi
    end subroutine eval_profiles 

    subroutine write_profiles
    use file_names_data, only : name_density_profile_file,name_charge_profile_file
    use profiles_data
    implicit none
    integer i

      open(unit=333,file=trim(name_density_profile_file),recl=300)
      do i = 1, N_BINS_ZZ
        write(333,*) Z_scale(i), RA_zp_mol_density(:,i)
      enddo
      close(333)
      open(unit=333,file=trim(name_charge_profile_file),recl=300)
      do i = 1, N_BINS_ZZ
        write(333,*) Z_scale(i), RA_zp_charge_plus(i),RA_zp_charge_minus(i),RA_zp_charge(i)
      enddo
      close(333)
 
    end subroutine write_profiles

  end module z_profiles 
