
 module statistics_eval

 implicit none

 ! 1st order statistics: Temperature | density | charge + and - | charge mod(+,-)| charge + | charge -; 
public :: Z_profiles_evals

 contains

subroutine Z_profiles_evals
! don t forget to eval some MOL properties
use profiles_data
use all_atoms_data, only : zz,xx,yy,xxx,yyy,zzz,fxx,fyy,fzz,Natoms, i_type_atom,l_WALL_CTRL, &
                           all_p_charges,all_g_charges, is_charge_distributed,&
                           all_dipoles_xx,all_dipoles_yy,all_dipoles_zz,all_dipoles,is_dummy,&
                           is_sfield_constrained, atom_dof
use all_mols_data
use all_rigid_mols_data
use physical_constants
use sim_cel_data
use atom_type_data, only : N_TYPE_ATOMS,statistics_AtomPair_type
use mol_type_data, only : N_type_molecules,statistics_MolPair_type
use boundaries, only : adjust_box,cel_properties
use mol_utils, only : get_all_mol_properties
use CTRLs_data, only : l_DIP_CTRL
use rsmd_data
use integrate_data, only : l_do_QN_CTRL, integration_step
use sizes_data, only :  N_type_atoms_for_statistics, N_type_mols_for_statistics
use collect_data

implicit none
real(8) inv_dZ, inv_dz_s, dz,dV_1,dv_1_small,z,itype,inv_dx,inv_dy,dv_1_small_x,dv_1_small_y
integer i,j,k,N,N1,NB,NBs,NN1,NBx,NBy
real(8) dof,tmp
real(8) re_center
real(8) cos_i,cos_i2,cos_i3,cos_i4,cos_i6,mu, temp
real(8),allocatable :: mol_zz(:)
integer , allocatable :: NNB(:),NNBs(:),itypevct(:),NNBx(:),NNBy(:)
real(8) , allocatable :: V1(:,:), local_zz2(:)
logical l_proceed

allocate(V1(Nmols,10),local_zz2(Nmols))
   call adjust_box
   call cel_properties(.true.)

   call get_all_mol_properties

where (is_sfield_constrained)
  RA_fi(:) = RA_fi(:) + atom_profile(:)%fi
endwhere
  RA_fi_counts  = RA_fi_counts + 1
!print*,'ara =', atom_profile(1)%fi/Volt_to_internal_field, atom_profile(500)%fi/Volt_to_internal_field
!print*,'ra=',RA_fi(1)/RA_fi_counts /Volt_to_internal_field,RA_fi(500)/RA_fi_counts/Volt_to_internal_field


inv_dZ = dble(N_BINS_ZZ-1) / sim_cel(9) 
inv_dx = dble(N_BINS_XX-1) / sim_cel(1)
inv_dy = dble(N_BINS_YY-1) / sim_cel(5)

inv_dZ_s = dble(N_BINS_ZZs-1) / sim_cel(9)
dz = 1.0d0/inv_dz
dV_1 = 1.0d0 / (Volume)
dv_1_small = dv_1*dble(N_BINS_ZZ) ! only the volume of a small little piece!
dv_1_small_x =dv_1  * dble(N_BINS_XX)
dv_1_small_y = dv_1 * dble(N_BINS_YY)
allocate(NNB(Natoms),itypevct(Natoms),NNBs(Natoms),NNBx(Natoms),NNBy(Natoms))
do i = 1, Natoms
   j = i_type_atom(i)
   itype = statistics_AtomPair_type(j) ! statistics_pair_type < 0 if no statistics to be done
   l_proceed = itype > 0
   z = zz(i) ! zz is centered around 0
   NB = INT((Z+0.5d0*sim_cel(9))*inv_dZ) + 1
   NBx = INT((xx(i)+0.5d0*sim_cel(1))*inv_dx) + 1
   NBy = INT((yy(i)+0.5d0*sim_cel(5))*inv_dy) + 1
   
   if (NB==N_BINS_ZZ+1) NB=N_BINS_ZZ  ! LAST ONE IF NASTY map it back... it will not happen and if it 
   if (NBx==N_BINS_xx+1) NB=1
   if (NBy==N_BINS_yy+1) NB=1
   NNB(i) = NB
   NNBx(i) = NBx
   NNBy(i) = NBy
   if (NB > N_BINS_ZZ+1) then
       print*, 'Something terrible bad in Z_profiles_evals: STOP NB>MAXBINS',NB, N_BINS_ZZ
       STOP 
   endif
   if (NB < 1) then
        print*, 'Something terrible bad in Z_profiles_evals: STOP NB < 1',NB,zzz(i),zz(i),i,mol_xyz(1,:)
        STOP
   endif
   if (NBx < 1) then
        print*, 'Something terrible bad in Z_profiles_evals: STOP NBx < 1',NBx,xxx(i),xx(i),i,mol_xyz(1,:)
        STOP
   endif
   if (NBy < 1) then
        print*, 'Something terrible bad in Z_profiles_evals: STOP NBy < 1',NBy,yyy(i),yy(i),i,mol_xyz(1,:)
        STOP
   endif

   if (l_proceed)  then 
       counter_ATOMS_global(NB,itype) = counter_ATOMS_global(NB,itype) + 1
       counter_ATOMS_global_x(NBx,itype) = counter_ATOMS_global_x(NBx,itype) + 1
       counter_ATOMS_global_y(NBy,itype) = counter_ATOMS_global_y(NBy,itype) + 1
   endif
enddo

if (l_need_1st_profile) then
do i = 1,Natoms
   NB = NNB(i)
   j = i_type_atom(i)
   itype = statistics_AtomPair_type(j)
   l_proceed = itype > 0
   zp1_atom(NB)%kin = zp1_atom(NB)%kin + atom_profile(i)%kin
   dof = atom_dof(i)
   zp1_atom(NB)%DOF = zp1_atom(NB)%DOF + dof
   if (l_proceed) zp1_atom(NB)%density(itype) = zp1_atom(NB)%density(itype) + dv_1_small
   zp1_atom(NB)%p_charge(1) = zp1_atom(NB)%p_charge(1) + all_p_charges(i)
   zp1_atom(NB)%p_charge(4) = zp1_atom(NB)%p_charge(4) + dabs(all_p_charges(i))
   if (all_p_charges(i) > 0.0d0) then
       zp1_atom(NB)%p_charge(2) = zp1_atom(NB)%p_charge(2) + all_p_charges(i)
   else
       zp1_atom(NB)%p_charge(3) = zp1_atom(NB)%p_charge(3) + all_p_charges(i)
   endif
       zp1_atom(NB)%g_charge(1) = zp1_atom(NB)%g_charge(1) + all_g_charges(i)
       zp1_atom(NB)%g_charge(4) = zp1_atom(NB)%g_charge(4) + dabs(all_g_charges(i))
   if (all_p_charges(i) > 0.0d0) then
       zp1_atom(NB)%g_charge(2) = zp1_atom(NB)%g_charge(2) + all_g_charges(i)
   else
       zp1_atom(NB)%g_charge(3) = zp1_atom(NB)%g_charge(3) + all_g_charges(i)
   endif
enddo 


do i = 1,Natoms
   NB = NNBx(i)
   j = i_type_atom(i)
   itype = statistics_AtomPair_type(j)
   l_proceed = itype > 0
   zp1_atom_x(NB)%kin = zp1_atom_x(NB)%kin + atom_profile(i)%kin
   dof = atom_dof(i)
   zp1_atom_x(NB)%DOF = zp1_atom_x(NB)%DOF + dof
   if (l_proceed) zp1_atom_x(NB)%density(itype) = zp1_atom_x(NB)%density(itype) + dv_1_small_x
   zp1_atom_x(NB)%p_charge(1) = zp1_atom_x(NB)%p_charge(1) + all_p_charges(i)
   zp1_atom_x(NB)%p_charge(4) = zp1_atom_x(NB)%p_charge(4) + dabs(all_p_charges(i))
   if (all_p_charges(i) > 0.0d0) then
       zp1_atom_x(NB)%p_charge(2) = zp1_atom_x(NB)%p_charge(2) + all_p_charges(i)
   else
       zp1_atom_x(NB)%p_charge(3) = zp1_atom_x(NB)%p_charge(3) + all_p_charges(i)
   endif
       zp1_atom_x(NB)%g_charge(1) = zp1_atom_x(NB)%g_charge(1) + all_g_charges(i)
       zp1_atom_x(NB)%g_charge(4) = zp1_atom_x(NB)%g_charge(4) + dabs(all_g_charges(i))
   if (all_p_charges(i) > 0.0d0) then
       zp1_atom_x(NB)%g_charge(2) = zp1_atom_x(NB)%g_charge(2) + all_g_charges(i)
   else
       zp1_atom_x(NB)%g_charge(3) = zp1_atom_x(NB)%g_charge(3) + all_g_charges(i)
   endif
enddo

do i = 1,Natoms
   NB = NNBy(i)
   j = i_type_atom(i)
   itype = statistics_AtomPair_type(j)
   l_proceed = itype > 0
   zp1_atom_y(NB)%kin = zp1_atom_y(NB)%kin + atom_profile(i)%kin
   dof = atom_dof(i)
   zp1_atom_y(NB)%DOF = zp1_atom_y(NB)%DOF + dof
   if (l_proceed) zp1_atom_y(NB)%density(itype) = zp1_atom_y(NB)%density(itype) + dv_1_small_y
   zp1_atom_y(NB)%p_charge(1) = zp1_atom_y(NB)%p_charge(1) + all_p_charges(i)
   zp1_atom_y(NB)%p_charge(4) = zp1_atom_y(NB)%p_charge(4) + dabs(all_p_charges(i))
   if (all_p_charges(i) > 0.0d0) then
       zp1_atom_y(NB)%p_charge(2) = zp1_atom_x(NB)%p_charge(2) + all_p_charges(i)
   else
       zp1_atom_y(NB)%p_charge(3) = zp1_atom_x(NB)%p_charge(3) + all_p_charges(i)
   endif
       zp1_atom_y(NB)%g_charge(1) = zp1_atom_y(NB)%g_charge(1) + all_g_charges(i)
       zp1_atom_y(NB)%g_charge(4) = zp1_atom_y(NB)%g_charge(4) + dabs(all_g_charges(i))
   if (all_p_charges(i) > 0.0d0) then
       zp1_atom_y(NB)%g_charge(2) = zp1_atom_y(NB)%g_charge(2) + all_g_charges(i)
   else
       zp1_atom_y(NB)%g_charge(3) = zp1_atom_y(NB)%g_charge(3) + all_g_charges(i)
   endif
enddo


if (l_DIP_CTRL) then
do i = 1, Natoms
 NB = NNB(i)
 j = i_type_atom(i)
 itype = statistics_AtomPair_type(j)
   if (is_charge_distributed(i)) then
   zp1_atom(NB)%g_dipole(1) = zp1_atom(NB)%g_dipole(1) + all_dipoles_xx(i)
   zp1_atom(NB)%g_dipole(2) = zp1_atom(NB)%g_dipole(2) + all_dipoles_yy(i)
   zp1_atom(NB)%g_dipole(3) = zp1_atom(NB)%g_dipole(3) + all_dipoles_zz(i)
   zp1_atom(NB)%g_dipole(4) = zp1_atom(NB)%g_dipole(4) + all_dipoles(i)
   else
   zp1_atom(NB)%p_dipole(1) = zp1_atom(NB)%p_dipole(1) + all_dipoles_xx(i)
   zp1_atom(NB)%p_dipole(2) = zp1_atom(NB)%p_dipole(2) + all_dipoles_yy(i)
   zp1_atom(NB)%p_dipole(3) = zp1_atom(NB)%p_dipole(3) + all_dipoles_zz(i)
   zp1_atom(NB)%p_dipole(4) = zp1_atom(NB)%p_dipole(4) + all_dipoles(i)
   endif
enddo

do i = 1, Natoms
 NB = NNBx(i)
 j = i_type_atom(i)
 itype = statistics_AtomPair_type(j)
   if (is_charge_distributed(i)) then
   zp1_atom_x(NB)%g_dipole(1) = zp1_atom_x(NB)%g_dipole(1) + all_dipoles_xx(i)
   zp1_atom_x(NB)%g_dipole(2) = zp1_atom_x(NB)%g_dipole(2) + all_dipoles_yy(i)
   zp1_atom_x(NB)%g_dipole(3) = zp1_atom_x(NB)%g_dipole(3) + all_dipoles_zz(i)
   zp1_atom_x(NB)%g_dipole(4) = zp1_atom_x(NB)%g_dipole(4) + all_dipoles(i)
   else
   zp1_atom_x(NB)%p_dipole(1) = zp1_atom_x(NB)%p_dipole(1) + all_dipoles_xx(i)
   zp1_atom_x(NB)%p_dipole(2) = zp1_atom_x(NB)%p_dipole(2) + all_dipoles_yy(i)
   zp1_atom_x(NB)%p_dipole(3) = zp1_atom_x(NB)%p_dipole(3) + all_dipoles_zz(i)
   zp1_atom_x(NB)%p_dipole(4) = zp1_atom_x(NB)%p_dipole(4) + all_dipoles(i)
   endif
enddo
do i = 1, Natoms
 NB = NNB(i)
 j = i_type_atom(i)
 itype = statistics_AtomPair_type(j)
   if (is_charge_distributed(i)) then
   zp1_atom_y(NB)%g_dipole(1) = zp1_atom_y(NB)%g_dipole(1) + all_dipoles_xx(i)
   zp1_atom_y(NB)%g_dipole(2) = zp1_atom_y(NB)%g_dipole(2) + all_dipoles_yy(i)
   zp1_atom_y(NB)%g_dipole(3) = zp1_atom_y(NB)%g_dipole(3) + all_dipoles_zz(i)
   zp1_atom_y(NB)%g_dipole(4) = zp1_atom_y(NB)%g_dipole(4) + all_dipoles(i)
   else
   zp1_atom_y(NB)%p_dipole(1) = zp1_atom_y(NB)%p_dipole(1) + all_dipoles_xx(i)
   zp1_atom_y(NB)%p_dipole(2) = zp1_atom_y(NB)%p_dipole(2) + all_dipoles_yy(i)
   zp1_atom_y(NB)%p_dipole(3) = zp1_atom_y(NB)%p_dipole(3) + all_dipoles_zz(i)
   zp1_atom_y(NB)%p_dipole(4) = zp1_atom_y(NB)%p_dipole(4) + all_dipoles(i)
   endif
enddo

endif ! l_DIP_CTRL

endif ! l_need_1st_profile
 
if (l_need_2nd_profile) then
do i = 1, Natoms
 NB = NNB(i)
   j = i_type_atom(i)
   itype = statistics_AtomPair_type(j)
   l_proceed = itype > 0
   if (l_proceed) then
    zp2_atom(NB)%pot(itype) = zp2_atom(NB)%pot(itype) + atom_profile(i)%pot
    zp2_atom(NB)%fi(itype)  = zp2_atom(NB)%fi(itype)  + atom_profile(i)%fi
    zp2_atom(NB)%Qpot(itype)  = zp2_atom(NB)%Qpot(itype)  + atom_profile(i)%Qpot
   endif
enddo
endif


 allocate(mol_zz(Nmols))
 if (i_boundary_CTRL==0.or.i_boundary_CTRL==1) then 
   re_center = re_center_ZZ
   mol_zz(:) = mol_xyz(:,3) - re_center/2.0d0
 else if  (i_boundary_CTRL==2.or.i_boundary_CTRL==3) then
   mol_zz(:) = mol_xyz(:,3) - &
   sim_cel(9)*(dble(INT(2.0d0*(mol_xyz(:,3)/sim_cel(9)))) -dble(INT((mol_xyz(:,3)/sim_cel(9)))) )
 else
print*, 'in statistic not implemented yet i_type_boundary_CTRL=',i_boundary_CTRL
STOP
 endif

deallocate(itypevct,NNB,NNBs)
allocate(NNB(Nmols),itypevct(Nmols),NNBs(Nmols))

do i = 1, Nmols
  j = i_type_molecule(i)
  itype = statistics_MolPair_type(j)
  l_proceed = itype > 0
  z = mol_zz(i)
  NB = INT((Z+0.5d0*sim_cel(9))*inv_dZ) + 1    ;  NB = min(NB,N_BINS_ZZ)
  NBs = INT((Z+0.5d0*sim_cel(9))*inv_dZ_s) + 1 ; NBs = min(NB,N_BINS_ZZs)
  itypevct(i) = itype
  NNBs(i) = NBs
  NNB(i) = NB
   if (NB < 1) then
       print*, ' Code crashes : in Z_profiles_evals: STOP NB < 1',NB
       STOP
   endif
 if (l_proceed) counter_MOLS_global(NB,itype) = counter_MOLS_global(NB,itype) + 1  
enddo

if (l_need_1st_profile) then
do i = 1, Nmols
  NB=NNB(i)
  j = i_type_molecule(i)
  itype = statistics_MolPair_type(j)
  l_proceed = itype > 0
    if (l_proceed) zp1_mol(NB)%density(itype) = zp1_mol(NB)%density(itype) + dv_1_small
    zp1_mol(NB)%p_charge(1) = zp1_mol(NB)%p_charge(1) + all_mol_p_charges(i)
    zp1_mol(NB)%p_charge(4) = zp1_mol(NB)%p_charge(4) + dabs(all_mol_p_charges(i))
    if (all_mol_p_charges(i) > 0.0d0) then
       zp1_mol(NB)%p_charge(2) = zp1_mol(NB)%p_charge(2) + all_mol_p_charges(i)
    else
       zp1_mol(NB)%p_charge(3) = zp1_mol(NB)%p_charge(3) + all_mol_p_charges(i)
    endif 
       zp1_mol(NB)%g_charge(1) = zp1_mol(NB)%g_charge(1) + all_mol_g_charges(i) 
       zp1_mol(NB)%g_charge(4) = zp1_mol(NB)%g_charge(4) + dabs(all_mol_g_charges(i))
    if (all_mol_p_charges(i) > 0.0d0) then
       zp1_mol(NB)%g_charge(2) = zp1_mol(NB)%g_charge(2) + all_mol_g_charges(i)
    else
       zp1_mol(NB)%g_charge(3) = zp1_mol(NB)%g_charge(3) + all_mol_g_charges(i)
    endif 
   if (l_proceed) then
     mu = dsqrt(dot_product(mol_dipole(i,1:3),mol_dipole(i,1:3)))
     if (mu < 1.0d-10) then
       cos_i = 0.0d0
     else
       cos_i = mol_dipole(i,3)/dsqrt(dot_product(mol_dipole(i,1:3),mol_dipole(i,1:3))) ! the director is OZ
     endif
     cos_i2 = cos_i*cos_i   ; cos_i4 = cos_i2*cos_i2; cos_i6 = cos_i2*cos_i4
     cos_i3 = cos_i2*cos_i
     zp1_mol(NB)%OP(itype,1) =  zp1_mol(NB)%OP(itype,1) + cos_i
     zp1_mol(NB)%OP(itype,2) =  zp1_mol(NB)%OP(itype,2) + (3.0d0*cos_i2-1.0d0)*0.5d0
     zp1_mol(NB)%OP(itype,3) =  zp1_mol(NB)%OP(itype,3) + (5.0d0*cos_i3-3.0d0*cos_i)*0.5d0
     zp1_mol(NB)%OP(itype,4) =  zp1_mol(NB)%OP(itype,4) + (35.0d0*cos_i4-30.0d0*cos_i2+3.0d0)*0.125d0
     zp1_mol(NB)%OP(itype,5) =  zp1_mol(NB)%OP(itype,5) + (63.0d0*cos_i4*cos_i-70.0d0*cos_i3+15.0d0*cos_i)*0.125d0
     zp1_mol(NB)%OP(itype,6) =  zp1_mol(NB)%OP(itype,6) + (693.0d0*cos_i6-945*cos_i4+315*cos_i2-15.0d0)/48.0d0
   endif !l_proceed
enddo


     local_zz2(:) = (mol_xyz(:,3))**2 + (mol_xyz(:,2))**2 + (mol_xyz(:,1))**2
     rmsd_xyz_med_2(:) = rmsd_xyz_med_2(:) + local_zz2(:)
     rmsd_xyz_med(:,1:3) = rmsd_xyz_med(:,1:3) +  mol_xyz(:,1:3)

   if (mod(integration_step,collect_length)==0) then
    do i = 1, Nmols
     NB=NNBs(i)
     j = i_type_molecule(i)
     itype = statistics_MolPair_type(j)
     if (itype>0)then
       tmp = dot_product(rmsd_xyz_med(i,1:3),rmsd_xyz_med(i,1:3))
       zp_translate_cryterion(NB,itype) = (rmsd_xyz_med_2(i) - tmp/di_collections_short)/di_collections_short
     endif
    enddo

   endif

     if (l_do_QN_CTRL) then ! quaternions
        rmsd_qn_med(:,:) = rmsd_qn_med(:,:) + qn(:,:)
!       V1(:,1:4) = qn(:,1:4)**2
!       V1(:,5) = qn(:,1)*qn(:,2)
!       V1(:,6) = qn(:,1)*qn(:,3)
!       V1(:,7) = qn(:,1)*qn(:,4)
!       V1(:,8) = qn(:,2)*qn(:,3)
!       V1(:,9) = qn(:,2)*qn(:,4)
!       V1(:,10) = qn(:,3)*qn(:,4)
!       rmsd_qn_med_2(:,1:10) =  rmsd_qn_med_2(:,1:10) + V1(:,1:10)
!       do i = 1, Nmols
!         NB=NNBs(i)
!         itype = itypevct(i)
!         zp_rmsd_qn_med_2(NB,1:10,itype) = zp_rmsd_qn_med_2(NB,1:10,itype) + V1(i,1:10)
!         zp_rmsd_qn_med(NB,1:4,itype) = qn(i,1:4)
!       enddo 
     endif

  
endif ! (l_need_1st_profile) 

if (l_need_2nd_profile) then
do i = 1, Nmols
  NB=NNB(i)
  j = i_type_molecule(i)
  itype = statistics_MolPair_type(j)
  l_proceed = itype > 0
  if(l_proceed) then 
     zp2_mol(NB)%pot(itype) = zp2_mol(NB)%pot(itype) + mol_potential(i)
     zp2_mol(NB)%Qpot(itype)  = zp2_mol(NB)%Qpot(itype)  +  mol_potential_Q(i) 
!    zp2_mol(NB)%stress(itype,1:3) = zp2_mol(NB)%stress(itype,1:3) + mol_pressure(i,1:3) * dV_1_small
!    zp2_mol(NB)%stress(itype,4) = zp2_mol(NB)%stress(itype,4) + sum(mol_pressure(i,1:3)) * dV_1_small/3.0d0
!    zp2_mol(NB)%stress(itype,5:10) = zp2_mol(NB)%stress(itype,5:10) + mol_pressure(i,5:10) * dV_1_small
!    zp2_mol(NB)%superficial_tension(1) = zp2_mol(NB)%superficial_tension(1) + &
!    (mol_pressure(i,3)-0.5d0*(mol_pressure(i,1)+mol_pressure(i,2) ) ) * dV_1_small * dz
!    zp2_mol(NB)%superficial_tension(2) = zp2_mol(NB)%superficial_tension(2) + &
!    (mol_pressure(i,3)-0.5d0*(mol_pressure(i,1)+mol_pressure(i,2) ) ) * dV_1_small
    zp2_mol(NB)%force(itype,1:3) = zp2_mol(NB)%force(itype,1:3) + mol_force(i,1:3)
    zp2_mol(NB)%force(itype,4) = zp2_mol(NB)%force(itype,4) + dot_product(mol_force(i,1:3),mol_force(i,1:3))   
  endif ! l_proceed
enddo
endif

deallocate(V1,local_zz2)
deallocate(mol_zz)
deallocate(NNB,itypevct,NNBs)
deallocate(NNBx,NNBy)
end subroutine Z_profiles_evals

end  module statistics_eval


module scalar_statistics_eval
 use generic_statistics_module
  implicit none
 private :: stress_counter_details
 public :: update_scalar_statistics
  contains
   subroutine update_scalar_statistics
   use rolling_averages_data
   use energies_data 
   use stresses_data
   use ensamble_data, only : T_eval, Temperature_trans_Calc,Temperature_rot_Calc
   use generic_statistics_module
   use ALL_atoms_data, only : xxx,yyy,zzz,xx,yy,zz,all_p_charges,all_g_charges,all_charges,&
                              all_dipoles_xx,all_dipoles_yy,all_dipoles_zz,all_dipoles,&
                              fxx,fyy,fzz,all_atoms_mass,vxx,vyy,vzz,&
                              is_sfield_constrained, is_dipole_polarizable,&
                              Natoms
   use field_constrain_data, only : N_atoms_field_constrained, N_dipol_polarizable
   use integrate_data, only : l_do_QN_CTRL
   use ALL_rigid_mols_data, only : mol_MOM
   use profiles_data, only : l_need_2nd_profile,atom_profile
   use physical_constants, only : Volt_to_internal_field
   use saved_energies, only : update_other_energies

   implicit none
   real(8) vct(100) 
   logical, save :: l_very_first_pas=.true.
   integer i,j,k,i11,i11_0,i11_1

   call update_other_energies
   vct(1:17)=energy(1:17)

   call stress_counter_details
   call statistics_5_eval(RA_energy(1:17),   vct(1:17))
   call statistics_5_eval(RA_stress(1:10),   stress(1:10))
   call statistics_5_eval(RA_pressure(1:10), pressure(1:10))
   call statistics_5_eval(RA_Temperature, T_eval)
   if (N_atoms_field_constrained>0) then
   if (l_need_2nd_profile) then
    vct(1:4) = 0.0d0 ; i11=0
    do i = 1, Natoms
     if (is_sfield_constrained(i)) then
      i11=i11+1
      if (i11<N_atoms_field_constrained/2+1) then
        vct(1) = vct(1) + all_charges(i)
      else
        vct(2) = vct(2) + all_charges(i)
      endif
     endif
    enddo

    i11=0 ; vct(3:4) = 0.0d0
    i11_0=0; i11_1=0
    do i = 1, Natoms
     if (is_sfield_constrained(i)) then
      i11=i11+1
      if (i11<N_atoms_field_constrained/2+1) then
        vct(3) = vct(3) + atom_profile(i)%fi
        i11_0=i11_0+1
      else
        vct(4) = vct(4) + atom_profile(i)%fi
        i11_1=i11_1+1
      endif
     endif
    enddo 
    vct(3) = vct(3) / dble(i11_0) ; 
    vct(4)  =  vct(4) / dble(i11_1)
!print*,atom_profile(1)%fi/Volt_to_internal_field,atom_profile(500)%fi/Volt_to_internal_field
!print*,'vct 3 4=',vct(3:4)/Volt_to_internal_field,N_atoms_field_constrained
!read(*,*)
    call statistics_5_eval(RA_sfc(1:4), vct(1:4))
   endif

   endif ! (N_atoms_field_constrained>0)

   if (N_dipol_polarizable>0) then
      vct(1:4) = 0.0d0; 
      do i = 1, Natoms
        if (is_dipole_polarizable(i)) then
          vct(1) = vct(1) + all_dipoles_xx(i)
          vct(2) = vct(2) + all_dipoles_yy(i)
          vct(3) = vct(3) + all_dipoles_zz(i)
          vct(4) = vct(4) + all_dipoles(i)
        endif
      enddo
   call statistics_5_eval(RA_dip(1:4), vct(1:4))
   endif ! (N_dipol_polarizable>0) 
   if (l_do_QN_CTRL) then 
    call statistics_5_eval(RA_Temperature_trans, Temperature_trans_Calc)
    call statistics_5_eval(RA_Temperature_rot,   Temperature_rot_Calc)
   endif 
   if (l_do_QN_CTRL) then
    vct(1) = sum(mol_MOM(:,1));  vct(2) = sum(mol_MOM(:,2)); vct(3) = sum(mol_MOM(:,3))
   else
    vct(1) = dot_product(vxx,all_atoms_mass) 
    vct(2) = dot_product(vyy,all_atoms_mass) 
    vct(3) = dot_product(vzz,all_atoms_mass)
   endif
   call statistics_5_eval(RA_MOM_0(1:3), vct(1:3))
!   call zero_5_type(RA_msd2, )
!   call zero_5_type(RA_diffusion)
   vct(1) = sum(all_p_charges); vct(2) = sum(all_g_charges) ; vct(3) = sum(vct(1:2))
   call statistics_5_eval(RA_sum_charge, vct(1:3))

   if (l_very_first_pas) then
    call first_pass_min_max
    l_very_first_pas=.false.
   endif

  contains
   subroutine first_pass_min_max
     call min_max_stat_1st_pass(RA_energy, RA_energy%val)
     call min_max_stat_1st_pass(RA_stress, RA_stress%val)
     call min_max_stat_1st_pass(RA_pressure,RA_pressure%val)
     call min_max_stat_1st_pass(RA_Temperature, RA_Temperature%val)
     if (l_do_QN_CTRL) then
      call min_max_stat_1st_pass(RA_Temperature_trans, RA_Temperature_trans%val)
      call min_max_stat_1st_pass(RA_Temperature_rot,   RA_Temperature_rot%val)
     endif
     call min_max_stat_1st_pass(RA_MOM_0,RA_MOM_0%val)
     call min_max_stat_1st_pass(RA_msd2,RA_msd2%val)
     call min_max_stat_1st_pass(RA_diffusion, RA_diffusion%val)
     call min_max_stat_1st_pass(RA_sum_charge, RA_sum_charge%val)
     if (N_atoms_field_constrained>0) then
      call min_max_stat_1st_pass(RA_sfc, RA_sfc%val)
     endif
     if (N_dipol_polarizable>0) then
      call min_max_stat_1st_pass(RA_dip, RA_dip%val)
     endif

    end subroutine first_pass_min_max
  end subroutine update_scalar_statistics

 
 subroutine stress_counter_details
  use stresses_data
  use RA1_stresses_data
  implicit none
     RA1_stress_kin(:)=RA1_stress_kin(:) + stress_kin(:)
     RA1_stress_shake(:)=RA1_stress_shake(:) + stress_shake(:)
     RA1_stress_bond(:)=RA1_stress_bond(:) + stress_bond(:)
     RA1_stress_angle(:)=RA1_stress_angle(:) + stress_angle(:)
     RA1_stress_dih(:)=RA1_stress_dih(:) + stress_dih(:)
     RA1_stress_deform(:)=RA1_stress_deform(:) + stress_deform(:)
     RA1_stress_dummy(:)=RA1_stress_dummy(:) + stress_dummy(:)
     RA1_stress_vdw(:)= RA1_stress_vdw(:) + stress_vdw(:)
     RA1_stress_Qreal(:)=RA1_stress_Qreal(:) + stress_Qreal(:)
     RA1_stress_Qcmplx(:)=RA1_stress_Qcmplx(:) + stress_Qcmplx(:)
     RA1_stress_Qcmplx_k_eq_0(:) = RA1_stress_Qcmplx_k_eq_0(:) + stress_Qcmplx_k_eq_0(:)
     RA1_stress_excluded(:) = RA1_stress_excluded(:) + stress_excluded(:)
     RA1_stress_Qcmplx_as_in_3D(:)=RA1_stress_Qcmplx_as_in_3D(:)+stress_Qcmplx_as_in_3D(:)
     RA1_stress_thetering(:) = RA1_stress_thetering(:) + stress_thetering
     RA1_stress_counts = RA1_stress_counts + 1.0d0
     

 end subroutine stress_counter_details

 end module scalar_statistics_eval

