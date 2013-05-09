
module pair_short_ENERGY


contains 

!------------------
subroutine pair_short_Q_ENERGY
 use sys_data
 use paralel_env_data
 use math_constants
 use boundaries
 use ALL_atoms_data
 use atom_type_data
 use max_sizes_data, only : MX_list_nonbonded
 use non_bonded_lists_data, only : list_nonbonded, size_list_nonbonded
 use profiles_data
 use energies_data
 use stresses_data
 use atom_type_data
 use interpolate_data
 use variables_short_pairs
 use physical_constants, only : Volt_to_internal_field
 use integrate_data, only : integration_step
 use CTRLs_data, only : l_ANY_DIPOLE_CTRL, l_DIP_CTRL
 use compute_14_module, only : compute_14_interactions_driver_ENERGY
 use REAL_part_intra_correction_sfc_sfc
 implicit none
 real(8), parameter :: en_factor = 0.5d0
 integer i,j,i1,imol,itype,N,neightot,k,nneigh, iStyle,jStyle
 real(8) local_energy,En0,En00
 real(8) ppp, vk,vk1,vk2, B1_THOLE_DERIV, B0_THOLE_DERIV, B2,B1,B0, G1,G2, pipj, didj
 real(8) B0_THOLE, B1_THOLE
 real(8) dipole_i_times_Rij, dipole_j_times_Rij, a0
 real(8) t0,t1,t2
 real(8), allocatable :: local_force(:,:)
   integer i_pair,ndx
   real(8) fx,fy,fz,sxx,sxy,sxz,syx,syy,syz,szx,szy,szz
   real(8) En,gk,gk1,gk2,x,y,z,Inverse_r_squared,r,r2,Inverse_r


 en_vdw = 0.0d0
 en_Qreal = 0.0d0
 irdr = 1.0d0/rdr
 allocate(dx(Natoms),dy(Natoms),dz(Natoms),dr_sq(Natoms))
 allocate(in_list_Q(Natoms))
 do i = 1+rank, Natoms , nprocs
  imol = atom_in_which_molecule(i)
  iStyle  = i_Style_atom(i)
  i1 = 0
  do k =  1, size_list_nonbonded(i)
    i1 = i1 + 1
    j = list_nonbonded(i,k)
    dx(i1) = xxx(i) - xxx(j)
    dy(i1) = yyy(i) - yyy(j)
    dz(i1) = zzz(i) - zzz(j)
    in_list_Q(i1) = j
  enddo
  neightot=i1

  call periodic_images(dx(1:i1),dy(1:i1),dz(1:i1))
  dr_sq(1:i1) = dx(1:i1)*dx(1:i1) + dy(1:i1)*dy(1:i1) + dz(1:i1)*dz(1:i1)
  neightot = i1
!  call nonbonded_vdw_2_forces(i,iStyle,size_list_nonbonded(i)) ! put it back

    do k =  1, neightot
    j = list_nonbonded(i,k)
    r2 = dr_sq(k)
    if ( r2 < cut_off_sq ) then
     jstyle = i_style_atom(j)
     i_pair = which_atomStyle_pair(istyle,jstyle) ! can replace it by a formula?
     a0 = atom_Style2_vdwPrm(0,i_pair)
     if (a0 > SYS_ZERO) then
        r = dsqrt(r2)
        Inverse_r_squared = 1.0d0/r2
        NDX = max(1,int(r*irdr))
        ppp = (r*irdr) - dble(ndx)
        vk  = vvdw(ndx,i_pair)  ;  vk1 = vvdw(ndx+1,i_pair) ; vk2 = vvdw(ndx+2,i_pair)
        t1 = vk  + (vk1 - vk )*ppp
        t2 = vk1 + (vk2 - vk1)*(ppp - 1.0d0)
        En = (t1 + (t2-t1)*ppp*0.5d0)
        en_vdw = en_vdw + En
      endif ! (a0 > 1.0d-10
   endif
  enddo ! j index of the double loop
 
  qi = all_charges(i) ! electricity = charge no matter of what kind.
  dipole_xx_i = all_dipoles_xx(i) ; dipole_yy_i = all_dipoles_yy(i) ; dipole_zz_i = all_dipoles_zz(i)
  dipole_i2 = dipole_xx_i*dipole_xx_i+dipole_yy_i*dipole_yy_i+dipole_zz_i*dipole_zz_i
  if (l_DIP_CTRL) then

    do k =  1, neightot
    j = in_list_Q(k)
    r2 = dr_sq(k)
    if ( r2 < cut_off_sq ) then
     jStyle = i_Style_atom(j)


     i_pair = which_atomStyle_pair(iStyle,jStyle) ! can replace it by a formula?
     r = dsqrt(r2)
     Inverse_r = 1.0d0/r
     NDX = max(1,int(r*irdr))
     ppp = (r*irdr) - dble(ndx)
     x = dx(k)   ;  y = dy(k)    ; z = dz(k)
        qj = all_charges(j)
        dipole_xx_j = all_dipoles_xx(j) ; dipole_yy_j=all_dipoles_yy(j); dipole_zz_j=all_dipoles_zz(j)
        dipole_i_times_Rij = x*dipole_xx_i+y*dipole_yy_i+z*dipole_zz_i
        dipole_j_times_Rij = x*dipole_xx_j+y*dipole_yy_j+z*dipole_zz_j
        pipj = dipole_xx_i*dipole_xx_j + dipole_yy_i*dipole_yy_j+ dipole_zz_i*dipole_zz_j
        didj = dipole_i_times_Rij*dipole_j_times_Rij
        G1 = - dipole_i_times_Rij*qj + dipole_j_times_Rij*qi + pipj
        G2 = - didj
        qij = qi*qj
        include 'interpolate_3.frg'
        include 'interpolate_THOLE_ALL.frg'

        En =  B0*qij + B1*G1 + B2*G2 +    B0_THOLE*pipj + B1_THOLE*G2
        En_Qreal = En_Qreal + En
      endif ! if ( r2 < cut_off_sq )
    enddo ! k =  1, neightot

  else
!    call Q_2_forces(i,iStyle,neightot)
  do k =  1, neightot
    j = in_list_Q(k)
    r2 = dr_sq(k)
    if ( r2 < cut_off_sq ) then
     jstyle = i_style_atom(j)
     i_pair = which_atomStyle_pair(istyle,jstyle) ! can replace it by a formula?
     r = dsqrt(r2)
     Inverse_r = 1.0d0/r
     NDX = max(1,int(r*irdr))
     ppp = (r*irdr) - dble(ndx)
     x = dx(k)   ;  y = dy(k)    ; z = dz(k)

        qj = all_charges(j)
        qij = qi*qj
        vk  = vele_G(ndx,i_pair)  ;  vk1 = vele_G(ndx+1,i_pair) ; vk2 = vele_G(ndx+2,i_pair)
        t1 = vk  + (vk1 - vk )*ppp
        t2 = vk1 + (vk2 - vk1)*(ppp - 1.0d0)
        En0 = (t1 + (t2-t1)*ppp*0.5d0)
        En = En0 * qij
        En_Qreal = En_Qreal + En
      endif
    enddo
  endif

  enddo ! i 

  call compute_14_interactions_driver_ENERGY
  call REAL_part_intra_correction_sfc_sfc_driver_ENERGY
local_energy = 0.0d0
do i = 1, Natoms
if(is_dipole_polarizable(i))then
  En0 = all_DIPOLE_pol(i)*(all_dipoles_xx(i)**2+all_dipoles_yy(i)**2+all_dipoles_zz(i)**2) * 0.5d0
  local_energy = local_energy + En0
endif
enddo
en_Qreal = en_Qreal + local_energy
En_Q = En_Q + en_Qreal
en_induced_dip = local_energy

 deallocate(dr_sq)
 deallocate(dx,dy,dz)
 deallocate(in_list_Q)

end subroutine pair_short_Q_ENERGY
   
   
!---------------
!-------------------
!------------


 subroutine pair_short_vdw_ENERGY ! do only vdw
  use sys_data
  use paralel_env_data
  use math_constants
  use boundaries
  use ALL_atoms_data
  use atom_type_data
  use max_sizes_data, only : MX_list_nonbonded
  use sizes_data, only : N_pairs_14
  use non_bonded_lists_data, only : list_nonbonded, size_list_nonbonded
  use profiles_data
  use energies_data
  use stresses_data
  use atom_type_data
  use interpolate_data
  use variables_short_pairs
 use rdfs_data, only : rdfs
 use integrate_data, only : integration_step
 use rdfs_collect_module, only : rdfs_collect
 use compute_14_module, only : compute_14_interactions_vdw_ENERGY
 use connectivity_ALL_data, only : l_red_14_vdw_CTRL

  implicit none
  real(8), parameter :: en_factor = 0.5d0
  integer i,j,i1,imol,itype,N,neightot,k,nneigh, istyle,jStyle,i_pair
  real(8), allocatable :: local_force(:,:)
  real(8) a0, t1,t2,B0,B1,vk,vk1,vk2,ppp,En,Inverse_r_squared,r2,r
  integer ndx


  en_vdw = 0.0d0
  irdr = 1.0d0/rdr
  do i = 1+rank, Natoms , nprocs
    imol = atom_in_which_molecule(i)
    istyle  = i_style_atom(i)
    i1 = size_list_nonbonded(i)
!    qi = all_p_charges(i)
    do k =  1, i1
      j = list_nonbonded(i,k)
      dx(k) = xxx(i) - xxx(j)
      dy(k) = yyy(i) - yyy(j)
      dz(k) = zzz(i) - zzz(j)
    enddo
    call periodic_images(dx(1:i1),dy(1:i1),dz(1:i1))
    dr_sq(1:i1) = dx(1:i1)*dx(1:i1) + dy(1:i1)*dy(1:i1) + dz(1:i1)*dz(1:i1)
    neightot = i1
!    call nonbonded_vdw_2_forces(i,istyle,neightot)
    do k =  1, neightot
    j = list_nonbonded(i,k)
    r2 = dr_sq(k)
    if ( r2 < cut_off_sq ) then
     jstyle = i_style_atom(j)
     i_pair = which_atomStyle_pair(istyle,jstyle) ! can replace it by a formula?
     a0 = atom_Style2_vdwPrm(0,i_pair)
     if (a0 > SYS_ZERO) then
        r = dsqrt(r2)
        Inverse_r_squared = 1.0d0/r2
        NDX = max(1,int(r*irdr))
        ppp = (r*irdr) - dble(ndx)
        vk  = vvdw(ndx,i_pair)  ;  vk1 = vvdw(ndx+1,i_pair) ; vk2 = vvdw(ndx+2,i_pair)
        t1 = vk  + (vk1 - vk )*ppp
        t2 = vk1 + (vk2 - vk1)*(ppp - 1.0d0)
        En = (t1 + (t2-t1)*ppp*0.5d0)
        en_vdw = en_vdw + En
    endif !(a0 > SYS_ZERO)
    endif !( r2 < cut_off_sq )
    enddo ! k =  1, neightot

  enddo  ! main cycle

  if (N_pairs_14>0.and.l_red_14_vdw_CTRL) call compute_14_interactions_vdw_ENERGY


  deallocate(dr_sq)
  deallocate(dx,dy,dz)

end  subroutine pair_short_vdw_ENERGY


!---------------
!-------------------
!------------

end module pair_short_ENERGY


