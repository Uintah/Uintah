 subroutine Ew_2D_k0_Q_SLOW_ENERGY
!  for test purposes
    use sim_cel_data
    use ALL_atoms_data, only : xx,yy,zz,all_p_charges,all_g_charges,Natoms, fzz, &
                               is_charge_distributed, i_type_atom, all_charges,all_dipoles_zz,&
                               atom_in_which_molecule,zzz
    use Ewald_data
    use connectivity_ALL_data, only : list_excluded_HALF,size_list_excluded_HALF,MX_excluded
    use boundaries, only : periodic_images_ZZ
    use energies_data
    use profiles_data, only : atom_profile, l_need_2nd_profile
    use sizes_data, only : N_TYPE_ATOMS
    use stresses_data, only : stress,stress_Qcmplx_k_eq_0

    implicit none
    real(8), parameter :: sqrt_Pi = 1.77245385090552d0
    real(8), parameter :: Pi = 3.14159265358979d0
    real(8), parameter :: Pi2 = Pi*2.0d0
    real(8) CC_alpha,CC_beta,CC_gamma, Ew_beta, Ew_gamma, qi,qj,qij,z_k,xxx,field,zk,dij_zz
    real(8) CC , CC2
    integer i,j,k,itype,jtype
    logical l_i,l_j
    real(8) local_energy_pp,local_energy, local_stress
    real(8) i_Area, En, En0,En1,En2,ff,field0, field1,a_pot_i, suma
    real(8) q_d,di_zz,dj_zz ,  derf_x, dexp_x2, CC_2, CC_3


    CC_alpha = sqrt_Pi/Ewald_alpha
    CC_2 = sqrt_Pi * 4.0d0 * Ewald_alpha
    CC_3 = 8.0d0*sqrt_Pi*Ewald_alpha**3
    i_Area = 1.0d0/area_xy
    local_energy_pp=0.0d0
    local_stress = 0.0d0

    CC = CC_alpha

 do i = 1, Natoms
 qi = all_charges(i)
 di_zz=all_dipoles_zz(i)
 a_pot_i = 0.0d0
 do j = i+1, Natoms
if(i/=j)then
! jtype = i_type_atom(j)
 zk = zzz(i) - zzz(j)
     qj = all_charges(j)
     dj_zz = all_dipoles_zz(j)
     dij_zz = di_zz*dj_zz
     qij = qi*qj
     q_d = (qi*dj_zz-di_zz*qj)
     xxx = Ewald_alpha*zk
     derf_x = derf(xxx)
     dexp_x2 = dexp(-xxx*xxx)
     field0 =  (CC_alpha*dexp_x2+zk*Pi*derf_x) * ( (-2.0d0*i_area) )
     field1 = derf_x*Pi*( 2.0d0*i_area)
     En0 = field0*qij  ! charge-charge
     En1 = q_d * derf_x * Pi2  ! charge-dipole
     En2 = dij_zz * dexp_x2 * CC_2  ! dipole-dipole
     En = En0 + (En1 + En2 ) * i_Area
     local_energy_pp = local_energy_pp + En
endif!i<>j
 enddo
 enddo

 do i = 1, Natoms
    qi = all_charges(i)
    di_zz = all_dipoles_zz(i)
    field0 =  CC_alpha *  (-2.0d0*i_area )
    En0 = field0*qi*qi
    En1 = 0.0d0
    En2 = di_zz*di_zz*CC_2
    En = En0 + (En1 + En2)*i_area
    local_energy_pp = local_energy_pp + En*0.5d0
 enddo


local_energy = local_energy_pp

En_Q_k0_cmplx = En_Q_k0_cmplx + local_energy
En_Q_cmplx = En_Q_cmplx + local_energy
En_Q = En_Q + local_energy

 end subroutine Ew_2D_k0_Q_SLOW_ENERGY

