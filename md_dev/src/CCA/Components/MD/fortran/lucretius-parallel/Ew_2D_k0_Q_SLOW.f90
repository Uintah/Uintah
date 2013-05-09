 subroutine Ew_2D_k0_Q_SLOW
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
    real(8), allocatable :: local_force(:),a_pot(:),a_fi(:)
    real(8), allocatable :: buffer(:,:)

    allocate(local_force(Natoms)) ; local_force=0.0d0
    allocate(a_pot(Natoms)); a_pot = 0.0d0
    allocate(a_fi(Natoms)) ; a_fi = 0.0d0


    allocate(buffer(Natoms,3)); buffer=0.0d0 ! DELETE IT ; 

    CC_alpha = sqrt_Pi/Ewald_alpha
    CC_2 = sqrt_Pi * 4.0d0 * Ewald_alpha
    CC_3 = 8.0d0*sqrt_Pi*Ewald_alpha**3
    i_Area = 1.0d0/area_xy
    local_energy_pp=0.0d0
    local_stress = 0.0d0

    CC = CC_alpha

 do i = 1, Natoms
! itype = i_type_atom(i)
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
     a_fi(i) = a_fi(i) + qj*field0+dj_zz*field1
     a_fi(j) = a_fi(j) + qi*field0-di_zz*field1
buffer(i,3) = buffer(i,3) + (-Pi2*qj*derf_x + CC_2*dexp_x2*dj_zz) * (i_Area )
buffer(j,3) = buffer(j,3) + (+Pi2*qi*derf_x + CC_2*dexp_x2*di_zz) * (i_Area )

     local_energy_pp = local_energy_pp + En
     ff = (qij*Pi2*derf_x   - (q_d*CC_2 - dij_zz*CC_3*zk)*dexp_x2 )* i_Area
     a_pot_i = a_pot_i + En
     a_pot(j) = a_pot(j) + En
     local_force(i) = local_force(i) + ff
     local_force(j) = local_force(j) - ff
     local_stress = local_stress + zk*ff
!if (sum(a_pot) > 0.0d0) then
!print*,i,j,'En sum=',local_energy_pp,sum(a_pot)/2.0d0
!endif
endif!i<>j
 enddo
 a_pot(i) = a_pot(i) + a_pot_i
!if (sum(a_pot) > 0.0d0) then
!print*, i, local_energy_pp,sum(a_pot)/2.0d0
!read(*,*)
!endif
 enddo

 do i = 1, Natoms
    qi = all_charges(i)
    di_zz = all_dipoles_zz(i)
    field0 =  CC_alpha *  (-2.0d0*i_area )
    En0 = field0*qi*qi
    En1 = 0.0d0
    En2 = di_zz*di_zz*CC_2
    En = En0 + (En1 + En2)*i_area
    a_fi(i) = a_fi(i) + qi*field0 ! the dipole component of field is zero (erf(0)=0) at k=0 term.
!buffer(i,1) = buffer(i,1) + 0.0d0
!buffer(i,2) = buffer(i,2) + 0.0d0
buffer(i,3) = buffer(i,3) + ( + CC_2*di_zz) * (i_Area )

    local_energy_pp = local_energy_pp + En*0.5d0
    a_pot(i) = a_pot(i) + En
 enddo


local_energy = local_energy_pp

En_Q_k0_cmplx = En_Q_k0_cmplx + local_energy
En_Q_cmplx = En_Q_cmplx + local_energy
En_Q = En_Q + local_energy
stress(1:2) = stress(1:2) + local_energy
stress(3) = stress(3) + local_stress
stress(4) = sum(stress(1:3))
stress_Qcmplx_k_eq_0(1:2)=local_energy
stress_Qcmplx_k_eq_0(3)  =local_stress
stress_Qcmplx_k_eq_0(4)  = (local_energy+local_energy+local_stress)/3.0d0
stress_Qcmplx_k_eq_0(5:10)=0.0d0
stress = stress + stress_Qcmplx_k_eq_0

!print*,'local_enedgy=',local_energy, sum(a_pot)/2.0d0
!STOP
!print*,'sum local foreces = ',sum(local_force)
!print*, 'local_stress=',local_stress


fzz(:) = fzz(:) + local_force(:)


if (l_need_2nd_profile) then
 atom_profile%pot = atom_profile%pot + a_pot
 atom_profile%Qpot = atom_profile%Qpot + a_pot
 atom_profile%fi = atom_profile%fi + a_fi
 do i =1,Natoms; atom_profile(i)%buffer3(3) = atom_profile(i)%buffer3(3) + buffer(i,3);print*,buffer(i,3); enddo
! atom_profile%szz = atom_profile%szz + stress_33
! atom_profile%EE_zz = atom_profile%EE_zz + e_zz  
 ! TO DO THE STRESSES
endif

!open(unit=77,file='fort.77')
!write(77,*) Natoms
!do i = 1, Natoms
!write(77,*) i,' ------'
!write(77,*) a_fi(i),atom_profile(i)%fi
!write(77,*) buffer(i,:)
!enddo
!close(77)

 deallocate(local_force,a_pot,a_fi)
 end subroutine Ew_2D_k0_Q_SLOW

