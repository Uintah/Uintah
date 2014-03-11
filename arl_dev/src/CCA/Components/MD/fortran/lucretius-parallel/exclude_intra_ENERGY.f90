module exclude_intra_ENERGY
implicit none
public :: exclude_2D_at_k0_Q_DIP_ENERGY
public :: exclude_2D_at_k0_Q_ENERGY
public :: exclude_intra_Q_ENERGY
public :: exclude_intra_Q_DIP_ENERGY

contains

!-------------------------------------------------------------
!***************************************************************
!**************************************************************
!------------------------------------------------------------

subroutine exclude_2D_at_k0_Q_DIP_ENERGY
 call exclude_2D_at_k0_Q_ENERGY ! the same subroutine 
end subroutine exclude_2D_at_k0_Q_DIP_ENERGY


subroutine exclude_2D_at_k0_Q_ENERGY
!  self and intra corrections for 2D at k=0
!   point charges only
use connectivity_ALL_data, only : list_excluded_HALF_no_SFC,size_list_excluded_HALF_no_SFC,MX_excluded
use all_atoms_data, only : xxx,yyy,zzz,all_p_charges,Natoms, fzz, all_G_charges, &
                           is_charge_distributed, i_type_atom, all_charges, all_dipoles_zz,&
                           i_style_atom, is_sfield_constrained
use Ewald_data
use spline_z_k0_module
use sim_cel_data
use boundaries, only : periodic_images_ZZ
use energies_data
implicit none
integer i,j,k,i1,itype,jtype
real(8), parameter :: Pi = 3.14159265358979d0
real(8), parameter :: sqrt_Pi = 1.77245385090552d0
real(8), parameter :: Pi2 = 2.0d0*Pi
real(8) x_x_x,qi,qj,qij,a_f_i,poti,ff,En,En2,En1,local_energy, i_Area,zij, CC1,CC_2
real(8) a_force_i,e_xx_i,e_yy_i,e_zz_i,ff0,En0, a_fi_i, szz, stress_i,field
real(8) local_stress_zz, local_intra, local_self
real(8), allocatable :: dz(:)
real(8) fieldG, EnG, qiG, pref, di_zz,dj_zz, q_d,dij_zz,dexp_x2,derf_x
logical l_i,l_j, is_sfc_i,is_sfc_j,l_proceed
!real(8), allocatable :: buffer(:) ! coment it
!allocate(buffer(Natoms)); buffer=0.0d0 !coment it

allocate(dz(MX_excluded))

local_energy=0.0d0
i_Area=1.0d0/Area_xy
CC1 = sqrt_Pi/Ewald_alpha
CC_2 = sqrt_Pi * 4.0d0 * Ewald_alpha

    do i = 1, Natoms
      l_i = is_charge_distributed(i)
!      is_sfc_i = is_sfield_constrained(i) 
      itype = i_style_atom(i)
      qi = all_charges(i)
      di_zz = all_dipoles_zz(i)
      i1 = size_list_excluded_HALF_no_SFC(i)
      do k = 1, i1
         j = list_excluded_HALF_no_SFC(i,k)
         dz(k) = zzz(i)-zzz(j)
      enddo
      call periodic_images_ZZ(dz(1:i1))
      do k = 1, i1
         j = list_excluded_HALF_no_SFC(i,k)
         l_j = is_charge_distributed(j)
!         is_sfc_j = is_sfield_constrained(j)
         zij = dz(k)
         qj = all_charges(j) 
         dj_zz = all_dipoles_zz(j)
         q_d = (qi*dj_zz-di_zz*qj)
         dij_zz = dj_zz*di_zz
         x_x_x = Ewald_alpha * zij
         pref = CC1
         qij = qi*qj
         derf_x = derf(x_x_x)
         dexp_x2 = dexp(-x_x_x*x_x_x)
         En0 = -(2.0d0*i_Area)*(pref*dexp_x2+zij*Pi*derf_x)
         En1 = q_d * derf_x * Pi2  ! charge-dipole
         En2 = dij_zz * dexp_x2 * CC_2  ! dipole-dipole
         En = En0*qij + (En1 + En2 ) * i_Area
         local_energy = local_energy + En
      enddo
    enddo

En_Q_cmplx_k0_CORR =  - local_energy
En_Q_cmplx = En_Q_cmplx + En_Q_cmplx_k0_CORR
En_Q_k0_cmplx = En_Q_k0_cmplx + En_Q_cmplx_k0_CORR
En_Q = En_Q + En_Q_cmplx_k0_CORR


deallocate(dz)

end subroutine exclude_2D_at_k0_Q_ENERGY


! - ------------------------

subroutine exclude_intra_Q_ENERGY
! I have now both QP and QG charges
use connectivity_ALL_data, only : list_excluded_HALF_no_SFC,size_list_excluded_HALF_no_SFC,MX_excluded
use all_atoms_data, only : xx,yy,zz,all_p_charges,Natoms, fzz,fyy,fxx, i_type_atom, is_charge_distributed,&
                           all_G_charges, all_charges,i_style_atom,xxx,yyy,zzz,is_sfield_constrained
use Ewald_data
use spline_z_k0_module
use sim_cel_data
use boundaries, only : periodic_images
use energies_data
use stresses_data
use atom_type_data, only : atom_type_1GAUSS_charge_distrib
use connectivity_ALL_data, only : red_14_Q,red_14_Q_mu,red_14_mu_mu,list_14,size_list_14,l_red_14_Q_CTRL
use sizes_data, only : N_pairs_14
use max_sizes_data, only : MX_in_list_14
use interpolate_data, only : vele,gele,rdr,irdr
implicit none
integer i,j,k,i1
real(8), parameter :: Pi = 3.14159265358979d0
real(8), parameter :: sqrt_Pi = 1.77245385090552d0
real(8) x_x_x,qi,qj,qij,a_f_i,poti,ff,En,local_energy, i_Area,zij, CC1, CC2,CC3,AAA
real(8) a_force_i,e_xx_i,e_yy_i,e_zz_i,ff0,En0, a_fi_i, field
real(8), allocatable :: dx(:),dy(:),dz(:),dr_sq(:)
real(8) x,y,z
real(8) inv_rij, rij,vk1,vk2,vk0,vk,t1,t2
real(8) En0G,En0GG, qiG, pref,eta,inv_r_B0,inv_r_B1,inv_r3,B0,B1,r,ppp,inv_r2
integer itype,jtype
logical l_i,l_j,is_sfc_i,is_sfc_j,l_proceed
integer N
integer NDX

N=max(MX_excluded,MX_in_list_14)
allocate(dx(N),dy(N),dz(N),dr_sq(N))

  CC1 = 2.0d0*Ewald_alpha/sqrt_Pi
  local_energy = 0.0d0

  do i = 1, Natoms-1
!      is_sfc_i = is_sfield_constrained(i)
      l_i = is_charge_distributed(i)
      qi = all_charges(i)
      i1 = size_list_excluded_HALF_no_SFC(i)
      do k = 1, i1
         j = list_excluded_HALF_no_SFC(i,k)
         dx(k) = xxx(i)-xxx(j)
         dy(k) = yyy(i)-yyy(j)
         dz(k) = zzz(i)-zzz(j)
      enddo
      if (i1 > 0) then
        call periodic_images(dx(1:i1),dy(1:i1),dz(1:i1))
        dr_sq(1:i1) = dx(1:i1)*dx(1:i1)+dy(1:i1)*dy(1:i1)+dz(1:i1)*dz(1:i1)
      endif
      do k = 1, i1
        j = list_excluded_HALF_no_sfc(i,k)
!        is_sfc_j = is_sfield_constrained(j)
        x = dx(k) ; y = dy(k) ; z = dz(k) ; rij = dsqrt(dr_sq(k)); ;r=rij
        NDX = max(1,int(r*irdr))
        ppp = (rij*irdr) - dble(ndx)
        inv_rij = 1.0d0 / rij;
        inv_r2 = inv_rij*inv_rij;
        inv_r3 = inv_r2*inv_rij ;

         qj = all_charges(j)
         qij = qi*qj
         inv_rij = 1.0d0 / rij        

      vk  = vele(ndx)  ;  vk1 = vele(ndx+1) ; vk2 = vele(ndx+2)
       t1 = vk  + (vk1 - vk )*ppp
       t2 = vk1 + (vk2 - vk1)*(ppp - 1.0d0)
       B0 = (t1 + (t2-t1)*(ppp*0.5d0))
      vk  = gele(ndx)  ;  vk1 = gele(ndx+1) ; vk2 = gele(ndx+2)
       t1 = vk  + (vk1 - vk )*ppp
       t2 = vk1 + (vk2 - vk1)*(ppp - 1.0d0)
       B1 = (t1 + (t2-t1)*(ppp*0.5d0))
        inv_r_B0 = inv_rij-B0
        inv_r_B1 = inv_r3-B1
        En = qij*inv_r_B0
        En0 = inv_r_B0
        local_energy = local_energy + En

      enddo ! k
  enddo  ! i

  En_Q_intra_corr = -local_energy 

ew_self = 0.0d0

  do i = 1, Natoms
    qi = all_charges(i)
       En0 = 2.0d0*qi*(Ewald_alpha/sqrt_Pi)
       local_energy = local_energy + En0*0.5d0*qi
       ew_self = ew_self - En0*0.5d0*qi
  enddo

  En_Q_Gausian_self = 0.0d0
  do i = 1, Natoms
  if (is_charge_distributed(i)) then
    itype = i_style_atom(i)
    qi = all_G_charges(i)
       En0 = -2.0d0*ewald_eta(itype,itype)*qi/sqrt_Pi
       local_energy = local_energy + En0*0.5d0*qi
       En_Q_Gausian_self = En_Q_Gausian_self -  En0*0.5d0*qi
  endif
  enddo


 En_Q = En_Q - local_energy

deallocate(dx,dy,dz,dr_sq)

end subroutine exclude_intra_Q_ENERGY



subroutine exclude_intra_Q_DIP_ENERGY
use connectivity_ALL_data, only : list_excluded_HALF_no_SFC,size_list_excluded_HALF_no_SFC,MX_excluded
use all_atoms_data, only : xx,yy,zz,all_p_charges,Natoms, fzz,fyy,fxx, i_type_atom, is_charge_distributed,&
                           all_G_charges, all_charges, all_dipoles,&
                           all_dipoles_xx,all_dipoles_yy,all_dipoles_zz, i_style_atom,xxx,yyy,zzz,&
                           is_sfield_constrained
use Ewald_data
use spline_z_k0_module
use sim_cel_data
use boundaries, only : periodic_images
use profiles_data, only : atom_profile, l_need_2nd_profile
use energies_data
use stresses_data
use atom_type_data, only : atom_type_1GAUSS_charge_distrib
use non_bonded_lists_data, only : list_nonbonded, size_list_nonbonded!DELETE IT
use connectivity_ALL_data, only : red_14_Q,red_14_Q_mu,red_14_mu_mu,list_14,size_list_14
use sizes_data, only : N_pairs_14
use max_sizes_data, only : MX_in_list_14
use interpolate_data, only : iRDR,RDR,vele,gele,vele2,vele3
implicit none
integer i,j,k,i1
real(8), parameter :: Pi = 3.14159265358979d0
real(8), parameter :: sqrt_Pi = 1.77245385090552d0
real(8) x_x_x,qi,qj,qij,a_f_i,poti,ff,En,local_energy, i_Area,zij, CC0,CC1, CC2,CC3,AAA
real(8) a_force_i,e_xx_i,e_yy_i,e_zz_i,ff0,En0, a_fi_i, field
real(8), allocatable :: dx(:),dy(:),dz(:),dr_sq(:)
real(8) x,y,z
real(8) inv_rij, rij,r
real(8) En0G,En0GG, qiG, pref,eta,eta_sq
integer itype,jtype
logical l_i,l_j,is_sfc_i,is_sfc_j,l_proceed
real(8) B0,B1,B2,B3,fct_UP,fct_DOWN,ratio_B1,ratio_B2,ratio_B3
real(8) inv_r2,inv_r3,inv_r5,inv_r7,EEE
real(8) dipole_xx_i,dipole_yy_i,dipole_zz_i,dipole_xx_j,dipole_yy_j,dipole_zz_j
real(8) dipole_i_times_Rij, dipole_j_times_Rij
real(8) pipj, didj,G1,G2,ew2
real(8) nabla_G1_xx, nabla_G1_yy, nabla_G1_zz
real(8) nabla_G2_xx, nabla_G2_yy, nabla_G2_zz
real(8) inv_r_B0,inv_r_B1,inv_r_B2,inv_r_B3
real(8) vk,vk1,vk2,gk,gk1,gk2,t1,t2,t3,ppp
real(8),allocatable :: buffer(:,:)
integer N,ndx



N=max(MX_excluded,MX_in_list_14)

allocate(dx(N),dy(N),dz(N),dr_sq(N))

  CC1 = 2.0d0*Ewald_alpha/sqrt_Pi
  CC0 = CC1 * 0.5d0
!  fct_UP = 2.0d0*Ewald_alpha*Ewald_alpha
!  fct_DOWN = 1.0d0/Ewald_alpha/sqrt_Pi
!  ratio_B1 = fct_UP*fct_DOWN
!  ratio_B2 = ratio_B1*fct_UP
!  ratio_B3 = ratio_B2*fct_UP

  local_energy = 0.0d0

  do i = 1, Natoms-1
      l_i = is_charge_distributed(i)
!      is_sfc_i = is_sfield_constrained(i)
      qi = all_charges(i)
      dipole_xx_i = all_dipoles_xx(i) ; dipole_yy_i=all_dipoles_yy(i); dipole_zz_i=all_dipoles_zz(i)
      i1 = size_list_excluded_HALF_no_SFC(i)
      do k = 1, i1
         j = list_excluded_HALF_no_SFC(i,k)
         dx(k) = xxx(i)-xxx(j)
         dy(k) = yyy(i)-yyy(j)
         dz(k) = zzz(i)-zzz(j)
      enddo
      if (i1 > 0) then
        call periodic_images(dx(1:i1),dy(1:i1),dz(1:i1))
        dr_sq(1:i1) = dx(1:i1)*dx(1:i1)+dy(1:i1)*dy(1:i1)+dz(1:i1)*dz(1:i1)
      endif
      do k = 1, i1
        j = list_excluded_HALF_no_SFC(i,k)
!        is_sfc_j = is_sfield_constrained(j)
        x = dx(k) ; y = dy(k) ; z = dz(k) ; rij = dsqrt(dr_sq(k)); ;r=rij
        NDX = max(1,int(r*irdr))
        ppp = (rij*irdr) - dble(ndx)
        inv_rij = 1.0d0 / rij; 
        inv_r2 = inv_rij*inv_rij; 
        inv_r3 = inv_r2*inv_rij ; 
        inv_r5 = inv_r3*inv_r2
!        x_x_x=Ewald_alpha*rij
        qj = all_charges(j)
        dipole_xx_j = all_dipoles_xx(j) ; dipole_yy_j=all_dipoles_yy(j); dipole_zz_j=all_dipoles_zz(j)
        dipole_i_times_Rij = x*dipole_xx_i+y*dipole_yy_i+z*dipole_zz_i
        dipole_j_times_Rij = x*dipole_xx_j+y*dipole_yy_j+z*dipole_zz_j
        pipj = dipole_xx_i*dipole_xx_j + dipole_yy_i*dipole_yy_j+ dipole_zz_i*dipole_zz_j
        didj = dipole_i_times_Rij*dipole_j_times_Rij
        G1 = - dipole_i_times_Rij*qj + dipole_j_times_Rij*qi + pipj
        G2 = - didj

      vk  = vele(ndx)  ;  vk1 = vele(ndx+1) ; vk2 = vele(ndx+2)
       t1 = vk  + (vk1 - vk )*ppp
       t2 = vk1 + (vk2 - vk1)*(ppp - 1.0d0)
       B0 = (t1 + (t2-t1)*(ppp*0.5d0))
      vk  = gele(ndx)  ;  vk1 = gele(ndx+1) ; vk2 = gele(ndx+2)
       t1 = vk  + (vk1 - vk )*ppp
       t2 = vk1 + (vk2 - vk1)*(ppp - 1.0d0)
       B1 = (t1 + (t2-t1)*(ppp*0.5d0))
      vk  = vele2(ndx)  ;  vk1 = vele2(ndx+1) ; vk2 = vele2(ndx+2)
       t1 = vk  + (vk1 - vk )*ppp
       t2 = vk1 + (vk2 - vk1)*(ppp - 1.0d0)
       B2 = (t1 + (t2-t1)*(ppp*0.5d0))

        qij = qi*qj

        inv_r_B0 = inv_rij-B0
        inv_r_B1 = inv_r3-B1
        inv_r_B2 = 3.0d0*inv_r5-B2
        En = qij*inv_r_B0 + G1*inv_r_B1 + G2*inv_r_B2
        local_energy = local_energy + En
    enddo
  enddo  ! i

  En_Q_intra_corr = - local_energy 

ew_self = 0.0d0
ew2=Ewald_alpha*Ewald_alpha
  do i = 1, Natoms
    qi = all_charges(i)
       En = CC0*(qi*qi+2.0d0*ew2/3.0d0*all_dipoles(i)**2)
       local_energy = local_energy + En
       ew_self = ew_self - En
  enddo

  En_Q_Gausian_self = 0.0d0
  do i = 1, Natoms
  if (is_charge_distributed(i)) then
    itype = i_style_atom(i)
    qi = all_charges(i)
       eta = ewald_eta(itype,itype)
       eta_sq = eta*eta
       CC0 = eta/sqrt_Pi
       En0 = -2.0d0*qi*CC0  
       En = -CC0*(qi*qi + 2.0d0*eta_sq/3.0d0*all_dipoles(i)**2)
       local_energy = local_energy + En
       En_Q_Gausian_self = En_Q_Gausian_self -  En
  endif
  enddo


 En_Q = En_Q - local_energy

deallocate(dx,dy,dz,dr_sq)

end subroutine exclude_intra_Q_DIP_ENERGY


end module exclude_intra_ENERGY



