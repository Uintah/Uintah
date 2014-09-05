module exclude_intra
implicit none

public :: exclude_2D_at_k0_Q_DIP
public :: exclude_2D_at_k0_Q
public :: exclude_intra_Q
public :: exclude_intra_Q_DIP

contains

subroutine exclude_2D_at_k0_Q_DIP
 call exclude_2D_at_k0_Q ! the same subroutine 
end subroutine exclude_2D_at_k0_Q_DIP


subroutine exclude_2D_at_k0_Q
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
use profiles_data, only : atom_profile, l_need_2nd_profile
use energies_data
implicit none
integer i,j,k,i1,itype,jtype
real(8), parameter :: Pi = 3.14159265358979d0
real(8), parameter :: sqrt_Pi = 1.77245385090552d0
real(8), parameter :: Pi2 = 2.0d0*Pi
real(8) x_x_x,qi,qj,qij,a_f_i,poti,ff,En,En2,En1,local_energy, i_Area,zij, CC1,CC_2,CC_3
real(8) a_force_i,e_xx_i,e_yy_i,e_zz_i,ff0,En0, a_fi_i, szz, stress_i,field
real(8) local_stress_zz, local_intra, local_self
real(8), allocatable :: a_pot(:),local_force(:),dz(:),stress_33(:),a_fi(:)
real(8), allocatable :: e_xx(:),e_yy(:),e_zz(:)
real(8) fieldG, EnG, qiG, pref, di_zz,dj_zz, q_d,dij_zz,dexp_x2,derf_x
logical l_i,l_j, is_sfc_i,is_sfc_j,l_proceed
!real(8), allocatable :: buffer(:) ! coment it
!allocate(buffer(Natoms)); buffer=0.0d0 !coment it

allocate(dz(MX_excluded),local_force(Natoms))
if (l_need_2nd_profile) then
  allocate(a_pot(Natoms) , a_fi(Natoms), stress_33(Natoms))
  allocate(e_xx(Natoms),e_yy(Natoms),e_zz(Natoms))
  a_pot=0.0d0 ; a_fi = 0.0d0 ; stress_33=0.0d0
  e_xx=0.0d0; e_yy=0.0d0; e_zz=0.0d0
endif

local_force=0.0d0
local_energy=0.0d0
local_stress_zz=0.0d0
i_Area=1.0d0/Area_xy
CC1 = sqrt_Pi/Ewald_alpha
CC_2 = sqrt_Pi * 4.0d0 * Ewald_alpha
CC_3 = 8.0d0*sqrt_Pi*Ewald_alpha**3

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
      a_fi_i = 0.0d0 ; poti = 0.0d0 ;
      a_force_i=0.0d0
      e_zz_i=0.0d0 ; e_yy_i=0.0d0 ; e_xx_i=0.0d0
      stress_i = 0.0d0
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

         ff = (qij*Pi2*derf_x   - (q_d*CC_2 - dij_zz*CC_3*zij)*dexp_x2 )* i_Area
         local_energy = local_energy + En
         local_force(j) = local_force(j) - ff
         a_force_i = a_force_i + ff
         szz = ff*zij
         local_stress_zz = local_stress_zz + szz
if (l_need_2nd_profile) then
         a_pot(j) = a_pot(j)+En
         a_fi(j) = a_fi(j) + En0*qi
         a_fi_i = a_fi_i + En0*qj
!buffer(i) = buffer(i) + (-Pi2*qj*derf_x + CC_2*dexp_x2*dj_zz)*i_Area
!buffer(j) = buffer(j) + ( Pi2*qi*derf_x + CC_2*dexp_x2*dj_zz)*i_Area
         poti = poti + En
         stress_33(j) = stress_33(j) + szz
         stress_i = stress_i + szz
endif
      enddo
if (l_need_2nd_profile) then
      a_pot(i) = a_pot(i) + poti
      a_fi(i) = a_fi(i) + a_fi_i
      stress_33(i) = stress_33(i) + stress_i
endif
      local_force(i) = local_force(i) + a_force_i
    enddo

!print*, 'stress=',sum(stress_33)/2.0d0,local_stress_zz
!print*, 'apot=',sum(a_pot)/2.0d0 , local_energy
!print*, 'f=0?=',sum(local_force)
!print*,'-localenergy=',-local_energy, -sum(a_pot)/2.0d0
!print*, 'local force=',local_force(1:30)
!read(*,*)
!do i = 1, Natoms
! print*, i,a_pot(i),a_fi(i)*all_p_charges(i)
!enddo

En_Q_cmplx_k0_CORR =  - local_energy
En_Q_cmplx = En_Q_cmplx + En_Q_cmplx_k0_CORR
En_Q_k0_cmplx = En_Q_k0_cmplx + En_Q_cmplx_k0_CORR
En_Q = En_Q + En_Q_cmplx_k0_CORR
fzz = fzz - local_force

if (l_need_2nd_profile) then
 atom_profile%pot = atom_profile%pot - a_pot
 atom_profile%Qpot = atom_profile%Qpot - a_pot
 atom_profile%fi = atom_profile%fi - a_fi
!do i=1,Natoms; atom_profile(i)%buffer3(3)=atom_profile(i)%buffer3(3)-buffer(i); enddo
! atom_profile%szz = atom_profile%szz - stress_33
! atom_profile%EE_xx = atom_profile%EE_xx - e_xx
! atom_profile%EE_yy = atom_profile%EE_yy - e_yy
! atom_profile%EE_zz = atom_profile%EE_zz - e_zz

endif



deallocate(dz,local_force)
if (l_need_2nd_profile) then
 deallocate(a_pot,a_fi,stress_33)
 deallocate(e_xx,e_yy,e_zz)
endif

end subroutine exclude_2D_at_k0_Q


! - ------------------------

subroutine exclude_intra_Q
! I have now both QP and QG charges
use connectivity_ALL_data, only : list_excluded_HALF_no_SFC,size_list_excluded_HALF_no_SFC,MX_excluded
use all_atoms_data, only : xx,yy,zz,all_p_charges,Natoms, fzz,fyy,fxx, i_type_atom, is_charge_distributed,&
                           all_G_charges, all_charges,i_style_atom,xxx,yyy,zzz,is_sfield_constrained
use Ewald_data
use spline_z_k0_module
use sim_cel_data
use boundaries, only : periodic_images
use profiles_data, only : atom_profile, l_need_2nd_profile
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
real(8), allocatable :: a_pot(:),lfx(:),lfy(:),lfz(:),dx(:),dy(:),dz(:),dr_sq(:)
real(8), allocatable :: lstr_xx(:),lstr_yy(:),lstr_zz(:),lstr_xy(:),lstr_xz(:),lstr_yz(:)
real(8), allocatable :: a_fi(:)
real(8), allocatable :: e_xx(:),e_yy(:),e_zz(:)
real(8) x,y,z,fx,fy,fz,sxx,syy,szz,sxy,sxz,syz
real(8) local_stress_xx,local_stress_yy,local_stress_zz
real(8) local_stress_xy,local_stress_xz,local_stress_yz
real(8) stress_i_xx,stress_i_yy,stress_i_zz
real(8) stress_i_xy,stress_i_xz,stress_i_yz
real(8) a_force_i_xx,a_force_i_yy,a_force_i_zz
real(8) inv_rij, rij,vk1,vk2,vk0,vk,t1,t2
real(8) En0G,En0GG, qiG, pref,eta,inv_r_B0,inv_r_B1,inv_r3,B0,B1,r,ppp,inv_r2
integer itype,jtype
logical l_i,l_j,is_sfc_i,is_sfc_j,l_proceed
integer N
integer NDX

N=max(MX_excluded,MX_in_list_14)
allocate(dx(N),dy(N),dz(N),dr_sq(N))
allocate(lfx(Natoms),lfy(Natoms),lfz(Natoms)) ! local_force

if (l_need_2nd_profile) then
  allocate(a_pot(Natoms) , a_fi(Natoms))
  allocate(lstr_xx(Natoms),lstr_yy(Natoms), lstr_zz(Natoms), lstr_xy(Natoms), lstr_xz(Natoms), lstr_yz(Natoms))
  allocate(e_xx(Natoms),e_yy(Natoms),e_zz(Natoms))
  a_pot=0.0d0 ; a_fi = 0.0d0 ;
  lstr_xx=0.0d0; lstr_yy=0.0d0; lstr_zz=0.0d0; lstr_xy=0.0d0; lstr_xz=0.0d0; lstr_yz=0.0d0
  e_xx=0.0d0; e_yy=0.0d0; e_zz=0.0d0
endif

  CC1 = 2.0d0*Ewald_alpha/sqrt_Pi
  local_energy = 0.0d0
  lfx = 0.0d0; lfy = 0.0d0; lfz = 0.0d0
  local_stress_xx=0.0d0; local_stress_yy=0.0d0; local_stress_zz=0.0d0
  local_stress_xy=0.0d0; local_stress_xz=0.0d0; local_stress_yz=0.0d0

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
      a_fi_i = 0.0d0 ; poti = 0.0d0 ;
      a_force_i_xx=0.0d0; a_force_i_yy=0.0d0; a_force_i_zz=0.0d0
      e_zz_i=0.0d0 ; e_yy_i=0.0d0 ; e_xx_i=0.0d0
      stress_i_xx = 0.0d0 ; stress_i_xy = 0.0d0
      stress_i_yy = 0.0d0 ; stress_i_xz = 0.0d0
      stress_i_zz = 0.0d0 ; stress_i_yz = 0.0d0
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

!         x_x_x=Ewald_alpha*rij
!         pref = CC1
!         En0 = derf(x_x_x)*inv_rij  !2.0d0*i_Area
!         En = Qij*En0
!         ff0 =  ( pref*dexp(-x_x_x*x_x_x) - En0 ) * (inv_rij**2)
!         ff = Qij*ff0

        inv_r_B0 = inv_rij-B0
        inv_r_B1 = inv_r3-B1
        En = qij*inv_r_B0
        En0 = inv_r_B0
        local_energy = local_energy + En
        ff=inv_r_B1*qij

!write(66,*)i,j,Qij,xxx,inv_rij,En0,En,local_energy
!read(*,*)

         fx = ff * x ; fy = ff * y ; fz = ff * z
         sxx = fx * x ; syy = fy * y ; szz = fz * z
         sxy = fx * y ; sxz = fx * z ; syz = fy * z
!print*,i,j,ff/418.4,fx/418.4,fy/418.4,fz/418.4!deleteit
!print*,rij
!print*,'En=',En/418.4, B0, derfc(Ewald_alpha*rij)/rij
!read(*,*)
         a_force_i_xx = a_force_i_xx + fx
         a_force_i_yy = a_force_i_yy + fy
         a_force_i_zz = a_force_i_zz + fz
         lfx(j) = lfx(j) - fx
         lfy(j) = lfy(j) - fy
         lfz(j) = lfz(j) - fz
         local_stress_xx = local_stress_xx + sxx
         local_stress_yy = local_stress_yy + syy
         local_stress_zz = local_stress_zz + szz
         local_stress_xy = local_stress_xy + sxy
         local_stress_xz = local_stress_xz + sxz
         local_stress_yz = local_stress_yz + syz

if (l_need_2nd_profile) then
         a_pot(j) = a_pot(j)+En
         a_fi(j) = a_fi(j) + En0*qi
         a_fi_i = a_fi_i + En0*qj
         poti = poti + En

!         lstr_xx(j) = lstr_xx(j) + sxx
!         lstr_yy(j) = lstr_yy(j) + syy
!         lstr_zz(j) = lstr_zz(j) + szz
!         lstr_xy(j) = lstr_xy(j) + sxy
!         lstr_xz(j) = lstr_xz(j) + sxz
!         lstr_yz(j) = lstr_yz(j) + syz
!         stress_i_xx = stress_i_xx + sxx ;
!         stress_i_xy = stress_i_xy + sxy
!         stress_i_yy = stress_i_yy + syy ;
!         stress_i_xz = stress_i_xz + sxz
!         stress_i_zz = stress_i_zz + szz
!         stress_i_yz = stress_i_yz+ syz

!         e_xx(j) = e_xx(j) - (ff0*x)*qi
!         e_xx_i = e_xx_i +   (ff0*x)*qj
!         e_yy(j) = e_yy(j) - (ff0*y)*qi
!         e_yy_i = e_yy_i +   (ff0*y)*qj
!         e_zz(j) = e_zz(j) - (ff0*z)*qi
!         e_zz_i = e_zz_i +   (ff0*z)*qj

endif ! l_need_2nd_profile
      enddo ! k
      lfx(i) = lfx(i) + a_force_i_xx
      lfy(i) = lfy(i) + a_force_i_yy
      lfz(i) = lfz(i) + a_force_i_zz
if (l_need_2nd_profile) then
         a_pot(i) = a_pot(i)+ poti
         a_fi(i) = a_fi(i) + a_fi_i
!         lstr_xx(i) = lstr_xx(i) + stress_i_xx
!         lstr_yy(i) = lstr_yy(i) + stress_i_yy
!         lstr_zz(i) = lstr_zz(i) + stress_i_zz
!         lstr_xy(i) = lstr_xy(i) + stress_i_xy
!         lstr_xz(i) = lstr_xz(i) + stress_i_xz
!         lstr_yz(i) = lstr_yz(i) + stress_i_yz
!         e_xx(i) = e_xx(i) + e_xx_i
!         e_yy(i) = e_yy(i) + e_yy_i
!         e_zz(i) = e_zz(i) + e_zz_i
endif

  enddo  ! i

  En_Q_intra_corr = -local_energy 

ew_self = 0.0d0

  do i = 1, Natoms
    qi = all_charges(i)
       En0 = 2.0d0*qi*(Ewald_alpha/sqrt_Pi)
       local_energy = local_energy + En0*0.5d0*qi
       ew_self = ew_self - En0*0.5d0*qi
       if (l_need_2nd_profile) then
         a_pot(i) = a_pot(i) + En0*qi
         a_fi(i) = a_fi(i) + En0
       endif
  enddo

  En_Q_Gausian_self = 0.0d0
  do i = 1, Natoms
  if (is_charge_distributed(i)) then
    itype = i_style_atom(i)
    qi = all_G_charges(i)
       En0 = -2.0d0*ewald_eta(itype,itype)*qi/sqrt_Pi
       local_energy = local_energy + En0*0.5d0*qi
       En_Q_Gausian_self = En_Q_Gausian_self -  En0*0.5d0*qi
       if (l_need_2nd_profile) then
         a_pot(i) = a_pot(i) + En0*qi
         a_fi(i) = a_fi(i) + En0
       endif
  endif
  enddo


 En_Q = En_Q - local_energy

!print*, 'En_Q_intra_corr=',En_Q_intra_corr
!print*,'ew_self=',ew_self
!print*,'En_Q_Gausian_self=',En_Q_Gausian_self

 fxx = fxx - lfx
 fyy = fyy - lfy
 fzz = fzz - lfz

 stress_excluded(1) = local_stress_xx
 stress_excluded(2) = local_stress_yy
 stress_excluded(3) = local_stress_zz
 stress_excluded(4) = (local_stress_xx+local_stress_yy+local_stress_zz)/3.0d0
 stress_excluded(5) = local_stress_xy
 stress_excluded(6) = local_stress_xz
 stress_excluded(7) = local_stress_yz
 stress_excluded(8) = local_stress_xy
 stress_excluded(9) = local_stress_xz
 stress_excluded(10)= local_stress_yz

 stress(:) = stress(:) - stress_excluded(:)
 
if (l_need_2nd_profile) then
   atom_profile%pot = atom_profile%pot - a_pot
   atom_profile%Qpot = atom_profile%Qpot - a_pot
!   atom_profile%sxx = atom_profile%sxx - lstr_xx
!   atom_profile%sxy = atom_profile%sxy - lstr_xy
!   atom_profile%sxz = atom_profile%sxz - lstr_xz
!   atom_profile%syx = atom_profile%syx - lstr_xy
!   atom_profile%syy = atom_profile%syy - lstr_yy
!   atom_profile%syz = atom_profile%syz - lstr_yz
!   atom_profile%szx = atom_profile%szx - lstr_xz
!   atom_profile%szy = atom_profile%szy - lstr_yz
!   atom_profile%szz = atom_profile%szz - lstr_zz
   atom_profile%fi = atom_profile%fi  - a_fi
!   atom_profile%EE_xx = atom_profile%EE_xx -  e_xx
!   atom_profile%EE_yy = atom_profile%EE_yy -  e_yy
!   atom_profile%EE_zz = atom_profile%EE_zz -  e_zz
endif


deallocate(dx,dy,dz,dr_sq)
deallocate(lfx,lfy,lfz) ! local_force
if (l_need_2nd_profile) then
  deallocate(a_pot , a_fi)
  deallocate(lstr_xx,lstr_yy,lstr_zz,lstr_xy,lstr_xz,lstr_yz)
  deallocate(e_xx,e_yy,e_zz)
endif

end subroutine exclude_intra_Q



subroutine exclude_intra_Q_DIP
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
real(8), allocatable :: a_pot(:),lfx(:),lfy(:),lfz(:),dx(:),dy(:),dz(:),dr_sq(:)
real(8), allocatable :: lstr_xx(:),lstr_yy(:),lstr_zz(:),lstr_xy(:),lstr_xz(:),lstr_yz(:)
real(8), allocatable :: a_fi(:)
real(8), allocatable :: e_xx(:),e_yy(:),e_zz(:)
real(8) x,y,z,fx,fy,fz,sxx,syy,szz,sxy,sxz,syz,syx,szy,szx
real(8) local_stress_xx,local_stress_yy,local_stress_zz
real(8) local_stress_xy,local_stress_xz,local_stress_yz
real(8) local_stress_yx,local_stress_zx,local_stress_zy
real(8) stress_i_xx,stress_i_yy,stress_i_zz
real(8) stress_i_xy,stress_i_xz,stress_i_yz
real(8) a_force_i_xx,a_force_i_yy,a_force_i_zz
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
allocate(lfx(Natoms),lfy(Natoms),lfz(Natoms)) ! local_force

if (l_need_2nd_profile) then
  allocate(a_pot(Natoms) , a_fi(Natoms))
  allocate(lstr_xx(Natoms),lstr_yy(Natoms), lstr_zz(Natoms), lstr_xy(Natoms), lstr_xz(Natoms), lstr_yz(Natoms))
  allocate(e_xx(Natoms),e_yy(Natoms),e_zz(Natoms))
  a_pot=0.0d0 ; a_fi = 0.0d0 ;
  lstr_xx=0.0d0; lstr_yy=0.0d0; lstr_zz=0.0d0; lstr_xy=0.0d0; lstr_xz=0.0d0; lstr_yz=0.0d0
  e_xx=0.0d0; e_yy=0.0d0; e_zz=0.0d0
  allocate(buffer(Natoms,3)); buffer=0.0d0
endif

  CC1 = 2.0d0*Ewald_alpha/sqrt_Pi
  CC0 = CC1 * 0.5d0
!  fct_UP = 2.0d0*Ewald_alpha*Ewald_alpha
!  fct_DOWN = 1.0d0/Ewald_alpha/sqrt_Pi
!  ratio_B1 = fct_UP*fct_DOWN
!  ratio_B2 = ratio_B1*fct_UP
!  ratio_B3 = ratio_B2*fct_UP

  local_energy = 0.0d0
  lfx = 0.0d0; lfy = 0.0d0; lfz = 0.0d0
  local_stress_xx=0.0d0; local_stress_yy=0.0d0; local_stress_zz=0.0d0
  local_stress_xy=0.0d0; local_stress_xz=0.0d0; local_stress_yz=0.0d0
  local_stress_yx=0.0d0; local_stress_zx=0.0d0; local_stress_zy=0.0d0

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
      a_fi_i = 0.0d0 ; poti = 0.0d0 ;
      a_force_i_xx=0.0d0; a_force_i_yy=0.0d0; a_force_i_zz=0.0d0
      e_zz_i=0.0d0 ; e_yy_i=0.0d0 ; e_xx_i=0.0d0
      stress_i_xx = 0.0d0 ; stress_i_xy = 0.0d0
      stress_i_yy = 0.0d0 ; stress_i_xz = 0.0d0
      stress_i_zz = 0.0d0 ; stress_i_yz = 0.0d0
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
        inv_r7 = inv_r5*inv_r2
!        x_x_x=Ewald_alpha*rij
        qj = all_charges(j)
        dipole_xx_j = all_dipoles_xx(j) ; dipole_yy_j=all_dipoles_yy(j); dipole_zz_j=all_dipoles_zz(j)
        dipole_i_times_Rij = x*dipole_xx_i+y*dipole_yy_i+z*dipole_zz_i
        dipole_j_times_Rij = x*dipole_xx_j+y*dipole_yy_j+z*dipole_zz_j
        pipj = dipole_xx_i*dipole_xx_j + dipole_yy_i*dipole_yy_j+ dipole_zz_i*dipole_zz_j
        didj = dipole_i_times_Rij*dipole_j_times_Rij
        G1 = - dipole_i_times_Rij*qj + dipole_j_times_Rij*qi + pipj
        G2 = - didj
        nabla_G1_xx = qj*dipole_xx_i-qi*dipole_xx_j
        nabla_G1_yy = qj*dipole_yy_i-qi*dipole_yy_j
        nabla_G1_zz = qj*dipole_zz_i-qi*dipole_zz_j
        nabla_G2_xx = dipole_j_times_Rij*dipole_xx_i + dipole_i_times_Rij * dipole_xx_j
        nabla_G2_yy = dipole_j_times_Rij*dipole_yy_i + dipole_i_times_Rij * dipole_yy_j
        nabla_G2_zz = dipole_j_times_Rij*dipole_zz_i + dipole_i_times_Rij * dipole_zz_j
!        EEE = dexp(-x_x_x**2)
!        B0 = derfc(x_x_x)*inv_rij
!        B1 =       (B0 + ratio_B1*EEE) * inv_r2
!        B2 = (3.0d0*B1 + ratio_B2*EEE) * inv_r2
!        B3 = (5.0d0*B2 + ratio_B3*EEE) * inv_r2


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
      vk  = vele3(ndx)  ;  vk1 = vele3(ndx+1) ; vk2 = vele3(ndx+2)
       t1 = vk  + (vk1 - vk )*ppp
       t2 = vk1 + (vk2 - vk1)*(ppp - 1.0d0)
       B3 = (t1 + (t2-t1)*(ppp*0.5d0))

        qij = qi*qj

        inv_r_B0 = inv_rij-B0
        inv_r_B1 = inv_r3-B1
        inv_r_B2 = 3.0d0*inv_r5-B2
        inv_r_B3 = 15.0d0*inv_r7-B3
        En = qij*inv_r_B0 + G1*inv_r_B1 + G2*inv_r_B2
        ff0 = qij*inv_r_B1 + G1*inv_r_B2 + G2*inv_r_B3
        fx = ff0*x  + nabla_G1_xx*inv_r_B1 + nabla_G2_xx*inv_r_B2
        fy = ff0*y  + nabla_G1_yy*inv_r_B1 + nabla_G2_yy*inv_r_B2
        fz = ff0*z  + nabla_G1_zz*inv_r_B1 + nabla_G2_zz*inv_r_B2
        local_energy = local_energy + En
        sxx = fx * x ; syy = fy * y ; szz = fz * z
        sxy = fx * y ; sxz = fx * z ; syz = fy * z
        syx = fy * x ; szx = fz * x ; szy = fz * y 
        a_force_i_xx = a_force_i_xx + fx
        a_force_i_yy = a_force_i_yy + fy
        a_force_i_zz = a_force_i_zz + fz
        lfx(j) = lfx(j) - fx
        lfy(j) = lfy(j) - fy
        lfz(j) = lfz(j) - fz
        local_stress_xx = local_stress_xx + sxx
        local_stress_yy = local_stress_yy + syy
        local_stress_zz = local_stress_zz + szz
        local_stress_xy = local_stress_xy + sxy
        local_stress_xz = local_stress_xz + sxz
        local_stress_yz = local_stress_yz + syz
        local_stress_yx = local_stress_yx + syx
        local_stress_zx = local_stress_zx + szx
        local_stress_zy = local_stress_zy + szy

if (l_need_2nd_profile) then
         a_pot(j) = a_pot(j)+En
         a_fi(j) = a_fi(j) +  inv_r_B0 * qi - inv_r_B1 * dipole_i_times_Rij
         a_fi_i  = a_fi_i  +  inv_r_B0 * qj + inv_r_B1 * dipole_j_times_Rij

!buffer(j,1) = buffer(j,1) + ( qi*x + dipole_xx_i)*inv_r_B1 - x*(dipole_i_times_Rij*inv_r_B2)
!buffer(j,2) = buffer(j,2) + ( qi*y + dipole_yy_i)*inv_r_B1 - y*(dipole_i_times_Rij*inv_r_B2)
!buffer(j,3) = buffer(j,3) + ( qi*z + dipole_zz_i)*inv_r_B1 - z*(dipole_i_times_Rij*inv_r_B2)
!buffer(i,1) = buffer(i,1) + (- qj*x + dipole_xx_j)*inv_r_B1 - x*(dipole_j_times_Rij*inv_r_B2)
!buffer(i,2) = buffer(i,2) + (- qj*y + dipole_yy_j)*inv_r_B1 - y*(dipole_j_times_Rij*inv_r_B2)
!buffer(i,3) = buffer(i,3) + (- qj*z + dipole_zz_j)*inv_r_B1 - z*(dipole_j_times_Rij*inv_r_B2)
         poti = poti + En

!         lstr_xx(j) = lstr_xx(j) + sxx
!         lstr_yy(j) = lstr_yy(j) + syy
!         lstr_zz(j) = lstr_zz(j) + szz
!         lstr_xy(j) = lstr_xy(j) + sxy
!         lstr_xz(j) = lstr_xz(j) + sxz
!         lstr_yz(j) = lstr_yz(j) + syz
!         stress_i_xx = stress_i_xx + sxx ;
!         stress_i_xy = stress_i_xy + sxy
!         stress_i_yy = stress_i_yy + syy ;
!         stress_i_xz = stress_i_xz + sxz
!         stress_i_zz = stress_i_zz + szz
!         stress_i_yz = stress_i_yz+ syz
!
!         e_xx(j) = e_xx(j) - (ff0*x)*qi
!         e_xx_i = e_xx_i +   (ff0*x)*qj
!         e_yy(j) = e_yy(j) - (ff0*y)*qi
!         e_yy_i = e_yy_i +   (ff0*y)*qj
!         e_zz(j) = e_zz(j) - (ff0*z)*qi
!         e_zz_i = e_zz_i +   (ff0*z)*qj

endif
    enddo


      lfx(i) = lfx(i) + a_force_i_xx
      lfy(i) = lfy(i) + a_force_i_yy
      lfz(i) = lfz(i) + a_force_i_zz
if (l_need_2nd_profile) then
         a_pot(i) = a_pot(i)+ poti
         a_fi(i) = a_fi(i) + a_fi_i
!         lstr_xx(i) = lstr_xx(i) + stress_i_xx
!         lstr_yy(i) = lstr_yy(i) + stress_i_yy
!         lstr_zz(i) = lstr_zz(i) + stress_i_zz
!         lstr_xy(i) = lstr_xy(i) + stress_i_xy
!         lstr_xz(i) = lstr_xz(i) + stress_i_xz
!         lstr_yz(i) = lstr_yz(i) + stress_i_yz
!         e_xx(i) = e_xx(i) + e_xx_i
!         e_yy(i) = e_yy(i) + e_yy_i
!         e_zz(i) = e_zz(i) + e_zz_i
endif

  enddo  ! i

  En_Q_intra_corr = - local_energy 

ew_self = 0.0d0
ew2=Ewald_alpha*Ewald_alpha
  do i = 1, Natoms
    qi = all_charges(i)
       En = CC0*(qi*qi+2.0d0*ew2/3.0d0*all_dipoles(i)**2)
       local_energy = local_energy + En
       ew_self = ew_self - En
       if (l_need_2nd_profile) then
         a_pot(i) = a_pot(i) + En*2.0d0 
          a_fi(i) = a_fi(i) + qi*(CC0 * 2.0d0)
!buffer(i,1) = buffer(i,1) + all_dipoles_xx(i) * ((CC0*2.0d0)*(2.0d0*ew2/3.0d0))
!buffer(i,2) = buffer(i,2) + all_dipoles_yy(i) * ((CC0*2.0d0)*(2.0d0*ew2/3.0d0))
!buffer(i,3) = buffer(i,3) + all_dipoles_zz(i) * ((CC0*2.0d0)*(2.0d0*ew2/3.0d0))
! I convinced myself that the scalar self-field given by dipole is zero
! see Eq 13 from http://www.pubmedcentral.nih.gov/articlerender.fcgi?artid=2176076 ; the term dot(miu,R12) is zero.
       endif
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
       if (l_need_2nd_profile) then
         a_pot(i) = a_pot(i) + En*2.0d0
          a_fi(i) = a_fi(i)  - (qi * (CC0 * 2.0d0))
buffer(i,1) = buffer(i,1) - all_dipoles_xx(i) * ((CC0*2.0d0)*(2.0d0*eta_sq/3.0d0))
buffer(i,2) = buffer(i,2) - all_dipoles_yy(i) * ((CC0*2.0d0)*(2.0d0*eta_sq/3.0d0))
buffer(i,3) = buffer(i,3) - all_dipoles_zz(i) * ((CC0*2.0d0)*(2.0d0*eta_sq/3.0d0))
       endif
  endif
  enddo


 En_Q = En_Q - local_energy


!print*, 'En_Q_intra_corr=',En_Q_intra_corr
!print*,'ew_self=',ew_self
!print*,'En_Q_Gausian_self=',En_Q_Gausian_self
!print*, 'verify',En_Q_intra_corr-En_Q_Gausian_self,-sum(a_pot)/2.0d0
!print*, 'sum forces=',sum(lfx),sum(lfy),sum(lfz)

 fxx = fxx - lfx
 fyy = fyy - lfy
 fzz = fzz - lfz

 stress_excluded(1) = local_stress_xx
 stress_excluded(2) = local_stress_yy
 stress_excluded(3) = local_stress_zz
 stress_excluded(4) = (local_stress_xx+local_stress_yy+local_stress_zz)/3.0d0
 stress_excluded(5) = local_stress_xy
 stress_excluded(6) = local_stress_xz
 stress_excluded(7) = local_stress_yz
 stress_excluded(8) = local_stress_xy
 stress_excluded(9) = local_stress_xz
 stress_excluded(10)= local_stress_yz

 stress(:) = stress(:) - stress_excluded(:)

if (l_need_2nd_profile) then
   atom_profile%pot = atom_profile%pot - a_pot
   atom_profile%Qpot = atom_profile%Qpot - a_pot
!   atom_profile%sxx = atom_profile%sxx - lstr_xx
!   atom_profile%sxy = atom_profile%sxy - lstr_xy
!   atom_profile%sxz = atom_profile%sxz - lstr_xz
!   atom_profile%syx = atom_profile%syx - lstr_xy
!   atom_profile%syy = atom_profile%syy - lstr_yy
!   atom_profile%syz = atom_profile%syz - lstr_yz
!   atom_profile%szx = atom_profile%szx - lstr_xz
!   atom_profile%szy = atom_profile%szy - lstr_yz
!   atom_profile%szz = atom_profile%szz - lstr_zz
   atom_profile%fi = atom_profile%fi  - a_fi
do i=1,Natoms;atom_profile(i)%buffer3(1:3) = atom_profile(i)%buffer3(1:3) - buffer(i,1:3);enddo
!   atom_profile%EE_xx = atom_profile%EE_xx -  e_xx
!   atom_profile%EE_yy = atom_profile%EE_yy -  e_yy
!   atom_profile%EE_zz = atom_profile%EE_zz -  e_zz
endif


deallocate(dx,dy,dz,dr_sq)
deallocate(lfx,lfy,lfz) ! local_force
if (l_need_2nd_profile) then
  deallocate(a_pot , a_fi)
  deallocate(lstr_xx,lstr_yy,lstr_zz,lstr_xy,lstr_xz,lstr_yz)
  deallocate(e_xx,e_yy,e_zz)
  deallocate(buffer)
endif

end subroutine exclude_intra_Q_DIP


end module exclude_intra



