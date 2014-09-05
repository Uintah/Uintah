
module compute_14_module
implicit none
public :: compute_14_interactions_vdw_q_mu
public :: compute_14_interactions_vdw_q
public :: compute_14_interactions_vdw

public :: compute_14_interactions_vdw_q_mu_ENERGY
public :: compute_14_interactions_vdw_q_ENERGY
public :: compute_14_interactions_vdw_ENERGY

public :: compute_14_interactions_driver
public :: compute_14_interactions_driver_ENERGY
public :: compute_14_interactions_driver_MCmove_1atom

public :: compute_14_interactions_vdw_q_mu_MCmove_1atom
public :: compute_14_interactions_vdw_q_MCmove_1atom
public :: compute_14_interactions_vdw_MCmove_1atom
contains

subroutine compute_14_interactions_driver
use sizes_data, only : N_pairs_14
use CTRLs_data, only : l_DIP_CTRL
use connectivity_ALL_data, only : l_red_14_vdw_CTRL,l_red_14_Q_CTRL,l_red_14_Q_mu_CTRL,l_red_14_mu_mu_CTRL
implicit none 
logical l_1
if (N_pairs_14 < 0) RETURN


if (l_DIP_CTRL) then
  l_1 = (.not.l_red_14_Q_mu_CTRL).and.(.not.l_red_14_mu_mu_CTRL)
  if ((.not.l_red_14_vdw_CTRL).and.(.not.l_red_14_Q_CTRL).and.l_1) then
! Just do noting
  else if (l_red_14_vdw_CTRL.and.(.not.l_red_14_Q_CTRL).and.l_1) then
    call compute_14_interactions_vdw
  else if (l_red_14_vdw_CTRL.and.l_red_14_Q_CTRL.and.l_1) then
    call compute_14_interactions_vdw_Q
  else
    call compute_14_interactions_vdw_Q_mu  ! put it back
  endif
else
  if (l_red_14_vdw_CTRL.and.l_red_14_Q_CTRL) then
    call compute_14_interactions_vdw_Q
  else if (l_red_14_vdw_CTRL.and.(.not.l_red_14_Q_CTRL)) then
    call compute_14_interactions_vdw
  else if ((.not.l_red_14_vdw_CTRL).and.(.not.l_red_14_Q_CTRL)) then
! JUST DO NOTHING
  else
    call compute_14_interactions_vdw_Q
  endif
endif

end subroutine compute_14_interactions_driver

subroutine compute_14_interactions_driver_ENERGY
use sizes_data, only : N_pairs_14
use CTRLs_data, only : l_DIP_CTRL
use connectivity_ALL_data, only : l_red_14_vdw_CTRL,l_red_14_Q_CTRL,l_red_14_Q_mu_CTRL,l_red_14_mu_mu_CTRL
implicit none 
logical l_1
if (N_pairs_14 < 0) RETURN


if (l_DIP_CTRL) then
  l_1 = (.not.l_red_14_Q_mu_CTRL).and.(.not.l_red_14_mu_mu_CTRL)
  if ((.not.l_red_14_vdw_CTRL).and.(.not.l_red_14_Q_CTRL).and.l_1) then
! Just do noting
  else if (l_red_14_vdw_CTRL.and.(.not.l_red_14_Q_CTRL).and.l_1) then
    call compute_14_interactions_vdw_ENERGY
  else if (l_red_14_vdw_CTRL.and.l_red_14_Q_CTRL.and.l_1) then
    call compute_14_interactions_vdw_Q_ENERGY
  else
    call compute_14_interactions_vdw_Q_mu_ENERGY  ! put it back
  endif
else
  if (l_red_14_vdw_CTRL.and.l_red_14_Q_CTRL) then
    call compute_14_interactions_vdw_Q_ENERGY
  else if (l_red_14_vdw_CTRL.and.(.not.l_red_14_Q_CTRL)) then
    call compute_14_interactions_vdw_ENERGY
  else if ((.not.l_red_14_vdw_CTRL).and.(.not.l_red_14_Q_CTRL)) then
! JUST DO NOTHING
  else
    call compute_14_interactions_vdw_Q_ENERGY
  endif
endif

end subroutine compute_14_interactions_driver_ENERGY




subroutine compute_14_interactions_driver_MCmove_1atom(iwhich)
use sizes_data, only : N_pairs_14
use CTRLs_data, only : l_DIP_CTRL
use connectivity_ALL_data, only : l_red_14_vdw_CTRL,l_red_14_Q_CTRL,l_red_14_Q_mu_CTRL,l_red_14_mu_mu_CTRL
implicit none
integer, intent(IN) :: iwhich 
logical l_1
if (N_pairs_14 < 0) RETURN


if (l_DIP_CTRL) then
  l_1 = (.not.l_red_14_Q_mu_CTRL).and.(.not.l_red_14_mu_mu_CTRL)
  if ((.not.l_red_14_vdw_CTRL).and.(.not.l_red_14_Q_CTRL).and.l_1) then
! Just do noting
  else if (l_red_14_vdw_CTRL.and.(.not.l_red_14_Q_CTRL).and.l_1) then
    call compute_14_interactions_vdw_MCmove_1atom(iwhich)
  else if (l_red_14_vdw_CTRL.and.l_red_14_Q_CTRL.and.l_1) then
    call compute_14_interactions_vdw_Q_MCmove_1atom(iwhich)
  else
    call compute_14_interactions_vdw_Q_mu_MCmove_1atom(iwhich)  ! put it back
  endif
else
  if (l_red_14_vdw_CTRL.and.l_red_14_Q_CTRL) then
    call compute_14_interactions_vdw_Q_MCmove_1atom(iwhich)
  else if (l_red_14_vdw_CTRL.and.(.not.l_red_14_Q_CTRL)) then
    call compute_14_interactions_vdw_MCmove_1atom(iwhich)
  else if ((.not.l_red_14_vdw_CTRL).and.(.not.l_red_14_Q_CTRL)) then
! JUST DO NOTHING
  else
    call compute_14_interactions_vdw_Q_MCmove_1atom(iwhich)
  endif
endif

end subroutine compute_14_interactions_driver_MCmove_1atom


!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!


subroutine compute_14_interactions_vdw_q_mu
 use sys_data
 use paralel_env_data
 use math_constants
 use boundaries
 use ALL_atoms_data, only : Natoms, i_style_atom,fxx,fyy,fzz,xxx,yyy,zzz, &
     all_charges,all_dipoles_xx,all_dipoles_yy,all_dipoles_zz,all_dipoles,&
     fshort_xx,fshort_yy,fshort_zz
 use atom_type_data
 use max_sizes_data, only : MX_list_nonbonded
 use connectivity_ALL_data, only : list_14, size_list_14,&
       red_14_vdw, red_14_Q, red_14_Q_mu, &
       l_red_14_vdw_CTRL,l_red_14_Q_CTRL,l_red_14_Q_mu_CTRL,& 
       MX_in_list_14,l_red_14_mu_mu_CTRL,red_14_mu_mu
 use profiles_data
 use energies_data
 use stresses_data
 use atom_type_data
 use interpolate_data
 use variables_short_pairs, only : stress_xx,stress_xy,stress_xz,stress_yy,stress_yz,stress_zz,&
         stress_yx,stress_zx,stress_zy, &
         stress_vdw_xx,stress_vdw_xy,stress_vdw_xz,&
         stress_vdw_yx,stress_vdw_yy,stress_vdw_yz,&
         stress_vdw_zx,stress_vdw_zy,stress_vdw_zz,&
         a_pot_LJ,a_pot_Q,a_fi



implicit none
  integer i , istyle, neightot ! the outer atom index
  real(8) fx,fy,fz,sxx,sxy,sxz,syx,syy,syz,szx,szy,szz,x,y,z
  real(8) ppp, vk,vk1,vk2,En,gk,gk1,t1,t2,ff,Inverse_r_squared,r,r2,inverse_r
  real(8) apot_i_1, a_pot_Q_i
  integer NDX,jtype,i_pair,ju,k,j,jstyle,i1
  real(8) a0, B0,B1,B2,B3,B00,B11,B22,B33,inv_r_B0,inv_r_B1,inv_r_B2,inv_r_B3
  real(8) apress_i_11,apress_i_12,apress_i_13,apress_i_22,apress_i_23,apress_i_33
  real(8) af_i_1_x, af_i_1_y, af_i_1_z
  real(8), allocatable :: ddx(:),ddy(:),ddz(:),ddr_sq(:)
  integer, allocatable :: in_list(:)
 real(8)  dipole_xx_i, dipole_yy_i, dipole_zz_i,qi,qj,qi_G,qj_G,&
                     dipole_i,dipole_j,qij, dipole_i2,dipole_j2,&
          dipole_xx_j,dipole_yy_j,dipole_zz_j,dipole_i_times_Rij,dipole_j_times_Rij
 real(8)  fi_i,ff0,pipj,didj
 real(8) nabla_G1_xx, nabla_G1_yy,nabla_G1_zz,nabla_G2_xx,nabla_G2_yy,nabla_G2_zz,G1,G2
 real(8) inv_rij, inv_r3,inv_r5,inv_r7
 real(8) afs_i_xx,afs_i_yy,afs_i_zz


  allocate(ddx(MX_in_list_14),ddy(MX_in_list_14),ddz(MX_in_list_14),&
           ddr_sq(MX_in_list_14),in_list(MX_in_list_14))
       
    en_14 = 0.0d0   


    do i = 1+rank, Natoms, nprocs
       iStyle = i_style_atom(i)
       neightot = size_list_14(i)
       in_list(1:neightot) = list_14(i,1:neightot)
       i1 = 0 
       do k = 1, neightot
           j = in_list(k)
           i1 = i1 + 1
           ddx(i1) = xxx(i) - xxx(j)
           ddy(i1) = yyy(i) - yyy(j)
           ddz(i1) = zzz(i) - zzz(j)
       enddo ! j = 1,neightot
       if (neightot > 0) then
           call periodic_images(ddx(1:i1),ddy(1:i1),ddz(1:i1))
           ddr_sq(1:i1) = ddx(1:i1)*ddx(1:i1) + ddy(1:i1)*ddy(1:i1) + ddz(1:i1)*ddz(1:i1)
       endif
           qi = all_charges(i) 
           dipole_xx_i=all_dipoles_xx(i);dipole_yy_i=all_dipoles_yy(i);dipole_zz_i=all_dipoles_zz(i)
           af_i_1_x=0.0d0 ; af_i_1_y = 0.0d0 ; af_i_1_z = 0.0d0 ; apot_i_1 = 0.0d0
           if (l_need_2nd_profile) then
             a_pot_Q_i=0.0d0
             fi_i = 0.0d0
             apot_i_1= 0.0d0
           endif
       afs_i_xx=0.0d0;afs_i_yy=0.0d0;afs_i_zz=0.0d0
       do k = 1, neightot
        j = in_list(k)
        r2 = ddr_sq(k)
        if ( r2 < cut_off_sq ) then
            jstyle = i_style_atom(j)
            i_pair = which_atomStyle_pair(istyle,jstyle) ! can replace it by a formula?
            a0 = atom_Style2_vdwPrm(0,i_pair)
            r = dsqrt(r2)
            Inverse_r_squared = 1.0d0/r2
            NDX = max(1,int(r*irdr))
            ppp = (r*irdr) - dble(ndx)
            x = ddx(k)   ; y = ddy(k)    ; z = ddz(k)
            if (a0 > SYS_ZERO.and.l_red_14_vdw_CTRL) then       
                vk  = vvdw(ndx,i_pair)  ;  vk1 = vvdw(ndx+1,i_pair)
                t1 = vk  + (vk1 - vk )*ppp
                t2 = vk1 + (vvdw(ndx+2,i_pair) - vk1)*(ppp - 1.0d0)
                En = - red_14_vdw  *  (t1 + (t2-t1)*ppp*0.5d0)
                gk  = gvdw(ndx,i_pair)  ;  gk1 = gvdw(ndx+1,i_pair)
                t1 = gk  + (gk1 - gk )*ppp
                t2 = gk1 + (gvdw(ndx+2,i_pair) - gk1)*(ppp - 1.0d0)
                ff = - red_14_vdw  *  (t1 + (t2-t1)*ppp*0.5d0)*Inverse_r_squared
                
               fx = ff*x    ;  fy = ff*y   ;  fz = ff*z
              sxx = fx*x    ;  sxy = fx*y  ;  sxz = fx*z
                               syy = fy*y  ;  syz = fy*z
                                              szz = fz*z
afs_i_xx = afs_i_xx + fx
afs_i_yy = afs_i_yy + fy
afs_i_zz = afs_i_zz + fz
fshort_xx(j) = fshort_xx(j) - fx
fshort_yy(j) = fshort_yy(j) - fy
fshort_zz(j) = fshort_zz(j) - fz

                  stress_vdw_xx = stress_vdw_xx + sxx ;
                  stress_vdw_xy = stress_vdw_xy + sxy ;
                  stress_vdw_xz = stress_vdw_xz + sxz ;
                  stress_vdw_yy = stress_vdw_yy + syy ;
                  stress_vdw_yz = stress_vdw_yz + syz ;
                  stress_vdw_zz = stress_vdw_zz + szz ;
                 af_i_1_x = af_i_1_x+fx ; af_i_1_y = af_i_1_y + fy ; af_i_1_z = af_i_1_z + fz
                 fxx(j)   = fxx(j) - fx ; fyy(j)   = fyy(j) - fy   ; fzz(j)   = fzz(j) - fz
                 en_vdw = en_vdw + En
                 en_14 = en_14 + En
                 
                 if (l_need_2nd_profile) then
                    apot_i_1=apot_i_1 + En 
                    a_pot_LJ(j)=a_pot_LJ(j) + En
                 endif
                 
                 
            endif
            include 'interpolate_2.frg'
            vk  = vele(ndx)  ;  vk1 = vele(ndx+1) ; vk2 = vele(ndx+2)
            t1 = vk  + (vk1 - vk )*ppp
            t2 = vk1 + (vk2 - vk1)*(ppp - 1.0d0)
            B00 = (t1 + (t2-t1)*(ppp*0.5d0))
            vk  = gele(ndx)  ;  vk1 = gele(ndx+1) ; vk2 = gele(ndx+2)
            t1 = vk  + (vk1 - vk )*ppp
            t2 = vk1 + (vk2 - vk1)*(ppp - 1.0d0)
            B11 = (t1 + (t2-t1)*(ppp*0.5d0))
            inv_rij = 1.0d0 / r ;
            inv_r3 = Inverse_r_squared * inv_rij
            inv_r_B0 = inv_rij - (B0-B00)
            inv_r_B1 = inv_r3 - (B1-B11)
            qj = all_charges(j)
            qij = qi*qj
            if (l_red_14_Q_CTRL) then
              En = -(red_14_Q*qij) * inv_r_B0
              ff = -(red_14_Q*qij) * inv_r_B1
              
              fx = ff*x    ;  fy = ff*y   ;  fz = ff*z
              sxx = fx*x    ;  sxy = fx*y  ;  sxz = fx*z
                               syy = fy*y  ;  syz = fy*z
                                              szz = fz*z
afs_i_xx = afs_i_xx + fx
afs_i_yy = afs_i_yy + fy
afs_i_zz = afs_i_zz + fz
fshort_xx(j) = fshort_xx(j) - fx
fshort_yy(j) = fshort_yy(j) - fy
fshort_zz(j) = fshort_zz(j) - fz

                  stress_xx = stress_xx - sxx ;
                  stress_xy = stress_xy - sxy ;
                  stress_xz = stress_xz - sxz ;
                  stress_yy = stress_yy - syy ;
                  stress_yz = stress_yz - syz ;
                  stress_zz = stress_zz - szz ;
                 af_i_1_x = af_i_1_x+fx ; af_i_1_y = af_i_1_y + fy ; af_i_1_z = af_i_1_z + fz
                 fxx(j)   = fxx(j) - fx ; fyy(j)   = fyy(j) - fy   ; fzz(j)   = fzz(j) - fz
                 en_Qreal = en_Qreal + En
                 en_14 = en_14 + En
                 if (l_need_2nd_profile)  then
                     a_pot_Q_i = a_pot_Q_i + En
                     fi_i = fi_i + inv_r_B0 * qj*(-red_14_Q)
                     a_pot_Q(j) = a_pot_Q(j) + En
                     a_fi(j) = a_fi(j) +inv_r_B0 * qi * (-red_14_Q)
                 endif
              endif ! l_red_14_Q_CTRL
              if (l_red_14_Q_mu_CTRL.or.l_red_14_mu_mu_CTRL) then
      vk  = vele2_G(ndx,i_pair)  ;  vk1 = vele2_G(ndx+1,i_pair) ; vk2 = vele2_G(ndx+2,i_pair)
       t1 = vk  + (vk1 - vk )*ppp
       t2 = vk1 + (vk2 - vk1)*(ppp - 1.0d0)
       B2 = (t1 + (t2-t1)*(ppp*0.5d0))
      vk  = vele3_G(ndx,i_pair)  ;  vk1 = vele3_G(ndx+1,i_pair) ; vk2 = vele3_G(ndx+2,i_pair)
       t1 = vk  + (vk1 - vk )*ppp
       t2 = vk1 + (vk2 - vk1)*(ppp - 1.0d0)
       B3 = (t1 + (t2-t1)*(ppp*0.5d0))
      vk  = vele2(ndx)  ;  vk1 = vele2(ndx+1) ; vk2 = vele2(ndx+2)
       t1 = vk  + (vk1 - vk )*ppp
       t2 = vk1 + (vk2 - vk1)*(ppp - 1.0d0)
       B22 = (t1 + (t2-t1)*(ppp*0.5d0))
      vk  = vele3(ndx)  ;  vk1 = vele3(ndx+1) ; vk2 = vele3(ndx+2)
       t1 = vk  + (vk1 - vk )*ppp
       t2 = vk1 + (vk2 - vk1)*(ppp - 1.0d0)
       B33 = (t1 + (t2-t1)*(ppp*0.5d0))
       inv_r5 = inv_r3 * Inverse_r_squared
       inv_r7 = inv_r5 * Inverse_r_squared 
       inv_r_B2 = 3.0d0*inv_r5-(B2-B22)
       inv_r_B3 = 15.0d0*inv_r7-(B3-B33)

                   dipole_xx_j=all_dipoles_xx(j);dipole_yy_j=all_dipoles_yy(j);dipole_zz_j=all_dipoles_zz(j)
                   dipole_i_times_Rij = x*dipole_xx_i+y*dipole_yy_i+z*dipole_zz_i
                   dipole_j_times_Rij = x*dipole_xx_j+y*dipole_yy_j+z*dipole_zz_j
                   pipj = dipole_xx_i*dipole_xx_j + dipole_yy_i*dipole_yy_j+ dipole_zz_i*dipole_zz_j
                   didj = dipole_i_times_Rij*dipole_j_times_Rij
                   G1 = - dipole_i_times_Rij*qj + dipole_j_times_Rij*qi !+ pipj
                   G2 = - didj
                   nabla_G1_xx = qj*dipole_xx_i-qi*dipole_xx_j
                   nabla_G1_yy = qj*dipole_yy_i-qi*dipole_yy_j
                   nabla_G1_zz = qj*dipole_zz_i-qi*dipole_zz_j
                   nabla_G2_xx = dipole_j_times_Rij*dipole_xx_i + dipole_i_times_Rij * dipole_xx_j
                   nabla_G2_yy = dipole_j_times_Rij*dipole_yy_i + dipole_i_times_Rij * dipole_yy_j
                   nabla_G2_zz = dipole_j_times_Rij*dipole_zz_i + dipole_i_times_Rij * dipole_zz_j
                   En  =   inv_r_B1*G1*(-red_14_Q_mu) + (inv_r_B2*G2+pipj*inv_r_B1) *(-red_14_mu_mu)
                   ff0 =   inv_r_B2*G1*(-red_14_Q_mu) + (pipj*inv_r_B2+ inv_r_B3*G2)*(-red_14_mu_mu) 

                   fx = ff0*x  +  nabla_G1_xx*(inv_r_B1*(-red_14_Q_mu)) + nabla_G2_xx*(inv_r_B2*(-red_14_mu_mu))
                   fy = ff0*y  +  nabla_G1_yy*(inv_r_B1*(-red_14_Q_mu)) + nabla_G2_yy*(inv_r_B2*(-red_14_mu_mu))
                   fz = ff0*z  +  nabla_G1_zz*(inv_r_B1*(-red_14_Q_mu)) + nabla_G2_zz*(inv_r_B2*(-red_14_mu_mu))

                   sxx = fx*x  ;  sxy = fx*y  ;  sxz = fx*z
                                  syy = fy*y  ;  syz = fy*z
                                                 szz = fz*z
                  stress_xx = stress_xx + sxx ;
                  stress_xy = stress_xy + sxy ;
                  stress_xz = stress_xz + sxz ;
                  stress_yy = stress_yy + syy ;
                  stress_yz = stress_yz + syz ;
                  stress_zz = stress_zz + szz ;
                 af_i_1_x = af_i_1_x+fx ; af_i_1_y = af_i_1_y + fy ; af_i_1_z = af_i_1_z + fz
                 fxx(j)   = fxx(j) - fx ; fyy(j)   = fyy(j) - fy   ; fzz(j)   = fzz(j) - fz

                if (l_need_2nd_profile)  then
                     a_pot_Q_i = a_pot_Q_i + En
                     fi_i = fi_i + inv_r_B1 * dipole_j_times_Rij * (-red_14_Q_mu)
                     a_pot_Q(j) = a_pot_Q(j) + En
                     a_fi(j) = a_fi(j)  - inv_r_B1 * dipole_i_times_Rij * (-red_14_Q_mu)
                 endif
                 en_Qreal = en_Qreal + En 
                 en_14 = en_14 + En
            endif  !   l_red_14_Q_CTRL.or.l_red_14_Q_mu_CTRL
              
        endif
      enddo



    call finalize(i)
fshort_xx(i) = fshort_xx(i) + afs_i_xx
fshort_yy(i) = fshort_yy(i) + afs_i_yy
fshort_zz(i) = fshort_zz(i) + afs_i_zz

    enddo ! i = 1+rank, Natoms, nprocs

deallocate(ddx,ddy,ddz,ddr_sq,in_list) 
contains



   subroutine finalize(i)
   integer, intent(IN) :: i
   fxx(i) = fxx(i) + af_i_1_x
   fyy(i) = fyy(i) + af_i_1_y
   fzz(i) = fzz(i) + af_i_1_z
   if (l_need_2nd_profile) then
      a_pot_LJ(i)=a_pot_LJ(i)+apot_i_1
      a_pot_Q(i) = a_pot_Q(i) + a_pot_Q_i
      a_fi(i) = a_fi(i) + fi_i! field acting on charge as a result of the action of a nother charge
!      a_press_LJ_11(i)=a_press_LJ_11(i)+apress_i_11
!      a_press_LJ_22(i)=a_press_LJ_22(i)+apress_i_22
!      a_press_LJ_33(i)=a_press_LJ_33(i)+apress_i_33
!      a_press_LJ_12(i)=a_press_LJ_12(i)+apress_i_12
!      a_press_LJ_13(i)=a_press_LJ_13(i)+apress_i_13
!      a_press_LJ_23(i)=a_press_LJ_23(i)+apress_i_23
   endif
  end subroutine finalize


end subroutine compute_14_interactions_vdw_q_mu


!-------------------------------------------
!-------------------------------------------
!-------------------------------------------

subroutine compute_14_interactions_vdw_q_mu_ENERGY
 use sys_data
 use paralel_env_data
 use math_constants
 use boundaries
 use ALL_atoms_data, only : Natoms, i_style_atom,xxx,yyy,zzz, &
     all_charges,all_dipoles_xx,all_dipoles_yy,all_dipoles_zz,all_dipoles
 use atom_type_data
 use max_sizes_data, only : MX_list_nonbonded
 use connectivity_ALL_data, only : list_14, size_list_14,&
       red_14_vdw, red_14_Q, red_14_Q_mu, &
       l_red_14_vdw_CTRL,l_red_14_Q_CTRL,l_red_14_Q_mu_CTRL,& 
       MX_in_list_14,l_red_14_mu_mu_CTRL,red_14_mu_mu
 use profiles_data
 use energies_data
 use atom_type_data
 use interpolate_data
 use variables_short_pairs, only : a_pot_LJ,a_pot_Q



implicit none
  integer i , istyle, neightot ! the outer atom index
  real(8) x,y,z
  real(8) ppp, vk,vk1,vk2,En,gk,gk1,t1,t2,ff,Inverse_r_squared,r,r2,inverse_r
  real(8) apot_i_1, a_pot_Q_i
  integer NDX,jtype,i_pair,ju,k,j,jstyle,i1
  real(8) a0, B0,B1,B2,B3,B00,B11,B22,B33,inv_r_B0,inv_r_B1,inv_r_B2,inv_r_B3
  real(8), allocatable :: ddx(:),ddy(:),ddz(:),ddr_sq(:)
  integer, allocatable :: in_list(:)
 real(8)  dipole_xx_i, dipole_yy_i, dipole_zz_i,qi,qj,qi_G,qj_G,&
                     dipole_i,dipole_j,qij, dipole_i2,dipole_j2,&
          dipole_xx_j,dipole_yy_j,dipole_zz_j,dipole_i_times_Rij,dipole_j_times_Rij
 real(8)  fi_i,ff0,pipj,didj
 real(8) G1,G2
 real(8) inv_rij, inv_r3,inv_r5,inv_r7
 real(8) afs_i_xx,afs_i_yy,afs_i_zz


  allocate(ddx(MX_in_list_14),ddy(MX_in_list_14),ddz(MX_in_list_14),&
           ddr_sq(MX_in_list_14),in_list(MX_in_list_14))
       
    en_14 = 0.0d0   


    do i = 1+rank, Natoms, nprocs
       iStyle = i_style_atom(i)
       neightot = size_list_14(i)
       in_list(1:neightot) = list_14(i,1:neightot)
       i1 = 0 
       do k = 1, neightot
           j = in_list(k)
           i1 = i1 + 1
           ddx(i1) = xxx(i) - xxx(j)
           ddy(i1) = yyy(i) - yyy(j)
           ddz(i1) = zzz(i) - zzz(j)
       enddo ! j = 1,neightot
       if (neightot > 0) then
           call periodic_images(ddx(1:i1),ddy(1:i1),ddz(1:i1))
           ddr_sq(1:i1) = ddx(1:i1)*ddx(1:i1) + ddy(1:i1)*ddy(1:i1) + ddz(1:i1)*ddz(1:i1)
       endif
           qi = all_charges(i) 
           dipole_xx_i=all_dipoles_xx(i);dipole_yy_i=all_dipoles_yy(i);dipole_zz_i=all_dipoles_zz(i)
       do k = 1, neightot
        j = in_list(k)
        r2 = ddr_sq(k)
        if ( r2 < cut_off_sq ) then
            jstyle = i_style_atom(j)
            i_pair = which_atomStyle_pair(istyle,jstyle) ! can replace it by a formula?
            a0 = atom_Style2_vdwPrm(0,i_pair)
            r = dsqrt(r2)
            Inverse_r_squared = 1.0d0/r2
            inv_rij = 1.0d0 / r ;
            inv_r3 = Inverse_r_squared * inv_rij
            NDX = max(1,int(r*irdr))
            ppp = (r*irdr) - dble(ndx)
            x = ddx(k)   ; y = ddy(k)    ; z = ddz(k)
            if (a0 > SYS_ZERO.and.l_red_14_vdw_CTRL) then       
                vk  = vvdw(ndx,i_pair)  ;  vk1 = vvdw(ndx+1,i_pair)
                t1 = vk  + (vk1 - vk )*ppp
                t2 = vk1 + (vvdw(ndx+2,i_pair) - vk1)*(ppp - 1.0d0)
                En = - red_14_vdw  *  (t1 + (t2-t1)*ppp*0.5d0)
                
                 en_vdw = en_vdw + En
                 en_14 = en_14 + En
               
            endif
            include 'interpolate_2.frg'
            vk  = vele(ndx)  ;  vk1 = vele(ndx+1) ; vk2 = vele(ndx+2)
            t1 = vk  + (vk1 - vk )*ppp
            t2 = vk1 + (vk2 - vk1)*(ppp - 1.0d0)
            B00 = (t1 + (t2-t1)*(ppp*0.5d0))
            vk  = gele(ndx)  ;  vk1 = gele(ndx+1) ; vk2 = gele(ndx+2)
            t1 = vk  + (vk1 - vk )*ppp
            t2 = vk1 + (vk2 - vk1)*(ppp - 1.0d0)
            B11 = (t1 + (t2-t1)*(ppp*0.5d0))
            inv_rij = 1.0d0 / r ;
            inv_r_B0 = inv_rij - (B0-B00)         
            qj = all_charges(j)
            qij = qi*qj
            if (l_red_14_Q_CTRL) then
              En = -(red_14_Q*qij) * inv_r_B0 
              en_Qreal = en_Qreal + En
              en_14 = en_14 + En
                 
            endif ! l_red_14_Q_CTRL
            if (l_red_14_Q_mu_CTRL.or.l_red_14_mu_mu_CTRL) then
      vk  = vele2_G(ndx,i_pair)  ;  vk1 = vele2_G(ndx+1,i_pair) ; vk2 = vele2_G(ndx+2,i_pair)
       t1 = vk  + (vk1 - vk )*ppp
       t2 = vk1 + (vk2 - vk1)*(ppp - 1.0d0)
       B2 = (t1 + (t2-t1)*(ppp*0.5d0))
      vk  = vele3_G(ndx,i_pair)  ;  vk1 = vele3_G(ndx+1,i_pair) ; vk2 = vele3_G(ndx+2,i_pair)
       t1 = vk  + (vk1 - vk )*ppp
       t2 = vk1 + (vk2 - vk1)*(ppp - 1.0d0)
       B3 = (t1 + (t2-t1)*(ppp*0.5d0))
      vk  = vele2(ndx)  ;  vk1 = vele2(ndx+1) ; vk2 = vele2(ndx+2)
       t1 = vk  + (vk1 - vk )*ppp
       t2 = vk1 + (vk2 - vk1)*(ppp - 1.0d0)
       B22 = (t1 + (t2-t1)*(ppp*0.5d0))

        inv_r_B0 = inv_rij - (B0-B00)
        inv_r_B1 = inv_r3 - (B1-B11)

       inv_r5 = inv_r3 * Inverse_r_squared
       inv_r_B2 = 3.0d0*inv_r5-(B2-B22)

        dipole_xx_j=all_dipoles_xx(j);dipole_yy_j=all_dipoles_yy(j);dipole_zz_j=all_dipoles_zz(j)
        dipole_i_times_Rij = x*dipole_xx_i+y*dipole_yy_i+z*dipole_zz_i
        dipole_j_times_Rij = x*dipole_xx_j+y*dipole_yy_j+z*dipole_zz_j
        pipj = dipole_xx_i*dipole_xx_j + dipole_yy_i*dipole_yy_j+ dipole_zz_i*dipole_zz_j
        didj = dipole_i_times_Rij*dipole_j_times_Rij
        G1 = - dipole_i_times_Rij*qj + dipole_j_times_Rij*qi !+ pipj
        G2 = - didj
                   
        En  =   inv_r_B1*G1*(-red_14_Q_mu) + (inv_r_B2*G2+pipj*inv_r_B1) *(-red_14_mu_mu)
        en_Qreal = en_Qreal + En 
        en_14 = en_14 + En
        endif  !   l_red_14_Q_CTRL.or.l_red_14_Q_mu_CTRL
              
        endif
      enddo


    enddo ! i = 1+rank, Natoms, nprocs

deallocate(ddx,ddy,ddz,ddr_sq,in_list) 

end subroutine compute_14_interactions_vdw_q_mu_ENERGY



!---------------------------------------------
!-----------------------------------------------
!--------------------------------------------
subroutine compute_14_interactions_vdw_q
 use sys_data
 use paralel_env_data
 use math_constants
 use boundaries
 use ALL_atoms_data, only : Natoms, i_style_atom,fxx,fyy,fzz,xxx,yyy,zzz,all_charges,&
                            fshort_xx,fshort_yy,fshort_zz
 use atom_type_data
 use max_sizes_data, only : MX_list_nonbonded
 use connectivity_ALL_data, only : list_14, size_list_14,&
       red_14_vdw, red_14_Q, red_14_Q_mu, &
       l_red_14_vdw_CTRL,l_red_14_Q_CTRL,l_red_14_Q_mu_CTRL,&
       MX_in_list_14,l_red_14_mu_mu_CTRL,red_14_mu_mu
 use profiles_data
 use energies_data
 use stresses_data
 use atom_type_data
 use interpolate_data
 use variables_short_pairs, only : stress_xx,stress_xy,stress_xz,stress_yy,stress_yz,stress_zz,&
         stress_yx,stress_zx,stress_zy, &
         stress_vdw_xx,stress_vdw_xy,stress_vdw_xz,&
         stress_vdw_yx,stress_vdw_yy,stress_vdw_yz,&
         stress_vdw_zx,stress_vdw_zy,stress_vdw_zz,&
         a_pot_LJ,a_pot_Q,a_fi



implicit none
  integer i , istyle, neightot ! the outer atom index
  real(8) fx,fy,fz,sxx,sxy,sxz,syx,syy,syz,szx,szy,szz,x,y,z
  real(8) ppp, vk,vk1,vk2,En,gk,gk1,t1,t2,ff,Inverse_r_squared,r,r2,inverse_r
  real(8) apot_i_1, a_pot_Q_i
  integer NDX,jtype,i_pair,ju,k,j,jstyle,i1
  real(8) a0, B0,B1,B2,B3,B00,B11,B22,B33,inv_r_B0,inv_r_B1,inv_r_B2,inv_r_B3
  real(8) apress_i_11,apress_i_12,apress_i_13,apress_i_22,apress_i_23,apress_i_33
  real(8) af_i_1_x, af_i_1_y, af_i_1_z
  real(8), allocatable :: ddx(:),ddy(:),ddz(:),ddr_sq(:)
  integer, allocatable :: in_list(:)
 real(8)  qi,qj,qi_G,qj_G,qij
 real(8)  fi_i,ff0
 real(8) inv_r3, inv_rij
 real(8) afs_i_xx,afs_i_yy,afs_i_zz



  allocate(ddx(MX_in_list_14),ddy(MX_in_list_14),ddz(MX_in_list_14),&
           ddr_sq(MX_in_list_14),in_list(MX_in_list_14))

    en_14 = 0.0d0

    do i = 1+rank, Natoms, nprocs
       iStyle = i_style_atom(i)
       neightot = size_list_14(i)
       in_list(1:neightot) = list_14(i,1:neightot)
       i1 = 0
       do k = 1, neightot
           j = in_list(k)
           i1 = i1 + 1
           ddx(i1) = xxx(i) - xxx(j)
           ddy(i1) = yyy(i) - yyy(j)
           ddz(i1) = zzz(i) - zzz(j)
       enddo ! j = 1,neightot
       if (neightot > 0) then
           call periodic_images(ddx(1:i1),ddy(1:i1),ddz(1:i1))
           ddr_sq(1:i1) = ddx(1:i1)*ddx(1:i1) + ddy(1:i1)*ddy(1:i1) + ddz(1:i1)*ddz(1:i1)
       endif
           qi = all_charges(i)
           af_i_1_x=0.0d0 ; af_i_1_y = 0.0d0 ; af_i_1_z = 0.0d0 ; apot_i_1 = 0.0d0
           if (l_need_2nd_profile) then
             a_pot_Q_i=0.0d0
             fi_i = 0.0d0
             apot_i_1= 0.0d0
           endif
       afs_i_xx=0.0d0;afs_i_yy=0.0d0;afs_i_zz=0.0d0
       do k = 1, neightot
        j = in_list(k)
        r2 = ddr_sq(k)
        if ( r2 < cut_off_sq ) then
            jstyle = i_style_atom(j)
            i_pair = which_atomStyle_pair(istyle,jstyle) ! can replace it by a formula?
            a0 = atom_Style2_vdwPrm(0,i_pair)
            r = dsqrt(r2)
            Inverse_r_squared = 1.0d0/r2
            NDX = max(1,int(r*irdr))
            ppp = (r*irdr) - dble(ndx)
            x = ddx(k)   ; y = ddy(k)    ; z = ddz(k)
            if (a0 > SYS_ZERO.and.l_red_14_vdw_CTRL) then
                vk  = vvdw(ndx,i_pair)  ;  vk1 = vvdw(ndx+1,i_pair)
                t1 = vk  + (vk1 - vk )*ppp
                t2 = vk1 + (vvdw(ndx+2,i_pair) - vk1)*(ppp - 1.0d0)
                En = - red_14_vdw  *  (t1 + (t2-t1)*ppp*0.5d0)
                gk  = gvdw(ndx,i_pair)  ;  gk1 = gvdw(ndx+1,i_pair)
                t1 = gk  + (gk1 - gk )*ppp
                t2 = gk1 + (gvdw(ndx+2,i_pair) - gk1)*(ppp - 1.0d0)
                ff = - red_14_vdw  *  (t1 + (t2-t1)*ppp*0.5d0)*Inverse_r_squared

               fx = ff*x    ;  fy = ff*y   ;  fz = ff*z
              sxx = fx*x    ;  sxy = fx*y  ;  sxz = fx*z
                               syy = fy*y  ;  syz = fy*z
                                              szz = fz*z

afs_i_xx = afs_i_xx + fx
afs_i_yy = afs_i_yy + fy
afs_i_zz = afs_i_zz + fz
fshort_xx(j) = fshort_xx(j) - fx
fshort_yy(j) = fshort_yy(j) - fy
fshort_zz(j) = fshort_zz(j) - fz

                  stress_vdw_xx = stress_vdw_xx + sxx ;
                  stress_vdw_xy = stress_vdw_xy + sxy ;
                  stress_vdw_xz = stress_vdw_xz + sxz ;
                  stress_vdw_yy = stress_vdw_yy + syy ;
                  stress_vdw_yz = stress_vdw_yz + syz ;
                  stress_vdw_zz = stress_vdw_zz + szz ;
                 af_i_1_x = af_i_1_x+fx ; af_i_1_y = af_i_1_y + fy ; af_i_1_z = af_i_1_z + fz
                 fxx(j)   = fxx(j) - fx ; fyy(j)   = fyy(j) - fy   ; fzz(j)   = fzz(j) - fz
                 en_vdw = en_vdw + En
                 en_14 = en_14 + En

                 if (l_need_2nd_profile) then
                    apot_i_1=apot_i_1 + En
                    a_pot_LJ(j)=a_pot_LJ(j) + En
                 endif


            endif
            include 'interpolate_2.frg'
            qj = all_charges(j)
            qij = qi*qj
            if (l_red_14_Q_CTRL) then
           vk  = vele(ndx)  ;  vk1 = vele(ndx+1) ; vk2 = vele(ndx+2)
           t1 = vk  + (vk1 - vk )*ppp
           t2 = vk1 + (vk2 - vk1)*(ppp - 1.0d0)
           B00 = (t1 + (t2-t1)*(ppp*0.5d0))
           vk  = gele(ndx)  ;  vk1 = gele(ndx+1) ; vk2 = gele(ndx+2)
           t1 = vk  + (vk1 - vk )*ppp
           t2 = vk1 + (vk2 - vk1)*(ppp - 1.0d0)
           B11 = (t1 + (t2-t1)*(ppp*0.5d0))
           inv_rij = 1.0d0 / r ;
           inv_r3 = Inverse_r_squared * inv_rij
           inv_r_B0 = inv_rij - (B0-B00)
           inv_r_B1 = inv_r3 - (B1-B11)

              En = -(red_14_Q*qij) * inv_r_B0
              ff = -(red_14_Q*qij) * inv_r_B1

              fx = ff*x    ;  fy = ff*y   ;  fz = ff*z
              sxx = fx*x    ;  sxy = fx*y  ;  sxz = fx*z
                               syy = fy*y  ;  syz = fy*z
                                              szz = fz*z

afs_i_xx = afs_i_xx + fx
afs_i_yy = afs_i_yy + fy
afs_i_zz = afs_i_zz + fz
fshort_xx(j) = fshort_xx(j) - fx
fshort_yy(j) = fshort_yy(j) - fy
fshort_zz(j) = fshort_zz(j) - fz

                  stress_xx = stress_xx + sxx ;
                  stress_xy = stress_xy + sxy ;
                  stress_xz = stress_xz + sxz ;
                  stress_yy = stress_yy + syy ;
                  stress_yz = stress_yz + syz ;
                  stress_zz = stress_zz + szz ;
                 af_i_1_x = af_i_1_x+fx ; af_i_1_y = af_i_1_y + fy ; af_i_1_z = af_i_1_z + fz
                 fxx(j)   = fxx(j) - fx ; fyy(j)   = fyy(j) - fy   ; fzz(j)   = fzz(j) - fz
                 en_Qreal = en_Qreal + En
                 en_14 = en_14 + En
                 if (l_need_2nd_profile)  then
                     a_pot_Q_i = a_pot_Q_i + En
                     fi_i = fi_i + inv_r_B0 * qj*(-red_14_Q)
                     a_pot_Q(j) = a_pot_Q(j) + En
                     a_fi(j) = a_fi(j) + inv_r_B0 * qi * (-red_14_Q)
                 endif
              endif ! l_red_14_Q_CTRL
        endif
      enddo



    call finalize(i)
fshort_xx(i) = fshort_xx(i) + afs_i_xx
fshort_yy(i) = fshort_yy(i) + afs_i_yy
fshort_zz(i) = fshort_zz(i) + afs_i_zz

    enddo ! i = 1+rank, Natoms, nprocs
deallocate(ddx,ddy,ddz,ddr_sq,in_list)
contains



   subroutine finalize(i)
   integer, intent(IN) :: i
   fxx(i) = fxx(i) + af_i_1_x
   fyy(i) = fyy(i) + af_i_1_y
   fzz(i) = fzz(i) + af_i_1_z
   if (l_need_2nd_profile) then
      a_pot_LJ(i)=a_pot_LJ(i)+apot_i_1
      a_pot_Q(i) = a_pot_Q(i) + a_pot_Q_i
      a_fi(i) = a_fi(i) + fi_i! field acting on charge as a result of the action of a nother charge
!      a_press_LJ_11(i)=a_press_LJ_11(i)+apress_i_11
!      a_press_LJ_22(i)=a_press_LJ_22(i)+apress_i_22
!      a_press_LJ_33(i)=a_press_LJ_33(i)+apress_i_33
!      a_press_LJ_12(i)=a_press_LJ_12(i)+apress_i_12
!      a_press_LJ_13(i)=a_press_LJ_13(i)+apress_i_13
!      a_press_LJ_23(i)=a_press_LJ_23(i)+apress_i_23
   endif
  end subroutine finalize


end subroutine compute_14_interactions_vdw_q

!-----------------------------
!-----------------------------
!-----------------------------
subroutine compute_14_interactions_vdw_q_ENERGY
 use sys_data
 use paralel_env_data
 use math_constants
 use boundaries
 use ALL_atoms_data, only : Natoms, i_style_atom,xxx,yyy,zzz,all_charges
 use atom_type_data
 use max_sizes_data, only : MX_list_nonbonded
 use connectivity_ALL_data, only : list_14, size_list_14,&
       red_14_vdw, red_14_Q, red_14_Q_mu, &
       l_red_14_vdw_CTRL,l_red_14_Q_CTRL,l_red_14_Q_mu_CTRL,&
       MX_in_list_14,l_red_14_mu_mu_CTRL,red_14_mu_mu
 use energies_data
 use atom_type_data
 use interpolate_data
 use variables_short_pairs, only :  a_pot_LJ,a_pot_Q



implicit none
  integer i , istyle, neightot ! the outer atom index
  real(8) fx,fy,fz,sxx,sxy,sxz,syx,syy,syz,szx,szy,szz,x,y,z
  real(8) ppp, vk,vk1,vk2,En,gk,gk1,t1,t2,ff,Inverse_r_squared,r,r2,inverse_r
  real(8) apot_i_1, a_pot_Q_i
  integer NDX,jtype,i_pair,ju,k,j,jstyle,i1
  real(8) a0, B0,B1,B2,B3,B00,B11,B22,B33,inv_r_B0,inv_r_B1,inv_r_B2,inv_r_B3
  real(8) apress_i_11,apress_i_12,apress_i_13,apress_i_22,apress_i_23,apress_i_33
  real(8) af_i_1_x, af_i_1_y, af_i_1_z
  real(8), allocatable :: ddx(:),ddy(:),ddz(:),ddr_sq(:)
  integer, allocatable :: in_list(:)
 real(8)  qi,qj,qi_G,qj_G,qij
 real(8)  fi_i,ff0
 real(8) inv_r3, inv_rij
 real(8) afs_i_xx,afs_i_yy,afs_i_zz



  allocate(ddx(MX_in_list_14),ddy(MX_in_list_14),ddz(MX_in_list_14),&
           ddr_sq(MX_in_list_14),in_list(MX_in_list_14))

    en_14 = 0.0d0

    do i = 1+rank, Natoms, nprocs
       iStyle = i_style_atom(i)
       neightot = size_list_14(i)
       in_list(1:neightot) = list_14(i,1:neightot)
       i1 = 0
       do k = 1, neightot
           j = in_list(k)
           i1 = i1 + 1
           ddx(i1) = xxx(i) - xxx(j)
           ddy(i1) = yyy(i) - yyy(j)
           ddz(i1) = zzz(i) - zzz(j)
       enddo ! j = 1,neightot
       if (neightot > 0) then
           call periodic_images(ddx(1:i1),ddy(1:i1),ddz(1:i1))
           ddr_sq(1:i1) = ddx(1:i1)*ddx(1:i1) + ddy(1:i1)*ddy(1:i1) + ddz(1:i1)*ddz(1:i1)
       endif
           qi = all_charges(i)
       do k = 1, neightot
        j = in_list(k)
        r2 = ddr_sq(k)
        if ( r2 < cut_off_sq ) then
            jstyle = i_style_atom(j)
            i_pair = which_atomStyle_pair(istyle,jstyle) ! can replace it by a formula?
            a0 = atom_Style2_vdwPrm(0,i_pair)
            r = dsqrt(r2)
            Inverse_r_squared = 1.0d0/r2
            inv_rij = 1.0d0 / r ;
            inv_r3 = Inverse_r_squared * inv_rij

            NDX = max(1,int(r*irdr))
            ppp = (r*irdr) - dble(ndx)
            x = ddx(k)   ; y = ddy(k)    ; z = ddz(k)
            if (a0 > SYS_ZERO.and.l_red_14_vdw_CTRL) then
                vk  = vvdw(ndx,i_pair)  ;  vk1 = vvdw(ndx+1,i_pair)
                t1 = vk  + (vk1 - vk )*ppp
                t2 = vk1 + (vvdw(ndx+2,i_pair) - vk1)*(ppp - 1.0d0)
                En = - red_14_vdw  *  (t1 + (t2-t1)*ppp*0.5d0)

                 en_vdw = en_vdw + En
                 en_14 = en_14 + En

            endif
      vk  = vele_G(ndx,i_pair)  ;  vk1 = vele_G(ndx+1,i_pair) ; vk2 = vele_G(ndx+2,i_pair)
       t1 = vk  + (vk1 - vk )*ppp
       t2 = vk1 + (vk2 - vk1)*(ppp - 1.0d0)
       B0 = (t1 + (t2-t1)*(ppp*0.5d0))

            qj = all_charges(j)
            qij = qi*qj
            if (l_red_14_Q_CTRL) then
              vk  = vele(ndx)  ;  vk1 = vele(ndx+1) ; vk2 = vele(ndx+2)
              t1 = vk  + (vk1 - vk )*ppp
              t2 = vk1 + (vk2 - vk1)*(ppp - 1.0d0)
              B00 = (t1 + (t2-t1)*(ppp*0.5d0))
              inv_rij = 1.0d0 / r ;
              inv_r_B0 = inv_rij - (B0-B00)
              En = -(red_14_Q*qij) * inv_r_B0
               en_Qreal = en_Qreal + En
               en_14 = en_14 + En
 
              endif ! l_red_14_Q_CTRL
        endif
      enddo



    enddo ! i = 1+rank, Natoms, nprocs
deallocate(ddx,ddy,ddz,ddr_sq,in_list)

end subroutine compute_14_interactions_vdw_q_ENERGY
!-----------------------------------------
!-----------------------------------------
!-----------------------------------------

subroutine compute_14_interactions_vdw
 use sys_data
 use paralel_env_data
 use math_constants
 use boundaries
 use ALL_atoms_data, only : Natoms, i_style_atom,fxx,fyy,fzz,xxx,yyy,zzz,&
                            fshort_xx,fshort_yy,fshort_zz
 use atom_type_data
 use max_sizes_data, only : MX_list_nonbonded
 use connectivity_ALL_data, only : list_14, size_list_14,&
       red_14_vdw,l_red_14_vdw_CTRL,MX_in_list_14
 use profiles_data
 use energies_data
 use stresses_data
 use atom_type_data
 use interpolate_data
 use variables_short_pairs, only : a_pot_LJ,&
         stress_vdw_xx,stress_vdw_xy,stress_vdw_xz,&
         stress_vdw_yx,stress_vdw_yy,stress_vdw_yz,&
         stress_vdw_zx,stress_vdw_zy,stress_vdw_zz

  implicit none
  integer i , istyle, neightot ! the outer atom index
  real(8) fx,fy,fz,sxx,sxy,sxz,syx,syy,syz,szx,szy,szz,x,y,z
  real(8) ppp, vk,vk1,vk2,En,gk,gk1,t1,t2,ff,Inverse_r_squared,r,r2,inverse_r
  real(8) apot_i_1, a_pot_Q_i
  integer NDX,jtype,i_pair,ju,k,j,jstyle,i1
  real(8) a0
  real(8) apress_i_11,apress_i_12,apress_i_13,apress_i_22,apress_i_23,apress_i_33
  real(8) af_i_1_x, af_i_1_y, af_i_1_z
  real(8), allocatable :: ddx(:),ddy(:),ddz(:),ddr_sq(:)
  integer, allocatable :: in_list(:)
  real(8) afs_i_xx,afs_i_yy,afs_i_zz

  if (.not.l_red_14_vdw_CTRL ) RETURN

  allocate(ddx(MX_in_list_14),ddy(MX_in_list_14),ddz(MX_in_list_14),&
           ddr_sq(MX_in_list_14),in_list(MX_in_list_14))

  en_14 = 0.0d0

    do i = 1+rank, Natoms, nprocs
       iStyle = i_style_atom(i)
       neightot = size_list_14(i)
       in_list(1:neightot) = list_14(i,1:neightot)
       i1 = 0
       do k = 1, neightot
           j = in_list(k)
           i1 = i1 + 1
           ddx(i1) = xxx(i) - xxx(j)
           ddy(i1) = yyy(i) - yyy(j)
           ddz(i1) = zzz(i) - zzz(j)
       enddo ! j = 1,neightot
       if (neightot > 0) then
           call periodic_images(ddx(1:i1),ddy(1:i1),ddz(1:i1))
           ddr_sq(1:i1) = ddx(1:i1)*ddx(1:i1) + ddy(1:i1)*ddy(1:i1) + ddz(1:i1)*ddz(1:i1)
       endif
           af_i_1_x=0.0d0 ; af_i_1_y = 0.0d0 ; af_i_1_z = 0.0d0 ; apot_i_1 = 0.0d0
           if (l_need_2nd_profile) then
             apot_i_1= 0.0d0
           endif
       afs_i_xx=0.0d0; afs_i_yy=0.0d0;afs_i_zz=0.0d0
       do k = 1, neightot
        j = in_list(k)
        r2 = ddr_sq(k)
        if ( r2 < cut_off_sq) then
            jstyle = i_style_atom(j)
            i_pair = which_atomStyle_pair(istyle,jstyle) ! can replace it by a formula?
            a0 = atom_Style2_vdwPrm(0,i_pair)
         if (a0 > SYS_ZERO ) then
            r = dsqrt(r2)
            Inverse_r_squared = 1.0d0/r2
            NDX = max(1,int(r*irdr))
            ppp = (r*irdr) - dble(ndx)
            x = ddx(k)   ; y = ddy(k)    ; z = ddz(k)
                vk  = vvdw(ndx,i_pair)  ;  vk1 = vvdw(ndx+1,i_pair)
                t1 = vk  + (vk1 - vk )*ppp
                t2 = vk1 + (vvdw(ndx+2,i_pair) - vk1)*(ppp - 1.0d0)
                En = - red_14_vdw  *  (t1 + (t2-t1)*ppp*0.5d0)
                gk  = gvdw(ndx,i_pair)  ;  gk1 = gvdw(ndx+1,i_pair)
                t1 = gk  + (gk1 - gk )*ppp
                t2 = gk1 + (gvdw(ndx+2,i_pair) - gk1)*(ppp - 1.0d0)
                ff = - red_14_vdw  *  (t1 + (t2-t1)*ppp*0.5d0)*Inverse_r_squared

               fx = ff*x    ;  fy = ff*y   ;  fz = ff*z
              sxx = fx*x    ;  sxy = fx*y  ;  sxz = fx*z
                               syy = fy*y  ;  syz = fy*z
                                              szz = fz*z
afs_i_xx = afs_i_xx + fx
afs_i_yy = afs_i_yy + fy
afs_i_zz = afs_i_zz + fz
fshort_xx(j) = fshort_xx(j) - fx
fshort_yy(j) = fshort_yy(j) - fy
fshort_zz(j) = fshort_zz(j) - fz

                  stress_vdw_xx = stress_vdw_xx + sxx ;
                  stress_vdw_xy = stress_vdw_xy + sxy ;
                  stress_vdw_xz = stress_vdw_xz + sxz ;
                  stress_vdw_yy = stress_vdw_yy + syy ;
                  stress_vdw_yz = stress_vdw_yz + syz ;
                  stress_vdw_zz = stress_vdw_zz + szz ;
                 af_i_1_x = af_i_1_x+fx ; af_i_1_y = af_i_1_y + fy ; af_i_1_z = af_i_1_z + fz
                 fxx(j)   = fxx(j) - fx ; fyy(j)   = fyy(j) - fy   ; fzz(j)   = fzz(j) - fz
                 en_vdw = en_vdw + En
                 en_14 = en_14 + En

                 if (l_need_2nd_profile) then
                    apot_i_1=apot_i_1 + En
                    a_pot_LJ(j)=a_pot_LJ(j) + En
                 endif

            endif ! a0>0
            endif ! within cut and proceed

      enddo   ! k  = 1, neightot



    call finalize(i)
fshort_xx(i) = fshort_xx(i) + afs_i_xx
fshort_yy(i) = fshort_yy(i) + afs_i_yy
fshort_zz(i) = fshort_zz(i) + afs_i_zz

  enddo ! i = 1+rank, Natoms, nprocs


deallocate(ddx,ddy,ddz,ddr_sq,in_list)
contains



   subroutine finalize(i)
   integer, intent(IN) :: i
   fxx(i) = fxx(i) + af_i_1_x
   fyy(i) = fyy(i) + af_i_1_y
   fzz(i) = fzz(i) + af_i_1_z
   if (l_need_2nd_profile) then
      a_pot_LJ(i)=a_pot_LJ(i)+apot_i_1
! field acting on charge as a result of the action of a nother charge
!      a_press_LJ_11(i)=a_press_LJ_11(i)+apress_i_11
!      a_press_LJ_22(i)=a_press_LJ_22(i)+apress_i_22
!      a_press_LJ_33(i)=a_press_LJ_33(i)+apress_i_33
!      a_press_LJ_12(i)=a_press_LJ_12(i)+apress_i_12
!      a_press_LJ_13(i)=a_press_LJ_13(i)+apress_i_13
!      a_press_LJ_23(i)=a_press_LJ_23(i)+apress_i_23
   endif
  end subroutine finalize
end subroutine compute_14_interactions_vdw


!---------------
!---------------
!-----------------
subroutine compute_14_interactions_vdw_ENERGY
 use sys_data
 use paralel_env_data
 use math_constants
 use boundaries
 use ALL_atoms_data, only : Natoms, i_style_atom,xxx,yyy,zzz
 use atom_type_data
 use max_sizes_data, only : MX_list_nonbonded
 use connectivity_ALL_data, only : list_14, size_list_14,&
       red_14_vdw,l_red_14_vdw_CTRL,MX_in_list_14
 use energies_data
 use atom_type_data
 use interpolate_data
 use variables_short_pairs, only : a_pot_LJ
 
  implicit none
  integer i , istyle, neightot ! the outer atom index
  real(8) x,y,z
  real(8) ppp, vk,vk1,vk2,En,gk,gk1,t1,t2,ff,Inverse_r_squared,r,r2,inverse_r
  real(8) apot_i_1, a_pot_Q_i
  integer NDX,jtype,i_pair,ju,k,j,jstyle,i1
  real(8) a0
  real(8) af_i_1_x, af_i_1_y, af_i_1_z
  real(8), allocatable :: ddx(:),ddy(:),ddz(:),ddr_sq(:)
  integer, allocatable :: in_list(:)
  real(8) afs_i_xx,afs_i_yy,afs_i_zz

  if (.not.l_red_14_vdw_CTRL ) RETURN

  allocate(ddx(MX_in_list_14),ddy(MX_in_list_14),ddz(MX_in_list_14),&
           ddr_sq(MX_in_list_14),in_list(MX_in_list_14))

  en_14 = 0.0d0

    do i = 1+rank, Natoms, nprocs
       iStyle = i_style_atom(i)
       neightot = size_list_14(i)
       in_list(1:neightot) = list_14(i,1:neightot)
       i1 = 0
       do k = 1, neightot
           j = in_list(k)
           i1 = i1 + 1
           ddx(i1) = xxx(i) - xxx(j)
           ddy(i1) = yyy(i) - yyy(j)
           ddz(i1) = zzz(i) - zzz(j)
       enddo ! j = 1,neightot
       if (neightot > 0) then
           call periodic_images(ddx(1:i1),ddy(1:i1),ddz(1:i1))
           ddr_sq(1:i1) = ddx(1:i1)*ddx(1:i1) + ddy(1:i1)*ddy(1:i1) + ddz(1:i1)*ddz(1:i1)
       endif

       do k = 1, neightot
        j = in_list(k)
        r2 = ddr_sq(k)
        if ( r2 < cut_off_sq) then
            jstyle = i_style_atom(j)
            i_pair = which_atomStyle_pair(istyle,jstyle) ! can replace it by a formula?
            a0 = atom_Style2_vdwPrm(0,i_pair)
            if (a0 > 0.0d0) then
            r = dsqrt(r2)
            Inverse_r_squared = 1.0d0/r2
            NDX = max(1,int(r*irdr))
            ppp = (r*irdr) - dble(ndx)
            x = ddx(k)   ; y = ddy(k)    ; z = ddz(k)
                vk  = vvdw(ndx,i_pair)  ;  vk1 = vvdw(ndx+1,i_pair)
                t1 = vk  + (vk1 - vk )*ppp
                t2 = vk1 + (vvdw(ndx+2,i_pair) - vk1)*(ppp - 1.0d0)
                En = - red_14_vdw  *  (t1 + (t2-t1)*ppp*0.5d0)
                en_vdw = en_vdw + En
                en_14 = en_14 + En

            endif  ! a0>0
            endif ! within cut and proceed

      enddo   ! k  = 1, neightot


  enddo ! i = 1+rank, Natoms, nprocs


deallocate(ddx,ddy,ddz,ddr_sq,in_list)

end subroutine compute_14_interactions_vdw_ENERGY
!---------------------
!---------------------
!---------------------








!MC MC MC MC MC MC MC MC MC MC MC MC MC MC MC MC MC MC MC MC
!MC MC MC MC MC MC MC MC MC MC MC MC MC MC MC MC MC MC MC MC
!MC MC MC MC MC MC MC MC MC MC MC MC MC MC MC MC MC MC MC MC
!MC MC MC MC MC MC MC MC MC MC MC MC MC MC MC MC MC MC MC MC




subroutine compute_14_interactions_vdw_q_mu_MCmove_1atom(iwhich)
 use sys_data
 use paralel_env_data
 use math_constants
 use boundaries
 use ALL_atoms_data, only : Natoms, i_style_atom,xxx,yyy,zzz, &
     all_charges,all_dipoles_xx,all_dipoles_yy,all_dipoles_zz,all_dipoles
 use atom_type_data
 use max_sizes_data, only : MX_list_nonbonded
 use connectivity_ALL_data, only : list_14, size_list_14,&
       red_14_vdw, red_14_Q, red_14_Q_mu, &
       l_red_14_vdw_CTRL,l_red_14_Q_CTRL,l_red_14_Q_mu_CTRL,& 
       MX_in_list_14,l_red_14_mu_mu_CTRL,red_14_mu_mu
 use profiles_data
 use d_energies_data
 use atom_type_data
 use interpolate_data
 use variables_short_pairs, only : a_pot_LJ,a_pot_Q



implicit none
  integer, intent(IN) :: iwhich
  integer i , istyle, neightot ! the outer atom index
  real(8) x,y,z
  real(8) ppp, vk,vk1,vk2,En,gk,gk1,t1,t2,ff,Inverse_r_squared,r,r2,inverse_r
  real(8) apot_i_1, a_pot_Q_i
  integer NDX,jtype,i_pair,ju,k,j,jstyle,i1
  real(8) a0, B0,B1,B2,B3,B00,B11,B22,B33,inv_r_B0,inv_r_B1,inv_r_B2,inv_r_B3
  real(8), allocatable :: ddx(:),ddy(:),ddz(:),ddr_sq(:)
  integer, allocatable :: in_list(:)
 real(8)  dipole_xx_i, dipole_yy_i, dipole_zz_i,qi,qj,qi_G,qj_G,&
                     dipole_i,dipole_j,qij, dipole_i2,dipole_j2,&
          dipole_xx_j,dipole_yy_j,dipole_zz_j,dipole_i_times_Rij,dipole_j_times_Rij
 real(8)  fi_i,ff0,pipj,didj
 real(8) G1,G2
 real(8) inv_rij, inv_r3,inv_r5,inv_r7
 real(8) afs_i_xx,afs_i_yy,afs_i_zz


  allocate(ddx(MX_in_list_14),ddy(MX_in_list_14),ddz(MX_in_list_14),&
           ddr_sq(MX_in_list_14),in_list(MX_in_list_14))
       
    d_en_14 = 0.0d0   


    do i = 1+rank, Natoms, nprocs
       iStyle = i_style_atom(i)
       neightot = size_list_14(i)
       in_list(1:neightot) = list_14(i,1:neightot)
       i1 = 0 
	   if (i /= iwhich) then
       do k = 1, neightot
           j = in_list(k)
		   if (j == iwhich) then
            i1 = i1 + 1
            ddx(i1) = xxx(i) - xxx(j)
            ddy(i1) = yyy(i) - yyy(j)
            ddz(i1) = zzz(i) - zzz(j)
		   endif
       enddo ! j = 1,neightot
	   else ! the case i==iwhich
	   do k = 1, neightot
           j = in_list(k)
            i1 = i1 + 1
            ddx(i1) = xxx(i) - xxx(j)
            ddy(i1) = yyy(i) - yyy(j)
            ddz(i1) = zzz(i) - zzz(j)
       enddo ! j = 1,neightot
	   endif
	enddo    !!!!!



       if (neightot > 0) then
           call periodic_images(ddx(1:i1),ddy(1:i1),ddz(1:i1))
           ddr_sq(1:i1) = ddx(1:i1)*ddx(1:i1) + ddy(1:i1)*ddy(1:i1) + ddz(1:i1)*ddz(1:i1)
       endif
           qi = all_charges(iwhich) 
           dipole_xx_i=all_dipoles_xx(iwhich);dipole_yy_i=all_dipoles_yy(iwhich);dipole_zz_i=all_dipoles_zz(iwhich)




       istyle = i_style_atom(iwhich)
       do k = 1, neightot
        j = in_list(k)
        r2 = ddr_sq(k)
        if ( r2 < cut_off_sq ) then
            jstyle = i_style_atom(j)
            i_pair = which_atomStyle_pair(istyle,jstyle) ! can replace it by a formula?
            a0 = atom_Style2_vdwPrm(0,i_pair)
            r = dsqrt(r2)
            Inverse_r_squared = 1.0d0/r2
            inv_rij = 1.0d0 / r ;
            inv_r3 = Inverse_r_squared * inv_rij
            NDX = max(1,int(r*irdr))
            ppp = (r*irdr) - dble(ndx)
            x = ddx(k)   ; y = ddy(k)    ; z = ddz(k)
            if (a0 > SYS_ZERO.and.l_red_14_vdw_CTRL) then       
                vk  = vvdw(ndx,i_pair)  ;  vk1 = vvdw(ndx+1,i_pair)
                t1 = vk  + (vk1 - vk )*ppp
                t2 = vk1 + (vvdw(ndx+2,i_pair) - vk1)*(ppp - 1.0d0)
                En = - red_14_vdw  *  (t1 + (t2-t1)*ppp*0.5d0)
                
                 d_en_vdw = d_en_vdw + En
                 d_en_14 = d_en_14 + En
               
            endif
            include 'interpolate_2.frg'
            vk  = vele(ndx)  ;  vk1 = vele(ndx+1) ; vk2 = vele(ndx+2)
            t1 = vk  + (vk1 - vk )*ppp
            t2 = vk1 + (vk2 - vk1)*(ppp - 1.0d0)
            B00 = (t1 + (t2-t1)*(ppp*0.5d0))
            vk  = gele(ndx)  ;  vk1 = gele(ndx+1) ; vk2 = gele(ndx+2)
            t1 = vk  + (vk1 - vk )*ppp
            t2 = vk1 + (vk2 - vk1)*(ppp - 1.0d0)
            B11 = (t1 + (t2-t1)*(ppp*0.5d0))
            inv_rij = 1.0d0 / r ;
            inv_r_B0 = inv_rij - (B0-B00)         
            qj = all_charges(j)
            qij = qi*qj
            if (l_red_14_Q_CTRL) then
              En = -(red_14_Q*qij) * inv_r_B0 
              d_en_Qreal = d_en_Qreal + En
              d_en_14 = d_en_14 + En
                 
            endif ! l_red_14_Q_CTRL
            if (l_red_14_Q_mu_CTRL.or.l_red_14_mu_mu_CTRL) then
      vk  = vele2_G(ndx,i_pair)  ;  vk1 = vele2_G(ndx+1,i_pair) ; vk2 = vele2_G(ndx+2,i_pair)
       t1 = vk  + (vk1 - vk )*ppp
       t2 = vk1 + (vk2 - vk1)*(ppp - 1.0d0)
       B2 = (t1 + (t2-t1)*(ppp*0.5d0))
      vk  = vele3_G(ndx,i_pair)  ;  vk1 = vele3_G(ndx+1,i_pair) ; vk2 = vele3_G(ndx+2,i_pair)
       t1 = vk  + (vk1 - vk )*ppp
       t2 = vk1 + (vk2 - vk1)*(ppp - 1.0d0)
       B3 = (t1 + (t2-t1)*(ppp*0.5d0))
      vk  = vele2(ndx)  ;  vk1 = vele2(ndx+1) ; vk2 = vele2(ndx+2)
       t1 = vk  + (vk1 - vk )*ppp
       t2 = vk1 + (vk2 - vk1)*(ppp - 1.0d0)
       B22 = (t1 + (t2-t1)*(ppp*0.5d0))

        inv_r_B0 = inv_rij - (B0-B00)
        inv_r_B1 = inv_r3 - (B1-B11)

       inv_r5 = inv_r3 * Inverse_r_squared
       inv_r_B2 = 3.0d0*inv_r5-(B2-B22)

        dipole_xx_j=all_dipoles_xx(j);dipole_yy_j=all_dipoles_yy(j);dipole_zz_j=all_dipoles_zz(j)
        dipole_i_times_Rij = x*dipole_xx_i+y*dipole_yy_i+z*dipole_zz_i
        dipole_j_times_Rij = x*dipole_xx_j+y*dipole_yy_j+z*dipole_zz_j
        pipj = dipole_xx_i*dipole_xx_j + dipole_yy_i*dipole_yy_j+ dipole_zz_i*dipole_zz_j
        didj = dipole_i_times_Rij*dipole_j_times_Rij
        G1 = - dipole_i_times_Rij*qj + dipole_j_times_Rij*qi !+ pipj
        G2 = - didj
                   
        En  =   inv_r_B1*G1*(-red_14_Q_mu) + (inv_r_B2*G2+pipj*inv_r_B1) *(-red_14_mu_mu)
        d_en_Qreal = d_en_Qreal + En 
        d_en_14 = d_en_14 + En
        endif  !   l_red_14_Q_CTRL.or.l_red_14_Q_mu_CTRL
              
        endif
      enddo



deallocate(ddx,ddy,ddz,ddr_sq,in_list) 

end subroutine compute_14_interactions_vdw_q_mu_MCmove_1atom


!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
subroutine compute_14_interactions_vdw_q_MCmove_1atom(iwhich)
 use sys_data
 use paralel_env_data
 use math_constants
 use boundaries
 use ALL_atoms_data, only : Natoms, i_style_atom,xxx,yyy,zzz,all_charges
 use atom_type_data
 use max_sizes_data, only : MX_list_nonbonded
 use connectivity_ALL_data, only : list_14, size_list_14,&
       red_14_vdw, red_14_Q, red_14_Q_mu, &
       l_red_14_vdw_CTRL,l_red_14_Q_CTRL,l_red_14_Q_mu_CTRL,&
       MX_in_list_14,l_red_14_mu_mu_CTRL,red_14_mu_mu
 use d_energies_data
 use atom_type_data
 use interpolate_data
 use variables_short_pairs, only :  a_pot_LJ,a_pot_Q



implicit none
  integer, intent(IN) :: iwhich
  integer i , istyle, neightot ! the outer atom index
  real(8) fx,fy,fz,sxx,sxy,sxz,syx,syy,syz,szx,szy,szz,x,y,z
  real(8) ppp, vk,vk1,vk2,En,gk,gk1,t1,t2,ff,Inverse_r_squared,r,r2,inverse_r
  real(8) apot_i_1, a_pot_Q_i
  integer NDX,jtype,i_pair,ju,k,j,jstyle,i1
  real(8) a0, B0,B1,B2,B3,B00,B11,B22,B33,inv_r_B0,inv_r_B1,inv_r_B2,inv_r_B3
  real(8) apress_i_11,apress_i_12,apress_i_13,apress_i_22,apress_i_23,apress_i_33
  real(8) af_i_1_x, af_i_1_y, af_i_1_z
  real(8), allocatable :: ddx(:),ddy(:),ddz(:),ddr_sq(:)
  integer, allocatable :: in_list(:)
 real(8)  qi,qj,qi_G,qj_G,qij
 real(8)  fi_i,ff0
 real(8) inv_r3, inv_rij
 real(8) afs_i_xx,afs_i_yy,afs_i_zz



  allocate(ddx(MX_in_list_14),ddy(MX_in_list_14),ddz(MX_in_list_14),&
           ddr_sq(MX_in_list_14),in_list(MX_in_list_14))

    d_en_14 = 0.0d0

    do i = 1+rank, Natoms, nprocs
       iStyle = i_style_atom(i)
       neightot = size_list_14(i)
       in_list(1:neightot) = list_14(i,1:neightot)
       i1 = 0 
	   if (i /= iwhich) then
       do k = 1, neightot
           j = in_list(k)
		   if (j == iwhich) then
            i1 = i1 + 1
            ddx(i1) = xxx(i) - xxx(j)
            ddy(i1) = yyy(i) - yyy(j)
            ddz(i1) = zzz(i) - zzz(j)
		   endif
       enddo ! j = 1,neightot
	   else ! the case i==iwhich
	   do k = 1, neightot
           j = in_list(k)
            i1 = i1 + 1
            ddx(i1) = xxx(i) - xxx(j)
            ddy(i1) = yyy(i) - yyy(j)
            ddz(i1) = zzz(i) - zzz(j)
       enddo ! j = 1,neightot
	   endif
	enddo    !!!!!





       if (neightot > 0) then
           call periodic_images(ddx(1:i1),ddy(1:i1),ddz(1:i1))
           ddr_sq(1:i1) = ddx(1:i1)*ddx(1:i1) + ddy(1:i1)*ddy(1:i1) + ddz(1:i1)*ddz(1:i1)
       endif
	   iStyle = i_style_atom(iwhich)
       qi = all_charges(iwhich)
       do k = 1, neightot
        j = in_list(k)
        r2 = ddr_sq(k)
        if ( r2 < cut_off_sq ) then
            jstyle = i_style_atom(j)
            i_pair = which_atomStyle_pair(istyle,jstyle) ! can replace it by a formula?
            a0 = atom_Style2_vdwPrm(0,i_pair)
            r = dsqrt(r2)
            Inverse_r_squared = 1.0d0/r2
            inv_rij = 1.0d0 / r ;
            inv_r3 = Inverse_r_squared * inv_rij

            NDX = max(1,int(r*irdr))
            ppp = (r*irdr) - dble(ndx)
            x = ddx(k)   ; y = ddy(k)    ; z = ddz(k)
            if (a0 > SYS_ZERO.and.l_red_14_vdw_CTRL) then
                vk  = vvdw(ndx,i_pair)  ;  vk1 = vvdw(ndx+1,i_pair)
                t1 = vk  + (vk1 - vk )*ppp
                t2 = vk1 + (vvdw(ndx+2,i_pair) - vk1)*(ppp - 1.0d0)
                En = - red_14_vdw  *  (t1 + (t2-t1)*ppp*0.5d0)

                 d_en_vdw = d_en_vdw + En
                 d_en_14 = d_en_14 + En

            endif
       vk  = vele_G(ndx,i_pair)  ;  vk1 = vele_G(ndx+1,i_pair) ; vk2 = vele_G(ndx+2,i_pair)
       t1 = vk  + (vk1 - vk )*ppp
       t2 = vk1 + (vk2 - vk1)*(ppp - 1.0d0)
       B0 = (t1 + (t2-t1)*(ppp*0.5d0))

            qj = all_charges(j)
            qij = qi*qj
            if (l_red_14_Q_CTRL) then
              vk  = vele(ndx)  ;  vk1 = vele(ndx+1) ; vk2 = vele(ndx+2)
              t1 = vk  + (vk1 - vk )*ppp
              t2 = vk1 + (vk2 - vk1)*(ppp - 1.0d0)
              B00 = (t1 + (t2-t1)*(ppp*0.5d0))
              inv_rij = 1.0d0 / r ;
              inv_r_B0 = inv_rij - (B0-B00)
              En = -(red_14_Q*qij) * inv_r_B0
              d_en_Qreal = d_en_Qreal + En
              d_en_14 = d_en_14 + En
             endif ! l_red_14_Q_CTRL
        endif
      enddo



deallocate(ddx,ddy,ddz,ddr_sq,in_list)

end subroutine compute_14_interactions_vdw_q_MCmove_1atom


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

subroutine compute_14_interactions_vdw_MCmove_1atom(iwhich)
 use sys_data
 use paralel_env_data
 use math_constants
 use boundaries
 use ALL_atoms_data, only : Natoms, i_style_atom,xxx,yyy,zzz
 use atom_type_data
 use max_sizes_data, only : MX_list_nonbonded
 use connectivity_ALL_data, only : list_14, size_list_14,&
       red_14_vdw,l_red_14_vdw_CTRL,MX_in_list_14
 use d_energies_data
 use atom_type_data
 use interpolate_data
 use variables_short_pairs, only : a_pot_LJ
 
  implicit none
  integer, intent(IN) :: iwhich
  integer i , istyle, neightot ! the outer atom index
  real(8) x,y,z
  real(8) ppp, vk,vk1,vk2,En,gk,gk1,t1,t2,ff,Inverse_r_squared,r,r2,inverse_r
  real(8) apot_i_1, a_pot_Q_i
  integer NDX,jtype,i_pair,ju,k,j,jstyle,i1
  real(8) a0
  real(8) af_i_1_x, af_i_1_y, af_i_1_z
  real(8), allocatable :: ddx(:),ddy(:),ddz(:),ddr_sq(:)
  integer, allocatable :: in_list(:)
  real(8) afs_i_xx,afs_i_yy,afs_i_zz

  if (.not.l_red_14_vdw_CTRL ) RETURN

  allocate(ddx(MX_in_list_14),ddy(MX_in_list_14),ddz(MX_in_list_14),&
           ddr_sq(MX_in_list_14),in_list(MX_in_list_14))

  d_en_14 = 0.0d0

    do i = 1+rank, Natoms, nprocs
       iStyle = i_style_atom(i)
       neightot = size_list_14(i)
       in_list(1:neightot) = list_14(i,1:neightot)
       i1 = 0 
	   if (i /= iwhich) then
       do k = 1, neightot
           j = in_list(k)
		   if (j == iwhich) then
            i1 = i1 + 1
            ddx(i1) = xxx(i) - xxx(j)
            ddy(i1) = yyy(i) - yyy(j)
            ddz(i1) = zzz(i) - zzz(j)
		   endif
       enddo ! j = 1,neightot
	   else ! the case i==iwhich
	   do k = 1, neightot
           j = in_list(k)
            i1 = i1 + 1
            ddx(i1) = xxx(i) - xxx(j)
            ddy(i1) = yyy(i) - yyy(j)
            ddz(i1) = zzz(i) - zzz(j)
       enddo ! j = 1,neightot
	   endif
	enddo    !!!!!	   

       if (neightot > 0) then
           call periodic_images(ddx(1:i1),ddy(1:i1),ddz(1:i1))
           ddr_sq(1:i1) = ddx(1:i1)*ddx(1:i1) + ddy(1:i1)*ddy(1:i1) + ddz(1:i1)*ddz(1:i1)
       endif

       iStyle = i_style_atom(i)

       do k = 1, neightot
        j = in_list(k)
        r2 = ddr_sq(k)
        if ( r2 < cut_off_sq) then
            jstyle = i_style_atom(j)
            i_pair = which_atomStyle_pair(istyle,jstyle) ! can replace it by a formula?
            a0 = atom_Style2_vdwPrm(0,i_pair)
            if (a0 > 0.0d0) then
            r = dsqrt(r2)
            Inverse_r_squared = 1.0d0/r2
            NDX = max(1,int(r*irdr))
            ppp = (r*irdr) - dble(ndx)
            x = ddx(k)   ; y = ddy(k)    ; z = ddz(k)
                vk  = vvdw(ndx,i_pair)  ;  vk1 = vvdw(ndx+1,i_pair)
                t1 = vk  + (vk1 - vk )*ppp
                t2 = vk1 + (vvdw(ndx+2,i_pair) - vk1)*(ppp - 1.0d0)
                En = - red_14_vdw  *  (t1 + (t2-t1)*ppp*0.5d0)
                d_en_vdw = d_en_vdw + En
                d_en_14 = d_en_14 + En

            endif  ! a0>0
            endif ! within cut and proceed

      enddo   ! k  = 1, neightot




deallocate(ddx,ddy,ddz,ddr_sq,in_list)

end subroutine compute_14_interactions_vdw_MCmove_1atom

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

end module compute_14_module
