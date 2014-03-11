
module REAL_part_intra_correction_sfc_sfc
private :: REAL_part_intra_correction_sfc_sfc_Q_DIP
private :: REAL_part_intra_correction_sfc_sfc_Q
public :: REAL_part_intra_correction_sfc_sfc_driver
public :: REAL_part_intra_correction_sfc_sfc_driver_ENERGY

CONTAINS

subroutine REAL_part_intra_correction_sfc_sfc_driver
use CTRLs_data, only : l_DIP_CTRL
use connectivity_ALL_data, only : N_pairs_sfc_sfc_123
implicit none
if (N_pairs_sfc_sfc_123>0) then
if (l_DIP_CTRL)  then
 call REAL_part_intra_correction_sfc_sfc_Q_DIP
else
 call REAL_part_intra_correction_sfc_sfc_Q
endif
endif !N_pairs_sfc_sfc_123

end subroutine REAL_part_intra_correction_sfc_sfc_driver

subroutine REAL_part_intra_correction_sfc_sfc_driver_ENERGY
use CTRLs_data, only : l_DIP_CTRL
use connectivity_ALL_data, only : N_pairs_sfc_sfc_123
implicit none
if (N_pairs_sfc_sfc_123>0) then
if (l_DIP_CTRL)  then
 call REAL_part_intra_correction_sfc_sfc_Q_DIP_ENERGY
else
 call REAL_part_intra_correction_sfc_sfc_Q_ENERGY
endif
endif !N_pairs_sfc_sfc_123

end subroutine 
subroutine REAL_part_intra_correction_sfc_sfc_Q_DIP 
! add 1-2 and 1-3 interactions for sfc-sfc pairs.
 use sys_data
 use paralel_env_data
 use math_constants
 use boundaries
 use ALL_atoms_data, only : Natoms, i_style_atom,fxx,fyy,fzz,xxx,yyy,zzz, &
     all_charges,all_dipoles_xx,all_dipoles_yy,all_dipoles_zz,all_dipoles,&
     fshort_xx,fshort_yy,fshort_zz
 use atom_type_data
 use max_sizes_data, only : MX_excluded
 use connectivity_ALL_data, only : list_excluded_sfc_iANDj_HALF,&
                                   size_list_excluded_sfc_iANDj_HALF
 use profiles_data
 use energies_data
 use stresses_data
 use atom_type_data
 use interpolate_data
 use variables_short_pairs, only : stress_xx,stress_xy,stress_xz,stress_yy,stress_yz,stress_zz,&
         stress_yx,stress_zx,stress_zy,a_pot_Q,a_fi

implicit none
integer i,j,k,i1
real(8) x_x_x,qi,qj,qij,a_f_i,poti,ff,En,local_energy,g
real(8) a_force_i,ff0,En0, a_fi_i
real(8), allocatable :: dx(:),dy(:),dz(:),dr_sq(:)
real(8) x,y,z,fx,fy,fz,sxx,syy,szz,sxy,sxz,syz,syx,szy,szx,fsx,fsy,fsz
real(8) af_i_1_x,af_i_1_y,af_i_1_z,f_s_i_xx,f_s_i_yy,f_s_i_zz,ffs
real(8) r,trunc_and_shift
real(8) En0G,En0GG, qiG, pref,eta,eta_sq
integer itype,jtype,i_pair
logical l_i,l_j,is_sfc_i,is_sfc_j,l_proceed
real(8) B0,B1,B2,B3,B0_THOLE,B1_THOLE,B2_THOLE,B0_THOLE_DERIV,B1_THOLE_DERIV
real(8) dipole_xx_i,dipole_yy_i,dipole_zz_i,dipole_xx_j,dipole_yy_j,dipole_zz_j
real(8) dipole_i_times_Rij, dipole_j_times_Rij
real(8) pipj, didj,G1,G2
real(8) nabla_G1_xx, nabla_G1_yy, nabla_G1_zz
real(8) nabla_G2_xx, nabla_G2_yy, nabla_G2_zz
real(8) inv_r_B0,inv_r_B1,inv_r_B2,inv_r_B3
real(8) vk,vk1,vk2,gk,gk1,gk2,t1,t2,t3,ppp, i_displacement,a_pot_Q_i,fi_i
real(8),allocatable :: buffer(:,:)
integer N,ndx

N=MX_excluded

allocate(dx(N),dy(N),dz(N),dr_sq(N))
i_displacement=1.0d0/displacement
if (l_need_2nd_profile) then
!  allocate(buffer(Natoms,3)); buffer=0.0d0
endif

  local_energy = 0.0d0

a_pot_Q=0.0d0!delete it

  do i = 1, Natoms
      qi = all_charges(i)
      dipole_xx_i = all_dipoles_xx(i) ; dipole_yy_i=all_dipoles_yy(i); dipole_zz_i=all_dipoles_zz(i)
      i1 = size_list_excluded_sfc_iANDj_HALF(i)
      do k = 1, i1
         j = list_excluded_sfc_iANDj_HALF(i,k)
         dx(k) = xxx(i)-xxx(j)
         dy(k) = yyy(i)-yyy(j)
         dz(k) = zzz(i)-zzz(j)
      enddo
      if (i1 > 0) then
        call periodic_images(dx(1:i1),dy(1:i1),dz(1:i1))
        dr_sq(1:i1) = dx(1:i1)*dx(1:i1)+dy(1:i1)*dy(1:i1)+dz(1:i1)*dz(1:i1)
      endif
      fi_i = 0.0d0 ; a_pot_Q_i = 0.0d0 ;
      f_s_i_xx=0.0d0;f_s_i_yy=0.0d0;f_s_i_zz=0.0d0
      af_i_1_x=0.0d0; af_i_1_y=0.0d0; af_i_1_x=0.0d0;
      do k = 1, i1
        j = list_excluded_sfc_iANDj_HALF(i,k)
        x = dx(k) ; y = dy(k) ; z = dz(k) ; r = dsqrt(dr_sq(k)); 
        NDX = max(1,int(r*irdr))
        ppp = (r*irdr) - dble(ndx)
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
        qij = qi*qj
        include 'interpolate_4.frg'
        include 'interpolate_THOLE_ALL.frg'

        En =  B0*qij + B1*G1 + B2*G2 +    B0_THOLE*pipj + B1_THOLE*G2
        ff0 = (B1*qij) +  B2*G1 + B3*G2 +    B0_THOLE_DERIV*pipj + B1_THOLE_DERIV*G2
        fx = ff0*x  +  nabla_G1_xx*B1 + nabla_G2_xx*(B2+B1_THOLE)
        fy = ff0*y  +  nabla_G1_yy*B1 + nabla_G2_yy*(B2+B1_THOLE)
        fz = ff0*z  +  nabla_G1_zz*B1 + nabla_G2_zz*(B2+B1_THOLE)
        
if (r<cut_off_short)then
if (r>(cut_off_short-displacement)) then
 g = (r-(cut_off_short-displacement))*i_displacement
 trunc_and_shift = 1.0d0+(g*g*(2.0d0*g-3.0d0))
 ffs = (B1*qij) * trunc_and_shift
else
 ffs = (B1*qij)
endif
 fsx = ffs*x ; fsy = ffs*y; fsz = ffs*z
 f_s_i_xx = f_s_i_xx + fsx
 f_s_i_yy = f_s_i_yy + fsy
 f_s_i_zz = f_s_i_zz + fsz
 fshort_xx(j) = fshort_xx(j) - fsx
 fshort_yy(j) = fshort_yy(j) - fsy
 fshort_zz(j) = fshort_zz(j) - fsz
endif
        
        sxx = fx * x ; syy = fy * y ; szz = fz * z
        sxy = fx * y ; sxz = fx * z ; syz = fy * z
        syx = fy * x ; szx = fz * x ; szy = fz * y
        af_i_1_x = af_i_1_x+fx ; af_i_1_y = af_i_1_y + fy ; af_i_1_z = af_i_1_z + fz
        fxx(j)   = fxx(j) - fx ; fyy(j)   = fyy(j) - fy   ; fzz(j)   = fzz(j) - fz        
        local_energy = local_energy + En
        stress_xx = stress_xx + sxx ;
        stress_xy = stress_xy + sxy ;
        stress_xz = stress_xz + sxz ;
        stress_yy = stress_yy + syy ;
        stress_yz = stress_yz + syz ;
        stress_zz = stress_zz + szz ;
        stress_yx = stress_yx + sxy ;
        stress_zx = stress_zx + sxz ;
        stress_zy = stress_zy + syz ;


if (l_need_2nd_profile) then
        a_pot_Q_i = a_pot_Q_i + En
        fi_i =       fi_i + B0 * qj + B1 * dipole_j_times_Rij
        a_fi(j) = a_fi(j) + B0 * qi - B1 * dipole_i_times_Rij
        a_pot_Q(j) = a_pot_Q(j) + En


!buffer(j,1) = buffer(j,1) + ( qi*x + dipole_xx_i)*inv_r_B1 - x*(dipole_i_times_Rij*inv_r_B2)
!buffer(j,2) = buffer(j,2) + ( qi*y + dipole_yy_i)*inv_r_B1 - y*(dipole_i_times_Rij*inv_r_B2)
!buffer(j,3) = buffer(j,3) + ( qi*z + dipole_zz_i)*inv_r_B1 - z*(dipole_i_times_Rij*inv_r_B2)
!buffer(i,1) = buffer(i,1) + (- qj*x + dipole_xx_j)*inv_r_B1 - x*(dipole_j_times_Rij*inv_r_B2)
!buffer(i,2) = buffer(i,2) + (- qj*y + dipole_yy_j)*inv_r_B1 - y*(dipole_j_times_Rij*inv_r_B2)
!buffer(i,3) = buffer(i,3) + (- qj*z + dipole_zz_j)*inv_r_B1 - z*(dipole_j_times_Rij*inv_r_B2)


endif
    enddo


      fxx(i) = fxx(i) + af_i_1_x
      fyy(i) = fyy(i) + af_i_1_y
      fzz(i) = fzz(i) + af_i_1_z
      
      fshort_xx(i) = fshort_xx(i) + f_s_i_xx
      fshort_yy(i) = fshort_yy(i) + f_s_i_yy
      fshort_zz(i) = fshort_zz(i) + f_s_i_zz

if (l_need_2nd_profile) then
      a_pot_Q(i) = a_pot_Q(i) + a_pot_Q_i
      a_fi(i) = a_fi(i) + fi_i
endif

  enddo  ! i

 En_Q = En_Q + local_energy
 En_Qreal = En_Qreal + local_energy
 EN_Qreal_sfc_sfc_intra = local_energy

print*,'local en sum(apoq)=',local_energy,sum(a_pot_Q)
stop

deallocate(dx,dy,dz,dr_sq)
if (l_need_2nd_profile) then
  deallocate(a_pot_Q , a_fi)
!  deallocate(buffer)
endif

end subroutine REAL_part_intra_correction_sfc_sfc_Q_DIP



subroutine REAL_part_intra_correction_sfc_sfc_Q 
! add 1-2 and 1-3 interactions for sfc-sfc pairs.
 use sys_data
 use paralel_env_data
 use math_constants
 use boundaries
 use ALL_atoms_data, only : Natoms, i_style_atom,fxx,fyy,fzz,xxx,yyy,zzz, &
     all_charges,all_dipoles_xx,all_dipoles_yy,all_dipoles_zz,all_dipoles,&
     fshort_xx,fshort_yy,fshort_zz
 use atom_type_data
 use max_sizes_data, only : MX_excluded
 use connectivity_ALL_data, only : list_excluded_sfc_iANDj_HALF,&
                                   size_list_excluded_sfc_iANDj_HALF
 use profiles_data
 use energies_data
 use stresses_data
 use atom_type_data
 use interpolate_data
 use variables_short_pairs, only : stress_xx,stress_xy,stress_xz,stress_yy,stress_yz,stress_zz,&
         stress_yx,stress_zx,stress_zy,a_pot_Q,a_fi

implicit none
integer i,j,k,i1
real(8) x_x_x,qi,qj,qij,a_f_i,poti,ff,En,local_energy
real(8) a_force_i,e_xx_i,e_yy_i,e_zz_i,ff0,En0, a_fi_i, field,g
real(8), allocatable :: dx(:),dy(:),dz(:),dr_sq(:)
real(8) x,y,z,fx,fy,fz,sxx,syy,szz,sxy,sxz,syz,syx,szy,szx,fsx,fsy,fsz,ffs
real(8) af_i_1_x,af_i_1_y,af_i_1_z,f_s_i_xx,f_s_i_yy,f_s_i_zz
real(8) r,trunc_and_shift
real(8) En0G,En0GG, qiG, pref,eta,eta_sq
integer itype,jtype,i_pair
logical l_i,l_j,is_sfc_i,is_sfc_j,l_proceed
real(8) B0,B1
real(8) vk,vk1,vk2,gk,gk1,gk2,t1,t2,t3,ppp,i_displacement,a_pot_Q_i,fi_i
real(8),allocatable :: buffer(:,:)
integer N,ndx

N=MX_excluded

allocate(dx(N),dy(N),dz(N),dr_sq(N))
i_displacement=1.0d0/displacement
if (l_need_2nd_profile) then
!  allocate(buffer(Natoms,3)); buffer=0.0d0
endif

  local_energy = 0.0d0


  do i = 1, Natoms
      qi = all_charges(i)
      i1 = size_list_excluded_sfc_iANDj_HALF(i)
      do k = 1, i1
         j = list_excluded_sfc_iANDj_HALF(i,k)
         dx(k) = xxx(i)-xxx(j)
         dy(k) = yyy(i)-yyy(j)
         dz(k) = zzz(i)-zzz(j)
      enddo
      if (i1 > 0) then
        call periodic_images(dx(1:i1),dy(1:i1),dz(1:i1))
        dr_sq(1:i1) = dx(1:i1)*dx(1:i1)+dy(1:i1)*dy(1:i1)+dz(1:i1)*dz(1:i1)
      endif
      fi_i = 0.0d0 ; a_pot_Q_i = 0.0d0 ;
      f_s_i_xx=0.0d0;f_s_i_yy=0.0d0;f_s_i_zz=0.0d0
      af_i_1_x=0.0d0; af_i_1_y=0.0d0; af_i_1_x=0.0d0;
      do k = 1, i1
        j = list_excluded_sfc_iANDj_HALF(i,k)
        x = dx(k) ; y = dy(k) ; z = dz(k) ; r = dsqrt(dr_sq(k)); 
        NDX = max(1,int(r*irdr))
        ppp = (r*irdr) - dble(ndx)
        qj = all_charges(j)
        qij = qi*qj
        include 'interpolate_2.frg'

        En =  B0*qij
        ff0 = (B1*qij) 
        fx = ff0*x  
        fy = ff0*y  
        fz = ff0*z  
        
if (r<cut_off_short)then
if (r>(cut_off_short-displacement)) then
 g = (r-(cut_off_short-displacement))*i_displacement
 trunc_and_shift = 1.0d0+(g*g*(2.0d0*g-3.0d0))
 ffs = (B1*qij) * trunc_and_shift
else
 ffs = (B1*qij)
endif
 fsx = ffs*x ; fsy = ffs*y; fsz = ffs*z
 f_s_i_xx = f_s_i_xx + fsx
 f_s_i_yy = f_s_i_yy + fsy
 f_s_i_zz = f_s_i_zz + fsz
 fshort_xx(j) = fshort_xx(j) - fsx
 fshort_yy(j) = fshort_yy(j) - fsy
 fshort_zz(j) = fshort_zz(j) - fsz
endif
        
        sxx = fx * x ; syy = fy * y ; szz = fz * z
        sxy = fx * y ; sxz = fx * z ; syz = fy * z
        syx = fy * x ; szx = fz * x ; szy = fz * y
        af_i_1_x = af_i_1_x+fx ; af_i_1_y = af_i_1_y + fy ; af_i_1_z = af_i_1_z + fz
        fxx(j)   = fxx(j) - fx ; fyy(j)   = fyy(j) - fy   ; fzz(j)   = fzz(j) - fz        
        local_energy = local_energy + En
        stress_xx = stress_xx + sxx ;
        stress_xy = stress_xy + sxy ;
        stress_xz = stress_xz + sxz ;
        stress_yy = stress_yy + syy ;
        stress_yz = stress_yz + syz ;
        stress_zz = stress_zz + szz ;
        stress_yx = stress_yx + sxy ;
        stress_zx = stress_zx + sxz ;
        stress_zy = stress_zy + syz ;


if (l_need_2nd_profile) then
        a_pot_Q_i = a_pot_Q_i + En
        fi_i =       fi_i + B0 * qj 
        a_fi(j) = a_fi(j) + B0 * qi 
        a_pot_Q(j) = a_pot_Q(j) + En


!buffer(j,1) = buffer(j,1) + ( qi*x + dipole_xx_i)*inv_r_B1 - x*(dipole_i_times_Rij*inv_r_B2)
!buffer(j,2) = buffer(j,2) + ( qi*y + dipole_yy_i)*inv_r_B1 - y*(dipole_i_times_Rij*inv_r_B2)
!buffer(j,3) = buffer(j,3) + ( qi*z + dipole_zz_i)*inv_r_B1 - z*(dipole_i_times_Rij*inv_r_B2)
!buffer(i,1) = buffer(i,1) + (- qj*x + dipole_xx_j)*inv_r_B1 - x*(dipole_j_times_Rij*inv_r_B2)
!buffer(i,2) = buffer(i,2) + (- qj*y + dipole_yy_j)*inv_r_B1 - y*(dipole_j_times_Rij*inv_r_B2)
!buffer(i,3) = buffer(i,3) + (- qj*z + dipole_zz_j)*inv_r_B1 - z*(dipole_j_times_Rij*inv_r_B2)


endif
    enddo


      fxx(i) = fxx(i) + af_i_1_x
      fyy(i) = fyy(i) + af_i_1_y
      fzz(i) = fzz(i) + af_i_1_z
      
      fshort_xx(i) = fshort_xx(i) + f_s_i_xx
      fshort_yy(i) = fshort_yy(i) + f_s_i_yy
      fshort_zz(i) = fshort_zz(i) + f_s_i_zz

if (l_need_2nd_profile) then
      a_pot_Q(i) = a_pot_Q(i) + a_pot_Q_i
      a_fi(i) = a_fi(i) + fi_i
endif

  enddo  ! i

 En_Q = En_Q + local_energy
 En_Qreal = En_Qreal + local_energy
 EN_Qreal_sfc_sfc_intra = local_energy


deallocate(dx,dy,dz,dr_sq)
if (l_need_2nd_profile) then
  deallocate(a_pot_Q , a_fi)
!  deallocate(buffer)
endif

end subroutine REAL_part_intra_correction_sfc_sfc_Q



subroutine REAL_part_intra_correction_sfc_sfc_Q_DIP_ENERGY
implicit none
print*, 'REAL_part_intra_correction_sfc_sfc_Q_DIP_ENERGY not implemented '; STOP
end subroutine REAL_part_intra_correction_sfc_sfc_Q_DIP_ENERGY

subroutine REAL_part_intra_correction_sfc_sfc_Q_ENERGY
implicit none
print*, 'REAL_part_intra_correction_sfc_sfc_Q_ENERGY not implemented '; STOP
end subroutine REAL_part_intra_correction_sfc_sfc_Q_ENERGY

end module REAL_part_intra_correction_sfc_sfc

