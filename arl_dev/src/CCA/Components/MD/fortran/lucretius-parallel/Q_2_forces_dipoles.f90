  subroutine Q_2_forces_dipoles(i,iStyle,neightot)
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
 use interpolate_data
 use variables_short_pairs
 use cut_off_data

   implicit none
    integer, intent(IN) :: i,iStyle,neightot
   integer k,jStyle,i_pair,ndx,j
   real(8 )fx,fy,fz,sxx,sxy,sxz,syx,syy,syz,szx,szy,szz
   real(8) ppp, vk,vk1,vk2,En,gk,gk1,t1,t2,ff,x,y,z,Inverse_r_squared,r,r2,Inverse_r
   real(8) EE_i_xx,EE_i_yy,EE_i_zz
   real(8) ff0,ff00,En0,En00, a_pot_Q_i
   real(8) B0,B1,B2,B3,G1,G2,nabla_G1_xx,nabla_G1_yy,nabla_G1_zz,nabla_G2_xx,nabla_G2_yy,nabla_G2_zz
   real(8) B0_THOLE, B1_THOLE, B0_THOLE_DERIV,B1_THOLE_DERIV
   real(8) ff1_xx,ff1_yy,ff1_zz
   real(8) ff2_xx,ff2_yy,ff2_zz
   real(8) f0,f_1_x,f_1_y,f_1_z,fxi,fyi,fzi,fxj,fyj,fzj
   real(8) pipj,didj,dipole_i_times_Rij,dipole_j_times_Rij
   real(8) i_displacement,g,trunc_and_shift,fsx,fsy,fsz,ffs
   real(8) f_s_i_xx,f_s_i_yy,f_s_i_zz
 !  integer ii,V(1000)

  call initialize_inside_loop
  f_s_i_xx=0.0d0;f_s_i_yy=0.0d0;f_s_i_zz=0.0d0
  i_displacement = 1.0d0/displacement
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
! Dipol-Charge
!I am entering with  qi be point ;
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


!if (i==1.or.j==1)then
!write(14,*)i,j,fx/418.4d0,fy/418.4d0,fz/418.4d0,&
!(af_i_1_x+fx)/418.4d0,(af_i_1_y+fy)/418.4d0,(af_i_1_z+fz)/418.4d0
!endif
     
        sxx = fx*x  ;  sxy = fx*y  ;  sxz = fx*z
                       syy = fy*y  ;  syz = fy*z
                                      szz = fz*z
!print*, i,j,r,En,fx,fy,fz
!print*,sxx,syy,szz
!print*,sxy,sxz,syz
!read(*,*)
        stress_xx = stress_xx + sxx ;
        stress_xy = stress_xy + sxy ;
        stress_xz = stress_xz + sxz ;
        stress_yy = stress_yy + syy ;
        stress_yz = stress_yz + syz ;
        stress_zz = stress_zz + szz ;

        af_i_1_x = af_i_1_x+fx ; af_i_1_y = af_i_1_y + fy ; af_i_1_z = af_i_1_z + fz
        fxx(j)   = fxx(j) - fx ; fyy(j)   = fyy(j) - fy   ; fzz(j)   = fzz(j) - fz
        En_Qreal = En_Qreal + En

if (l_need_2nd_profile) then

        a_pot_Q_i = a_pot_Q_i + En !
        fi_i =       fi_i + B0 * qj + B1 * dipole_j_times_Rij
        a_fi(j) = a_fi(j) + B0 * qi - B1 * dipole_i_times_Rij
        a_pot_Q(j) = a_pot_Q(j) + En

        ! delete this

!buffer(i,1) = buffer(i,1) + (-x*qj+dipole_xx_j)*B1 - x*(dipole_j_times_Rij*B2)
!buffer(i,2) = buffer(i,2) + (-y*qj+dipole_yy_j)*B1 - y*(dipole_j_times_Rij*B2)
!buffer(i,3) = buffer(i,3) + (-z*qj+dipole_zz_j)*B1 - z*(dipole_j_times_Rij*B2)
!
!buffer(j,1) = buffer(j,1) + ( x*qi+dipole_xx_i)*B1 - x*(dipole_i_times_Rij*B2)
!buffer(j,2) = buffer(j,2) + ( y*qi+dipole_yy_i)*B1 - y*(dipole_i_times_Rij*B2)
!buffer(j,3) = buffer(j,3) + ( z*qi+dipole_zz_i)*B1 - z*(dipole_i_times_Rij*B2)
        ! \\\ delete this

!        G_EE_i_xx = G_EE_i_xx + (ff0 * qj) * x
!        G_EE_i_yy = G_EE_i_yy + (ff0 * qj) * y
!        G_EE_i_zz = G_EE_i_zz + (ff0 * qj) * z
!
!        G_a_EE_xx(j) = G_a_EE_xx(j) - (ff0 * qi) * x
!        G_a_EE_yy(j) = G_a_EE_yy(j) - (ff0 * qi) * y
!        G_a_EE_zz(j) = G_a_EE_zz(j) - (ff0 * qi) * z
!
!        a_press_i_11=a_press_i_11 + sxx
!        a_press_i_22=a_press_i_22 + syy
!        a_press_i_33=a_press_i_33 + szz
!        a_press_i_12=a_press_i_12 + sxy
!        a_press_i_13=a_press_i_13 + sxz
!        a_press_i_23=a_press_i_23 + syz
!        a_press_Q_11(j)=a_press_Q_11(j) + sxx
!        a_press_Q_22(j)=a_press_Q_22(j) + syy
!        a_press_Q_33(j)=a_press_Q_33(j) + szz
!       a_press_Q_12(j)=a_press_Q_12(j) + sxy
!       a_press_Q_13(j)=a_press_Q_13(j) + sxz
!        a_press_Q_23(j)=a_press_Q_23(j) + syz
endif

   endif  ! cut_off
  enddo ! j index of the do
 call finalize_inside_loop
 fshort_xx(i) = fshort_xx(i) + f_s_i_xx
 fshort_yy(i) = fshort_yy(i) + f_s_i_yy
 fshort_zz(i) = fshort_zz(i) + f_s_i_zz


contains

 subroutine initialize_inside_loop
use profiles_data
use variables_short_pairs
implicit none

 if (l_need_2nd_profile) then
! potential
 a_pot_Q_i=0.0d0
 fi_i = 0.0d0
! on charge (either point or gauss)
! vect field xx

 endif
end   subroutine initialize_inside_loop

subroutine finalize_inside_loop
 use profiles_data
 use variables_short_pairs
 implicit none
 if (l_need_2nd_profile) then

! potential
 a_pot_Q(i) = a_pot_Q(i) + a_pot_Q_i
 a_fi(i) = a_fi(i) + fi_i
! vect field xx
 endif
end subroutine finalize_inside_loop

 end  subroutine Q_2_forces_dipoles
