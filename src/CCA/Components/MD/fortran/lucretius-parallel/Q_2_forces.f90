  subroutine Q_2_forces(i,istyle,neightot)
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
    integer, intent(IN) :: i,istyle,neightot
   integer k,jstyle,i_pair,ndx,j
   real(8) fx,fy,fz,sxx,sxy,sxz,syx,syy,syz,szx,szy,szz
   real(8) ppp, vk,vk1,vk2,En,gk,gk1,gk2,t1,t2,ff,x,y,z,Inverse_r_squared,r,r2,Inverse_r
   real(8) EE_i_xx,EE_i_yy,EE_i_zz
   real(8) ff0,ff00,En0,En00, a_pot_Q_i,f_s_i_yy,f_s_i_xx,f_s_i_zz,ffs,fsx,fsy,fsz
   real(8) ff1_xx,ff1_yy,ff1_zz
   real(8) ff2_xx,ff2_yy,ff2_zz
   real(8) g,trunc_and_shift,i_displacement

  call initialize_inside_loop
  f_s_i_yy=0.0d0; f_s_i_xx=0.0d0;f_s_i_zz=0.0d0
  i_displacement=1.0d0/displacement
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
        gk  = gele_G(ndx,i_pair)  ;  gk1 = gele_G(ndx+1,i_pair) ; gk2 = gele_G(ndx+2,i_pair)
        t1 = gk  + (gk1 - gk )*ppp
        t2 = gk1 + (gk2 - gk1)*(ppp - 1.0d0)
        ff0 = (t1 + (t2-t1)*ppp*0.5d0)
        ff = ff0 * qij
        fx = ff*x   ;  fy = ff*y   ;  fz = ff*z
        sxx = fx*x  ;  sxy = fx*y  ;  sxz = fx*z
                        syy = fy*y  ;  syz = fy*z
                                        szz = fz*z
!print*,i,j,r,ndx,i_pair,vele_G(ndx,i_pair), En0, En
!read(*,*)
!print*, i,j,r,En,fx,fy,fz
!print*,sxx,syy,szz
!print*,sxy,sxz,syz
!read(*,*)
if (r<cut_off_short)then
if (r>(cut_off_short-displacement)) then
 g = (r-(cut_off_short-displacement))*i_displacement
 trunc_and_shift = 1.0d0+(g*g*(2.0d0*g-3.0d0))
 ffs = ff * trunc_and_shift
else
 ffs = ff
endif
 fsx = ffs*x ; fsy = ffs*y; fsz = ffs*z
 f_s_i_xx = f_s_i_xx + fsx
 f_s_i_yy = f_s_i_yy + fsy
 f_s_i_zz = f_s_i_zz + fsz
 fshort_xx(j) = fshort_xx(j) - fsx
 fshort_yy(j) = fshort_yy(j) - fsy
 fshort_zz(j) = fshort_zz(j) - fsz
endif

        stress_xx = stress_xx + sxx ;
        stress_xy = stress_xy + sxy ;
        stress_xz = stress_xz + sxz ;
        stress_yy = stress_yy + syy ;
        stress_yz = stress_yz + syz ;
        stress_zz = stress_zz + szz ;

        af_i_1_x = af_i_1_x+fx ; af_i_1_y = af_i_1_y + fy ; af_i_1_z = af_i_1_z + fz
        fxx(j)   = fxx(j) - fx ; fyy(j)   = fyy(j) - fy   ; fzz(j)   = fzz(j) - fz
        En_Qreal = En_Qreal + En


!if (i==1.or.j==1)then
!write(14,*) i,j,En/418.4,r
!endif

!if (i==7) write(14,*) i,j,En,En_Qreal
!if (i>7) stop
!if (i==8402.or.j==8402) then
!suma_temp = suma_temp + En
!write(24,'(I7,1X,I7,1x,3(F16.8,1X))', advance='yes') i,j, En, suma_temp,r
!endif


if (l_need_2nd_profile) then

        a_pot_Q_i = a_pot_Q_i + En

        fi_i = fi_i + En0 * qj
        a_pot_Q(j) = a_pot_Q(j) + En
        a_fi(j) = a_fi(j) + En0 * qi
!
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

 fshort_xx(i) = fshort_xx(i) + f_s_i_xx
 fshort_yy(i) = fshort_yy(i) + f_s_i_yy
 fshort_zz(i) = fshort_zz(i) + f_s_i_zz

 call finalize_inside_loop
contains

 subroutine initialize_inside_loop
use profiles_data
use variables_short_pairs
implicit none

 if (l_need_2nd_profile) then
! potential
 a_pot_Q_i=0.0d0
! on charge (either point or gauss)
 fi_i = 0.0d0 ! field acting on charge as a result of the action of a nother charge
! vect field xx
! P_EE_i_xx = 0.0d0
! G_EE_i_xx = 0.0d0
! P_EE_i_yy = 0.0d0
! G_EE_i_yy = 0.0d0
! P_EE_i_zz = 0.0d0
! G_EE_i_zz = 0.0d0

 endif
end   subroutine initialize_inside_loop

subroutine finalize_inside_loop
 use profiles_data
 use variables_short_pairs
 implicit none
 if (l_need_2nd_profile) then

! potential
 a_pot_Q(i) = a_pot_Q(i) + a_pot_Q_i
 a_fi(i) = a_fi(i) + fi_i! field acting on charge as a result of the action of a nother charge
! vect field xx
! P_a_EE_xx(i) = P_a_EE_xx(i) + P_EE_i_xx
! G_a_EE_xx(i) = G_a_EE_xx(i) + G_EE_i_xx
! P_a_EE_yy(i) = P_a_EE_yy(i) + P_EE_i_yy
! G_a_EE_yy(i) = G_a_EE_yy(i) + G_EE_i_yy
! P_a_EE_zz(i) = P_a_EE_zz(i) + P_EE_i_zz
! G_a_EE_zz(i) = G_a_EE_zz(i) + G_EE_i_zz
 endif
end subroutine finalize_inside_loop

 end  subroutine Q_2_forces
