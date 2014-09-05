  subroutine nonbonded_vdw_2_forces(i,istyle,neightot)
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
 use cut_off_data

  integer, intent(IN) :: i , istyle, neightot ! the outer atom index
  real(8) local_stress_xx,local_stress_xy,local_stress_xz,&
          local_stress_yy,local_stress_yz,local_stress_zz,&
          fx,fy,fz,sxx,sxy,sxz,syx,syy,syz,szx,szy,szz,x,y,z
  real(8) ppp, vk,vk1,vk2,En,gk,gk1,gk2,t1,t2,ff,Inverse_r_squared,r,r2,inverse_r
  real(8) apot_i_1
  integer NDX,jtype,i_pair,ju,k,j,jstyle
  real(8) a0
  real(8) apress_i_11,apress_i_12,apress_i_13,apress_i_22,apress_i_23,apress_i_33
  real(8) f_s_i_yy,f_s_i_xx,f_s_i_zz,ffs,fsx,fsy,fsz
  real(8) g,trunc_and_shift,i_displacement

  call initialize
! formula as when generate them (in vdw_def and ewald_def
 f_s_i_xx=0.0d0;f_s_i_yy = 0.0d0; f_s_i_zz=0.0d0
 i_displacement = 1.0d0/displacement
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

!if (i==25.or.j==25)then
!write(14,*) i,j,En/418.4,r
!endif
!if (i==2) then
!print*,dx(k),dy(k),dz(k),r2
!print*, 'ppp rrr rdr',ppp,r,rdr
!read(*,*)
!endif
!print*,i,j,xxx(i),yyy(i),zzz(i),xxx(j),yyy(j),zzz(j),dsqrt(r2),En,&
!'vk=',vk,vk1,vvdw(ndx+2,i_pair)
!
!read(*,*)


       gk  = gvdw(ndx,i_pair)  ;  gk1 = gvdw(ndx+1,i_pair) ; gk2 = gvdw(ndx+2,i_pair)
        t1 = gk  + (gk1 - gk )*ppp
        t2 = gk1 + (gk2 - gk1)*(ppp - 1.0d0)
        ff = (t1 + (t2-t1)*ppp*0.5d0)*Inverse_r_squared
        x = dx(k)   ; y = dy(k)    ; z = dz(k)
        fx = ff*x   ;  fy = ff*y   ;  fz = ff*z
        sxx = fx*x  ;  sxy = fx*y  ;  sxz = fx*z
                        syy = fy*y  ;  syz = fy*z
                                        szz = fz*z


!print*,i,j
!print*,'En=',En/418.4d0
!print*,'ff=',ff/418.4d0
!print*,'f=',fx/418.4d0,fy/418.4d0,fz/418.4d0
!print*,'s=',sxx/418.4d0,syy/418.4d0,szz/418.4d0,&
!            sxy/418.4d0,sxz/418.4d0,syz/418.4d0
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


        stress_vdw_xx = stress_vdw_xx + sxx ;
        stress_vdw_xy = stress_vdw_xy + sxy ;
        stress_vdw_xz = stress_vdw_xz + sxz ;
        stress_vdw_yy = stress_vdw_yy + syy ;
        stress_vdw_yz = stress_vdw_yz + syz ;
        stress_vdw_zz = stress_vdw_zz + szz ;
        af_i_1_x = af_i_1_x+fx ; af_i_1_y = af_i_1_y + fy ; af_i_1_z = af_i_1_z + fz
        fxx(j)   = fxx(j) - fx ; fyy(j)   = fyy(j) - fy   ; fzz(j)   = fzz(j) - fz
        en_vdw = en_vdw + En

        if (l_need_2nd_profile) call update_second_profile_vdw

      endif ! (a0 > 1.0d-10
   endif
  enddo ! j index of the double loop
!print*, i,'en_vdw=',en_vdw
!read(*,*)
  call finalize(i)
 fshort_xx(i) = fshort_xx(i) + f_s_i_xx
 fshort_yy(i) = fshort_yy(i) + f_s_i_yy
 fshort_zz(i) = fshort_zz(i) + f_s_i_zz

  contains
subroutine update_second_profile_vdw
        apot_i_1=apot_i_1 + En
        apress_i_11=apress_i_11 + sxx
        apress_i_22=apress_i_22 + syy
        apress_i_33=apress_i_33 + szz
        apress_i_12=apress_i_12 + sxy
        apress_i_13=apress_i_13 + sxz
        apress_i_23=apress_i_23 + syz

        a_pot_LJ(j)=a_pot_LJ(j) + En
  !      a_press_LJ_11(j)=a_press_LJ_11(j) + sxx
  !      a_press_LJ_22(j)=a_press_LJ_22(j) + syy
  !      a_press_LJ_33(j)=a_press_LJ_33(j) + szz
  !      a_press_LJ_12(j)=a_press_LJ_12(j) + sxy
  !      a_press_LJ_13(j)=a_press_LJ_13(j) + sxz
  !      a_press_LJ_23(j)=a_press_LJ_23(j) + syz
end subroutine update_second_profile_vdw

  subroutine initialize
    af_i_1_x=0.0d0 ; af_i_1_y = 0.0d0 ; af_i_1_z = 0.0d0 ; apot_i_1 = 0.0d0
    if (l_need_2nd_profile) then
        apot_i_1= 0.0d0
        apress_i_11= 0.0d0
        apress_i_22= 0.0d0
        apress_i_33= 0.0d0
        apress_i_12= 0.0d0
        apress_i_13= 0.0d0
        apress_i_23= 0.0d0
    endif
  end subroutine initialize
   subroutine finalize(i)
   integer, intent(IN) :: i
   fxx(i) = fxx(i) + af_i_1_x
   fyy(i) = fyy(i) + af_i_1_y
   fzz(i) = fzz(i) + af_i_1_z
   if (l_need_2nd_profile) then
      a_pot_LJ(i)=a_pot_LJ(i)+apot_i_1
!      a_press_LJ_11(i)=a_press_LJ_11(i)+apress_i_11
!      a_press_LJ_22(i)=a_press_LJ_22(i)+apress_i_22
!      a_press_LJ_33(i)=a_press_LJ_33(i)+apress_i_33
!      a_press_LJ_12(i)=a_press_LJ_12(i)+apress_i_12
!      a_press_LJ_13(i)=a_press_LJ_13(i)+apress_i_13
!      a_press_LJ_23(i)=a_press_LJ_23(i)+apress_i_23
   endif
  end subroutine finalize

  end subroutine nonbonded_vdw_2_forces
