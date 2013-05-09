
subroutine initialize_inside_loop
use profiles_data
use variables_short_pairs
implicit none

 if (l_need_2nd_profile) then

! potential 
 QQ_GP_apot_i=0.0d0;
 QQ_PG_apot_i=0.0d0;
 QQ_PP_apot_i=0.0d0;
 QQ_GG_apot_i=0.0d0
!q-d
 QD_PP_apot_i=0.0d0;
 DQ_PP_apot_i=0.0d0;
 QD_GP_apot_i=0.0d0;
 DQ_PG_apot_i=0.0d0
!d-d
 DD_PP_apot_i=0.0d0

! scalar field
! on charge (either point or gauss)
 P_fi_i = 0.0d0 ! field acting on charge as a result of the action of a nother charge
 G_fi_i = 0.0d0
 D_fi_i = 0.0d0
! on dipole

! \potential
 
! vect field xx
 P_EE_i_xx = 0.0d0
 G_EE_i_xx = 0.0d0
 D_EE_i_xx = 0.0d0
 P_EE_i_yy = 0.0d0
 G_EE_i_yy = 0.0d0
 D_EE_i_yy = 0.0d0
 P_EE_i_zz = 0.0d0
 G_EE_i_zz = 0.0d0
 D_EE_i_zz = 0.0d0

! \vect field zz

 endif
end   subroutine initialize_inside_loop
