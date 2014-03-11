
subroutine finalize_inside_loop
 use profiles_data
 use variables_short_pairs
 implicit none
 if (l_need_2nd_profile) then

! potential 
 QQ_PP_a_pot(i)=QQ_PP_a_pot(i)+QQ_PP_apot_i 
 QQ_PG_a_pot(i)=QQ_PG_a_pot(i)+QQ_PG_apot_i
 QQ_GP_a_pot(i)=QQ_GP_a_pot(i)+QQ_GP_apot_i 
 QQ_GG_a_pot(i)=QQ_GG_a_pot(i)+QQ_GG_apot_i

 QD_PP_a_pot(i)=QD_PP_a_pot(i)+QD_PP_apot_i 
 DQ_PP_a_pot(i)=DQ_PP_a_pot(i)+DQ_PP_apot_i 
 QD_GP_a_pot(i)=QD_GP_a_pot(i)+QD_GP_apot_i 
 DQ_PG_a_pot(i)=DQ_PG_a_pot(i)+DQ_PG_apot_i 

 DD_PP_a_pot(i)=DD_PP_a_pot(i)+DD_PP_apot_i 
! \potential

! fields

 P_a_fi(i) = P_a_fi(i) + P_fi_i! field acting on charge as a result of the action of a nother charge
 G_a_fi(i) = G_a_fi(i) + G_fi_i
 D_a_fi(i) = D_a_fi(i) + D_fi_i
! on dipole

! \potential

! vect field xx
 P_a_EE_xx(i) = P_a_EE_xx(i) + P_EE_i_xx
 G_a_EE_xx(i) = G_a_EE_xx(i) + G_EE_i_xx
 D_a_EE_xx(i) = D_a_EE_xx(i) + D_EE_i_xx
 P_a_EE_yy(i) = P_a_EE_yy(i) + P_EE_i_yy
 G_a_EE_yy(i) = G_a_EE_yy(i) + G_EE_i_yy
 D_a_EE_yy(i) = D_a_EE_yy(i) + D_EE_i_yy
 P_a_EE_zz(i) = P_a_EE_zz(i) + P_EE_i_zz
 G_a_EE_zz(i) = G_a_EE_zz(i) + G_EE_i_zz
 D_a_EE_zz(i) = D_a_EE_zz(i) + D_EE_i_zz

! \fields 

 endif
end subroutine finalize_inside_loop 
