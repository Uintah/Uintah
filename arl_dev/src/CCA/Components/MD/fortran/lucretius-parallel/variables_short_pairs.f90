module variables_short_pairs
implicit none

 real(8), allocatable ::  dx(:),dy(:),dz(:),dr_sq(:)
 real(8), allocatable :: a_pot_LJ(:), a_press_LJ_11(:),a_press_LJ_22(:),&
    a_press_LJ_33(:),a_press_LJ_12(:),a_press_LJ_13(:),a_press_LJ_23(:),&
    a_pot_Q(:), a_press_Q_11(:),a_press_Q_22(:),&
    a_press_Q_33(:),a_press_Q_12(:),a_press_Q_13(:),a_press_Q_23(:), &
    a_press_Q_21(:),a_press_Q_31(:),a_press_Q_32(:),&
    a_fi(:),a_EE_xx(:),a_EE_yy(:),a_EE_zz(:)

 integer, allocatable :: in_list_Q(:)
 real(8)  dipole_xx_i, dipole_yy_i, dipole_zz_i,qi,qj,qi_G,qj_G,&
                     dipole_i,dipole_j,qij, dipole_i2,dipole_j2,&
          dipole_xx_j,dipole_yy_j,dipole_zz_j

 real(8)  P_fi_i,G_fi_i,D_fi_i
 real(8)  P_EE_i_xx,G_EE_i_xx,D_EE_i_xx
 real(8)  P_EE_i_yy,G_EE_i_yy,D_EE_i_yy
 real(8)  P_EE_i_zz,G_EE_i_zz,D_EE_i_zz
 real(8)  fi_i

 
 real(8)  QQ_PP_en_Qreal,QQ_PG_en_Qreal,QQ_GP_en_Qreal,QQ_GG_en_Qreal,&
                     QD_PP_en_Qreal,DQ_PP_en_Qreal,QD_GP_en_Qreal,DQ_PG_en_Qreal,DD_PP_en_Qreal


 real(8) stress_xx,stress_xy,stress_xz,stress_yy,stress_yz,stress_zz,&
         stress_yx,stress_zx,stress_zy

 real(8) stress_vdw_xx,stress_vdw_xy,stress_vdw_xz,&
         stress_vdw_yx,stress_vdw_yy,stress_vdw_yz,&
         stress_vdw_zx,stress_vdw_zy,stress_vdw_zz


 real(8)  a_press_i_11, a_press_i_12, a_press_i_13, &
                     a_press_i_21,a_press_i_22,a_press_i_23,&
                     a_press_i_31,a_press_i_32,a_press_i_33


  real(8)  af_i_1_x, af_i_1_y, af_i_1_z

  real(8)  QQ_PP_apot_i , QQ_PG_apot_i,QQ_GP_apot_i,QQ_GG_apot_i,QD_PP_apot_i,&
 DQ_PP_apot_i,QD_GP_apot_i,DQ_PG_apot_i,DD_PP_apot_i
        
  real(8), allocatable :: work3(:,:),work31(:,:),work30(:,:)

  real(8), allocatable :: buffer(:,:) ! delete this

  real(8) suma_temp

end module variables_short_pairs
