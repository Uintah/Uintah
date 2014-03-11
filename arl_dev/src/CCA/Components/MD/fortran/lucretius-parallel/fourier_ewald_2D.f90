
 module fourier_ewald_3D
 
 implicit none
 
 contains

   subroutine get_ewald_self_correct
   use energies_data, only : ewself
   use math_constants , only : rootpi
   use ALL_atoms_data, only : Natoms, i_type_atom
   use atom_type_data, only : q_reduced, is_charge_distributed
   use Ewald_data, only : ewald_alpha, ewald_beta, ewald_gamma
   use profiles_data , only : l_2nd_profile_CTRL,  atom_profile
   use physical_constants, only : temp_cvt_en_ALL, temp_cvt_en

      implicit none
      integer iat,itype
      real(8) ci,term_distrib,term_NOT_distrib,term_distrib_1,term_distrib_2
      ewself = 0.0d0
       term_NOT_distrib=0.0d0 ; term_distrib=0.0d0
       do iat = 1,Natoms
          itype = i_type_atom(iat)
          ci = q_reduced(itype)
          if (is_charge_distributed(itype)) then
            term_distrib = term_distrib + ci*ci
          else
            term_NOT_distrib = term_NOT_distrib + ci*ci
          endif
        end do
        term_NOT_distrib = term_NOT_distrib * ewald_alpha /rootpi
        term_distrib_1 = term_distrib * ewald_beta / rootpi
        term_distrib_2 = term_distrib * ewald_gamma / (rootpi*dsqrt(2.0d0))
        ewself = term_NOT_distrib + term_distrib_1 - term_distrib_2
    if (l_2nd_profile_CTRL) then
    endif 
   end subroutine get_ewald_self_correct

subroutine smpe_2D
  ! does smpe in p+g charge + dipols.

   
end subroutine smpe_2D

subroutine ewald_fourier_2D_slow
      use paralel_env_data
      use profiles_data, only : l_need_2nd_profile, atom_profile
      use Ewald_data
      use ALL_atoms_data, only : Natoms, atom_in_which_molecule, i_type_atom, xxx, yyy, zzz, &
                            fxx , fyy, fzz
      use atom_type_data, only : q_reduced, atom_type_charge, is_charge_distributed
      use sim_cel_data
      use boundaries
      use energies_data
      use stresses_data, only :  stress_Qcmplx_xx,stress_Qcmplx_xy,stress_Qcmplx_xz,&
         stress_Qcmplx_yx,stress_Qcmplx_yy,stress_Qcmplx_yz,&
         stress_Qcmplx_zx,stress_Qcmplx_zy,stress_Qcmplx_zz
      use physical_constants, only : temp_cvt_en_ALL, temp_cvt_en


      implicit none
      integer len 
      integer n,nn,m,mm,l,ll,nmin,mmin, i,j,k,iii,kk,kkk
      real(8), allocatable :: e_x_cos(:,:),e_y_cos(:,:),e_z_cos(:,:)
      real(8), allocatable :: e_x_sin(:,:),e_y_sin(:,:),e_z_sin(:,:)
      real(8), allocatable :: cos_KR(:),sin_KR(:),sin_xy(:),cos_xy(:)
      real(8), allocatable :: F_temp(:,:)
      real(8), allocatable :: a_pot(:), p_press_12(:), p_press_13(:), p_press_23(:),&
                              p_press_11(:), p_press_22(:), p_press_33(:)
      real(8) En, Eni, reciprocal_cut, reciprocal_cut_sq, expfct, rk_sq, rx,ry,rz,d
      real(8) Rec_x1, Rec_x2, Rec_x, Rec_y1, Rec_y2, Rec_y, Rec_z1, Rec_z2, Rec_z
      real(8) AK,SUM_cos_KR,SUM_sin_KR,ff,en_factor,vir,vir0,viri
      real(8) sxx,sxy,sxz,syx,syy,syz,szx,szy,szz
      real(8) local_stress_xx, local_stress_xy,local_stress_xz
      real(8) local_stress_yx, local_stress_yy,local_stress_yz
      real(8) local_stress_zx, local_stress_zy,local_stress_zz

 allocate(e_x_cos(len,0:K_MAX_X),e_y_cos(len,0:K_MAX_Y),e_z_cos(len,0:K_MAX_Z))
 allocate(e_x_sin(len,0:K_MAX_X),e_y_sin(len,0:K_MAX_Y),e_z_sin(len,0:K_MAX_Z))
 allocate(cos_KR(len),sin_KR(len),sin_xy(len),cos_xy(len))
 allocate(F_temp(len,3))



 call cel_properties(.true.)



     ! ORDER 0 terms.

  cos_G=0.0d0 ; cos_P = 0.0d0 ; sin_G=0.0d0; sin_P=0.0d0 ! It is OK to initialize before k - cycle
  cos_f=0.0d0; sin_f=0.0d0

  reciprocal_cut = min(dble(k_max_x)*Reciprocal_perp(1),&
                            dble(k_max_y)*Reciprocal_perp(2))
!                            dble(k_max_z)*Reciprocal_perp(3))
  reciprocal_cut_sq=reciprocal_cut**2


   k_vct = 0 
!   h_step = / ....????????
   do ix = -K_MAX_X, K_MAX_X
   tmp = dble(ix) 
   rec_xx = tmp*Reciprocal_cel(1)
   rec_yx = tmp*Reciprocal_cel(4)
!   rec_zx = tmp*Reciprocal_cel(7)
   do iy = -K_MAX_Y, K_MAX_Y
   tmp = dble(iy)
   rec_xy = tmp*Reciprocal_cel(2)
   rec_yy = tmp*Reciprocal_cel(5)
!   rec_zy = tmp*Reciprocal_cel(8)
   if (ix**2+iy**2 /= 0) then  ! EXCLUDE DIVERGENCY AT (0,0)
   do iz = -K_MAX_Z, K_MAX_Z
   tmp = dble(iz)
!   rec_xz = tmp*Reciprocal_cel(3)
!   rec_yz = tmp*Reciprocal_cel(6)
   rec_zz = tmp*h_step  ! zz in build differently 
   kx = rec_xx + rec_xy
   ky = rec_yx + rec_yy
   kz = rec_zz

   ew_coef = -1.0d0/(4.0d0*Ewald_alpha**2)
   ew_coef_1 = 1.0d0/(2.0d0*Ewald_alpha**2) ! for stress
   d2 = kx*kx + ky*ky + kz*kz
     k_vct = k_vct + 1 
     SK_P_Real = 0.0d0; SK_P_Imag = 0.0d0
     SK_G_Real = 0.0d0; SK_G_Imag = 0.0d0
     SK_D_Real = 0.0d0; SK_D_Imag = 0.0d0
     exp_fct = dexp(d2*ew_coef)/d2
     do i = 1, Natoms
      pot_i = 0.0d0
      a_f_i = 0.0d0
      qi = charge(i)
      xx = xxx(i); yy = yyy(i) ; zz = zzz(i)
      di_xx = dipol_xx(i) 
      di_yy = dipol_yy(i)
      di_zz = dipol_zz(i)
      KR = kx * xx + ky * yy + kz * zz
      cos_KR = dcos(KR)
      sin_KR = dsin(KR)
      p_dip = di_xx*rec_xx + di_yy*rec_yy + di_zz*rec_zz
       if (is_charge_distributed(i)) then
         qi = all_g_charges(i)
         p = qi*pref_in
         v_SK_G_real(i) = p*cos_KR
         v_SK_G_imag(i) = p*sin_KR 
         SK_G_Real = SK_G_Real + v_SK_G_real(i)
         SK_G_Imag = SK_G_Imag + v_SK_G_imag(i)
if (l_need_2nd_profile) then
         fi_field_G_real(i) = pref_in*cos_KR 
         fi_field_G_imag(i) = pref_in*sin_KR 
endif
       else
         qi = all_p_charges(i)
         p = qi
         v_SK_P_real(i) = p*cos_KR
         v_SK_P_imag(i) = p*sin_KR
         SK_P_Real = SK_P_Real + v_SK_P_real(i)
         SK_P_Imag = SK_P_Imag + v_SK_P_imag(i)
if (l_need_2nd_profile) then
         fi_field_P_real(i) = cos_KR 
         fi_field_P_imag(i) = sin_KR 
endif
       endif
       v_SK_D_Real(i) = - p_dip * cos_KR  !  !Re{i*pK*exp(i*KR)} = -pK*sin(KR)
       v_SK_D_Imag(i) =  p_dip * sin_KR   !  !Im{i*pK*exp(i*KR)} = pk*cos(KR)
       SK_D_Real = SK_D_Real + v_SK_D_Real(i) !Re{i*pK*exp(i*KR)} = -pK*sin(KR)
       SK_D_Imag = SK_D_Imag + v_SK_D_Imag(i) !Im{i*pK*exp(i*KR)} = pk*cos(KR)
     enddo ! i = 1, Natoms
     SK_real = SK_P_Real + SK_G_Real + SK_D_Real
     SK_imag = SK_P_imag + SK_G_imag + SK_D_imag
     S2 = SK_real*SK_real+SK_imag*SK_imag
     local_energy = local_energy + exp_fct*S2
! Forces 
    do i = 1, Natoms
       if (is_charge_distributed(i)) then
         qi = all_g_charges(i)
         ff = (-v_SK_G_Imag(i)+v_SK_D_Imag(i))*SK_real+&
              (v_SK_G_real(i)-v_SK_D_Real(i))*SK_imag
if (l_need_2nd_profile) then
       tmp_Re = fi_field_G_real(i)
       tmp_Im = fi_field_G_Imag(i)
       fi_i = tmp_Re*SK_real+tmp_Im*SK_imag
       fi(i) = fi(i) + fi_i*exp_fct*2.0d0
       a_pot(i) = apot(i) + fi(i)*qi
endif
       else
         qi = all_p_charges(i)
         p = qi
         ff = (-v_SK_P_Imag(i)+v_SK_D_Imag(i))*SK_real+&
              (v_SK_P_real(i)-v_SK_D_Real(i))*SK_imag
if (l_need_2nd_profile) then
       tmp_Re = fi_field_P_real(i)
       tmp_Im = fi_field_P_Imag(i)
       fi_i = tmp_Re*SK_real+tmp_Im*SK_imag
       fi(i) = fi(i) + fi_i*exp_fct*2.0d0
       a_pot(i) = apot(i) + fi(i)*qi
endif
       endif
    ff = ff * (-2.0d0*exp_fct)
    fx = ff * kx ; fy = ff * ky ; fz = ff * kz 
    fxx(i) = fxx(i) + fx ; fyy(i) = fyy(i) + fy ; fzz(i) = fzz(i) + fz
   enddo ! Natoms
   vir = 2.0d0/d2+ew_coef_1
!   S2_exp_fct=S2*exp_fct
!   sxx = (1.0d0-kx*kx*vir)*S2_exp_fct ; 
!   sxy = (1.0d0-kx*kx*vir)*S2_exp_fct 
!   syx = sxy 
!   sxz = (1.0d0-kx*kx*vir)*S2_exp_fct 
   endif ! ix**2+iy**2 /=0
   enddo ! iy
   enddo ! iz























 call initialize







 

  len = Natoms
  do i=1,len
       j=i
        rx=Reciprocal_cel(1)*xxx(j)+Reciprocal_cel(4)*yyy(j)+Reciprocal_cel(7)*zzz(j)
        ry=Reciprocal_cel(2)*xxx(j)+Reciprocal_cel(5)*yyy(j)+Reciprocal_cel(8)*zzz(j)
        rz=Reciprocal_cel(3)*xxx(j)+Reciprocal_cel(6)*yyy(j)+Reciprocal_cel(9)*zzz(j)

        e_x_cos(i,1)=dcos(rx*xxx(j)) ; e_y_cos(i,1)=dcos(ry*yyy(j)) ; e_z_cos(i,1)=dcos(rz*zzz(j))
        e_x_sin(i,1)=dsin(rx*xxx(j)) ; e_y_sin(i,1)=dsin(ry*yyy(j)) ; e_z_sin(i,1)=dsin(rz*zzz(j))
!        if (i_type_atom(j).eq.1) then
!        print*,i,atom_in_which_molecule(j), e_z_cos(i,1), e_z_sin(i,1) ,atom_xyz(j,3)
!        endif
    enddo !i=1,len

    do i=1,len
    do k=2,K_MAX_X
           e_x_cos(i,k)=e_x_cos(i,k-1)*e_x_cos(i,1)-e_x_sin(i,k-1)*e_x_sin(i,1)
           e_x_sin(i,k)=e_x_sin(i,k-1)*e_x_cos(i,1)+e_x_cos(i,k-1)*e_x_sin(i,1)
         enddo
    enddo
    do i=1,len
    do k=2,K_MAX_Y
           e_y_cos(i,k)=e_y_cos(i,k-1)*e_y_cos(i,1)-e_y_sin(i,k-1)*e_y_sin(i,1)
           e_y_sin(i,k)=e_y_sin(i,k-1)*e_y_cos(i,1)+e_y_cos(i,k-1)*e_y_sin(i,1)
     enddo
     enddo
     do i=1,len
     do k=2,K_MAX_Z
           e_z_cos(i,k)=e_z_cos(i,k-1)*e_z_cos(i,1)-e_z_sin(i,k-1)*e_z_sin(i,1)
           e_z_sin(i,k)=e_z_sin(i,k-1)*e_z_cos(i,1)+e_z_cos(i,k-1)*e_z_sin(i,1)
     enddo
     enddo  
    
     expfct=-0.25d0/ewald_alpha**2
     mmin=0
     nmin=1
     kkk=0
     do ll = 0 , K_MAX_X
       l=ll
       d = dble(ll)
       Rec_x1=Reciprocal_cel(1)*d ; Rec_y1=Reciprocal_cel(4)*d ; Rec_z1=Reciprocal_cel(7)*d
       do mm=mmin,K_MAX_Y
          m=iabs(mm)
          d = dble(mm)
       Rec_x2=Rec_x1+Reciprocal_cel(2)*d ; Rec_y2=Rec_y1+Reciprocal_cel(5)*d ; Rec_z2=Rec_z1+Reciprocal_cel(8)*d 
          if (mm.ge.0) then
              cos_xy(1:len)=e_x_cos(1:len,l)*e_y_cos(1:len,m)-e_x_sin(1:len,l)*e_y_sin(1:len,m)
              sin_xy(1:len)=e_x_sin(1:len,l)*e_y_cos(1:len,m)+e_y_sin(1:len,m)*e_x_cos(1:len,l)
          else
              cos_xy(1:len)=e_x_cos(1:len,l)*e_y_cos(1:len,m)+e_x_sin(1:len,l)*e_y_sin(1:len,m)
              sin_xy(1:len)=e_x_sin(1:len,l)*e_y_cos(1:len,m)-e_y_sin(1:len,m)*e_x_cos(1:len,l)
          endif
          do nn=NMIN,K_MAX_Z
          n=iabs(nn)
          kk=ll*ll+mm*mm+nn*nn
          d = dble(nn)
       Rec_x=Rec_x2+Reciprocal_cel(3)*d ; Rec_y=Rec_y2+Reciprocal_cel(6)*d ; Rec_z=Rec_z2+Reciprocal_cel(9)*d
          rk_sq = Rec_x*Rec_x+Rec_y*Rec_y+Rec_z*Rec_z
          if (rk_sq.le.reciprocal_cut_sq) then
          kkk=kkk+1
          if (mod(kkk-1,nprocs) == rank ) then    !test if this k vector correspond to the wanted CPU.
          if (nn.ge.0) then
             cos_KR(1:len)=cos_xy(1:len)*e_z_cos(1:len,n)-sin_xy(1:len)*e_z_sin(1:len,n)
             sin_KR(1:len)=sin_xy(1:len)*e_z_cos(1:len,n)+cos_xy(1:len)*e_z_sin(1:len,n)
          else
             cos_KR(1:len)=cos_xy(1:len)*e_z_cos(1:len,n)+sin_xy(1:len)*e_z_sin(1:len,n)
             sin_KR(1:len)=sin_xy(1:len)*e_z_cos(1:len,n)-cos_xy(1:len)*e_z_sin(1:len,n)
          endif !(nn.ge.0)
          cos_KR(1:len)=cos_KR(1:len)*q_reduced(i_type_atom(1:len))
          sin_KR(1:len)=sin_KR(1:len)*q_reduced(i_type_atom(1:len))
          SUM_cos_KR=SUM(cos_KR(1:len))  ;  SUM_sin_KR=SUM(sin_KR(1:len))
          if (rk_sq > 0.0d0) then
             AK=dexp(expfct*rk_sq)/rk_sq
          else
             AK=0.0d0
          endif
          En = AK*(SUM_cos_KR*SUM_cos_KR+SUM_sin_KR*SUM_sin_KR)
          vir0 = 2.0d0*(1.0d0/rk_sq-expfct)
          vir=vir0*En
          En_Q_cmplx = En_Q_cmplx + En
          do i = 1, len
             ff = (sin_KR(i)*SUM_cos_KR-cos_KR(i)*SUM_sin_KR)
             F_temp(i,1) = F_temp(i,1) + Rec_x*ff
             F_temp(i,2) = F_temp(i,2) + Rec_y*ff
             F_temp(i,3) = F_temp(i,3) + Rec_z*ff
          enddo
          sxx = vir*(Rec_x*Rec_x) ; sxy = vir*(Rec_x*Rec_y)  ; sxy = vir*(Rec_x*Rec_z)
                                  ; syy = vir*(Rec_y*Rec_y)  ; syz = vir*(Rec_y*Rec_z)
                                                             ; szz = vir*(Rec_z*Rec_z)

          local_stress_xx = local_stress_xx - sxx ; 
          local_stress_xy = local_stress_xy - sxy ;
          local_stress_xz = local_stress_xz - sxz ;
          local_stress_yy = local_stress_yy - syy ;
          local_stress_yz = local_stress_yz - syz ;
          local_stress_zz = local_stress_zz - szz ;


if (l_need_2nd_profile) then
      do iii=1,len
             Eni = AK*(cos_KR(iii)*SUM_cos_KR+sin_KR(iii)*SUM_sin_KR)
             a_pot(iii) = a_pot(iii) + Eni
             viri = vir0 * Eni
             p_press_11(iii) =  p_press_11(iii) - viri*(Rec_x*Rec_x)
             p_press_22(iii) =  p_press_22(iii) - viri*(Rec_y*Rec_y)
             p_press_33(iii) =  p_press_33(iii) - viri*(Rec_z*Rec_z)
             p_press_12(iii) =  p_press_12(iii) - viri*(Rec_x*Rec_y)
             p_press_13(iii) =  p_press_13(iii) - viri*(Rec_x*Rec_z)
             p_press_23(iii) =  p_press_23(iii) - viri*(Rec_y*Rec_z)
      enddo
endif

           endif     !do the job on this rank.(CPU)
          endif !(kk.le.K_SQ_MAX_in_FOURIER_EWALD)

         enddo !nn=NMIN,K_MAX_Z
         NMIN=-K_MAX_Z
        enddo !mm=mmin,K_MAX_Y
        MMIN=-K_MAX_Y
        enddo !ll = 0 , K_MAX_X
  
    call finalize   ! finish the stuff; rescale; assign global profiles
  
 contains
 subroutine initialize
 F_temp=0.0d0
 En_Q_cmplx=0.0d0
 e_x_cos(1:len,0)=1.0d0 ;  e_y_cos(1:len,0)=1.0d0 ; e_z_cos(1:len,0)=1.0d0
 e_x_sin(1:len,0)=0.0d0 ;  e_y_sin(1:len,0)=0.0d0 ; e_z_sin(1:len,0)=0.0d0

 local_stress_xx = 0.0d0 ; local_stress_xy = 0.0d0 ; local_stress_xz = 0.0d0
                           local_stress_yy = 0.0d0 ; local_stress_yz = 0.0d0
                                                     local_stress_zz = 0.0d0
if (l_need_2nd_profile) then
          allocate(p_press_11(len),p_press_22(len),p_press_33(len))
          allocate(p_press_12(len),p_press_13(len),p_press_23(len))
          allocate(a_pot(len)) ;  a_pot=0.0d0
          p_press_11 = 0.0d0 ; p_press_12 = 0.0d0 ; p_press_13 = 0.0d0
          p_press_22 = 0.0d0 ; p_press_23 = 0.0d0
          p_press_33 = 0.0d0
endif
 end subroutine initialize

 subroutine finalize
      en_factor = reciprocal_volume  !* 0.5d0  * 2.0d0 (multiplication by 2 is done because I used symmetry)
! REMEMBER that the force pre-factor is twice as much en_factor

      stress_Qcmplx_xx = (local_stress_xx + En_Q_cmplx) * en_factor
      stress_Qcmplx_xy = local_stress_xy * en_factor
      stress_Qcmplx_xz = local_stress_xz * en_factor
      stress_Qcmplx_yy = (local_stress_yy + En_Q_cmplx) * en_factor
      stress_Qcmplx_yz = local_stress_yz * en_factor
      stress_Qcmplx_zz = (local_stress_zz + En_Q_cmplx) * en_factor

      fxx(1:len) = fxx(1:len) + F_temp(1:len,1)*(2.0d0*en_factor)
      fyy(1:len) = fyy(1:len) + F_temp(1:len,2)*(2.0d0*en_factor)
      fzz(1:len) = fzz(1:len) + F_temp(1:len,3)*(2.0d0*en_factor)

      En_Q_cmplx = En_Q_cmplx * en_factor

if (l_need_2nd_profile) then
   do i = 1, len
             atom_profile(i)%pot = atom_profile(i)%pot  + a_pot(i)*en_factor
             atom_profile(i)%sxx = atom_profile(i)%sxx + (p_press_11(i) + a_pot(i))*en_factor
             atom_profile(i)%sxy = atom_profile(i)%sxy + p_press_12(i)*en_factor
             atom_profile(i)%sxz = atom_profile(i)%sxz + p_press_13(i)*en_factor
             atom_profile(i)%syx = atom_profile(i)%syx + p_press_12(i)*en_factor
             atom_profile(i)%syy = atom_profile(i)%syy + (p_press_22(i) + a_pot(i))*en_factor
             atom_profile(i)%syz = atom_profile(i)%syz + p_press_23(i)*en_factor
             atom_profile(i)%szx = atom_profile(i)%szx + p_press_13(i)*en_factor
             atom_profile(i)%szy = atom_profile(i)%szy + p_press_23(i)*en_factor
             atom_profile(i)%szz = atom_profile(i)%szz + (p_press_33(i) + a_pot(i))*en_factor
     enddo

     deallocate(a_pot)
     deallocate(p_press_12,p_press_13,p_press_23)
     deallocate(p_press_11,p_press_22,p_press_33)
endif
 deallocate(F_temp)
 deallocate(cos_KR,sin_KR,sin_xy,cos_xy)
 deallocate(e_x_sin,e_y_sin,e_z_sin)
 deallocate(e_x_cos,e_y_cos,e_z_cos)

 end subroutine finalize
      
 end subroutine ewald_fourier_3D_slow

 subroutine ewald_intra123_correct()
      use math_constants , only : rootpi
      use paralel_env_data
      use profiles_data, only : l_need_2nd_profile, atom_profile
      use Ewald_data
      use ALL_atoms_data, only : Natoms, atom_in_which_molecule, i_type_atom, xxx, yyy, zzz, &
                            fxx , fyy, fzz
      use atom_type_data, only : q_reduced, atom_type_charge, is_charge_distributed
      use sim_cel_data
      use boundaries
      use energies_data, only : En_Q123, ewself
      use stresses_data, only :  stress_Q123_xx,stress_Q123_xy,stress_Q123_xz,&
         stress_Q123_yx,stress_Q123_yy,stress_Q123_yz,&
         stress_Q123_zx,stress_Q123_zy,stress_Q123_zz
      use lists_data, only : list_excluded, size_list_excluded
      use physical_constants, only : temp_cvt_en_ALL, temp_cvt_en

      implicit none
      real(8), parameter :: en_factor = 0.5d0 ! to avoid overcounting of the energies
      integer kk,jj,iat,itype,neightot,k,jat,jtype,itypee
      integer i,j, i1
      real(8) qq,xij1,xij2,xij3,z,r,alphar,zinv,rinv,dalphar,r2,i_r2,i_r
      real(8) En, ff, fx,fy,fz, sxx,sxy,sxz,syx,syy,syz,szx,szy,szz
      real(8) ffcorrect,f1,f2,f3,r3inv,unbdcorrect
      real(8) ci,cj ! added in v1.1
      real(8) xx,yy,zz
      real(8), save :: twoalphapi
      logical, save :: l_first_pass = .true.
      integer, save :: N_size_dr_array
      real(8) a_p_i, a_str_i_11,a_str_i_12,a_str_i_13,a_str_i_22,a_str_i_23,a_str_i_33
      real(8) fx_i, fy_i, fz_i
      real(8), allocatable :: dx(:),dy(:),dz(:), dr_sq(:)
      
      call initialize
     
      i1 = 0
      do i = 1+rank, Natoms - 1, nprocs
        itype = i_type_atom(i)
        ci=q_reduced(itype)
        neightot = size_list_excluded(i)
        fx_i = 0.0d0 ; fy_i = 0.0d0 ; fz_i = 0.0d0
 if (l_need_2nd_profile) then
        a_p_i = 0.0d0
        a_str_i_11 = 0.0d0
        a_str_i_12 = 0.0d0
        a_str_i_13 = 0.0d0
        a_str_i_22 = 0.0d0
        a_str_i_23 = 0.0d0
        a_str_i_33 = 0.0d0
 endif
        do k  = 1, neightot 
          i1 = i1 + 1
          xx = dx(i1)    ; yy = dy(i1)  ; zz = dz(i1)  ; 
          r2 = dr_sq(i)  ; r = dsqrt(r) ; i_r2 = 1.0d0/r2; i_r = 1.0d0/r
          j = list_excluded(k,i)
          jtype = i_type_atom(j)
          cj=q_reduced(jtype)
          qq = (ci*cj)
          r3inv = zinv*rinv
          dalphar = 1.0d0 - derfc(alphar)
          En = qq*dalphar*rinv
          En_Q123 = En_Q123 - En
          ff = qq*(dalphar*r3inv - twoalphapi*exp(-alphar*alphar)*zinv)
          fx = ff*xx  ;  fy = ff*yy   ;   fz = ff*zz
          fx_i = fx_i + fx ; fy_i = fy_i + fy ; fz_i = fz_i + fz
          fxx(j) = fxx(j) - fx
          fyy(j) = fyy(j) - fy
          fzz(j) = fzz(j) - fz
          sxx = fx*xx  ; sxy = fx*yy ; sxz = fx*zz
          syy = fy*yy  ; syz = fy*zz
          szz = fz*zz

          stress_Q123_xx = stress_Q123_xx + sxx
          stress_Q123_xy = stress_Q123_xy + sxy
          stress_Q123_xz = stress_Q123_xz + sxz
          stress_Q123_yy = stress_Q123_yy + syy
          stress_Q123_yz = stress_Q123_yz + syz
          stress_Q123_zz = stress_Q123_zz + szz 

if (l_need_2nd_profile) then
             a_p_i = a_p_i - En
             a_str_i_11 = a_str_i_11 + sxx
             a_str_i_12 = a_str_i_12 + sxy
             a_str_i_13 = a_str_i_13 + sxz
             a_str_i_22 = a_str_i_22 + syy
             a_str_i_23 = a_str_i_23 + syz
             a_str_i_33 = a_str_i_33 + szz
             atom_profile(j)%pot = atom_profile(j)%pot - (En*en_factor)
             atom_profile(j)%sxx = atom_profile(j)%sxx + (sxx*en_factor)
             atom_profile(j)%sxy = atom_profile(j)%sxy + (sxy*en_factor)
             atom_profile(j)%sxz = atom_profile(j)%sxz + (sxz*en_factor)
             atom_profile(j)%syx = atom_profile(j)%syx + (syx*en_factor)
             atom_profile(j)%syy = atom_profile(j)%syy + (syy*en_factor)
             atom_profile(j)%syz = atom_profile(j)%syz + (syz*en_factor)
             atom_profile(j)%szx = atom_profile(j)%szx + (szx*en_factor)
             atom_profile(j)%szy = atom_profile(j)%szy + (szy*en_factor)
             atom_profile(j)%szz = atom_profile(j)%szz + (szz*en_factor)
endif
         
        end do  ! j cycle
        fxx(i) = fxx(i) + fx_i
        fyy(i) = fyy(i) + fy_i
        fzz(i) = fzz(i) + fz_i
if (l_need_2nd_profile) then
             atom_profile(i)%pot = atom_profile(i)%pot - (a_p_i*en_factor)
             atom_profile(i)%sxx = atom_profile(i)%sxx + (a_str_i_11*en_factor)
             atom_profile(i)%sxy = atom_profile(i)%sxy + (a_str_i_12*en_factor)
             atom_profile(i)%sxz = atom_profile(i)%sxz + (a_str_i_13*en_factor)
             atom_profile(i)%syx = atom_profile(i)%syx + (a_str_i_12*en_factor)
             atom_profile(i)%syy = atom_profile(i)%syy + (a_str_i_22*en_factor)
             atom_profile(i)%syz = atom_profile(i)%syz + (a_str_i_23*en_factor)
             atom_profile(i)%szx = atom_profile(i)%szx + (a_str_i_13*en_factor)
             atom_profile(i)%szy = atom_profile(i)%szy + (a_str_i_23*en_factor)
             atom_profile(i)%szz = atom_profile(i)%szz + (a_str_i_33*en_factor)
endif

   enddo ! i cycle

  print*, 'corrected ewald=',En_Q123*temp_cvt_en, ewself*temp_cvt_en,&
      ( En_Q123 - ewself ) * temp_cvt_en 

  contains
   subroutine initialize
   integer i,j,i1
      if (l_first_pass) then
        l_first_pass = .false.
        twoalphapi = 2.0d0*ewald_alpha/rootpi
        i1 = 0
        do i = 1+rank, Natoms-1, nprocs
        do j = 1, size_list_excluded(i)
           i1 = i1 + 1
        enddo ; enddo
        N_size_dr_array = i1
      endif

     En_Q123 = 0.0d0
     stress_Q123_xx = 0.0d0 ; stress_Q123_yy = 0.0d0 ; stress_Q123_zz = 0.0d0
     stress_Q123_xy = 0.0d0 ; stress_Q123_xz = 0.0d0 ; stress_Q123_yz = 0.0d0

     allocate (dx(N_size_dr_array))
     allocate (dy(N_size_dr_array))
     allocate (dz(N_size_dr_array))
     allocate (dr_sq(N_size_dr_array))

      i1 = 0
      do i = 1+rank, Natoms-1 , nprocs
       do j = 1, size_list_excluded(i)
         i1 = i1 + 1
         dx(i1) = xxx(i) - xxx(j)
         dy(i1) = yyy(i) - yyy(j)
         dz(i1) = zzz(i) - zzz(j)
       enddo
      enddo

      call periodic_images(dx(1:i1),dy(1:i1),dz(1:i1))
      dr_sq(1:i1) = dx(1:i1)*dx(1:i1) + dy(1:i1)*dy(1:i1) + dz(1:i1)*dz(1:i1)

   end subroutine initialize

  end subroutine ewald_intra123_correct
 
  subroutine ShortRanged_14_correct 
  use math_constants , only : rootpi
  use paralel_env_data
  use profiles_data, only : l_need_2nd_profile, atom_profile
  use Ewald_data
  use ALL_atoms_data, only : Natoms, atom_in_which_molecule, i_type_atom, xxx, yyy, zzz, &
                        fxx , fyy, fzz
  use atom_type_data, only : q_reduced, atom_type_charge, is_charge_distributed, which_atom_pair, &
                        atom_type2_vdwPrm
  use sim_cel_data
  use boundaries
  use energies_data, only : En_14
  use stresses_data, only :  stress_14_xx,stress_14_xy,stress_14_xz,&
      stress_14_yx,stress_14_yy,stress_14_yz,&
      stress_14_zx,stress_14_zy,stress_14_zz
  use connectivity_ALL_data, only : list_14
  use physical_constants, only : temp_cvt_en_ALL, temp_cvt_en
  use sizes_data
  use interpolate_data
  use pairs_14_data


  implicit none
  logical, save :: l_first_pass = .true.
  real(8), parameter :: en_factor = 0.5d0
  real(8), allocatable :: dx(:),dy(:),dz(:),dr_sq(:)
  real(8) sxx,sxy,sxz,syx,syy,syz,szx,szy,szz
  real(8) xx,yy,zz,t1,t2,vk,vk1,gk,gk1,gk2,ppp,red_factor
  real(8) r,r2,r_squared,Inverse_r_squared
  real(8) fx,fy,fz,ff,En,Eni
  integer i,j,k, i1, N_size_array, i_pair, it,jt, itype, i_type_pair
  integer NDX
  
    call initialize

    i1 = 0
    do i_pair = 1+rank, N_pairs_14 , nprocs
    i1 = i1 + 1
     i = list_14(1,i_pair)
     j = list_14(2,i_pair)
     itype = list_14(3, i_pair)
     red_factor = prm_14_types(itype)
     it = i_type_atom(i) 
     jt = i_type_atom(j)
     i_type_pair = which_atom_pair(it,jt) 
     r = dsqrt(r2)
     Inverse_r_squared = 1.0d0/r2
     NDX = max(1,int(r*RDR))
     ppp = (r*RDR) - dble(ndx)
     if (atom_type2_vdwPrm(1,i_type_pair) > 1.0d-10) then
        vk  = vvdw(ndx,i_type_pair)  ;  vk1 = vvdw(ndx+1,i_type_pair)
        t1 = vk  + (vk1 - vk )*ppp
        t2 = vk1 + (vvdw(ndx+2,i_type_pair) - vk1)*(ppp - 1.0d0)
        En = (t1 + (t2-t1)*ppp*0.5d0)*red_factor
        gk  = gvdw(ndx,i_type_pair)  ;  gk1 = gvdw(ndx+1,i_type_pair)
        t1 = gk  + (gk1 - gk )*ppp
        t2 = gk1 + (gvdw(ndx+2,i_type_pair) - gk1)*(ppp - 1.0d0)
        ff = (t1 + (t2-t1)*ppp*0.5d0)*Inverse_r_squared*red_factor
        xx = dx(k)   ; yy = dy(k)    ; zz = dz(k)
        fx = ff*xx   ;  fy = ff*yy   ;  fz =  ff*zz
        sxx = fx*xx  ;  sxy = fx*yy  ;  sxz = fx*zz
                        syy = fy*yy  ;  syz = fy*zz
                                        szz = fz*zz
     endif         
     if (dabs(q_reduced(it)) > 1.0d-10 .and. dabs(q_reduced(jt)) > 1.0d-10 ) then
! In conjunction with Ewald 
        vk  = vele(ndx,i_type_pair)  ;  vk1 = vele(ndx+1,i_type_pair)
        t1 = vk  + (vk1 - vk )*ppp
        t2 = vk1 + (vele(ndx+2,i_type_pair) - vk1)*(ppp - 1.0d0)
        En = En + (t1 + (t2-t1)*ppp*0.5d0)*red_factor
        gk  = gele(ndx,i_type_pair)  ;  gk1 = gele(ndx+1,i_type_pair)
        t1 = gk  + (gk1 - gk )*ppp
        t2 = gk1 + (gele(ndx+2,i_type_pair) - gk1)*(ppp - 1.0d0)
        ff = (t1 + (t2-t1)*ppp*0.5d0)*Inverse_r_squared*red_factor
        xx = dx(k)   ; yy = dy(k)    ; zz = dz(k)
        fx = fx + ff*xx  ;  fy = fy + ff*yy  ;  fz = fz + ff*zz
        sxx = sxx+fx*xx  ;  sxy = sxy+fx*yy  ;  sxz = sxz+fx*zz
                            syy = syy+fy*yy  ;  syz = syz+fy*zz
                                                szz = szz+fz*zz
     endif
! Finalize
    stress_14_xx = stress_14_xx + sxx
    stress_14_yy = stress_14_yy + syy
    stress_14_zz = stress_14_zz + szz
    stress_14_xy = stress_14_xy + sxy
    stress_14_xz = stress_14_xz + sxz
    stress_14_yz = stress_14_yz + syz
    
    fxx(i) = fxx(i) + fx
    fyy(i) = fyy(i) + fy
    fzz(i) = fzz(i) + fz
    fxx(j) = fxx(j) - fx
    fyy(j) = fyy(j) - fy
    fzz(j) = fzz(j) - fz

    if (l_need_2nd_profile) then
     atom_profile(i)%pot = atom_profile(i)%pot + (En*en_factor)
     atom_profile(i)%sxx = atom_profile(i)%sxx + (sxx*en_factor)
     atom_profile(i)%sxy = atom_profile(i)%sxy + (sxy*en_factor)
     atom_profile(i)%sxz = atom_profile(i)%sxz + (sxz*en_factor)
     atom_profile(i)%syx = atom_profile(i)%syx + (sxy*en_factor)
     atom_profile(i)%syy = atom_profile(i)%syy + (syy*en_factor)
     atom_profile(i)%syz = atom_profile(i)%syz + (syz*en_factor)
     atom_profile(i)%szx = atom_profile(i)%szx + (sxz*en_factor)
     atom_profile(i)%szy = atom_profile(i)%szy + (syz*en_factor)
     atom_profile(i)%szz = atom_profile(i)%szz + (szz*en_factor)
     atom_profile(j)%pot = atom_profile(j)%pot + (En*en_factor)
     atom_profile(j)%sxx = atom_profile(j)%sxx + (sxx*en_factor)
     atom_profile(j)%sxy = atom_profile(j)%sxy + (sxy*en_factor)
     atom_profile(j)%sxz = atom_profile(j)%sxz + (sxz*en_factor)
     atom_profile(j)%syx = atom_profile(j)%syx + (sxy*en_factor)
     atom_profile(j)%syy = atom_profile(j)%syy + (syy*en_factor)
     atom_profile(j)%syz = atom_profile(j)%syz + (syz*en_factor)
     atom_profile(j)%szx = atom_profile(j)%szx + (sxz*en_factor)
     atom_profile(j)%szy = atom_profile(j)%szy + (syz*en_factor)
     atom_profile(j)%szz = atom_profile(j)%szz + (szz*en_factor)
    endif

    enddo

  deallocate(dx,dy,dz,dr_sq) 

  contains
  subroutine initialize
   integer i,j,i1, i_pair
   if (l_first_pass) then
      l_first_pass = .false.
       i1 = 0
       do i_pair = 1+rank, N_pairs_14 , nprocs
          i1 = i1 + 1
       enddo
       N_size_array = i1
    endif

    allocate(dx(N_size_array),dy(N_size_array),dz(N_size_array),dr_sq(N_size_array))
    i1 = 0
    do i_pair = 1+rank, N_pairs_14 , nprocs
     i1 = i1 + 1
     i = list_14(1,i_pair)
     j = list_14(2,i_pair)
     dx(i1) = xxx(i) - xxx(j)
     dy(i1) = yyy(i) - yyy(j)
     dz(i1) = zzz(i) - zzz(j)
    enddo
    dr_sq(1:i1) = dx(1:i1)*dx(1:i1) + dy(1:i1)*dy(1:i1) + dz(1:i1)*dz(1:i1)
  end subroutine initialize
  end subroutine ShortRanged_14_correct


 end module fourier_ewald_3D
