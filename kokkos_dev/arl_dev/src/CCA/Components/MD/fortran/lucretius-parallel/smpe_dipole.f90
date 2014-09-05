module smpe_dipole_module
 ! contains utilities to do SMPE with dipoles
 implicit none
 public :: set_Q_DIP_2D
 public :: set_Q_DIP_2D_ENERGY
 public :: smpe_eval1_Q_DIP_2D
 public :: smpe_eval1_Q_DIP_2D_ENERGY
 public :: smpe_eval2_Q_DIP_2D
 public :: set_Q_DIP_3D
 public :: smpe_eval1_Q_DIP_3D 
 public :: smpe_eval1_Q_DIP_3D_ENERGY
 public :: smpe_eval2_Q_DIP_3D
 contains

 subroutine set_Q_DIP_2D
! WIll set both potential and stress charges
    use variables_smpe
    use Ewald_data
    use sim_cel_data
    use math, only : invert3 
    use sizes_data, only : Natoms
    use all_atoms_data , only : all_charges,l_WALL, zz,is_charge_distributed,all_dipoles,&
        all_dipoles_xx,all_dipoles_yy,all_dipoles_zz,zzz

    implicit none
    real(8) cox(order_spline_xx),coy(order_spline_yy),coz(order_spline_zz)
    real(8) cox_DD(order_spline_xx),coy_DD(order_spline_yy),coz_DD(order_spline_zz)
    integer i,j,k,nx,ny,nz,ix,iy,iz,jx,jy,jz,kx,ky,kz
    integer ii_xx,ii_yy,ii_zz,i_adress,ii,jj,kk,i1
    integer kx1,ky1,kz1
    real(8) ci,z,tmpqz,tmpqyz,db(3)
    real(8) icel(9)
    real(8) inv_box(3),box(3), pref
    logical, save :: l_very_first_pass = .true.
    logical l_i
    real(8) dipole,tmpqdipyz,tmpqdipz,di_xx,di_yy,di_zz
    real(8) tmpz,tmp_y_z,tmp_y_dz,tmp_dy_z,tmpdz
    real(8) rec_xx,rec_xy,rec_yx,rec_yy,rec_zz,d2,kkx,kky,kkz,tmp,temp
    real(8) t_x,t_y,t_z
    real(8) dipole_term,q_term,p_nabla_ux,p_nabla_uy,p_nabla_uz
    real(8) i_sim_zz,axx,axy,ayx,ayy,azz

!    box(1) = sim_cel(1); box(2) = sim_cel(5); box(3) = sim_cel(9)
!    inv_box = 1.0d0/box
!    axx = 
    qqq1_Re = 0.0d0;
    qqq2_Re = 0.0d0; 
    
     i_sim_zz = Inverse_cel(9)
     axx=Inverse_cel(1) ; axy = Inverse_cel(2);
     ayx=Inverse_cel(4) ; ayy = Inverse_cel(5)

    do i = 1,Natoms
      ci = all_charges(i)
      dipole = all_dipoles(i) ; 
      di_xx = all_dipoles_xx(i); di_yy = all_dipoles_yy(i);di_zz=all_dipoles_zz(i)
        p_nabla_ux = di_xx*axx + di_yy*axy
        p_nabla_uy = di_xx*ayx + di_yy*ayy
        p_nabla_uz = di_zz
      cox(1:order_spline_xx) = spline2_REAL_pp_x(i,1:order_spline_xx)
      coy(1:order_spline_yy) = spline2_REAL_pp_y(i,1:order_spline_yy)
      coz(1:order_spline_zz) = spline2_REAL_pp_z(i,1:order_spline_zz)
      cox_DD(1:order_spline_xx) = spline2_REAL_dd_x(i,1:order_spline_xx)
      coy_DD(1:order_spline_yy) = spline2_REAL_dd_y(i,1:order_spline_yy)
      coz_DD(1:order_spline_zz) = spline2_REAL_dd_z(i,1:order_spline_zz)

      z = zzz(i)
      nx = int(tx(i)) - order_spline_xx  ! n = 0..nfftx-1-splines_xx
      ny = int(ty(i)) - order_spline_yy
      nz = int(z) - order_spline_zz
      iz = nz
      do jz = 0, order_spline_zz-1
      iz = iz + 1
      kz = iz +  h_cut_z ! h_cut_z = nfftz/2
      if (kz >= nfftz) then  ! it cannot be nfftz
        write(6,*) 'error in set_q_2D kz >= nfftz; choose more nfftz points'
        write(6,*) 'You need to make nfftz at least ',int(box(3)) , &
        'or the first 2^N integer'
        write(6,*) 'kz boxz nfftz=',kz, box(3), nfftz
      STOP
      endif
      if (kz < 0) then
        write(6,*) 'error in set_q_2D kz < 0 : lower the splines order or increase the nfft',kz
        write(6,*) 'order spline = ',order_spline_xx,order_spline_yy,order_spline_zz
        write(6,*) 'nfft hcutz =',nfftx,nffty,nfftz,h_cut_z
        STOP
      endif
      tmpqz = coz(jz+1)*ci
      tmpz =   coz(jz+1) ! iz = jz + 1
      tmpdz =  coz_DD(jz+1)
      iy = ny
      do jy = 0, order_spline_yy-1
        iy = iy + 1
        if (iy < 0) then
          ky = iy + nffty
        else
          ky = iy
        endif
        tmpqyz = tmpqz * coy(jy+1)
        tmp_y_z  = coy(jy+1)     * tmpz
        tmp_dy_z = coy_DD(jy+1)  * tmpz;
        tmp_y_dz = coy(jy+1)     * tmpdz;
        ix = nx
        do jx = 0, order_spline_xx-1
          ix = ix + 1
          if (ix < 0) then
             kx = ix + nfftx
          else
             kx = ix
          endif
          i_adress = (ky+kz*nffty)*nfftx + kx + 1
!print*,i,jx,jy,jz,'k=',kx,ky,kz,i_adress
!read(*,*)

          t_x= cox_DD(jx+1) * tmp_y_z   * dfftx
          t_y= cox(jx+1)    * tmp_dy_z  * dffty
          t_z= cox(jx+1)    * tmp_y_dz  !* dble(nfftz)


          q_term = tmpqyz*cox(jx+1)
          dipole_term = t_x * p_nabla_ux + t_y * p_nabla_uy + t_z * p_nabla_uz

          qqq1_Re(i_adress) = qqq1_Re(i_adress) + (q_term + dipole_term)! potential POINT CHARGE
          qqq2_Re(i_adress) = qqq2_Re(i_adress) + z*(q_term + dipole_term) ! stress POINT CHARGE

        enddo ! jx
        enddo ! jy
      enddo ! jz
    enddo ! i
    qqq1 = cmplx(qqq1_Re,0.0d0,kind=8)
    qqq2 = cmplx(qqq2_Re,0.0d0,kind=8)

  end subroutine set_Q_DIP_2D

!----------------------------------------------------------------


 subroutine set_Q_DIP_2D_ENERGY
! WIll set both potential and stress charges
    use variables_smpe
    use Ewald_data
    use sim_cel_data
    use math, only : invert3
    use sizes_data, only : Natoms
    use all_atoms_data , only : all_charges,l_WALL, zz,is_charge_distributed,all_dipoles,&
        all_dipoles_xx,all_dipoles_yy,all_dipoles_zz,zzz

    implicit none
    real(8) cox(order_spline_xx),coy(order_spline_yy),coz(order_spline_zz)
    real(8) cox_DD(order_spline_xx),coy_DD(order_spline_yy),coz_DD(order_spline_zz)
    integer i,j,k,nx,ny,nz,ix,iy,iz,jx,jy,jz,kx,ky,kz
    integer ii_xx,ii_yy,ii_zz,i_adress,ii,jj,kk,i1
    integer kx1,ky1,kz1
    real(8) ci,z,tmpqz,tmpqyz,db(3)
    real(8) icel(9)
    real(8) inv_box(3),box(3), pref
    logical, save :: l_very_first_pass = .true.
    logical l_i
    real(8) dipole,tmpqdipyz,tmpqdipz,di_xx,di_yy,di_zz
    real(8) tmpz,tmp_y_z,tmp_y_dz,tmp_dy_z,tmpdz
    real(8) rec_xx,rec_xy,rec_yx,rec_yy,rec_zz,d2,kkx,kky,kkz,tmp,temp
    real(8) t_x,t_y,t_z
    real(8) dipole_term,q_term,p_nabla_ux,p_nabla_uy,p_nabla_uz
    real(8) i_sim_zz,axx,axy,ayx,ayy,azz


    qqq1_Re = 0.0d0;


     i_sim_zz = Inverse_cel(9)
     axx=Inverse_cel(1) ; axy = Inverse_cel(2);
     ayx=Inverse_cel(4) ; ayy = Inverse_cel(5)
    qqq1_Re = 0.0d0;
    qqq2_Re = 0.0d0;

     i_sim_zz = Inverse_cel(9)
     axx=Inverse_cel(1) ; axy = Inverse_cel(2);
     ayx=Inverse_cel(4) ; ayy = Inverse_cel(5)

    do i = 1,Natoms
      ci = all_charges(i)
      dipole = all_dipoles(i) ;
      di_xx = all_dipoles_xx(i); di_yy = all_dipoles_yy(i);di_zz=all_dipoles_zz(i)
        p_nabla_ux = di_xx*axx + di_yy*axy
        p_nabla_uy = di_xx*ayx + di_yy*ayy
        p_nabla_uz = di_zz
      cox(1:order_spline_xx) = spline2_REAL_pp_x(i,1:order_spline_xx)
      coy(1:order_spline_yy) = spline2_REAL_pp_y(i,1:order_spline_yy)
      coz(1:order_spline_zz) = spline2_REAL_pp_z(i,1:order_spline_zz)
      cox_DD(1:order_spline_xx) = spline2_REAL_dd_x(i,1:order_spline_xx)
      coy_DD(1:order_spline_yy) = spline2_REAL_dd_y(i,1:order_spline_yy)
      coz_DD(1:order_spline_zz) = spline2_REAL_dd_z(i,1:order_spline_zz)

      z = zzz(i)
      nx = int(tx(i)) - order_spline_xx  ! n = 0..nfftx-1-splines_xx
      ny = int(ty(i)) - order_spline_yy
      nz = int(z) - order_spline_zz
      iz = nz
      do jz = 0, order_spline_zz-1
      iz = iz + 1
      kz = iz +  h_cut_z ! h_cut_z = nfftz/2
      if (kz >= nfftz) then  ! it cannot be nfftz
        write(6,*) 'error in set_q_2D kz >= nfftz; choose more nfftz points'
        write(6,*) 'You need to make nfftz at least ',int(box(3)) , &
        'or the first 2^N integer'
        write(6,*) 'kz boxz nfftz=',kz, box(3), nfftz
      STOP
      endif
      if (kz < 0) then
        write(6,*) 'error in set_q_2D kz < 0 : lower the splines order or increase the nfft',kz
        write(6,*) 'order spline = ',order_spline_xx,order_spline_yy,order_spline_zz
        write(6,*) 'nfft hcutz =',nfftx,nffty,nfftz,h_cut_z
        STOP
      endif
      tmpqz = coz(jz+1)*ci
      tmpz =   coz(jz+1) ! iz = jz + 1
      tmpdz =  coz_DD(jz+1)
      iy = ny
      do jy = 0, order_spline_yy-1
        iy = iy + 1
        if (iy < 0) then
          ky = iy + nffty
        else
          ky = iy
        endif
        tmpqyz = tmpqz * coy(jy+1)
        tmp_y_z  = coy(jy+1)     * tmpz
        tmp_dy_z = coy_DD(jy+1)  * tmpz;
        tmp_y_dz = coy(jy+1)     * tmpdz;
        ix = nx
        do jx = 0, order_spline_xx-1
          ix = ix + 1
          if (ix < 0) then
             kx = ix + nfftx
          else
             kx = ix
          endif
          i_adress = (ky+kz*nffty)*nfftx + kx + 1
!print*,i,jx,jy,jz,'k=',kx,ky,kz,i_adress
!read(*,*)

          t_x= cox_DD(jx+1) * tmp_y_z   * dfftx
          t_y= cox(jx+1)    * tmp_dy_z  * dffty
          t_z= cox(jx+1)    * tmp_y_dz  !* dble(nfftz)


          q_term = tmpqyz*cox(jx+1)
          dipole_term = t_x * p_nabla_ux + t_y * p_nabla_uy + t_z * p_nabla_uz

          qqq1_Re(i_adress) = qqq1_Re(i_adress) + (q_term + dipole_term)! potential POINT CHARGE


        enddo ! jx
        enddo ! jy
      enddo ! jz
    enddo ! i
    qqq1 = cmplx(qqq1_Re,0.0d0,kind=8)

  end subroutine set_Q_DIP_2D_ENERGY

!----------------------------------------------------------------
   subroutine smpe_eval1_Q_DIP_2D
! Eval stresses and energy for Q_POINT charges AND dipoles
     use variables_smpe
     use Ewald_data
     use sim_cel_data
     use boundaries, only : cel_properties, get_reciprocal_cut
     use sizes_data, only : Natoms
     use energies_data
     use cut_off_data
     use stresses_data, only : stress,stress_Qcmplx,stress_Qcmplx_as_in_3D

     implicit none
     real(8), parameter :: four_pi_sq = 39.4784176043574d0
     real(8), parameter :: Pi2 = 6.28318530717959d0
     integer i,j,k,ix,iy,iz,jx,jy,jz,nx,ny,nz,nx0,ny0,nz0
     integer m_iy1,m_iy2,m_ix1,m_ix2,m_iz1,m_iz2,m_ix,m_iy,m_iz
     integer ii_xx,ii_yy,ii_zz
     integer jy1,jx1,jz1,mz,mz1,i_index
     real(8) sxx,sxy,sxz,syy,syz,szz
     real(8) expfct,i4a2,i4a2_2, tmp, vterm,vterm2
     real(8) spline_product, En, Eni, ff, vir,vir0
     real(8) temp_vir_cz, t
     real(8) qim, qre, exp_fct
     real(8) rec_xx,rec_yy,rec_zz,rec_xy,rec_xz,rec_yx,rec_zx,rec_zykx
     real(8) d2,i_d2
     real(8) kx , ky, kz
     real(8) local_potential
     real(8) lsxx,lsyy,lszz,lsxy,lsxz,lsyz,lsyx,lszx,lszy,lsxz_prime,lsyz_prime,lszz_prime
     real(8) area, tem_vir_az, vir_az, q_az_re,q_az_im
     real(8) local_potential1, En1, local_potential_PP,local_potential_GG,local_potential_PG

     complex(8), allocatable :: qq1(:),qq2(:)

! array qqq1 contains data for energy
! array qqq2 contains data for stresses xz,yz,zz


     area = Area_xy

     call get_reciprocal_cut


     nx0 = nfftx/2 ; ny0 = nffty/2 ; nz0 = nfftz/2
     i4a2 = 0.25d0/(Ewald_alpha*Ewald_alpha)
     expfct = - i4a2
     i4a2_2 = 2.0d0 * i4a2

     local_potential = 0.0d0
     lsxx = 0.0d0; lsxy = 0.0d0; lsxz = 0.0d0
     lsyx = 0.0d0; lsyy = 0.0d0; lsyz = 0.0d0
     lszx = 0.0d0; lszy = 0.0d0; lszz = 0.0d0
     lsxz_prime=0.0d0;lsyz_prime=0.0d0;lszz_prime=0.0d0
     do jz = -nz0+1,nz0
       mz = (jz + nz0)
       mz1 = mz - 1
       if (mz.gt.nz0) mz1 = mz1 - nfftz
       rec_zz =  dble(mz1)*reciprocal_zz
       do jy = 1,nffty
          jy1 = jy - 1
          if (jy > ny0) jy1 = jy1 - nffty
          tmp = dble(jy1)
          rec_xy = tmp*Reciprocal_cel(2)
          rec_yy = tmp*Reciprocal_cel(5)
          do jx = 1, nfftx
            jx1 = jx -1
            if (jx > nx0) jx1 = jx1 - nfftx
            tmp = dble(jx1)
            rec_xx = tmp*Reciprocal_cel(1)
            rec_yx = tmp*Reciprocal_cel(4)
            i_index = ((jy-1)+(mz-1)*nffty)*nfftx + jx
            kz = rec_zz
            kx = rec_xx + rec_xy
            ky = rec_yx + rec_yy
            d2 = kx*kx + ky*ky + kz*kz
 if (jy1**2+jx1**2 > 0 .and. d2 < reciprocal_cut_sq) then
              i_d2 = 1.0d0/d2
              exp_fct = dexp(expfct*d2) * i_d2
              qre = real(qqq1(i_index),kind=8) ;    qim=dimag(qqq1(i_index))
              spline_product = spline2_CMPLX_xx(jx)*spline2_CMPLX_yy(jy)*spline2_CMPLX_zz(mz)
              vterm =  exp_fct / (area*spline_product) * reciprocal_zz
              vterm2 = vterm*2.0d0
              En = vterm*(qre*qre+qim*qim)
              vir0 = 2.0d0*(i_d2 + i4a2)
              vir = vir0 * En
              local_potential = local_potential + En
              lsxx = lsxx + En - vir*kx*kx ;
              lsxy = lsxy - vir*kx*ky  ;
              lsyy = lsyy + En - vir*ky*ky ;
              vir_az = vterm2 
              q_az_re = real(qqq2(i_index),kind=8) ;    q_az_im=dimag(qqq2(i_index))
              tem_vir_az = vir_az * (qre*q_az_im - qim*q_az_re)
              lsxz = lsxz + tem_vir_az*kx ;
              lsyz = lsyz + tem_vir_az*ky ;
              lszz = lszz + tem_vir_az*kz ;
              lsxz_prime = lsxz_prime -vir*kx*kz
              lsyz_prime = lsyz_prime -vir*ky*kz
              lszz_prime = lszz_prime + En - vir*kz*kz

              qqq1(i_index) = qqq1(i_index)*vterm


 else
              qqq1(i_index) = 0.0d0
              qqq2(i_index) = 0.0d0
 endif     !  reciprocal_cutt within cut off
        enddo
     enddo
    enddo


   En_Q_cmplx = En_Q_cmplx + local_potential
   En_Q = En_Q + local_potential
   stress_Qcmplx(1) =  lsxx
   stress_Qcmplx(2) =  lsyy
   stress_Qcmplx(3) =  lszz
   stress_Qcmplx(4) = (lsxx+lsyy+lszz)/3.0d0
   stress_Qcmplx(5) =  lsxy
   stress_Qcmplx(6) =  lsxz
   stress_Qcmplx(7) =  lsyz
   stress_Qcmplx(8) =  lsxy  ! sxy = syx if no dipols!!!!
   stress_Qcmplx(9) =  lsxz  ! change it when dipols are in
   stress_Qcmplx(10) = lsyz  ! change it when dipols are in
   stress_Qcmplx_as_in_3D(1:2)=0.0d0
   stress_Qcmplx_as_in_3D(3) = lszz_prime
   stress_Qcmplx_as_in_3D(4) = lszz_prime
   stress_Qcmplx_as_in_3D(5) = 0.0d0
   stress_Qcmplx_as_in_3D(6) = lsxz_prime
   stress_Qcmplx_as_in_3D(7) = lsyz_prime
   stress_Qcmplx_as_in_3D(8) = 0.0d0
   stress_Qcmplx_as_in_3D(9) = lsxz_prime
   stress_Qcmplx_as_in_3D(10) = lsyz_prime

   stress(:) = stress(:) + stress_Qcmplx(:)

!print*,'local_potential=',local_potential

 end subroutine smpe_eval1_Q_DIP_2D ! energy and stresses

!---------------------------------

   subroutine smpe_eval1_Q_DIP_2D_ENERGY
! Eval stresses and energy for Q_POINT charges AND dipoles
     use variables_smpe
     use Ewald_data
     use sim_cel_data
     use boundaries, only : cel_properties, get_reciprocal_cut
     use sizes_data, only : Natoms
     use energies_data
     use cut_off_data

     implicit none
     real(8), parameter :: four_pi_sq = 39.4784176043574d0
     real(8), parameter :: Pi2 = 6.28318530717959d0
     integer i,j,k,ix,iy,iz,jx,jy,jz,nx,ny,nz,nx0,ny0,nz0
     integer m_iy1,m_iy2,m_ix1,m_ix2,m_iz1,m_iz2,m_ix,m_iy,m_iz
     integer ii_xx,ii_yy,ii_zz
     integer jy1,jx1,jz1,mz,mz1,i_index
     real(8) sxx,sxy,sxz,syy,syz,szz
     real(8) expfct,i4a2,i4a2_2, tmp, vterm
     real(8) spline_product, En, Eni, ff, vir,vir0
     real(8) temp_vir_cz, t
     real(8) qim, qre, exp_fct
     real(8) rec_xx,rec_yy,rec_zz,rec_xy,rec_xz,rec_yx,rec_zx,rec_zykx
     real(8) d2,i_d2
     real(8) kx , ky, kz
     real(8) local_potential
     real(8) area, q_az_re,q_az_im
     real(8) local_potential1, En1, local_potential_PP,local_potential_GG,local_potential_PG

     complex(8), allocatable :: qq1(:),qq2(:)

! array qqq1 contains data for energy
! array qqq2 contains data for stresses xz,yz,zz


     area = Area_xy

     call get_reciprocal_cut

     nx0 = nfftx/2 ; ny0 = nffty/2 ; nz0 = nfftz/2
     i4a2 = 0.25d0/(Ewald_alpha*Ewald_alpha)
     expfct = - i4a2
     i4a2_2 = 2.0d0 * i4a2

     local_potential = 0.0d0
     do jz = -nz0+1,nz0
       mz = (jz + nz0)
       mz1 = mz - 1
       if (mz.gt.nz0) mz1 = mz1 - nfftz
       rec_zz =  dble(mz1)*reciprocal_zz
       do jy = 1,nffty
          jy1 = jy - 1
          if (jy > ny0) jy1 = jy1 - nffty
          tmp = dble(jy1)
          rec_xy = tmp*Reciprocal_cel(2)
          rec_yy = tmp*Reciprocal_cel(5)
          do jx = 1, nfftx
            jx1 = jx -1
            if (jx > nx0) jx1 = jx1 - nfftx
            tmp = dble(jx1)
            rec_xx = tmp*Reciprocal_cel(1)
            rec_yx = tmp*Reciprocal_cel(4)
            i_index = ((jy-1)+(mz-1)*nffty)*nfftx + jx
            kz = rec_zz
            kx = rec_xx + rec_xy
            ky = rec_yx + rec_yy
            d2 = kx*kx + ky*ky + kz*kz
 if (jy1**2+jx1**2 > 0 .and. d2 < reciprocal_cut_sq) then
              i_d2 = 1.0d0/d2
              exp_fct = dexp(expfct*d2) * i_d2
              qre = real(qqq1(i_index),kind=8) ;    qim=dimag(qqq1(i_index))
              spline_product = spline2_CMPLX_xx(jx)*spline2_CMPLX_yy(jy)*spline2_CMPLX_zz(mz)
              vterm =  exp_fct / (area*spline_product) * reciprocal_zz
              En = vterm*(qre*qre+qim*qim)
              local_potential = local_potential + En              

 endif     !  reciprocal_cutt within cut off
        enddo
     enddo
    enddo


   En_Q_cmplx = En_Q_cmplx + local_potential
   En_Q = En_Q + local_potential

 end subroutine smpe_eval1_Q_DIP_2D_ENERGY
!----------------------------------
 
 subroutine smpe_eval2_Q_DIP_2D ! forces

     use math, only : invert3
     use sizes_data, only : Natoms
     use sim_cel_data
     use ALL_atoms_data, only : all_p_charges,all_g_charges, fxx, fyy, fzz, xx, yy, zz,&
                                zzz,is_charge_distributed, all_charges,&
                                all_dipoles,all_dipoles_xx,all_dipoles_yy,all_dipoles_zz
     use profiles_data, only : atom_profile, l_need_2nd_profile
     use variables_smpe
     use Ewald_data
     implicit none

     real(8) temp,icel(9)
     real(8), allocatable :: fx(:),fy(:),fz(:)
     real(8), allocatable :: a_pot(:)
     integer i,j,k,ix,iy,iz,kx,ky,kz,nx,ny,nz,mx,my,mz, i_index
     integer ii_xx, ii_yy , ii_zz
     real(8) vterm, En, ci, exp_fct, Eni, ff, Eni0
     real(8) sxx,sxy,sxz,syx,syy,syz,szx,szy,szz
     real(8) tmp_x,tmp_y,tmp_z, tmp_x_y,tmp_x_z,tmp_y_z, tmp_y_dz, tmp_dy_z, tmp_x_dy,tmp_dx_y,&
             tmp_x_dz,tmp_dx_z, tmp_x_2,tmp_y_2,tmp_z_2,tmp_dx,tmp_dy,tmp_dz,&
             tmp_dx_2,tmp_dy_2,tmp_dz_2
     real(8) i_sim_zz
     real(8) qsum, spline_product
     real(8) z
     real(8) spl_xx(order_spline_xx), spl_yy(order_spline_yy),spl_zz(order_spline_zz)
     real(8) spl_xx_DRV(order_spline_xx), spl_yy_DRV(order_spline_yy),spl_zz_DRV(order_spline_zz)
     real(8) spl_xx_2_DRV(order_spline_xx), spl_yy_2_DRV(order_spline_yy),spl_zz_2_DRV(order_spline_zz)
     real(8) t, t_x, t_y, t_z
     real(8) axx,axy,axz,ayx,ayy,ayz,azx,azy,azz
     real(8) pref,di_xx,di_yy,di_zz,dipole,dipole_term
     real(8) tf_x,tf_y,tf_z,ffxx,ffyy,ffzz
     real(8) t2_zz,t2_yy,t2_xx,t2_x1y1z,t2_xy1z1,t2_x1yz1
     real(8), save :: eta
     logical l_i
     real(8) p_nabla_ux,p_nabla_uy,p_nabla_uz
     real(8), allocatable :: a_pot_GG(:),a_pot_PG(:),a_pot_GP(:),a_fi(:),a_fi1(:)
     real(8), allocatable :: buffer3(:,:)

allocate(buffer3(Natoms,3));buffer3=0.0d0
! qqq1 is charge array (actually field array)
! qqq2 is stress array

     i_sim_zz = Inverse_cel(9)
     axx = Inverse_cel(1) ; axy = Inverse_cel(2)
     ayx = Inverse_cel(4) ; ayy = Inverse_cel(5)
     azz = Inverse_cel(9)

     allocate(fx(Natoms),fy(Natoms),fz(Natoms) )
     allocate(a_pot(Natoms),a_fi(Natoms))
!allocate(a_pot_GG(Natoms),a_pot_PG(Natoms),a_pot_GP(Natoms))
     fx = 0.0d0 ; fy = 0.0d0 ; fz = 0.0d0 ;
     a_pot = 0.0d0; a_fi=0.0d0
!allocate(a_fi1(Natoms)); a_fi1=0.0d0
!a_pot_GG=0.0d0;a_pot_PG=0.0d0;a_pot_GP=0.0d0
     do i = 1, Natoms
     ci = all_charges(i)
     dipole = all_dipoles(i) ; di_xx = all_dipoles_xx(i); di_yy = all_dipoles_yy(i);di_zz=all_dipoles_zz(i)
      p_nabla_ux = di_xx*axx + di_yy*axy
      p_nabla_uy = di_xx*ayx + di_yy*ayy
      p_nabla_uz = di_zz

       z = zzz(i)
       nx = int(tx(i)) - order_spline_xx
       ny = int(ty(i)) - order_spline_yy
       nz = int(z) - order_spline_zz
       spl_xx(1:order_spline_xx) = spline2_REAL_pp_x(i,1:order_spline_xx)
       spl_yy(1:order_spline_yy) = spline2_REAL_pp_y(i,1:order_spline_yy)
       spl_zz(1:order_spline_zz) = spline2_REAL_pp_z(i,1:order_spline_zz)
       spl_xx_DRV(1:order_spline_xx) = spline2_REAL_dd_x(i,1:order_spline_xx)
       spl_yy_DRV(1:order_spline_yy) = spline2_REAL_dd_y(i,1:order_spline_yy)
       spl_zz_DRV(1:order_spline_zz) = spline2_REAL_dd_z(i,1:order_spline_zz)
       spl_xx_2_DRV(1:order_spline_xx) = spline2_REAL_dd_2_x(i,1:order_spline_xx)
       spl_yy_2_DRV(1:order_spline_yy) = spline2_REAL_dd_2_y(i,1:order_spline_yy)
       spl_zz_2_DRV(1:order_spline_zz) = spline2_REAL_dd_2_z(i,1:order_spline_zz)


        t = 0.0d0
        ii_zz = nz
        do iz=1,order_spline_zz
          ii_zz = ii_zz + 1
          kz = ii_zz + nfftz/2 + 1
          tmp_z =   spl_zz(iz)
          tmp_dz =  spl_zz_DRV(iz)
          tmp_dz_2 = spl_zz_2_DRV(iz)
          ii_yy = ny
          mz = kz !+ nfftz/2   ! use mz rather than kz because from fft freq are wrapped-arround
          if (mz > nfftz) mz = mz - nfftz
          do iy=1,order_spline_yy
            ii_yy = ii_yy + 1
            if (ii_yy < 0 ) then
              ky = ii_yy + nffty + 1
            else
              ky = ii_yy +1
            endif
            my = ky !+ nffty/2
            if (my > nffty) my = my - nffty
            tmp_y = spl_yy(iy)
            tmp_dy = spl_yy_DRV(iy)
            tmp_dy_2 = spl_yy_2_DRV(iy)
            tmp_y_z  = tmp_y     * tmp_z;
            tmp_dy_z = tmp_dy * tmp_z;
            tmp_y_dz = tmp_y     * tmp_dz;
           
            ii_xx = nx
            do ix=1,order_spline_xx
              ii_xx = ii_xx + 1
              if (ii_xx < 0 ) then
                kx = ii_xx + nfftx +1
              else
                kx = ii_xx +1
              endif
              mx = kx !+ nfftx/2
              if (mx > nfftx) mx = mx - nfftx
              if (mx**2+my**2 /= 0 ) then  ! skip (0,0)
                tmp_x    = spl_xx(ix)
                tmp_dx   = spl_xx_DRV(ix)
                tmp_dx_2 = spl_xx_2_DRV(ix)
                tmp_x_z = tmp_x * tmp_z
                tmp_y_z = tmp_y * tmp_z 
                tmp_x_y = tmp_x * tmp_y
                i_index = ((my-1)+(mz-1)*nffty)*nfftx + mx
                qsum = -real(qqq1 (i_index), kind=8)
                t_x= tmp_dx   * tmp_y_z   * dfftx
                t_y= tmp_x    * tmp_dy_z  * dffty
                t_z= tmp_x    * tmp_y_dz  !* dble(nfftz)
                t2_xx =tmp_dx_2 * tmp_y_z * dfftx2
                t2_yy =tmp_dy_2 * tmp_x_z * dffty2
                t2_zz =tmp_dz_2 * tmp_x_y   
                t2_x1y1z = tmp_dx*tmp_dy*tmp_z * (dfftxy)
                t2_x1yz1 = tmp_dx *tmp_y*tmp_dz *(dfftx)
                t2_xy1z1 = tmp_x *tmp_dy*tmp_dz *(dffty)

                tf_x = t_x
                tf_y = t_y
                tf_z = t_z

                spline_product =  tmp_x * tmp_y_z

                dipole_term = t_x * p_nabla_ux + t_y * p_nabla_uy + t_z * p_nabla_uz

                ffxx = ci*(tf_x*axx+tf_y*axy) + &
                       p_nabla_ux*(t2_xx   *axx    + t2_x1y1z*axy    )+&
                       p_nabla_uy*(t2_x1y1z*axx    + t2_yy   *axy    )+& 
                       p_nabla_uz*(t2_x1yz1*axx    + t2_xy1z1*axy    )

                ffyy = ci*(tf_x*ayx+tf_y*ayy) + &
                       p_nabla_ux*(t2_xx   *ayx    + t2_x1y1z*ayy    )+&
                       p_nabla_uy*(t2_x1y1z*ayx    + t2_yy   *ayy    )+&
                       p_nabla_uz*(t2_x1yz1*ayx    + t2_xy1z1*ayy    )

                ffzz = ci*(tf_z) + &
                       p_nabla_ux*(                                t2_x1yz1)+&
                       p_nabla_uy*(                                t2_xy1z1)+&
                       p_nabla_uz*(                                t2_zz   )
                fx(i) = fx(i) + ffxx*qsum
                fy(i) = fy(i) + ffyy*qsum
                fz(i) = fz(i) + ffzz*qsum

                Eni0 = - (qsum) * spline_product
                Eni =   ci*Eni0+dipole_term*(-qsum)
                a_pot(i) = a_pot(i) + Eni
                a_fi(i) = a_fi(i) + Eni0
!buffer3(i,1) = buffer3(i,1) + (-qsum)*(tf_x*axx+tf_y*axy)
!buffer3(i,2) = buffer3(i,2) + (-qsum)*(tf_x*ayx+tf_y*ayy)
!buffer3(i,3) = buffer3(i,3) + (-qsum)*t_z
            endif
            enddo
          enddo
        enddo

      enddo !i
    do i = 1, Natoms
     fxx(i) = fxx(i) + 2.0d0*fx(i)
     fyy(i) = fyy(i) + 2.0d0*fy(i)
     fzz(i) = fzz(i) + 2.0d0*fz(i)
    enddo

   if (l_need_2nd_profile) then
    atom_profile%pot = atom_profile%pot + a_pot*2.0d0
    atom_profile%Qpot = atom_profile%Qpot + a_pot*2.0d0
    atom_profile%fi = atom_profile%fi + a_fi*2.0d0
!    do i=1,Natoms; atom_profile(i)%buffer3(:) = atom_profile(i)%buffer3(:) + buffer3(i,:)*2.0d0 ; enddo
! stresses as well
   endif
!print*, 'sum a_pot=',sum(a_pot)
!open(unit=666,file='fort.666',recl=1000)
!do i = 1, Natoms
!write(666,*) i,a_pot(i),2.0d0*fx(i),2.0d0*fy(i),2.0d0*fz(i)
!enddo
   deallocate(fx,fy,fz)
   deallocate(a_pot,a_fi)
deallocate(buffer3)
   end subroutine smpe_eval2_Q_DIP_2D

!----------------------------------
!----------------------------------
!----------------------------------
!----------------------------------
!----------------------------------
!----------------------------------
!    3D   3D    3D     3D
!----------------------------------
!----------------------------------
!----------------------------------
!----------------------------------

  subroutine set_Q_DIP_3D
! WIll set both potential and stress charges
    use variables_smpe
    use Ewald_data
    use sim_cel_data
    use math, only : invert3
    use sizes_data, only : Natoms
    use all_atoms_data , only : all_charges,l_WALL, zz,is_charge_distributed,&
                                all_dipoles,all_dipoles_xx,all_dipoles_yy,all_dipoles_zz

    implicit none
    real(8) cox(order_spline_xx),coy(order_spline_yy),coz(order_spline_zz)
    real(8) cox_DD(order_spline_xx),coy_DD(order_spline_yy),coz_DD(order_spline_zz)
    integer i,j,k,nx,ny,nz,ix,iy,iz,jx,jy,jz,kx,ky,kz
    integer ii_xx,ii_yy,ii_zz,i_adress,ii,jj,kk,i1
    real(8) ci,di_xx,di_yy,di_zz,z,tmpqz,tmpqyz,db(3)
    real(8) inv_box(3),box(3), pref,icel(9)
    logical, save :: l_very_first_pass = .true.
    real(8), save :: eta
    logical l_i
    integer kx1,ky1,kz1
    real(8) rec_xx,rec_xy,rec_yx,rec_yy,rec_zz,d2,kkx,kky,kkz,tmp
    real(8) axx,axy,axz,ayx,ayy,ayz,azx,azy,azz
    real(8) dipole,dipole_term,q_term
    real(8) t_x,t_y,t_z,p_nabla_ux,p_nabla_uy,p_nabla_uz
    real(8) tmpz,tmpdz,tmp_y_z,tmp_x_y,tmp_x_z,tmp_dy_z,tmp_dx_y,tmp_dx_z,temp,tmp_y_dz


    qqq1_Re = 0.0d0

     axx = Inverse_cel(1) ; axy = Inverse_cel(2); axz = Inverse_cel(3)
     ayx = Inverse_cel(4) ; ayy = Inverse_cel(5); ayz = Inverse_cel(6)
     azx = Inverse_cel(7) ; azy = Inverse_cel(8); azz = Inverse_cel(9)

    do i = 1,Natoms
      ci = all_charges(i)
      dipole = all_dipoles(i) ; di_xx = all_dipoles_xx(i); di_yy = all_dipoles_yy(i);di_zz=all_dipoles_zz(i)
       p_nabla_ux = di_xx*axx + di_yy*axy + di_zz*axz
       p_nabla_uy = di_xx*ayx + di_yy*ayy + di_zz*ayz
       p_nabla_uz = di_xx*azx + di_yy*azy + di_zz*azz
      cox(1:order_spline_xx) = spline2_REAL_pp_x(i,1:order_spline_xx)
      coy(1:order_spline_yy) = spline2_REAL_pp_y(i,1:order_spline_yy)
      coz(1:order_spline_zz) = spline2_REAL_pp_z(i,1:order_spline_zz)
      cox_DD(1:order_spline_xx) = spline2_REAL_dd_x(i,1:order_spline_xx)
      coy_DD(1:order_spline_yy) = spline2_REAL_dd_y(i,1:order_spline_yy)
      coz_DD(1:order_spline_zz) = spline2_REAL_dd_z(i,1:order_spline_zz)


      nx = int(tx(i)) - order_spline_xx
      ny = int(ty(i)) - order_spline_yy
      nz = int(tz(i)) - order_spline_zz

      iz = nz
      do jz = 0, order_spline_zz-1
      iz = iz + 1
      if (iz < 0) then
         kz = iz + nfftz
      else
         kz = iz
      endif

      tmpqz = coz(jz+1)*ci
      tmpz  = coz(jz+1) ! iz = jz + 1
      tmpdz = coz_DD(jz+1)

      iy = ny
      do jy = 0, order_spline_yy-1
        iy = iy + 1
        if (iy < 0) then
          ky = iy + nffty
        else
          ky = iy
        endif

        tmpqyz = tmpqz * coy(jy+1)
        tmp_y_z  = coy(jy+1)     * tmpz
        tmp_dy_z = coy_DD(jy+1)  * tmpz;
        tmp_y_dz = coy(jy+1)     * tmpdz;

        ix = nx
        do jx = 0, order_spline_xx-1
          ix = ix + 1
          if (ix < 0) then
             kx = ix + nfftx
          else
             kx = ix
          endif


          t_x= cox_DD(jx+1) * tmp_y_z   * dfftx
          t_y= cox(jx+1)    * tmp_dy_z  * dffty
          t_z= cox(jx+1)    * tmp_y_dz  * dfftz

          i_adress = (ky+kz*nffty)*nfftx + kx + 1
          
          q_term = tmpqyz*cox(jx+1)
          dipole_term = t_x * p_nabla_ux + t_y * p_nabla_uy + t_z * p_nabla_uz

          qqq1_Re(i_adress) = qqq1_Re(i_adress) + q_term + dipole_term! potential POINT CHARGE


        enddo ! jx
        enddo ! jy
      enddo ! jz
    enddo ! i

    qqq1 = cmplx(qqq1_Re,0.0d0,kind=8)
!    qqq2 = cmplx(qqq2_Re,0.0d0,kind=8)

  end subroutine set_Q_DIP_3D


!-------------------------------------------------
!-------------------------------------------------

   subroutine smpe_eval1_Q_DIP_3D
! Eval stresses and energy for Q_POINT charges only
     use variables_smpe
     use Ewald_data
     use sim_cel_data
     use boundaries, only : cel_properties, get_reciprocal_cut
     use sizes_data, only : Natoms
     use energies_data
     use cut_off_data
     use stresses_data, only : stress, stress_Qcmplx

     implicit none
     real(8), parameter :: four_pi_sq = 39.4784176043574d0
     real(8), parameter :: Pi2 = 6.28318530717959d0
     integer i,j,k,ix,iy,iz,jx,jy,jz,nx,ny,nz,nx0,ny0,nz0
     integer m_iy1,m_iy2,m_ix1,m_ix2,m_iz1,m_iz2,m_ix,m_iy,m_iz
     integer ii_xx,ii_yy,ii_zz
     integer jy1,jx1,jz1,i_index
     real(8) sxx,sxy,sxz,syy,syz,szz
     real(8) expfct,i4a2,i4a2_2, tmp, vterm,vterm2
     real(8) spline_product, En, Eni, ff, vir,vir0
     real(8) temp_vir_cz, t
     real(8) qim, qre, exp_fct
     real(8) rec_xx,rec_yy,rec_zz,rec_xy,rec_yx,rec_xz,rec_zx,rec_zy,rec_yz
     real(8) d2,i_d2
     real(8) kx , ky, kz
     real(8) local_potential
     real(8) area
     real(8) local_potential1, En1, local_potential_PP,local_potential_GG,local_potential_PG
     real(8) lsxx,lsyy,lszz,lsxy,lsxz,lsyx,lsyz,lszx,lszy

     call get_reciprocal_cut

     nx0 = nfftx/2 ; ny0 = nffty/2 ; nz0 = nfftz/2
     i4a2 = 0.25d0/(Ewald_alpha*Ewald_alpha)
     expfct = - i4a2
     i4a2_2 = 2.0d0 * i4a2


     local_potential = 0.0d0
     lsxx = 0.0d0; lsxy = 0.0d0; lsxz = 0.0d0
     lsyx = 0.0d0; lsyy = 0.0d0; lsyz = 0.0d0
     lszx = 0.0d0; lszy = 0.0d0; lszz = 0.0d0

     do jz = 1,nfftz
       jz1 = jz-1
       if (jz > nz0) jz1 = jz1 - nfftz
       tmp = dble(jz1)
       rec_xz = tmp*Reciprocal_cel(3)
       rec_yz = tmp*Reciprocal_cel(6)
       rec_zz = tmp*Reciprocal_cel(9)
       do jy = 1,nffty
          jy1 = jy - 1
          if (jy > ny0) jy1 = jy1 - nffty
          tmp = dble(jy1)
          rec_xy = tmp*Reciprocal_cel(2)
          rec_yy = tmp*Reciprocal_cel(5)
          rec_zy = tmp*Reciprocal_cel(8)
!          rec_yy = dble(jy1) * Reciprocal_Sim_Box(2)
          do jx = 1, nfftx
            jx1 = jx -1
            if (jx > nx0) jx1 = jx1 - nfftx
            tmp = dble(jx1)
            rec_xx = tmp*Reciprocal_cel(1)
            rec_yx = tmp*Reciprocal_cel(4)
            rec_zx = tmp*Reciprocal_cel(7)
            i_index = ((jy-1)+(jz-1)*nffty)*nfftx + jx
            kz = rec_zx + rec_zy + rec_zz
            kx = rec_xx + rec_xy + rec_xz
            ky = rec_yx + rec_yy + rec_yz
            d2 = kx*kx + ky*ky + kz*kz
 if (d2 < reciprocal_cut_sq.and.jz1**2+jy1**2+jx1**2 /= 0) then
              i_d2 = 1.0d0/d2
              exp_fct = dexp(expfct*d2) * i_d2
!              qre = real(qqq1(i_index),kind=8) ;    qim=dimag(qqq1(i_index))
              spline_product = spline2_CMPLX_xx(jx)*spline2_CMPLX_yy(jy)*spline2_CMPLX_zz(jz)
              vterm =  exp_fct / (Volume*spline_product) * Pi2
              En = vterm*real(qqq1(i_index)*conjg(qqq1(i_index)),kind=8)  !*(qre*qre+qim*qim)
             local_potential = local_potential + En
             vir0 = 2.0d0*(i_d2 + i4a2)
             vir = vir0 * En
             lsxx = lsxx + En - vir*kx*kx ;
             lsxy = lsxy - vir*kx*ky  ;
             lsyy = lsyy + En - vir*ky*ky ;
             lsxz = lsxz - vir*kx*kz
             lsyz = lsyz - vir*ky*kz
             lszz = lszz + En - vir*kz*kz

!print*, jx,jy,jz,'term=',vir*kx*kx, vir*ky*ky, vir*kz*kz, vir*kx*ky,vir*kx*kz,vir*ky*kz
!print*,kx,vir0
!read(*,*)

             qqq1(i_index) = qqq1(i_index)*vterm

 else
              qqq1(i_index) = 0.0d0
 endif     !  reciprocal_cutt within cut off
        enddo
     enddo
    enddo

   En_Q_cmplx = En_Q_cmplx + local_potential
   En_Q = En_Q + local_potential
   stress_Qcmplx(1) =  lsxx
   stress_Qcmplx(2) =  lsyy
   stress_Qcmplx(3) =  lszz
   stress_Qcmplx(4) = (lsxx+lsyy+lszz)/3.0d0
   stress_Qcmplx(5) =  lsxy
   stress_Qcmplx(6) =  lsxz
   stress_Qcmplx(7) =  lsyz
   stress_Qcmplx(8) =  lsxy  ! sxy = syx if no dipols!!!!
   stress_Qcmplx(9) =  lsxz  ! change it when dipols are in
   stress_Qcmplx(10) = lsyz  ! change it when dipols are in

   stress(:) = stress(:) + stress_Qcmplx(:)

!print*, 'local energy = ',local_potential


 end subroutine smpe_eval1_Q_DIP_3D 

!-------------------------
!--------------------------

!------------

  subroutine smpe_eval1_Q_DIP_3D_ENERGY
! Eval stresses and energy for Q_POINT charges only
     use variables_smpe
     use Ewald_data
     use sim_cel_data
     use boundaries, only : cel_properties, get_reciprocal_cut
     use sizes_data, only : Natoms
     use energies_data
     use cut_off_data


     implicit none
     real(8), parameter :: four_pi_sq = 39.4784176043574d0
     real(8), parameter :: Pi2 = 6.28318530717959d0
     integer i,j,k,ix,iy,iz,jx,jy,jz,nx,ny,nz,nx0,ny0,nz0
     integer m_iy1,m_iy2,m_ix1,m_ix2,m_iz1,m_iz2,m_ix,m_iy,m_iz
     integer ii_xx,ii_yy,ii_zz
     integer jy1,jx1,jz1,i_index
     real(8) expfct,i4a2,i4a2_2, tmp, vterm,vterm2
     real(8) spline_product, En, Eni, ff, vir,vir0
     real(8) temp_vir_cz, t
     real(8) qim, qre, exp_fct
     real(8) rec_xx,rec_yy,rec_zz,rec_xy,rec_yx,rec_xz,rec_zx,rec_zy,rec_yz
     real(8) d2,i_d2
     real(8) kx , ky, kz
     real(8) local_potential
     real(8) area


     call get_reciprocal_cut

     nx0 = nfftx/2 ; ny0 = nffty/2 ; nz0 = nfftz/2
     i4a2 = 0.25d0/(Ewald_alpha*Ewald_alpha)
     expfct = - i4a2
     i4a2_2 = 2.0d0 * i4a2


     local_potential = 0.0d0
     do jz = 1,nfftz
       jz1 = jz-1
       if (jz > nz0) jz1 = jz1 - nfftz
       tmp = dble(jz1)
       rec_xz = tmp*Reciprocal_cel(3)
       rec_yz = tmp*Reciprocal_cel(6)
       rec_zz = tmp*Reciprocal_cel(9)
       do jy = 1,nffty
          jy1 = jy - 1
          if (jy > ny0) jy1 = jy1 - nffty
          tmp = dble(jy1)
          rec_xy = tmp*Reciprocal_cel(2)
          rec_yy = tmp*Reciprocal_cel(5)
          rec_zy = tmp*Reciprocal_cel(8)
!          rec_yy = dble(jy1) * Reciprocal_Sim_Box(2)
          do jx = 1, nfftx
            jx1 = jx -1
            if (jx > nx0) jx1 = jx1 - nfftx
            tmp = dble(jx1)
            rec_xx = tmp*Reciprocal_cel(1)
            rec_yx = tmp*Reciprocal_cel(4)
            rec_zx = tmp*Reciprocal_cel(7)
            i_index = ((jy-1)+(jz-1)*nffty)*nfftx + jx
            kz = rec_zx + rec_zy + rec_zz
            kx = rec_xx + rec_xy + rec_xz
            ky = rec_yx + rec_yy + rec_yz
            d2 = kx*kx + ky*ky + kz*kz
 if (d2 < reciprocal_cut_sq.and.jz1**2+jy1**2+jx1**2 /= 0) then
              i_d2 = 1.0d0/d2
              exp_fct = dexp(expfct*d2) * i_d2
!              qre = real(qqq1(i_index),kind=8) ;    qim=dimag(qqq1(i_index))
              spline_product = spline2_CMPLX_xx(jx)*spline2_CMPLX_yy(jy)*spline2_CMPLX_zz(jz)
              vterm =  exp_fct / (Volume*spline_product) * Pi2
              En = vterm*real(qqq1(i_index)*conjg(qqq1(i_index)),kind=8)  !*(qre*qre+qim*qim)
             local_potential = local_potential + En
endif     !  reciprocal_cutt within cut off
        enddo
     enddo
    enddo

   En_Q_cmplx = En_Q_cmplx + local_potential
   En_Q = En_Q + local_potential



 end subroutine smpe_eval1_Q_DIP_3D_ENERGY

!--------------------------
!-------------------------

   subroutine smpe_eval2_Q_DIP_3D ! forces

     use math, only : invert3
     use sizes_data, only : Natoms
     use sim_cel_data
     use ALL_atoms_data, only : all_p_charges,all_g_charges, fxx, fyy, fzz, xx, yy, zz,&
                                is_charge_distributed, all_charges,zzz,xxx,yyy,&
                                all_dipoles,all_dipoles_xx,all_dipoles_yy,all_dipoles_zz
     use profiles_data, only : atom_profile, l_need_2nd_profile
     use variables_smpe
     use Ewald_data
     implicit none

     real(8) temp,icel(9)
     real(8), allocatable :: fx(:),fy(:),fz(:)
     real(8), allocatable :: a_pot(:)
     integer i,j,k,ix,iy,iz,kx,ky,kz,nx,ny,nz,mx,my,mz, i_index
     integer ii_xx, ii_yy , ii_zz
     real(8) vterm, En, ci, exp_fct, Eni, ff, Eni0
     real(8) sxx,sxy,sxz,syx,syy,syz,szx,szy,szz
     real(8) tmp_x,tmp_y,tmp_z, tmp_x_z, tmp_x_y,tmp_y_z,tmp_dx,tmp_dy,tmp_dz
     real(8) tmp_y_dz, tmp_x_dy,tmp_dy_x,tmp_x_dz,tmp_dz_x,tmp_dy_z, tmp, tmpdz
     real(8) tmp_x_2,tmp_y_2,tmp_z_2,tmp_dx_2,tmp_dy_2,tmp_dz_2
     real(8) i_sim_zz
     real(8) qsum, spline_product
     real(8) z
     real(8) spl_xx(order_spline_xx), spl_yy(order_spline_yy),spl_zz(order_spline_zz)
     real(8) spl_xx_DRV(order_spline_xx), spl_yy_DRV(order_spline_yy),spl_zz_DRV(order_spline_zz)
     real(8) spl_xx_2_DRV(order_spline_xx), spl_yy_2_DRV(order_spline_yy),spl_zz_2_DRV(order_spline_zz)
     real(8) t, t_x, t_y, t_z
     real(8) i_cel_1,i_cel_2,i_cel_3,i_cel_4,i_cel_5,i_cel_6,i_cel_7,i_cel_8,i_cel_9
     real(8) pref,di_xx,di_yy,di_zz,dipole,dipole_term
     real(8) t2_zz,t2_yy,t2_xx,t2_x1y1z,t2_xy1z1,t2_x1yz1
     real(8), save :: eta
     logical l_i
     real(8) p_nabla_ux,p_nabla_uy,p_nabla_uz
     real(8)  tf_x,tf_y,tf_z, ffxx,ffyy,ffzz
     real(8) axx,axy,axz,ayx,ayy,ayz,azx,azy,azz 
     real(8), allocatable :: a_pot_GG(:),a_pot_PG(:),a_pot_GP(:),a_fi(:),a_fi1(:)
     real(8),allocatable :: buffer3(:,:)
allocate(buffer3(Natoms,3)); buffer3=0.0d0
! qqq1 is charge array (actually field array)
! qqq2 is stress array

     axx = Inverse_cel(1) ; axy = Inverse_cel(2); axz=Inverse_cel(3)
     ayx = Inverse_cel(4) ; ayy = Inverse_cel(5); ayz=Inverse_cel(6)
     azx = Inverse_cel(7) ; azy = Inverse_cel(8); azz=Inverse_cel(9)
     
     allocate(fx(Natoms),fy(Natoms),fz(Natoms) )
     allocate(a_pot(Natoms),a_fi(Natoms))
!allocate(a_pot_GG(Natoms),a_pot_PG(Natoms),a_pot_GP(Natoms))
     fx = 0.0d0 ; fy = 0.0d0 ; fz = 0.0d0 ;
     a_pot = 0.0d0; a_fi=0.0d0
!allocate(a_fi1(Natoms)); a_fi1=0.0d0
!a_pot_GG=0.0d0;a_pot_PG=0.0d0;a_pot_GP=0.0d0
     do i = 1, Natoms
       ci = all_charges(i)
       dipole = all_dipoles(i) ; 
       di_xx = all_dipoles_xx(i); di_yy = all_dipoles_yy(i);di_zz=all_dipoles_zz(i)
       z = zzz(i)
       nx = int(tx(i)) - order_spline_xx
       ny = int(ty(i)) - order_spline_yy
       nz = int(tz(i)) - order_spline_zz
       spl_xx(1:order_spline_xx) = spline2_REAL_pp_x(i,1:order_spline_xx)
       spl_yy(1:order_spline_yy) = spline2_REAL_pp_y(i,1:order_spline_yy)
       spl_zz(1:order_spline_zz) = spline2_REAL_pp_z(i,1:order_spline_zz)
       spl_xx_DRV(1:order_spline_xx) = spline2_REAL_dd_x(i,1:order_spline_xx)
       spl_yy_DRV(1:order_spline_yy) = spline2_REAL_dd_y(i,1:order_spline_yy)
       spl_zz_DRV(1:order_spline_zz) = spline2_REAL_dd_z(i,1:order_spline_zz)
       spl_xx_DRV(1:order_spline_xx) = spline2_REAL_dd_x(i,1:order_spline_xx)
       spl_yy_DRV(1:order_spline_yy) = spline2_REAL_dd_y(i,1:order_spline_yy)
       spl_zz_DRV(1:order_spline_zz) = spline2_REAL_dd_z(i,1:order_spline_zz)
       spl_xx_2_DRV(1:order_spline_xx) = spline2_REAL_dd_2_x(i,1:order_spline_xx)
       spl_yy_2_DRV(1:order_spline_yy) = spline2_REAL_dd_2_y(i,1:order_spline_yy)
       spl_zz_2_DRV(1:order_spline_zz) = spline2_REAL_dd_2_z(i,1:order_spline_zz)


        t = 0.0d0
        ii_zz = nz
        do iz=1,order_spline_zz
          tmp_z =   spl_zz(iz)
          tmp_dz =  spl_zz_DRV(iz)
          tmp_dz_2 = spl_zz_2_DRV(iz)
          ii_zz = ii_zz + 1
          if (ii_zz < 0 ) then
              kz = ii_zz + nfftz + 1
          else
              kz = ii_zz +1
          endif
          mz = kz

          if (mz > nfftz) mz = mz - nfftz
          ii_yy = ny
          mz = kz !+ nfftz/2   ! use mz rather than kz because from fft freq are wrapped-arround
          if (mz > nfftz) mz = mz - nfftz
          do iy=1,order_spline_yy
            ii_yy = ii_yy + 1
            if (ii_yy < 0 ) then
              ky = ii_yy + nffty + 1
            else
              ky = ii_yy +1
            endif
            my = ky !+ nffty/2
            if (my > nffty) my = my - nffty
            tmp_y = spl_yy(iy)
            tmp_dy = spl_yy_DRV(iy)
            tmp_dy_2 = spl_yy_2_DRV(iy)
            tmp_y_z  = tmp_y     * tmp_z;
            tmp_dy_z = tmp_dy * tmp_z;
            tmp_y_dz = tmp_y     * tmp_dz;

            ii_xx = nx
            do ix=1,order_spline_xx
              ii_xx = ii_xx + 1
              if (ii_xx < 0 ) then
                kx = ii_xx + nfftx +1
              else
                kx = ii_xx +1
              endif
              mx = kx !+ nfftx/2
              if (mx > nfftx) mx = mx - nfftx
                tmp_x    = spl_xx(ix)
                tmp_dx   = spl_xx_DRV(ix)
                tmp_dx_2 = spl_xx_2_DRV(ix)
                tmp_x_z = tmp_x * tmp_z
                tmp_y_z = tmp_y * tmp_z
                tmp_x_y = tmp_x * tmp_y
                i_index = ((my-1)+(mz-1)*nffty)*nfftx + mx
                qsum = -real(qqq1 (i_index), kind=8)
                t_x= tmp_dx   * tmp_y_z   * dfftx
                t_y= tmp_x    * tmp_dy_z  * dffty
                t_z= tmp_x    * tmp_y_dz  * dfftz
                t2_xx =tmp_dx_2 * tmp_y_z * dfftx2
                t2_yy =tmp_dy_2 * tmp_x_z * dffty2
                t2_zz =tmp_dz_2 * tmp_x_y * dfftz2
                t2_x1y1z = tmp_dx*tmp_dy_z* (dfftxy)
                t2_x1yz1 = tmp_dx *tmp_y_dz *(dfftxz)
                t2_xy1z1 = tmp_x *tmp_dy*tmp_dz *(dfftyz)

                tf_x = t_x
                tf_y = t_y
                tf_z = t_z

                spline_product =  tmp_x * tmp_y_z

                p_nabla_ux = di_xx*axx + di_yy*axy + di_zz*axz
                p_nabla_uy = di_xx*ayx + di_yy*ayy + di_zz*ayz
                p_nabla_uz = di_xx*azx + di_yy*azy + di_zz*azz
                dipole_term = t_x * p_nabla_ux + t_y * p_nabla_uy + t_z * p_nabla_uz

                ffxx = ci*(tf_x*axx+tf_y*axy+tf_z*axz) + &
                       p_nabla_ux*(t2_xx   *axx    + t2_x1y1z*axy  +  t2_x1yz1*axz )+&
                       p_nabla_uy*(t2_x1y1z*axx    + t2_yy   *axy  +  t2_xy1z1*axz )+&
                       p_nabla_uz*(t2_x1yz1*axx    + t2_xy1z1*axy  +  t2_xy1z1*axz )

                ffyy = ci*(tf_x*ayx+tf_y*ayy+tf_z*ayz) + &
                       p_nabla_ux*(t2_xx   *ayx    + t2_x1y1z*ayy  +  t2_x1yz1*ayz  )+&
                       p_nabla_uy*(t2_x1y1z*ayx    + t2_yy   *ayy  +  t2_xy1z1*ayz  )+&
                       p_nabla_uz*(t2_x1yz1*ayx    + t2_xy1z1*ayy  +  t2_xy1z1*ayz  )

                ffzz = ci*(tf_x*azx+tf_y*azy+tf_z*azz) + &
                       p_nabla_ux*(t2_xx   *azx    + t2_x1y1z*azy  +  t2_x1yz1*azz)+&
                       p_nabla_uy*(t2_x1y1z*azx    + t2_yy   *azy  +  t2_xy1z1*azz)+&
                       p_nabla_uz*(t2_x1yz1*azx    + t2_xy1z1*azy  +  t2_zz   *azz)
                fx(i) = fx(i) + ffxx*qsum
                fy(i) = fy(i) + ffyy*qsum
                fz(i) = fz(i) + ffzz*qsum

                Eni0 = - (qsum) * spline_product
                Eni =   ci*Eni0+dipole_term*(-qsum)
                a_pot(i) = a_pot(i) + Eni
                a_fi(i) = a_fi(i) + Eni0


buffer3(i,1) = buffer3(i,1) + (-qsum)*(tf_x*axx+tf_y*axy+tf_z*axz)
buffer3(i,2) = buffer3(i,2) + (-qsum)*(tf_x*ayx+tf_y*ayy+tf_z*ayz)
buffer3(i,3) = buffer3(i,3) + (-qsum)*(tf_x*azx+tf_y*azy+tf_z*azz)


            enddo
          enddo
        enddo

      enddo !i


    do i = 1, Natoms
     fxx(i) = fxx(i) + 2.0d0*fx(i)
     fyy(i) = fyy(i) + 2.0d0*fy(i)
     fzz(i) = fzz(i) + 2.0d0*fz(i)
    enddo

!print*, '---------------------\\\\\\\\\\ QP'
!print*, 'En_Q cmplx   =',sum(a_pot)!, dot_product(a_fi,all_charges)
!print*, 'sum forces =',sum(fxx),sum(fyy),sum(fzz)
!open(unit=666,file='fort.666',recl=1000)
!do i = 1, Natoms
!write(666,*) i,a_pot(i),2.0d0*fx(i),2.0d0*fy(i),2.0d0*fz(i)
!enddo
!print*, 'apot = ',a_pot(1:3)*2.0d0,a_pot(1003)*2.0d0
!print*,'fx 1 2 3 1003=',2.0d0*fx(1:3),2.0d0*fx(1003)
!print*,'fy 1 2 3 1003=',2.0d0*fy(1:3),2.0d0*fy(1003)
!print*,'fz 1 2 3 1003=',2.0d0*fz(1:3),2.0d0*fz(1003)
!print*, '---------------------\\\\\\\\\\\'

   ! The forces need to be recentered

   if (l_need_2nd_profile) then
    atom_profile%pot = atom_profile%pot + a_pot*2.0d0
    atom_profile%Qpot = atom_profile%Qpot + a_pot*2.0d0
    atom_profile%fi = atom_profile%fi + a_fi*2.0d0
!do i=1,Natoms;  atom_profile(i)%buffer3(:) = atom_profile(i)%buffer3(:)+2.0d0*buffer3(i,:); enddo
! stresses as well
   endif

   deallocate(fx,fy,fz)
   deallocate(a_pot,a_fi)
deallocate(buffer3)
   end subroutine smpe_eval2_Q_DIP_3D

 
  
end module smpe_dipole_module
