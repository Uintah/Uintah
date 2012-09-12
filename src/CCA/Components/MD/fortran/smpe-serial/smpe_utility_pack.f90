

  module  smpe_utility_pack_0
!basic stuff added here (get reduced coordinates ; set array vectors ; get splines
  implicit none
  public :: get_reduced_coordinates
  public :: get_ALL_spline2_coef_REAL
  public :: get_pp_spline2_coef_REAL
  public :: get_CMPLX_splines  
  public :: smpe_eval1_Q_3D   ! compute energy in Fourier space
  public :: smpe_eval2_Q_3D   ! compute fores on each atom in real space
  contains
 
  subroutine get_reduced_coordinates
  use variables_smpe, only : nfftx,nffty,nfftz,tx,ty,tz
  use Ewald_data
  use sim_cel_data
  use sizes_data, only : Natoms
  use all_atoms_data, only : xx,yy,zz,xxx,yyy,zzz
  implicit none
  real(8) db(3), inv_box(3), bbox(3)
  integer k,i

   db(1) = dble(nfftx); db(2) = dble(nffty) ; db(3) = dble ( nfftz )
   bbox(1) = sim_cel(1) ; bbox(2) = sim_cel(5) ; bbox(3) = sim_cel(9)
   inv_box = 1.0d0 / bbox
! To be changed latter
  allocate(tx(Natoms), ty(Natoms),tz(Natoms))
  do i = 1, Natoms  
   tx(i) = xxx(i)  - (int(2.0d0*(xxx(i)*inv_box(1))) - int(xxx(i)*inv_box(1)) ) * bbox(1) 
   tx(i) = db(1) * ( inv_box(1) * tx(i) + 0.5d0) 
   ty(i) = yyy(i) - (int(2.0d0*(yyy(i)*inv_box(2))) - int(yyy(i)*inv_box(2)) ) * bbox(2)
   ty(i) = db(2) * ( inv_box(2) * ty(i) + 0.5d0)
   tz(i) = zzz(i) - (int(2.0d0*(zzz(i)*inv_box(3))) - int(zzz(i)*inv_box(3)) ) * bbox(3)
   tz(i) = db(3) * ( inv_box(3) * tz(i) + 0.5d0)
   enddo

  end subroutine get_reduced_coordinates

   subroutine get_ALL_spline2_coef_REAL ! does it with respect to zzz as real coordinate rather than ....
! These are for real space (for Q and forces)
! zzz coordinate is treated differently and interpolated as real coordinate rather than reduced one
! that is one key point in doing FFT integral: treat zzz different
   use variables_smpe
   use Ewald_data
   use sim_cel_data
   use sizes_data, only : Natoms
   use ALL_atoms_data, only : zz,zzz
   use spline2

   implicit none
   real(8) x,y,z
   integer ox,oy,oz
   integer i
      ox = order_spline_xx ; oy = order_spline_yy ; oz = order_spline_zz
      if (i_boundary_CTRL == 1) then
      do i = 1, Natoms
        x = tx(i)-int(tx(i)) ! tt are the reduced coordinates
        y = ty(i)-int(ty(i))
        z = zzz(i)-int(zzz(i))   !beta_spline_REAL_coef(order,x,nfft,spline,spline_DERIV)
        call real_spline2_and_deriv(ox,x,spline2_REAL_pp_x(i,1:ox),spline2_REAL_dd_x(i,1:ox))
        call real_spline2_and_deriv(oy,y,spline2_REAL_pp_y(i,1:oy),spline2_REAL_dd_y(i,1:oy))
        call real_spline2_and_deriv(oz,z,spline2_REAL_pp_z(i,1:oz),spline2_REAL_dd_z(i,1:oz))
      enddo
      else
      do i = 1, Natoms
        x = tx(i)-int(tx(i)) ! tt are the reduced coordinates
        y = ty(i)-int(ty(i))
        z = tz(i)-int(tz(i))   !beta_spline_REAL_coef(order,x,nfft,spline,spline_DERIV)
        call real_spline2_and_deriv(ox,x,spline2_REAL_pp_x(i,1:ox),spline2_REAL_dd_x(i,1:ox))
        call real_spline2_and_deriv(oy,y,spline2_REAL_pp_y(i,1:oy),spline2_REAL_dd_y(i,1:oy))
        call real_spline2_and_deriv(oz,z,spline2_REAL_pp_z(i,1:oz),spline2_REAL_dd_z(i,1:oz))
      enddo
      endif
   end subroutine get_ALL_spline2_coef_REAL

   subroutine get_pp_spline2_coef_REAL ! does it with respect to zzz as real coordinate rather than ....
! These are for real space (for Q and forces)
! zzz coordinate is treated differently and interpolated as real coordinate rather than reduced one
! that is one key point in doing FFT integral: treat zzz different
   use variables_smpe
   use Ewald_data
   use sim_cel_data
   use sizes_data, only : Natoms
   use ALL_atoms_data, only : zz,zzz
   use spline2
   implicit none
   real(8) x,y,z
   integer ox,oy,oz
   integer i
      ox = order_spline_xx ; oy = order_spline_yy ; oz = order_spline_zz
      if (i_boundary_CTRL == 1) then
      do i = 1, Natoms
        x = tx(i)-int(tx(i)) ! tt are the reduced coordinates
        y = ty(i)-int(ty(i))
        z = zzz(i)-int(zzz(i))   !beta_spline_REAL_coef(order,x,nfft,spline,spline_DERIV)
        call real_spline2_pp(ox,x,spline2_REAL_pp_x(i,1:ox))
        call real_spline2_pp(oy,y,spline2_REAL_pp_y(i,1:oy))
        call real_spline2_pp(oz,z,spline2_REAL_pp_z(i,1:oz))
      enddo
      else
       do i = 1, Natoms
        x = tx(i)-int(tx(i)) ! tt are the reduced coordinates
        y = ty(i)-int(ty(i))
        z = tz(i)-int(tz(i))   !beta_spline_REAL_coef(order,x,nfft,spline,spline_DERIV)
        call real_spline2_pp(ox,x,spline2_REAL_pp_x(i,1:ox))
        call real_spline2_pp(oy,y,spline2_REAL_pp_y(i,1:oy))
        call real_spline2_pp(oz,z,spline2_REAL_pp_z(i,1:oz))
      enddo
      endif
   end subroutine get_pp_spline2_coef_REAL


   subroutine get_CMPLX_splines ! they are evaluated once in the beggining of simulations.
   use spline2, only : complex_spline2_xy, complex_spline2_z
   use variables_smpe
    implicit none
      call complex_spline2_xy(order_spline_xx,nfftx,spline2_CMPLX_xx(1:nfftx))
      call complex_spline2_xy(order_spline_yy,nffty,spline2_CMPLX_yy(1:nffty))
      call complex_spline2_xy(order_spline_zz,nfftz,spline2_CMPLX_zz(1:nfftz))
!      call complex_spline2_z (order_spline_zz,h_cut_z,nfftz,spline2_CMPLX_zz(1:nfftz)) 
! using complex_spline2_xy and complex_spline2_z gives the same splines, but in different order
! (shifted by Pi/2) . The splines as comes from complex_spline2_z are not 
! in a wrap-arround sequence whiile the splines from complex_spline2_xy are in 
! the right order for fft. 
   end  subroutine get_CMPLX_splines

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


   subroutine smpe_eval1_Q_3D
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


! array qqq1 contains data for energy
! array qqq2 contains data for stresses xz,yz,zz


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
   stress_Qcmplx(8) =  lsyx  ! sxy = syx if no dipols!!!!
   stress_Qcmplx(9) =  lszx  ! change it when dipols are in
   stress_Qcmplx(10) = lszy  ! change it when dipols are in

   stress(:) = stress(:) + stress_Qcmplx(:)

   end subroutine smpe_eval1_Q_3D

   subroutine smpe_eval2_Q_3D ! forces

     use math, only : invert3
     use sizes_data, only : Natoms
     use sim_cel_data
     use ALL_atoms_data, only : all_p_charges,all_g_charges, fxx, fyy, fzz, xx, yy, zz,&
                                is_charge_distributed, all_charges,xxx,yyy,zzz
     use variables_smpe
     use Ewald_data
     implicit none

     real(8) temp,icel(9)
     real(8), allocatable :: fx(:),fy(:),fz(:)
     real(8), allocatable :: a_pot(:),a_fi(:)
     integer i,j,k,ix,iy,iz,kx,ky,kz,nx,ny,nz,mx,my,mz, i_index
     integer ii_xx, ii_yy , ii_zz
     real(8) vterm, En, ci, exp_fct, Eni, ff, Eni0
     real(8) sxx,sxy,sxz,syx,syy,syz,szx,szy,szz
     real(8) tmpz, tmp_y_z, tmp_y_dz, tmp_dy_z, tmp, tmpdz
     real(8) qsum, spline_product
     real(8) z
     real(8) spl_xx(order_spline_xx), spl_yy(order_spline_yy),spl_zz(order_spline_zz)
     real(8) spl_xx_DRV(order_spline_xx), spl_yy_DRV(order_spline_yy),spl_zz_DRV(order_spline_zz)
     real(8) t, t_x, t_y, t_z
     real(8) i_cel_1,i_cel_2,i_cel_3,i_cel_4,i_cel_5,i_cel_6,i_cel_7,i_cel_8,i_cel_9
     real(8) pref
     real(8), save :: eta
     logical l_i

! qqq1 is charge array (actually field array)
! qqq2 is stress array

     i_cel_1 = Inverse_cel(1) ; i_cel_2 = Inverse_cel(2); i_cel_3=Inverse_cel(3)
     i_cel_4 = Inverse_cel(4) ; i_cel_5 = Inverse_cel(5); i_cel_6=Inverse_cel(6)
     i_cel_7 = Inverse_cel(7) ; i_cel_8 = Inverse_cel(8); i_cel_9=Inverse_cel(9)

     allocate(fx(Natoms),fy(Natoms),fz(Natoms) )
     allocate(a_pot(Natoms),a_fi(Natoms))
!allocate(a_pot_GG(Natoms),a_pot_PG(Natoms),a_pot_GP(Natoms))
     fx = 0.0d0 ; fy = 0.0d0 ; fz = 0.0d0 ;
     a_pot = 0.0d0; a_fi=0.0d0
!allocate(a_fi1(Natoms)); a_fi1=0.0d0
!a_pot_GG=0.0d0;a_pot_PG=0.0d0;a_pot_GP=0.0d0
     do i = 1, Natoms
     ci = all_charges(i)
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

        t = 0.0d0
        ii_zz = nz
        do iz=1,order_spline_zz
          tmpz =   spl_zz(iz)
          tmpdz =  spl_zz_DRV(iz)
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
            tmp_y_z  = spl_yy(iy)     * tmpz;
            tmp_dy_z = spl_yy_DRV(iy) * tmpz;
            tmp_y_dz = spl_yy(iy)     * tmpdz;
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
                i_index = ((my-1)+(mz-1)*nffty)*nfftx + mx
                qsum = -real(qqq1(i_index), kind=8)
                t_x=(ci*qsum) * spl_xx_DRV(ix)* tmp_y_z   * dfftx
                t_y=(ci*qsum) * spl_xx(ix)    * tmp_dy_z  * dffty
                t_z=(ci*qsum) * spl_xx(ix)    * tmp_y_dz  * dfftz
                spline_product =  spl_xx(ix) * tmp_y_z
                fx(i) = fx(i) + t_x*i_cel_1+t_y*i_cel_2+t_z*i_cel_3
                fy(i) = fy(i) + t_x*i_cel_4+t_y*i_cel_5+t_z*i_cel_6
                fz(i) = fz(i) + t_x*i_cel_7+t_y*i_cel_8+t_z*i_cel_9
                Eni0 = - (qsum) * spline_product
                Eni =   ci * Eni0
                a_pot(i) = a_pot(i) + Eni
                a_fi(i) = a_fi(i) + Eni0
!if (l_i) then
!a_fi1(i) = a_fi1(i) + real(qqq1(i_index), kind=8)
!endif
!                a_pot_GG(i) = a_pot_GG(i) + Eni
!                a_pot_GP(i) = a_pot_GP(i) - ci * (-qqq4_Re(i_index)) * spline_product
!                a_pot(i) = a_pot(i) + Eni
!                a_pot_PG(i) = a_pot_PG(i) - ci* (-qqq3_Re(i_index)) * spline_product

!              vct6(1:6) = stress_MAT(ii_xx,ii_yy,ii_zz,1:6) * spline_product * electricity(i)
!              p_press_11(i) =  p_press_11(i) + real(vct6(1),kind=8)
!              p_press_22(i) =  p_press_22(i) + real(vct6(2),kind=8)
!              p_press_33(i) =  p_press_33(i) + real(vct6(3),kind=8)
!              p_press_12(i) =  p_press_12(i) + real(vct6(4),kind=8)
!              p_press_13(i) =  p_press_13(i) + real(vct6(5),kind=8)
!              p_press_23(i) =  p_press_23(i) + real(vct6(6),kind=8)
            enddo
          enddo
        enddo

      enddo !i

!do i = 1, Natoms
! write(66,*) i,a_fi(i),a_fi1(i)
!enddo
!STOP



!print*, ' zero MOM ? ',sum(fx*2.0d0),sum(fy*2.0d0),sum(fz*2.0d0)

    do i = 1, Natoms
     fxx(i) = fxx(i) + 2.0d0*fx(i)
     fyy(i) = fyy(i) + 2.0d0*fy(i)
     fzz(i) = fzz(i) + 2.0d0*fz(i)
    enddo

!print*, '---------------------\\\\\\\\\\ QP'
!print*, 'En_Q cmplx PP  GG  =',sum(a_pot), dot_product(a_fi,all_charges)
!print*, 'sum forces =',sum(fxx),sum(fyy),sum(fzz)
!print*, 'apot = ',a_pot(1:3)*2.0d0,a_pot(1003)*2.0d0
!print*,'fx 1 2 3 1003=',2.0d0*fx(1:3),2.0d0*fx(1003)
!print*,'fy 1 2 3 1003=',2.0d0*fy(1:3),2.0d0*fy(1003)
!print*,'fz 1 2 3 1003=',2.0d0*fz(1:3),2.0d0*fz(1003)
!print*, '---------------------\\\\\\\\\\\'
!stop

   ! The forces need to be recentered


   deallocate(fx,fy,fz)
   deallocate(a_pot,a_fi)
   end subroutine smpe_eval2_Q_3D


  subroutine set_Q_3D
! WIll set both potential and stress charges
    use variables_smpe
    use Ewald_data
    use sim_cel_data
    use sizes_data, only : Natoms
    use all_atoms_data , only : all_charges,l_WALL, zz,is_charge_distributed

    implicit none
    real(8) cox(order_spline_xx),coy(order_spline_yy),coz(order_spline_zz)
    integer i,j,k,nx,ny,nz,ix,iy,iz,jx,jy,jz,kx,ky,kz
    integer ii_xx,ii_yy,ii_zz,i_adress,ii,jj,kk,i1
    real(8) ci,z,tmpqz,tmpqyz,db(3)
    real(8) inv_box(3),box(3), pref
    logical, save :: l_very_first_pass = .true.
    real(8), save :: eta
    logical l_i
    integer kx1,ky1,kz1

    real(8) rec_xx,rec_xy,rec_yx,rec_yy,rec_zz,d2,kkx,kky,kkz,tmp

    box(1) = sim_cel(1); box(2) = sim_cel(5); box(3) = sim_cel(9)
    inv_box = 1.0d0/box
    qqq1_Re = 0.0d0

    do i = 1,Natoms
      ci = all_charges(i)
      cox(1:order_spline_xx) = spline2_REAL_pp_x(i,1:order_spline_xx)
      coy(1:order_spline_yy) = spline2_REAL_pp_y(i,1:order_spline_yy)
      coz(1:order_spline_zz) = spline2_REAL_pp_z(i,1:order_spline_zz)

      nx = int(tx(i)) - order_spline_xx
      ny = int(ty(i)) - order_spline_yy
      nz = int(tz(i)) - order_spline_zz
!print*, i,' t=',tx(i),ty(i),tz(i)
!read(*,*)
      iz = nz
      do jz = 0, order_spline_zz-1
      iz = iz + 1
      if (iz < 0) then
         kz = iz + nfftz
      else
         kz = iz
      endif
!ii_zz=int(tz(i))-jz+2 -1
!if(ii_zz.gt.nfftz) ii_zz=ii_zz-nfftz  ;  if(ii_zz.lt.1)ii_zz=ii_zz+nfftz
!kz = ii_zz-1
      tmpqz = coz(jz+1)*ci
      iy = ny
      do jy = 0, order_spline_yy-1
        iy = iy + 1
        if (iy < 0) then
          ky = iy + nffty
        else
          ky = iy
        endif
!ii_yy=int(ty(i))-jy+2 -1
!if(ii_yy.gt.nffty) ii_yy=ii_yy-nffty   ;  if(ii_yy.lt.1)ii_yy=ii_yy+nffty
!ky = ii_yy-1
        tmpqyz = tmpqz * coy(jy+1)
        ix = nx
        do jx = 0, order_spline_xx-1
          ix = ix + 1
          if (ix < 0) then
             kx = ix + nfftx
          else
             kx = ix
          endif
!ii_xx=int(tx(i))-jx+2 -1
!if(ii_xx.gt.nfftx)ii_xx=ii_xx-nfftx  ;  if(ii_xx.lt.1)ii_xx=ii_xx+nfftx
!kx = ii_xx-1
!print*,kx,ky,kz, ' ii  ',ii_xx,ii_yy,ii_zz
          i_adress = (ky+kz*nffty)*nfftx + kx + 1
          qqq1_Re(i_adress) = qqq1_Re(i_adress) + tmpqyz*cox(jx+1) ! potential POINT CHARGE
        enddo ! jx
        enddo ! jy
      enddo ! jz
    enddo ! i

    qqq1 = cmplx(qqq1_Re,0.0d0,kind=8)

    
!    qqq2 = cmplx(qqq2_Re,0.0d0,kind=8)
  end subroutine set_Q_3D


  end module smpe_utility_pack_0

