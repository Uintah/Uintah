   subroutine smpe_eval2_Q_2D ! forces

     use math, only : invert3
     use sizes_data, only : Natoms
     use sim_cel_data
     use ALL_atoms_data, only : all_p_charges,all_g_charges, fxx, fyy, fzz, xx, yy, zz,&
                                zzz,is_charge_distributed, all_charges
     use profiles_data, only : atom_profile, l_need_2nd_profile
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
     real(8) i_sim_zz
     real(8) qsum, spline_product
     real(8) z
     real(8) spl_xx(order_spline_xx), spl_yy(order_spline_yy),spl_zz(order_spline_zz)
     real(8) spl_xx_DRV(order_spline_xx), spl_yy_DRV(order_spline_yy),spl_zz_DRV(order_spline_zz)
     real(8) t, t_x, t_y, t_z
     real(8) i_cel_1,i_cel_2,i_cel_4,i_cel_5
     real(8) pref
     real(8), save :: eta
     logical l_i

! qqq1 is charge array (actually field array)
! qqq2 is stress array

     i_sim_zz = Inverse_cel(9)
     i_cel_1 = Inverse_cel(1) ; i_cel_2 = Inverse_cel(2)
     i_cel_4 = Inverse_cel(4) ; i_cel_5 = Inverse_cel(5)

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
       nz = int(z) - order_spline_zz
       spl_xx(1:order_spline_xx) = spline2_REAL_pp_x(i,1:order_spline_xx)
       spl_yy(1:order_spline_yy) = spline2_REAL_pp_y(i,1:order_spline_yy)
       spl_zz(1:order_spline_zz) = spline2_REAL_pp_z(i,1:order_spline_zz)
       spl_xx_DRV(1:order_spline_xx) = spline2_REAL_dd_x(i,1:order_spline_xx)
       spl_yy_DRV(1:order_spline_yy) = spline2_REAL_dd_y(i,1:order_spline_yy)
       spl_zz_DRV(1:order_spline_zz) = spline2_REAL_dd_z(i,1:order_spline_zz)

        t = 0.0d0
        ii_zz = nz
        do iz=1,order_spline_zz
          ii_zz = ii_zz + 1
          kz = ii_zz + nfftz/2 + 1
          tmpz =   spl_zz(iz)
          tmpdz =  spl_zz_DRV(iz)
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
              if (mx**2+my**2 /= 0 ) then  ! skip (0,0)
                i_index = ((my-1)+(mz-1)*nffty)*nfftx + mx
                qsum = -real(qqq1(i_index), kind=8)
                t_x=(ci*qsum) * spl_xx_DRV(ix)* tmp_y_z   * dfftx
                t_y=(ci*qsum) * spl_xx(ix)    * tmp_dy_z  * dffty
                t_z=(ci*qsum) * spl_xx(ix)    * tmp_y_dz  !* dble(nfftz)
                spline_product =  spl_xx(ix) * tmp_y_z
                fx(i) = fx(i) + t_x*i_cel_1+t_y*i_cel_2
                fy(i) = fy(i) + t_x*i_cel_4+t_y*i_cel_5
                fz(i) = fz(i) + t_z
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
            endif
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
!print*, 'sum a_pot  =',sum(a_pot)
!print*, 'pot GP PG GP+PG = ',sum(a_pot_GP),sum(a_pot_PG),sum(a_pot_GP+a_pot_PG)
!print*, 'apot = ',a_pot(1:3)*2.0d0,a_pot(1003)*2.0d0
!print*,'fx 1 2 3 1003=',2.0d0*fx(1:3),2.0d0*fx(1003)
!print*,'fy 1 2 3 1003=',2.0d0*fy(1:3),2.0d0*fy(1003)
!print*,'fz 1 2 3 1003=',2.0d0*fz(1:3),2.0d0*fz(1003)
!print*, '---------------------\\\\\\\\\\\'
!stop

   ! The forces need to be recentered

   if (l_need_2nd_profile) then
    atom_profile%pot = atom_profile%pot + a_pot*2.0d0
    atom_profile%Qpot = atom_profile%Qpot + a_pot*2.0d0
    atom_profile%fi = atom_profile%fi + a_fi*2.0d0
! stresses as well
   endif

   deallocate(fx,fy,fz)
   deallocate(a_pot,a_fi)
   end subroutine smpe_eval2_Q_2D


