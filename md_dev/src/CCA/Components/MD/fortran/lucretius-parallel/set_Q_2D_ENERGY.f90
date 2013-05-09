  subroutine set_Q_2D_ENERGY
! WIll set both potential and stress charges
    use variables_smpe
    use Ewald_data
    use sim_cel_data
    use sizes_data, only : Natoms
    use all_atoms_data , only : all_charges,l_WALL, zz,zzz,is_charge_distributed

    implicit none
    real(8) cox(order_spline_xx),coy(order_spline_yy),coz(order_spline_zz)
    integer i,j,k,nx,ny,nz,ix,iy,iz,jx,jy,jz,kx,ky,kz
    integer ii_xx,ii_yy,ii_zz,i_adress,ii,jj,kk,i1
    integer kx1,ky1,kz1
    real(8) ci,z,tmpqz,tmpqyz,db(3)
    real(8) inv_box(3),box(3), pref
    logical, save :: l_very_first_pass = .true.
    real(8), save :: eta
    logical l_i

    real(8) rec_xx,rec_xy,rec_yx,rec_yy,rec_zz,d2,kkx,kky,kkz,tmp

    box(1) = sim_cel(1); box(2) = sim_cel(5); box(3) = sim_cel(9)
    inv_box = 1.0d0/box
    qqq1_Re = 0.0d0
    qqq2_Re = 0.0d0
    do i = 1,Natoms
      ci = all_charges(i)
      cox(1:order_spline_xx) = spline2_REAL_pp_x(i,1:order_spline_xx)
      coy(1:order_spline_yy) = spline2_REAL_pp_y(i,1:order_spline_yy)
      coz(1:order_spline_zz) = spline2_REAL_pp_z(i,1:order_spline_zz)

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
      iy = ny
      do jy = 0, order_spline_yy-1
        iy = iy + 1
        if (iy < 0) then
          ky = iy + nffty
        else
          ky = iy
        endif
        tmpqyz = tmpqz * coy(jy+1)
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
          qqq1_Re(i_adress) = qqq1_Re(i_adress) + tmpqyz*cox(jx+1) ! potential POINT CHARGE
        enddo ! jx
        enddo ! jy
      enddo ! jz
    enddo ! i

    qqq1 = cmplx(qqq1_Re,0.0d0,kind=8)
  end subroutine set_Q_2D_ENERGY

