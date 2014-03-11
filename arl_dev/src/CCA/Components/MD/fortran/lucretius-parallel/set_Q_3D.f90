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

