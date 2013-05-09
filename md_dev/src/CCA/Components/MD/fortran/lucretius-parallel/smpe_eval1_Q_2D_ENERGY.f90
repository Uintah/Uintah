   subroutine smpe_eval1_Q_2D_ENERGY
! Eval stresses and energy for Q_POINT charges only
     use variables_smpe
     use Ewald_data
     use sim_cel_data
     use boundaries, only : cel_properties, get_reciprocal_cut
     use sizes_data, only : Natoms
     use energies_data
     use cut_off_data
     use stresses_data, only : stress, stress_Qcmplx,stress_Qcmplx_as_in_3D

     implicit none
     real(8), parameter :: four_pi_sq = 39.4784176043574d0
     real(8), parameter :: Pi2 = 6.28318530717959d0
     integer i,j,k,ix,iy,iz,jx,jy,jz,nx,ny,nz,nx0,ny0,nz0
     integer m_iy1,m_iy2,m_ix1,m_ix2,m_iz1,m_iz2,m_ix,m_iy,m_iz
     integer ii_xx,ii_yy,ii_zz
     integer jy1,jx1,jz1,mz,mz1,i_index
     real(8) expfct,i4a2,i4a2_2, tmp, vterm,vterm2
     real(8) spline_product, En, Eni, ff, vir,vir0
     real(8) temp_vir_cz, t
     real(8) qim, qre, exp_fct
     real(8) rec_xx,rec_yy,rec_zz,rec_xy,rec_xz,rec_yx,rec_zx,rec_zykx
     real(8) d2,i_d2
     real(8) kx , ky, kz
     real(8) local_potential
     real(8) area, q_az_re, q_az_im

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
              vterm2 = vterm*2.0d0
              En = vterm*(qre*qre+qim*qim)
              local_potential = local_potential + En
 endif     !  reciprocal_cutt within cut off        
        enddo
     enddo
    enddo


   En_Q_cmplx = En_Q_cmplx + local_potential
   En_Q = En_Q + local_potential


 end subroutine smpe_eval1_Q_2D_ENERGY ! energy and stresses


