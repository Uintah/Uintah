 subroutine Ew_Q_SLOW_ENERGY

   use atom_type_data, only : N_type_atoms, atom_type_1GAUSS_charge_distrib
   use ALL_atoms_data, only : Natoms, all_charges,xx,yy,zz,fxx,fyy,fzz,&
                              i_type_atom,xxx,yyy,zzz, i_type_atom, &
                              all_dipoles_xx,all_dipoles_yy,all_dipoles_zz
   use Ewald_data
   use sim_cel_data
   use energies_data
   use boundaries, only : get_reciprocal_cut
   use cut_off_data, only : reciprocal_cut, reciprocal_cut_sq
! does 2D ewald at K non zero in a slow fashion
! this subroutine is for test only and no attempt to optimize it is made.

    implicit none
    real(8) , parameter :: Pi2=6.28318530717959d0
    integer i,j,k,ix,iy,iz,k_vct
    real(8) tmp, d2, i_d2, h_step, h_cut_off
    real(8) expfct, exp_fct
    real(8) qi,qj,qij
    real(8) kx,ky,kz,kr,rec_xx,rec_xy,rec_yx,rec_yy,rec_zz,rec_zx,rec_zy,rec_xz,rec_yz,K_P
    real(8) En, Eni,ff, local_energy
    real(8) SK_Re_G,SK_Im_G,SK_Re_P,SK_Im_P, SK_RE,SK_IM
    real(8) , allocatable :: sini(:),cosi(:),sin_KR(:),cos_KR(:),sin_K_P(:),cos_K_P(:)
    real(8) en_factor
    logical l_1
    real(8) fct
    real(8) SK_Im_z,SK_Re_z,tem_vir_az,vir_az


    allocate(cosi(Natoms),sini(Natoms),sin_KR(Natoms),cos_KR(Natoms),sin_K_P(Natoms),cos_K_P(Natoms))

    expfct = 1.0d0/(4.0d0*Ewald_alpha**2)

    local_energy = 0.0d0
    k_vct = 0

    if (i_boundary_CTRL == 1) then  ! 2D-slab
!       Reciprocal_cel(3) = 0.0d0; Reciprocal_cel(6) = 0.0d0 ; 
        h_cut_off = h_cut_off2D *  Reciprocal_cel(9)
        h_step = h_cut_off2D/dble(K_MAX_Z)
        if (h_step > 0.5d0) then
          write(6,*) 'ERROR in Ew_Q_SLOW with the step along zz: it must be at least HALF from Pi/Lz', h_step,K_MAX_Z
          write(6,*) 'Increase K_MAX_Z'
          STOP
        endif
       h_step = h_cut_off/dble(K_MAX_Z)

    endif

     call get_reciprocal_cut

    do ix = -K_MAX_X,K_MAX_X
      tmp = dble(ix)
      rec_xx = tmp*Reciprocal_cel(1)
      rec_yx = tmp*Reciprocal_cel(4)
      rec_zx = tmp*Reciprocal_cel(7)
      do iy = -K_MAX_Y,K_MAX_Y
        tmp = dble(iy)
        rec_xy = tmp*Reciprocal_cel(2)
        rec_yy = tmp*Reciprocal_cel(5)
        rec_zy = tmp*Reciprocal_cel(8)
        l_1 = i_boundary_CTRL /= 1 .or. (i_boundary_CTRL==1.and.ix**2+iy**2 /=0) 
        if (l_1) then
        do iz = -K_MAX_Z,K_MAX_Z
          tmp = dble(iz)
          rec_xz = tmp*Reciprocal_cel(3)
          rec_yz = tmp*Reciprocal_cel(6)
        if (i_boundary_CTRL==1) then 
          rec_zz = tmp * h_step
        else
          rec_zz = tmp*Reciprocal_cel(9) 
        endif
          kx = rec_xx + rec_xy + rec_xz
          ky = rec_yx + rec_yy + rec_yz
          kz = rec_zx + rec_zy + rec_zz
          d2 = kx*kx + ky*ky + kz*kz
   if (d2 < reciprocal_cut_sq.and.ix**2+iy**2+iz**2 /= 0) then
         k_vct = k_vct + 1
          i_d2 = 1.0d0/d2
          exp_fct = dexp(-expfct*d2) * i_d2
          SK_Re_P=0.0d0; SK_Im_P=0.0d0
          do i = 1, Natoms
            kr = xxx(i)*kx + yyy(i)*ky + zzz(i)*kz
            cosi(i) = dcos(kr)
            sini(i) = dsin(kr)
            qi = all_charges(i)
            K_P = all_dipoles_xx(i)*kx + all_dipoles_yy(i)*ky+all_dipoles_zz(i)*kz ! 
            cos_KR(i) = qi*cosi(i)
            sin_KR(i) = qi*sini(i)
            cos_K_P(i) = K_P*cosi(i)  ;   sin_K_P(i) =  K_P*sini(i)
            SK_Re_P = SK_Re_P + cos_KR(i) - sin_K_P(i)
            SK_Im_P = SK_Im_P + sin_KR(i) + cos_K_P(i)
          enddo ! i = 1, Natoms
          SK_RE =  SK_Re_P
          SK_IM =  SK_Im_P
          En = exp_fct*(SK_Re**2+SK_Im**2)
          local_energy = local_energy + En

    endif !   (d2 < reciprocal_cut_sq)

        enddo   ! IZ
        endif ! periodicity CTRL
      enddo ! iy
    enddo ! ix

  if (i_boundary_CTRL == 1) then
    en_factor = h_step /  Area_xy
  else
    en_factor = Pi2 /  Volume
  endif

  En_Q = En_Q + local_energy
  En_Q_cmplx = En_Q_cmplx + local_energy

  deallocate(cosi,sini,sin_KR,cos_KR,sin_K_P,cos_K_P)
 end subroutine Ew_Q_SLOW_ENERGY

