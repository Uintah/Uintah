module  slow_fourier_ewald_3D
contains
 subroutine Ew_Q_SLOW

   use ALL_atoms_data, only : Natoms, all_charges,xx,yy,zz,fxx,fyy,fzz,&
                              i_type_atom,xxx,yyy,zzz, i_type_atom, &
                              all_dipoles_xx,all_dipoles_yy,all_dipoles_zz
   use Ewald_data
   use sim_cel_data
   use energies_data
   use stresses_data
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
    real(8) , allocatable :: fi(:), a_pot(:),fip(:)
    real(8) , allocatable :: local_force_xx(:),local_force_yy(:),local_force_zz(:)
    real(8) , allocatable :: sini(:),cosi(:),sin_KR(:),cos_KR(:),sin_K_P(:),cos_K_P(:)
    real(8) en_factor
    logical l_1
    real(8) lsxx,lsyy,lszz,lsxy,lsxz,lsyx,lsyz,lszx,lszy, vir,vir0,fct
    real(8) sxx,sxy,sxz,syx,syy,syz,szx,szy,szz,lsxz_prime,lsyz_prime,lszz_prime
    real(8) SK_Im_z,SK_Re_z,tem_vir_az,vir_az
real(8), allocatable :: buffer(:,:) ! DELETE IT
allocate(buffer(Natoms,3)) ; buffer=0.0d0! DELETE IT


    allocate(fi(Natoms),fip(Natoms),a_pot(Natoms),local_force_xx(Natoms),local_force_yy(Natoms),local_force_zz(Natoms))
    allocate(cosi(Natoms),sini(Natoms),sin_KR(Natoms),cos_KR(Natoms),sin_K_P(Natoms),cos_K_P(Natoms))

    a_pot = 0.0d0 ; fi = 0.0d0; fip = 0.0d0
    local_force_xx = 0.0d0 ; local_force_yy = 0.0d0 ; local_force_zz = 0.0d0
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

    lsxx = 0.0d0; lsxy = 0.0d0; lsxz = 0.0d0
    lsyy = 0.0d0; lsyz = 0.0d0
    lszz = 0.0d0
    lsxz_prime=0.0d0; lsyz_prime=0.0d0 ; lszz_prime=0.0d0

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
          if ( i_boundary_CTRL == 1) then
           SK_RE_z=0.0d0;SK_Im_z=0.0d0
           do i = 1, Natoms 
            SK_RE_z = SK_RE_z + zzz(i)*(cos_KR(i) - cos_K_P(i))
            SK_Im_z = SK_Im_z + zzz(i)*(sin_KR(i) + cos_K_P(i))
           enddo
          endif
          SK_RE =  SK_Re_P
          SK_IM =  SK_Im_P
          En = exp_fct*(SK_Re**2+SK_Im**2)
          local_energy = local_energy + En

          vir0 = 2.0d0*(i_d2 + expfct)
          vir = vir0 * En
          sxx = En - vir*kx*kx 
          sxy = -vir*kx*ky  ;  
          syy = En - vir*ky*ky ;
    if ( i_boundary_CTRL /= 1) then   ! 3D geometry
          sxz = -vir*kx*kz
          syz = -vir*ky*kz
          szz = En - vir*kz*kz
    else  ! 2D (slab) geometry
          vir_az = exp_fct*2.0d0
          tem_vir_az = vir_az * (SK_RE*SK_Im_z-SK_RE_z*SK_IM)
          sxz = tem_vir_az * kx
          syz = tem_vir_az * ky          
          szz = tem_vir_az * kz
          lsxz_prime = lsxz_prime  -vir*kx*kz
          lsyz_prime = lsyz_prime  -vir*ky*kz
          lszz_prime = lszz_prime +En - vir*kz*kz
    endif
 
          lsxx = lsxx + sxx ; lsxy = lsxy + sxy ; lsxz = lsxz + sxz
          lsyy = lsyy + syy ; lsyz = lsyz + syz
          lszz = lszz + szz

          do i = 1, Natoms  ! forces + potential profile
               qi = all_charges(i)
               Eni = exp_fct*((cos_KR(i)-sin_K_P(i))*SK_Re + (sin_KR(i)+cos_K_P(i))*SK_Im)
               a_pot(i) = a_pot(i) + Eni*2.0d0
               ff = -exp_fct*((-sin_KR(i)-cos_K_P(i))*SK_Re + (cos_KR(i)-sin_K_P(i))*SK_Im)
               fi(i) = fi(i) + 2.0d0*exp_fct*( (cosi(i)*SK_Re + sini(i)*SK_Im) )
fct = 2.0d0*exp_fct*(cosi(i)*SK_Im-sini(i)*SK_Re)
buffer(i,1) = buffer(i,1) + kx*fct
buffer(i,2) = buffer(i,2) + ky*fct
buffer(i,3) = buffer(i,3) + kz*fct

!               fip(i) = fip(i) + 2.0d0*exp_fct*( ((-sin_K_P(i))*SK_Re + cos_K_P(i)*SK_Im) ) ! I am not interested in it
               local_force_xx(i) =  local_force_xx(i) + ff*kx
               local_force_yy(i) =  local_force_yy(i) + ff*ky
               local_force_zz(i) =  local_force_zz(i) + ff*kz
          enddo  !  i = 1, Natoms`wq
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

  a_pot = a_pot * en_factor
  fi = fi * en_factor
!  fip = fip * en_factor


  buffer=buffer * en_factor
  local_energy=local_energy*en_factor
  local_force_xx=local_force_xx*(2.0d0*en_factor)
  local_force_yy=local_force_yy*(2.0d0*en_factor)
  local_force_zz=local_force_zz*(2.0d0*en_factor)
  lsxx = lsxx * en_factor ; lsxy = lsxy * en_factor ; lsxz = lsxz * en_factor
  lsyx = lsyx * en_factor ; lsyy = lsyy * en_factor ; lsyz = lsyz * en_factor
  lszx = lszx * en_factor ; lszy = lszy * en_factor ; lszz = lszz * en_factor
  lszz_prime=lszz_prime*en_factor
  lsyz_prime=lsyz_prime*en_factor
  lsxz_prime=lsxz_prime*en_factor 

  En_Q = En_Q + local_energy
  En_Q_cmplx = En_Q_cmplx + local_energy

  fxx = fxx + local_force_xx
  fyy = fyy + local_force_yy
  fzz = fzz + local_force_zz

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
   if (i_boundary_CTRL == 1) then
   stress_Qcmplx_as_in_3D(1:2)=0.0d0
   stress_Qcmplx_as_in_3D(3) = lszz_prime 
   stress_Qcmplx_as_in_3D(4) = lszz_prime
   stress_Qcmplx_as_in_3D(5) = 0.0d0
   stress_Qcmplx_as_in_3D(6) = lsxz_prime
   stress_Qcmplx_as_in_3D(7) = lsyz_prime
   stress_Qcmplx_as_in_3D(8) = 0.0d0
   stress_Qcmplx_as_in_3D(9) = lsxz_prime
   stress_Qcmplx_as_in_3D(10) = lsyz_prime
   else
   stress_Qcmplx_as_in_3D=0.0d0
   endif

   stress(:) = stress(:) + stress_Qcmplx(:)
 




  deallocate(fi,fip,a_pot,local_force_xx,local_force_yy,local_force_zz)
  deallocate(cosi,sini,sin_KR,cos_KR,sin_K_P,cos_K_P)
  deallocate(buffer)
 end subroutine Ew_Q_SLOW
end module slow_fourier_ewald_3D
