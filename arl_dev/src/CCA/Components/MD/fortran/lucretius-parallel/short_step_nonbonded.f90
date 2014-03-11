

subroutine short_step_nonbonded(fxx,fyy,fzz)
 use sys_data
 use paralel_env_data
 use boundaries, only : periodic_images
 use ALL_atoms_data, only : xxx,yyy,zzz,Natoms,i_Style_atom, all_charges,i_type_atom
 use max_sizes_data, only : MX_list_nonbonded
 use non_bonded_lists_data, only : list_nonbonded_short, size_list_nonbonded_short, l_update_VERLET_LIST
 use connectivity_ALL_data, only : list_14,size_list_14, l_red_14_vdw_CTRL,l_red_14_Q_CTRL,&
                                   red_14_vdw,red_14_Q
 use atom_type_data, only : atom_Style2_vdwPrm,which_atomStyle_pair
 use interpolate_data, only : gvdw, gele_G_short,RDR , iRDR,MX_interpol_points
 use compute_14_module, only : compute_14_interactions_vdw_Q
 use sizes_data, only : N_pairs_14
 use cut_off_data
 use ewald_data, only : ewald_alpha
 use non_bonded_lists_builder, only : driver_link_nonbonded_pairs
 use water_Pt_ff_module, only : driver_water_surface_ff

 implicit none
 real(8), intent(INOUT) :: fxx(:),fyy(:),fzz(:)
 integer i,j,i1,imol,itype,N,neightot,k,nneigh, iStyle,jStyle,NDX,i_pair
 real(8), allocatable :: dx(:),dy(:),dz(:),dr_sq(:)
 integer, allocatable :: in_list(:)
 real(8) ff,fx,fy,fz,t1,t2,gk,gk1,a0,x,y,z,r,r2,Inverse_r,Inverse_r_squared
 real(8) ppp 
 real(8) qi,qj,qij,af_i_1_x,af_i_1_y,af_i_1_z,i_displacement,g,trunc_and_shift

 call  driver_link_nonbonded_pairs(l_update_VERLET_LIST) ! MANDATORY or else will fail

 irdr = 1.0d0/rdr
 N=max(maxval(size_list_nonbonded_short),maxval(size_list_14))

 allocate(dx(N),dy(N),dz(N),dr_sq(N))
 allocate(in_list(N))
 i_displacement=1.0d0/displacement
!fxx=0.0d0;fyy=0.0d0;fzz=0.0d0!delete it
 do i = 1+rank, Natoms , nprocs
   iStyle  = i_Style_atom(i)
   i1 = size_list_nonbonded_short(i)
   in_list(1:i1) = list_nonbonded_short(i,1:i1)
   do k =  1, i1
    j = in_list(k)
    dx(k) = xxx(i) - xxx(j)
    dy(k) = yyy(i) - yyy(j)
    dz(k) = zzz(i) - zzz(j)
   enddo
   call periodic_images(dx(1:i1),dy(1:i1),dz(1:i1))
   dr_sq(1:i1) = dx(1:i1)*dx(1:i1) + dy(1:i1)*dy(1:i1) + dz(1:i1)*dz(1:i1)
   neightot = i1
   qi = all_charges(i)
   af_i_1_x=0.0d0; af_i_1_y=0.0d0; af_i_1_z=0.0d0 
    do k =  1, neightot
      j = in_list(k)
      r2 = dr_sq(k)
      if ( r2 < cut_off_short_sq ) then
         jStyle  = i_Style_atom(j)
         i_pair = which_atomStyle_pair(istyle,jstyle) ! can replace it by a formula?
         r = dsqrt(r2)
         Inverse_r = 1.0d0/r ; Inverse_r_squared = Inverse_r*Inverse_r
         NDX = max(1,int(r*irdr))
         ppp = (r*irdr) - dble(ndx)
         x = dx(k)   ;  y = dy(k)    ; z = dz(k)
         a0 = atom_Style2_vdwPrm(0,i_pair)
         if (a0 > 0.0d0) then
                g = (r-(cut_off_short-displacement))*i_displacement
                trunc_and_shift = 1.0d0+(g*g*(2.0d0*g-3.0d0))
                gk  = gvdw(ndx,i_pair)  ;  gk1 = gvdw(ndx+1,i_pair)
                t1 = gk  + (gk1 - gk )*ppp
                t2 = gk1 + (gvdw(ndx+2,i_pair) - gk1)*(ppp - 1.0d0)
                if (r>(cut_off_short-displacement)) then
                  g = (r-(cut_off_short-displacement))*i_displacement
                  trunc_and_shift = 1.0d0+(g*g*(2.0d0*g-3.0d0))
                  ff = (t1 + (t2-t1)*ppp*0.5d0)*Inverse_r_squared * trunc_and_shift
                else
                  ff = (t1 + (t2-t1)*ppp*0.5d0)*Inverse_r_squared
                endif
                fx = ff*x    ;  fy = ff*y   ;  fz = ff*z
                af_i_1_x = af_i_1_x+fx ; af_i_1_y = af_i_1_y + fy ; af_i_1_z = af_i_1_z + fz
                fxx(j)   = fxx(j) - fx ; fyy(j)   = fyy(j) - fy   ; fzz(j)   = fzz(j) - fz
         endif
         qj = all_charges(j)
         qij = qi*qj
         gk  = gele_G_short(ndx,i_pair)  ;  gk1 = gele_G_short(ndx+1,i_pair)  ! gele_G_short has no thole
         t1 = gk  + (gk1 - gk )*ppp
         t2 = gk1 + (gele_G_short(ndx+2,i_pair) - gk1)*(ppp - 1.0d0)
         ff = qij * (t1 + (t2-t1)*ppp*0.5d0)
         fx = ff*x   ;  fy = ff*y   ;  fz = ff*z
         af_i_1_x = af_i_1_x+fx ; af_i_1_y = af_i_1_y + fy ; af_i_1_z = af_i_1_z + fz
         fxx(j)   = fxx(j) - fx ; fyy(j)   = fyy(j) - fy   ; fzz(j)   = fzz(j) - fz

       endif  ! whithin cut off
    enddo ! k
    fxx(i) = fxx(i) + af_i_1_x
    fyy(i) = fyy(i) + af_i_1_y
    fzz(i) = fzz(i) + af_i_1_z
  enddo  ! i

 if (N_pairs_14 > 0.and.l_red_14_Q_CTRL) then
  do i = 1, Natoms
   iStyle  = i_Style_atom(i)
   i1 = size_list_14(i)
   neightot = i1
   in_list(1:neightot) = list_14(i,1:neightot)
   do k =  1, i1
    j = in_list(k)
    dx(k) = xxx(i) - xxx(j)
    dy(k) = yyy(i) - yyy(j)
    dz(k) = zzz(i) - zzz(j)
   enddo
   if (neightot>0) then
     call periodic_images(dx(1:i1),dy(1:i1),dz(1:i1))
     dr_sq(1:i1) = dx(1:i1)*dx(1:i1) + dy(1:i1)*dy(1:i1) + dz(1:i1)*dz(1:i1)
   endif
   qi = all_charges(i)
   af_i_1_x=0.0d0 ; af_i_1_y = 0.0d0 ; af_i_1_z = 0.0d0 ; 
   do k = 1, neightot
      j = in_list(k)
      r2 = dr_sq(k)
      if ( r2 < cut_off_sq ) then ! this almost does not matter here
            jstyle = i_style_atom(j)
            i_pair = which_atomStyle_pair(istyle,jstyle) ! can replace it by a formula?
            r = dsqrt(r2)
            Inverse_r_squared = 1.0d0/r2
            NDX = max(1,int(r*irdr))
            ppp = (r*irdr) - dble(ndx)
            x = dx(k)   ; y = dy(k)    ; z = dz(k)
            a0 = atom_Style2_vdwPrm(0,i_pair)
            if (a0 > SYS_ZERO .and. l_red_14_vdw_CTRL ) then
                gk  = gvdw(ndx,i_pair)  ;  gk1 = gvdw(ndx+1,i_pair)
                t1 = gk  + (gk1 - gk )*ppp
                t2 = gk1 + (gvdw(ndx+2,i_pair) - gk1)*(ppp - 1.0d0)
                ff = - red_14_vdw * (t1 + (t2-t1)*ppp*0.5d0)*Inverse_r_squared
                fx = ff*x    ;  fy = ff*y   ;  fz = ff*z
                af_i_1_x = af_i_1_x+fx ; af_i_1_y = af_i_1_y + fy ; af_i_1_z = af_i_1_z + fz
                fxx(j)   = fxx(j) - fx ; fyy(j)   = fyy(j) - fy   ; fzz(j)   = fzz(j) - fz

            endif
            if (l_red_14_Q_CTRL) then
               qj = all_charges(j)
               qij = qi*qj
               gk  = gele_G_short(ndx,i_pair)  ;  gk1 = gele_G_short(ndx+1,i_pair)  ! gele_G
               t1 = gk  + (gk1 - gk )*ppp
               t2 = gk1 + (gele_G_short(ndx+2,i_pair) - gk1)*(ppp - 1.0d0)
               ff = - (red_14_Q * qij) * (t1 + (t2-t1)*ppp*0.5d0)
               fx = ff*x   ;  fy = ff*y   ;  fz = ff*z ! count 
               af_i_1_x = af_i_1_x+fx ; af_i_1_y = af_i_1_y + fy ; af_i_1_z = af_i_1_z + fz
               fxx(j)   = fxx(j) - fx ; fyy(j)   = fyy(j) - fy   ; fzz(j)   = fzz(j) - fz

            endif
       endif ! r2 < cut_off_sq ! which is actually not necesary
    enddo   ! do k = 1, neightot
    fxx(i) = fxx(i) + af_i_1_x
    fyy(i) = fyy(i) + af_i_1_y
    fzz(i) = fzz(i) + af_i_1_z
  enddo ! i
 endif ! N_pairs_14

!open(unit=14,file='fort.14',recl=300)
!write(14,*)Natoms
!do i = 1, Natoms
!write(14,'(I7,1X,3(F16.7,1X))') i,fxx(i)/418.4d0,fyy(i)/418.4d0,fzz(i)/418.4d0
!enddo
!close(14)
!STOP


 deallocate(dx,dy,dz,dr_sq)
 deallocate(in_list)
 call  driver_water_surface_ff
 end subroutine short_step_nonbonded

 
