
 subroutine pair_short_forces_vdw ! do only vdw
  use sys_data
  use paralel_env_data
  use math_constants
  use boundaries
  use ALL_atoms_data
  use atom_type_data
  use max_sizes_data, only : MX_list_nonbonded
  use sizes_data, only : N_pairs_14
  use non_bonded_lists_data, only : list_nonbonded, size_list_nonbonded
  use profiles_data
  use energies_data
  use stresses_data
  use atom_type_data
  use interpolate_data
  use variables_short_pairs
 use rdfs_data, only : rdfs
 use integrate_data, only : integration_step
 use rdfs_collect_module, only : rdfs_collect
 use compute_14_module, only : compute_14_interactions_vdw
 use connectivity_ALL_data, only : l_red_14_vdw_CTRL

  implicit none
  real(8), parameter :: en_factor = 0.5d0
  integer i,j,i1,imol,itype,N,neightot,k,nneigh, istyle
  real(8), allocatable :: local_force(:,:)

!  allocate(local_force(Natoms,3))
!  local_force(:,1) = fxx(:); local_force(:,2) = fyy(:); local_force(:,3) = fzz(:) 
  call initialize
  
  irdr = 1.0d0/rdr
  do i = 1+rank, Natoms , nprocs
    imol = atom_in_which_molecule(i)
    istyle  = i_style_atom(i)
    i1 = size_list_nonbonded(i)
!    qi = all_p_charges(i)
    do k =  1, i1
      j = list_nonbonded(i,k)
      dx(k) = xxx(i) - xxx(j)
      dy(k) = yyy(i) - yyy(j)
      dz(k) = zzz(i) - zzz(j)
    enddo
    call periodic_images(dx(1:i1),dy(1:i1),dz(1:i1))
    dr_sq(1:i1) = dx(1:i1)*dx(1:i1) + dy(1:i1)*dy(1:i1) + dz(1:i1)*dz(1:i1)
    neightot = i1
    call nonbonded_vdw_2_forces(i,istyle,neightot)


  if (rdfs%any_request) then
  if (mod(integration_step,rdfs%N_collect) == 0) then
  call rdfs_collect(i,neightot)   ! for radial distribution function if needed
  endif
  endif

  enddo  ! main cycle

  if (N_pairs_14>0.and.l_red_14_vdw_CTRL) call compute_14_interactions_vdw

  call finalize_scalars   ! here I do the stresses



  deallocate(dr_sq)
  deallocate(dx,dy,dz)
!  deallocate(local_force)
!print*, 'verify real energies =',sum(QQ_PP_a_pot)/2.0d0, En_Qreal
!print*, 'verify forces =',sum(fxx),sum(fyy),sum(fzz)
 call finalize ! deallocate cvt for profiles
 CONTAINS
 subroutine initialize
   integer N
   integer i,j,k

   en_vdw = 0.0d0

   N=maxval(size_list_nonbonded)

   allocate(dx(N),dy(N),dz(N),dr_sq(N))
   en_vdw = 0.0d0
   stress_vdw_xx = 0.0d0 ; stress_vdw_xy = 0.0d0 ; stress_vdw_xz = 0.0d0
   stress_vdw_yy = 0.0d0 ; stress_vdw_yz = 0.0d0 ;
   stress_vdw_zz = 0.0d0


   if (l_need_2nd_profile) then
     N = Natoms
     allocate(a_pot_LJ(N))      ;   a_pot_LJ = 0.0d0
!     allocate(a_press_LJ_11(N)) ;   a_press_LJ_11 = 0.0d0
!     allocate(a_press_LJ_22(N)) ;   a_press_LJ_22 = 0.0d0
!     allocate(a_press_LJ_33(N)) ;   a_press_LJ_33 = 0.0d0
!     allocate(a_press_LJ_12(N)) ;   a_press_LJ_12 = 0.0d0
!     allocate(a_press_LJ_13(N)) ;   a_press_LJ_13 = 0.0d0
 !    allocate(a_press_LJ_23(N)) ;   a_press_LJ_23 = 0.0d0
!
! \fields
   endif

 end subroutine initialize ! initializations before first loop

 subroutine finalize

 if (l_need_2nd_profile) then
 do i = 1, Natoms
   atom_profile(i)%pot = atom_profile(i)%pot + (a_pot_LJ(i)) ! the profiles cames 2x

!   atom_profile(i)%sxx = atom_profile(i)%sxx + a_press_LJ_11(i)
!   atom_profile(i)%sxy = atom_profile(i)%sxy + a_press_LJ_12(i)
!   atom_profile(i)%sxz = atom_profile(i)%sxz + a_press_LJ_13(i)
!
!   atom_profile(i)%syx = atom_profile(i)%syx + a_press_LJ_12(i)
!   atom_profile(i)%syy = atom_profile(i)%syy + a_press_LJ_22(i)
!   atom_profile(i)%syz = atom_profile(i)%syz + a_press_LJ_23(i)
!
!   atom_profile(i)%szx = atom_profile(i)%szx + a_press_LJ_13(i)
!   atom_profile(i)%szy = atom_profile(i)%szy + a_press_LJ_23(i)
!   atom_profile(i)%szz = atom_profile(i)%szz + a_press_LJ_33(i)
 enddo
endif


   if (l_need_2nd_profile) then
     deallocate(a_pot_LJ)      ;
!     deallocate(a_press_LJ_11) ;
!     deallocate(a_press_LJ_22) ;
!     deallocate(a_press_LJ_33) ;
!     deallocate(a_press_LJ_12) ;
!     deallocate(a_press_LJ_13) ;
!     deallocate(a_press_LJ_23) ;
   endif

 end subroutine finalize

   subroutine finalize_scalars
 use energies_data
 use stresses_data
 use variables_short_pairs
 implicit none
 integer i,j
 real(8) t(9)
 t=0.0d0

 stress_vdw(1) = stress_vdw_xx
 stress_vdw(2) = stress_vdw_yy
 stress_vdw(3) = stress_vdw_zz
 stress_vdw(4) = (stress_vdw_xx+stress_vdw_yy+stress_vdw_zz)/3.0d0
 stress_vdw(5) = stress_vdw_xy
 stress_vdw(6) = stress_vdw_xz
 stress_vdw(7) = stress_vdw_yz
 stress_vdw(8) = stress_vdw_xy
 stress_vdw(9) = stress_vdw_xz
 stress_vdw(10) = stress_vdw_yz

stress(:) = stress(:) + stress_vdw(:)


! pcharge-gcharge

 end subroutine finalize_scalars


 end subroutine pair_short_forces_vdw
