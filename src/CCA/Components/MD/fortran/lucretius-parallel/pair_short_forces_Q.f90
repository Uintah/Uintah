

subroutine pair_short_forces_Q
 use sys_data
 use paralel_env_data
 use math_constants
 use boundaries
 use ALL_atoms_data
 use atom_type_data
 use max_sizes_data, only : MX_list_nonbonded
 use non_bonded_lists_data, only : list_nonbonded, size_list_nonbonded
 use profiles_data
 use energies_data
 use stresses_data
 use atom_type_data
 use interpolate_data
 use variables_short_pairs
 use physical_constants, only : Volt_to_internal_field
 use rdfs_data, only : rdfs
 use integrate_data, only : integration_step
 use rdfs_collect_module, only : rdfs_collect
 use CTRLs_data, only : l_ANY_DIPOLE_CTRL, l_DIP_CTRL
 use compute_14_module, only : compute_14_interactions_driver
 use REAL_part_intra_correction_sfc_sfc
 implicit none
 real(8), parameter :: en_factor = 0.5d0
 integer i,j,i1,imol,itype,N,neightot,k,nneigh, iStyle
 real(8) local_energy,En0
 real(8), allocatable :: local_force(:,:)

! allocate(local_force(Natoms,3)) ; 
! local_force(:,1) = fxx(:); local_force(:,2) = fyy(:); local_force(:,3) = fzz(:)
 en_vdw = 0.0d0
 en_Qreal = 0.0d0
 call initialize
 irdr = 1.0d0/rdr

 do i = 1+rank, Natoms , nprocs
  imol = atom_in_which_molecule(i)
  iStyle  = i_Style_atom(i)
  i1 = 0
  do k =  1, size_list_nonbonded(i)
    i1 = i1 + 1
    j = list_nonbonded(i,k)
    dx(i1) = xxx(i) - xxx(j)
    dy(i1) = yyy(i) - yyy(j)
    dz(i1) = zzz(i) - zzz(j)
    in_list_Q(i1) = j
  enddo
  neightot=i1
 
  call periodic_images(dx(1:i1),dy(1:i1),dz(1:i1))
  dr_sq(1:i1) = dx(1:i1)*dx(1:i1) + dy(1:i1)*dy(1:i1) + dz(1:i1)*dz(1:i1)
  neightot = i1
  call nonbonded_vdw_2_forces(i,iStyle,size_list_nonbonded(i)) ! put it back
  qi = all_charges(i) ! electricity = charge no matter of what kind.
  dipole_xx_i = all_dipoles_xx(i) ; dipole_yy_i = all_dipoles_yy(i) ; dipole_zz_i = all_dipoles_zz(i)
  dipole_i2 = dipole_xx_i*dipole_xx_i+dipole_yy_i*dipole_yy_i+dipole_zz_i*dipole_zz_i
  call Qinner_initialize(i)
  if (l_DIP_CTRL) then
    call Q_2_forces_dipoles(i,iStyle,neightot)
  else  
    call Q_2_forces(i,iStyle,neightot)
  endif
  call Qinner_finalize(i)

  if (rdfs%any_request) then
  if (mod(integration_step,rdfs%N_collect) == 0) then
  call rdfs_collect(i,neightot)   ! for radial distribution function if needed
  endif
  endif

  enddo

  call compute_14_interactions_driver
  call REAL_part_intra_correction_sfc_sfc_driver

! Now deal with the polarization
local_energy = 0.0d0
do i = 1, Natoms
if(is_dipole_polarizable(i))then
  En0 = all_DIPOLE_pol(i)*(all_dipoles_xx(i)**2+all_dipoles_yy(i)**2+all_dipoles_zz(i)**2) * 0.5d0
  local_energy = local_energy + En0
  if (l_need_2nd_profile)then 
    a_pot_Q(i) = a_pot_Q(i) + En0 * 2.0d0 
  endif
endif
enddo
en_Qreal = en_Qreal + local_energy
en_induced_dip = local_energy

  call finalize_scalar_props
  call finalize

!open(unit=14,file='fort.14',recl=222)
!do i = 1, Natoms
!write(14,*) i,fxx(i)/418.4d0,fyy(i)/418.4d0,fzz(i)/418.4d0
!enddo
!print*, 'stress=',stress_xx/418.4d0,stress_yy/418.4d0,stress_zz/418.4d0,&
!stress_xy/418.4d0,stress_xz/418.4d0,stress_yz/418.4d0
!write(14,*)'enQreal=',(en_Qreal+en_vdw)/418.4d0, en_Qreal/418.4d0, en_vdw/418.4d0,local_energy/418.4d0
!stop
 deallocate(dr_sq)
 deallocate(dx,dy,dz)
 deallocate(in_list_Q)
! deallocate(local_force)

 CONTAINS

  subroutine initialize
   integer N
   integer i,j,k

   N=maxval(size_list_nonbonded)

   allocate(dx(N),dy(N),dz(N),dr_sq(N))
   allocate(in_list_Q(N))
! vdw
   en_vdw = 0.0d0
   en_Qreal=0.0d0
   stress_vdw_xx = 0.0d0 ; stress_vdw_xy = 0.0d0 ; stress_vdw_xz = 0.0d0
   stress_vdw_yy = 0.0d0 ; stress_vdw_yz = 0.0d0 ;
   stress_vdw_zz = 0.0d0
! Q stresses
   stress_xx = 0.0d0 ; stress_xy = 0.0d0 ; stress_xz = 0.0d0
   stress_yy = 0.0d0 ; stress_yz = 0.0d0 ; stress_yz = 0.0d0
   stress_zz = 0.0d0 ; stress_zx=0.0d0 ; stress_zy = 0.0d0

     if (l_need_2nd_profile) then
     N = Natoms
     allocate(a_fi(N))          ;   a_fi = 0.0d0
     allocate(a_pot_LJ(N))      ;   a_pot_LJ = 0.0d0
     allocate(a_pot_Q(N))       ;   a_pot_Q = 0.0d0
!     allocate(a_press_LJ_11(N)) ;   a_press_LJ_11 = 0.0d0
!     allocate(a_press_LJ_22(N)) ;   a_press_LJ_22 = 0.0d0
!     allocate(a_press_LJ_33(N)) ;   a_press_LJ_33 = 0.0d0
!     allocate(a_press_LJ_12(N)) ;   a_press_LJ_12 = 0.0d0
!     allocate(a_press_LJ_13(N)) ;   a_press_LJ_13 = 0.0d0
!     allocate(a_press_LJ_23(N)) ;   a_press_LJ_23 = 0.0d0
!     allocate(a_press_Q_11(Natoms)) ; a_press_Q_11=0.0d0
!     allocate(a_press_Q_12(Natoms)) ; a_press_Q_12=0.0d0
!     allocate(a_press_Q_13(Natoms)) ; a_press_Q_13=0.0d0
!     allocate(a_press_Q_22(Natoms)) ; a_press_Q_22=0.0d0
!     allocate(a_press_Q_23(Natoms)) ; a_press_Q_23=0.0d0
!     allocate(a_press_Q_33(Natoms)) ; a_press_Q_33=0.0d0
!     allocate(a_press_Q_21(Natoms)) ; a_press_Q_21=0.0d0
!     allocate(a_press_Q_31(Natoms)) ; a_press_Q_31=0.0d0
!     allocate(a_press_Q_32(Natoms)) ; a_press_Q_32=0.0d0
! potential
! \potential

! fields
 !P_a_fi(:)=0.0d0;G_a_fi(:)=0.0d0;D_a_fi(:)=0.0d0
 !P_a_EE_xx(:)=0.0d0;G_a_EE_xx(:)=0.0d0;D_a_EE_xx(:)=0.0d0
 !P_a_EE_yy(:)=0.0d0;G_a_EE_yy(:)=0.0d0;D_a_EE_yy(:)=0.0d0
 !P_a_EE_zz(:)=0.0d0;G_a_EE_zz(:)=0.0d0;D_a_EE_zz(:)=0.0d0

! \fields
   endif

 end subroutine initialize ! initializations before first loop

 subroutine Qinner_initialize(i) ! initializarions inside first loop
 integer, intent(IN) :: i
   af_i_1_x=0.0d0
   af_i_1_y=0.0d0
   af_i_1_z=0.0d0
   if (l_need_2nd_profile) then
      a_press_i_11=0.0d0
      a_press_i_22=0.0d0
      a_press_i_33=0.0d0
      a_press_i_12=0.0d0
      a_press_i_13=0.0d0
      a_press_i_23=0.0d0
      a_press_i_21=0.0d0 ! assymetric component for dipole
      a_press_i_31=0.0d0 ! assymetric component for dipole
      a_press_i_32=0.0d0 ! assymetric component for dipole
   endif

 end subroutine Qinner_initialize
 subroutine Qinner_finalize(i)
 integer, intent(IN) :: i
    fxx(i) = fxx(i) + af_i_1_x
    fyy(i) = fyy(i) + af_i_1_y
    fzz(i) = fzz(i) + af_i_1_z

    if (l_need_2nd_profile) then
!        a_press_Q_11(i)=a_press_Q_11(i) + a_press_i_11
!        a_press_Q_22(i)=a_press_Q_22(i) + a_press_i_22
!        a_press_Q_33(i)=a_press_Q_33(i) + a_press_i_33
!
!        a_press_Q_12(i)=a_press_Q_12(i) + a_press_i_12
!        a_press_Q_13(i)=a_press_Q_13(i) + a_press_i_13
!        a_press_Q_23(i)=a_press_Q_23(i) + a_press_i_23
!
!        a_press_Q_21(i)=a_press_Q_21(i) + a_press_i_21
!        a_press_Q_31(i)=a_press_Q_31(i) + a_press_i_31
!        a_press_Q_32(i)=a_press_Q_32(i) + a_press_i_32

    endif
 end subroutine Qinner_finalize

 subroutine finalize
if (l_need_2nd_profile) then
 do i = 1, Natoms
!print*, 'IN ERROR HERE a_pot_Q not defined';STOP
   atom_profile(i)%pot = atom_profile(i)%pot + (a_pot_LJ(i) + a_pot_Q(i))
   atom_profile(i)%Qpot = atom_profile(i)%Qpot + a_pot_Q(i)
   atom_profile(i)%fi =  atom_profile(i)%fi + a_fi(i)
!   atom_profile(i)%sxx = atom_profile(i)%sxx + (a_press_Q_11(i)+a_press_LJ_11(i))
!   atom_profile(i)%sxy = atom_profile(i)%sxy + (a_press_Q_12(i)+a_press_LJ_12(i))
!   atom_profile(i)%sxz = atom_profile(i)%sxz + (a_press_Q_13(i)+a_press_LJ_13(i))
!
!   atom_profile(i)%syx = atom_profile(i)%syx + (a_press_Q_21(i)+a_press_LJ_12(i))
!   atom_profile(i)%syy = atom_profile(i)%syy + (a_press_Q_22(i)+a_press_LJ_22(i))
!   atom_profile(i)%syz = atom_profile(i)%syz + (a_press_Q_23(i)+a_press_LJ_23(i))
!
!   atom_profile(i)%szx = atom_profile(i)%szx + (a_press_Q_31(i)+a_press_LJ_13(i))
!   atom_profile(i)%szy = atom_profile(i)%szy + (a_press_Q_32(i)+a_press_LJ_23(i))
!   atom_profile(i)%szz = atom_profile(i)%szz + (a_press_Q_33(i)+a_press_LJ_33(i))
 enddo
endif

   if (l_need_2nd_profile) then
     deallocate(a_fi)
     deallocate(a_pot_LJ)      ;
     deallocate(a_pot_Q)
!     deallocate(a_press_LJ_11) ;
!     deallocate(a_press_LJ_22) ;
!     deallocate(a_press_LJ_33) ;
!     deallocate(a_press_LJ_12) ;
!     deallocate(a_press_LJ_13) ;
!     deallocate(a_press_LJ_23) ;
!     deallocate(a_press_Q_11) ; 
!     deallocate(a_press_Q_12) ;
!     deallocate(a_press_Q_13) ; 
!     deallocate(a_press_Q_22) ; 
!     deallocate(a_press_Q_23) ; 
!     deallocate(a_press_Q_33) ; 
!     deallocate(a_press_Q_21) ; 
!     deallocate(a_press_Q_31) ;
!     deallocate(a_press_Q_32) ; 

   endif

 end subroutine finalize


 end subroutine pair_short_forces_Q

 
