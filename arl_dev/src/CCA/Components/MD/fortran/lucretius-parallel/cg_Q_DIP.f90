! conjugate gradient doing charge constrained with respect to a field.


module cg_Q_DIP_module 
implicit none
private :: finalize_charges_and_dipoles
private :: clean_up_and_finalize
private :: very_first_pass_init
private :: set_up_intramol_list12
private :: allocate_history
private :: set_up_TAGS_and_NDX_arrays
private :: finalize_history
private :: initialize
private :: re_assign_history
private :: set_local_lists
private :: first_iteration
private :: get_z0 ! put the preconditioner
private :: set_GG_arrays
private :: first_iter_free_term_REAL_SPACE
private :: first_iter_free_term_k0_2D
private :: first_iter_free_term_k0_2D_SLOW
private :: first_iter_free_term_k_NON_0_2D_SLOW
private :: first_iter_free_term_FOURIER_3D_SLOW
private :: first_iter_free_term_k_NON_0_2D
private :: first_iter_free_term_Fourier_3D
private :: first_iter_free_term_14
private :: get_AX_k_NON_0_2D
private :: get_AX_Fourier_3D
private :: set_grid_Q_DIP_init_2D
private :: set_grid_Q_DIP_init_3D
private :: set_grid_Q_DIP_cycle_2D
private :: set_grid_Q_DIP_cycle_3D
private :: smpe_eval1_Q_DIP_2D
private :: smpe_eval1_Q_DIP_3D
private :: smpe_eval2_Q_DIP_init_2D
private :: smpe_eval2_Q_DIP_init_3D
private :: smpe_eval2_Q_DIP_cycle_2D
private :: smpe_eval2_Q_DIP_cycle_3D
private :: first_iter_intra_correct_fourier
private :: first_iter_intra_correct_K_0
private :: get_Ax
private :: get_AX_intra_correct_fourier
private :: get_AX_REAL
private :: get_AX_14
private :: get_AX_self_interact
private :: get_Ax_at_k0_2D
private :: get_Ax_at_k0_2D_SLOW
private :: get_AX_k_NON_0_2D_SLOW
private :: get_Ax_fourier_3D_SLOW
private :: get_AX_intra_correct_k_0
private :: mask_vars
private :: get_sfield_free_term
private :: get_dipol_free_term

public :: cg_Q_DIP

real(8), private, save :: eta
real(8), private, save ::  h_cut_off,h_step, CC_alpha, i_Area
integer, private, save :: N_k_vct , TAG_SS,TAG_SP,TAG_PP
integer, private, save ::  NV, NDP, NVFC, NAFC
real(8), private, allocatable :: X(:),AX(:)
real(8), private, allocatable , save :: qq(:), bv(:,:),z_grid(:), MAT(:,:),vt(:)
real(8), private, allocatable :: MAT_cg(:,:), BB_cg(:), k_vector(:,:)
integer, private,save :: order, Ngrid, N_size_qq
real(8), private,save :: fct_UP,fct_DOWN,ratio_B1,ratio_B2,ratio_B3
real(8), private, allocatable :: X_predictor(:)
integer, private, save :: N_iterations
real(8), private, save :: TOLERANCE
logical, private, save :: l_skip_fourier_here, l_do_Fourier_here, use_picard
integer, private, save, allocatable :: is_sfield(:), is_dip(:)
logical, private :: is_convergent
integer, private, save :: number_of_nonconvergencies = 0
contains



subroutine cg_Q_DIP
use physical_constants, only : Red_Vacuum_EL_permitivity_4_Pi, Volt_to_internal_field
use cg_buffer, only : q, BB0, CG_TOLERANCE, cg_iterations, use_cg_preconditioner_CTRL,picard_dumping
use field_constrain_data, only : ndx_remap
use ALL_atoms_data, only : Natoms, all_DIPOLE_pol,i_type_atom
use atom_type_data, only : atom_type_name
use generic_statistics_module, only : statistics_5_eval,min_max_stat_1st_pass
use rolling_averages_data, only : RA_cg_iter
implicit none
logical, save :: very_first_pass = .true.
real(8), allocatable :: r(:),r1(:),Ad(:),b(:),d(:),d1(:)
real(8), allocatable :: q0(:),r0(:),d0(:)
real(8), allocatable :: z0(:), z1(:)
integer iter,i,j,k,i1, i1_1, i_adress
integer, save :: NNN
real(8) ERROR,ERROR1, alpha, beta,fct
logical stay_in

  if (very_first_pass) then 
     call very_first_pass_init
     very_first_pass = .false.
  endif
  call zero_initialize
  NNN = NVFC
  allocate(r(NNN),r1(NNN),Ax(NNN),X(NNN),Ad(NNN),b(NNN),d(NNN),d1(NNN))
  allocate(q0(NNN),r0(NNN),d0(NNN))
  if (use_cg_preconditioner_CTRL) allocate(z0(NNN),z1(NNN))
  call initialize
  call first_iteration
  call get_Ax(NNN,q(1:NNN),Ax(1:NNN))     ! Ax+b=0 ; this one gets Ax
  call get_sfield_free_term(NNN,b) ! Ax+b=0; this one gets b

fct=dsqrt(Red_Vacuum_EL_permitivity_4_Pi)
!use_picard=.false.!delete it
  if (use_picard)then ! only if dipol polarizable only
   stay_in = .true.
   r = b - Ax
   d = r
   q0=q
   q(1:NNN) = q0(1:NNN) + d(1:NNN)*vt(1:NNN)
   q0=q
   iter = 1
   ERROR = 1.0d90

   do while (iter < N_iterations .and. stay_in)
     iter=iter+1
     call get_Ax(NNN,q(1:NNN),Ax(1:NNN))
     d = b - Ax
     r(1:NNN) = q0(1:NNN) + d(1:NNN)*vt(1:NNN)
     q(1:NNN) = (1.0d0-picard_dumping)*q0(1:NNN) + picard_dumping*r(1:NNN)
     ERROR = maxval(dabs(q0-q))
     stay_in = ERROR > TOLERANCE
!print*, iter,maxval(dabs(q0-q))*fct
     q0=q
   enddo ! dowhile
!print*,'iter=',iter

  else  ! no piccard but cg iterations

! iterations starts here:

   !q=0.0d0
   ERROR = 1.0d90
   iter=0
   call get_Ax(NNN,q(1:NNN),Ax(1:NNN))
   r = b - Ax
   stay_in = .true.
if (use_cg_preconditioner_CTRL) then
   call get_z0(NNN,r,z0)
   d = z0
   ERROR = dot_product(r(1:NNN),z0(1:NNN))
 
   do while (iter < N_iterations .and. stay_in)
     iter = iter + 1
     call get_Ax(NNN,z0(1:NNN),Ad(1:NNN))
     alpha = ERROR/dot_product(z0(1:NNN),Ad(1:NNN))
     q(1:NNN) = q(1:NNN) + alpha*z0(1:NNN)
     r1(1:NNN) = r(1:NNN) - alpha*Ad(1:NNN)
     call get_z0(NNN,r1,z1)
     ERROR1 = dot_product(r1(1:NNN),z1(1:NNN))
     beta = ERROR1/ERROR
     d1(1:NNN) = z1(1:NNN) + beta*z0(1:NNN)
     z0(1:NNN) = d1(1:NNN) 
     r(1:NNN) = r1(1:NNN)
     ERROR = ERROR1
     stay_in = maxval(dabs(r(1:NNN))) > TOLERANCE
!print*, iter,  maxval(dabs(r(1:NNN)))*dsqrt(Red_Vacuum_EL_permitivity_4_Pi),N_iterations
   enddo
31 CONTINUE ! exit from cycle
else
   d = r


   do while (iter < N_iterations .and. stay_in) 
    iter = iter + 1
    call get_Ax(NNN,d(1:NNN),Ad(1:NNN))
    ERROR = dot_product(r(1:NNN),r(1:NNN))      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    alpha = ERROR/dot_product(d(1:NNN),Ad(1:NNN))   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    q(1:NNN) = q(1:NNN) + alpha*d(1:NNN)
    r1(1:NNN) = r(1:NNN) - alpha*Ad(1:NNN)
    ERROR1 = dot_product(r1(1:NNN),r1(1:NNN))           !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    beta = ERROR1/ERROR
    d1(1:NNN) = r1(1:NNN) + beta*d(1:NNN)
    d(1:NNN) = d1(1:NNN)
    r(1:NNN) = r1(1:NNN)
    stay_in = maxval(dabs(r(1:NNN))) > TOLERANCE
!print*,iter,maxval(dabs(r))*fct
   enddo

endif !  use_cg_preconditioner_CTRL
   
   endif ! use_picard

   cg_iterations = dble(iter)
   call statistics_5_eval(RA_cg_iter, cg_iterations)
   is_convergent  = iter < N_iterations

!open(unit=777,file='fort.777',recl=500)
!write(777,*) Natoms
!do k = 1, TAG_SP
!  write(777,*)k,'q1=',q(k)*dsqrt(Red_Vacuum_EL_permitivity_4_Pi)
!enddo
!do k = TAG_SS+1, TAG_PP
!i_adress = k - TAG_SS + TAG_SP
!write(777,*)k,ndx_remap%var(k),q(i_adress)*dsqrt(Red_Vacuum_EL_permitivity_4_Pi),&
!      q(i_adress+NDP)*dsqrt(Red_Vacuum_EL_permitivity_4_Pi),&
!      q(i_adress+2*NDP)*dsqrt(Red_Vacuum_EL_permitivity_4_Pi), &
!      dsqrt(q(i_adress)**2+q(i_adress+NDP)**2+q(i_adress+2*NDP)**2)*&
!           dsqrt(Red_Vacuum_EL_permitivity_4_Pi)
!enddo
!close(777)
!stop

   call finalize_charges_and_dipoles
   call clean_up_and_finalize
   deallocate(r,r1,Ax,Ad,b,d,d1,X)
   deallocate(q0,r0,d0)
   if (use_cg_preconditioner_CTRL) deallocate(z0,z1)
end subroutine cg_Q_DIP

subroutine get_z0(N,Vin,Vout)   ! put the preconditioner
 use ALL_atoms_data, only : i_Style_atom, is_charge_distributed
 use Ewald_data
 use field_constrain_data,only : ndx_remap, rec_ndx_remap
 use math_constants, only : two_per_sqrt_Pi,sqrt_Pi
 use preconditioner_data
implicit none
integer, intent(IN) :: N
real(8), intent(IN) :: Vin(N)
real(8), intent(OUT) :: Vout(N)

integer i,i1,itype, j,j1
real(8) CC_eta,f,Vi,r,qi


Vout = Vin ! No precontitioner
RETURN

Vout = 0.0d0

   do i1 = 1, TAG_SP  ! charges first
     i = ndx_remap%var(i1)
     if (is_charge_distributed(i)) then
      itype = i_Style_atom(i)
      CC_eta = Ewald_eta(itype,itype)
    else
      CC_eta = 0.0d0
    endif
     f = two_per_sqrt_Pi * (CC_eta - Ewald_alpha) 
     Vout(i1) = Vout(i1) + Vin(i1) / f ** 3  ! self interactions added here
   enddo


!   do i1 = 1, TAG_SP  ! charges first
!     i = ndx_remap%var(i1)
!     Vi = 0.0d0
!     qi = Vin(i1)
!     do j = 1, size_preconditioner(i)
!       j1 = rec_ndx_remap%var(j)
!       r = preconditioner_rr(i,j)
!       Vi = Vi + Vin(j1) / r
!       Vout(j1) = Vout(j1) + qi / r
!     enddo
!     Vout(i1) = Vout(i1) +  Vi   ! 
!   enddo
!print*,'Vout=',Vout(1:100)
! \End Try a pre-conditioner based on self-interactions for sfc-charges

!   do i1 = TAG_SS+1,TAG_PP
!     i = ndx_remap%var(i1)
!     
!   enddo


end subroutine get_z0

! -----------------------
subroutine first_iteration
use sim_cel_data, only : i_boundary_CTRL
use cg_buffer, only : BB0
use Ewald_data, only : i_type_EWALD_CTRL ! (slow or fast)
use sizes_data, only : N_pairs_14
implicit none
BB0(:) = 0.0d0

  call set_GG_arrays
  call first_iter_free_term_REAL_SPACE
  if (N_pairs_14>0) call first_iter_free_term_14
  if (i_boundary_CTRL == 1) then
  if (i_type_EWALD_CTRL == 1) then ! FAST
     call first_iter_free_term_k0_2D
     if(l_do_Fourier_here) call first_iter_free_term_k_NON_0_2D
  else ! SLOW
     call first_iter_free_term_k0_2D_SLOW
     if(l_do_Fourier_here) call first_iter_free_term_k_NON_0_2D_SLOW
  endif
  else  ! 3D-periodicity
  if (i_type_EWALD_CTRL == 1) then
     if (l_do_Fourier_here) call first_iter_free_term_Fourier_3D
  else ! SLOW
     if (l_do_Fourier_here) call first_iter_free_term_Fourier_3D_SLOW
  endif
  endif
if (l_do_Fourier_here) then 
   call first_iter_intra_correct_Fourier

else
   call first_iter_intra_correct_K_0 
endif

end subroutine first_iteration

!---------------------------------------------
subroutine get_Ax(N,Xin,AXout)
use sim_cel_data, only : i_boundary_CTRL
use Ewald_data, only : i_type_EWALD_CTRL
use sizes_data, only : N_pairs_14
implicit none
integer, intent(IN) :: N
real(8), intent(IN) :: Xin(N)
real(8), intent(OUT) :: AXout(N)

  X=Xin
  AX = 0.0d0
  call mask_vars
  call get_AX_REAL
  if (N_pairs_14>0) call get_AX_14
  call get_AX_self_interact!(N,X,AX)
 if (i_boundary_CTRL==1) then  ! 2D geometry
    if (i_type_EWALD_CTRL==1) then ! FAST
       call get_Ax_at_k0_2D
       if (l_do_Fourier_here) call get_AX_k_NON_0_2D
    else   ! SLOW
       call get_Ax_at_k0_2D_SLOW
       if (l_do_Fourier_here) call get_AX_k_NON_0_2D_SLOW
    endif
 else        ! 3D geometry
    if (i_type_EWALD_CTRL==1) then ! FAST
       if (l_do_Fourier_here) call get_AX_fourier_3D
    else
       if (l_do_Fourier_here) call get_AX_fourier_3D_SLOW
    endif
 endif
 if (l_do_Fourier_here) then 
   call get_AX_intra_correct_fourier ! This will remove all intramolecular interactions
 else
   if (i_boundary_CTRL==1) call get_AX_intra_correct_k_0 
 endif
 if (NDP > 0) call get_dipol_free_term(X,Ax) !(NNN,q(1:NNN))
Axout = AX
end subroutine get_Ax

!----------------------------------------------

subroutine finalize_charges_and_dipoles
   use field_constrain_data, only : ndx_remap
   use ALL_atoms_data, only : all_g_charges,all_p_charges,all_charges,is_charge_distributed,&
                       all_dipoles_xx,all_dipoles_yy,all_dipoles_zz,all_dipoles
   use cg_buffer, only : q, cg_predict_restart_CTRL, aspc_omega
   use integrate_data, only : integration_step
 implicit none
 integer i,j,k, i_adress,i1

if (is_convergent) then
  if (cg_predict_restart_CTRL == 3) then ! always stable predictor corrector 
     q(:) = aspc_omega*q(:) + (1.0d0-aspc_omega)*X_predictor(:)
  endif

  all_charges(ndx_remap%var(1:TAG_SP))= q(1:TAG_SP)
  do i1 = 1, NAFC
    i = ndx_remap%var(i1)
    if (is_charge_distributed(i)) then
      all_g_charges(i) = q(i1)
    else
      all_p_charges(i) = q(i1)
    endif
  enddo

  do i1 = TAG_SS+1,TAG_PP
   i_adress = i1 - TAG_SS + TAG_SP
   i = ndx_remap%var(i1)
     all_dipoles_xx(i) = q(i_adress)
     all_dipoles_yy(i) = q(i_adress +   NDP)
     all_dipoles_zz(i) = q(i_adress + 2*NDP)
     all_dipoles(i) = dsqrt( &
     all_dipoles_xx(i)*all_dipoles_xx(i)+all_dipoles_yy(i)*all_dipoles_yy(i)+all_dipoles_zz(i)*all_dipoles_zz(i))
  enddo
else ! did not converged
 number_of_nonconvergencies=number_of_nonconvergencies+1
 print*,' WARNING at integration step ', integration_step, ' cg_Q_DIP did not converged', &
 ' number_of_nonconvergencies = ', number_of_nonconvergencies,&
 ' The charges and/or dipoles will not be updated older value will be used' 
  if (number_of_nonconvergencies>1000) then 
   print*,' MORE than 1000 nonconvergencies in cg_Q_DIP ; something may be wrong ... The code will stop',&
   ' at cg_Q_DIP%finalize_charges_and_dipoles'
   STOP
  endif
endif
end subroutine finalize_charges_and_dipoles

subroutine clean_up_and_finalize
   use integrate_data, only : integration_step
   use cg_buffer, only : Ih_on_grid,Ih_on_grid_dx,Ih_on_grid_dy,Ih_on_grid_dz,sns,css,pre,&
                         GG_0,GG_1,GG_2,GG_0_excluded,GG_1_excluded,GG_2_excluded,BB,BB0,&
                         list1,list2,size_list1,size_list2,NNX,NNY,NNZ,q, GG_0_THOLE,GG_1_THOLE,&
                         GG_0_14,GG_1_14,GG_2_14,&
                         mask_qi,mask_di_xx,mask_di_yy,mask_di_zz,&
                         list1_14,list2_14,size_list1_14,size_list2_14,&
                         size_list2_ex,size_list1_ex,list2_ex,list1_ex,&
                         list1_sfc_sfc,size_list1_sfc_sfc,list2_sfc_sfc,size_list2_sfc_sfc

   use Ewald_data,only : i_type_EWALD_CTRL
   use sizes_data, only : N_pairs_14, N_pairs_sfc_sfc_123
   use variables_smpe, only : spline2_REAL_pp_x,spline2_REAL_pp_y,spline2_REAL_pp_z,&
                              spline2_REAL_dd_x,spline2_REAL_dd_y,spline2_REAL_dd_z,&
                              tx,ty,tz
   use allocate_them, only : smpe_DEalloc

 implicit none
  deallocate(list1); deallocate(size_list1)
  deallocate(list2); deallocate(size_list2)
  if (N_pairs_sfc_sfc_123>0)then
  deallocate(list1_sfc_sfc); deallocate(size_list1_sfc_sfc)
  deallocate(list2_sfc_sfc); deallocate(size_list2_sfc_sfc)
  endif
  deallocate(BB0,BB)
  deallocate(GG_0,GG_1,GG_2)
  if (N_pairs_14>0) then 
    deallocate(GG_0_14,GG_1_14,GG_2_14)
  endif
  deallocate(GG_0_excluded,GG_1_excluded,GG_2_excluded)
  deallocate(GG_0_THOLE,GG_1_THOLE)
  if(allocated(pre)) deallocate(pre)
  if(allocated(k_vector)) deallocate(k_vector)
  deallocate(q)

if (l_do_Fourier_here) then
if (i_type_EWALD_CTRL==2) then ! SLOW
   deallocate(sns,css)
else if(i_type_EWALD_CTRL==1) then !FAST
 deallocate(Ih_on_grid,Ih_on_grid_dx,Ih_on_grid_dy,Ih_on_grid_dz)
 deallocate(NNX,NNY,NNZ)
else
print*, 'ERROR : in cg&clean_up_and_finalize ewald method not defined i_type_EWALD_CTRL=',i_type_EWALD_CTRL
STOP
endif
endif

if (allocated(spline2_REAL_pp_x))deallocate(spline2_REAL_pp_x)
if (allocated(spline2_REAL_pp_y))deallocate(spline2_REAL_pp_y)
if (allocated(spline2_REAL_pp_z))deallocate(spline2_REAL_pp_z)
if (allocated(spline2_REAL_dd_x))deallocate(spline2_REAL_dd_x)
if (allocated(spline2_REAL_dd_y))deallocate(spline2_REAL_dd_y)
if (allocated(spline2_REAL_dd_z))deallocate(spline2_REAL_dd_z)
if(allocated(tx))deallocate(tx)
if(allocated(ty))deallocate(ty)
if(allocated(tz))deallocate(tz)

if (allocated(qq)) deallocate(qq)
if (allocated(bv)) deallocate(bv)
if (allocated(z_grid)) deallocate(z_grid)
if(allocated(vt)) deallocate(vt)

deallocate(mask_qi,mask_di_xx,mask_di_yy,mask_di_zz)
if (N_pairs_14>0)then
  deallocate(list1_14,size_list1_14)
  deallocate(list2_14,size_list2_14)
endif
deallocate(list1_ex,size_list1_ex)
deallocate(list2_ex,size_list2_ex)
call smpe_DEalloc

end subroutine clean_up_and_finalize

!-------------------------
subroutine very_first_pass_init
 use Ewald_data, only : Ewald_alpha, i_type_EWALD_CTRL,dfftx,dffty,dfftz
 use allocate_them, only : smpe_alloc
 use smpe_utility_pack_0, only : get_CMPLX_splines,get_CMPLX_splines_NEW
 use fft_3D
 use variables_smpe
 use cg_buffer, only : mask_qi,mask_di_xx,mask_di_yy,mask_di_zz, CG_TOLERANCE,cg_skip_fourier,l_try_picard_CTRL
 use math_constants, only : Pi,Pi2,sqrt_Pi
 use physical_constants, only : Red_Vacuum_EL_permitivity_4_Pi
 use sizes_data , only : N_pairs_14
 use ALL_atoms_data, only : all_DIPOLE_pol
 use field_constrain_data, only : ndx_remap,rec_ndx_remap
 use rolling_averages_data, only : RA_cg_iter
 use sizes_data, only : Natoms

 implicit none
 integer i,j,k,N, i1,j1,neightot
 logical ll, l1

 TOLERANCE = CG_TOLERANCE / dsqrt(Red_Vacuum_EL_permitivity_4_Pi)
 N_iterations = 200
 CC_alpha = sqrt_Pi/Ewald_alpha
     call allocate_history
     allocate(is_sfield(Natoms),is_dip(Natoms))


  fct_UP = 2.0d0*Ewald_alpha*Ewald_alpha
  fct_DOWN = 1.0d0/(Ewald_alpha*sqrt_Pi)
  ratio_B1 = fct_UP*fct_DOWN
  ratio_B2 = ratio_B1*fct_UP
  ratio_B3 = ratio_B2*fct_UP
 
  RA_cg_iter%val=0.0d0;RA_cg_iter%counts=0.0d0;RA_cg_iter%val_sq=0.0d0;RA_cg_iter%MIN=999999;RA_cg_iter%MAX=-99999

 if(.not.(allocated(ndx_remap%var)))      allocate(ndx_remap%var(Natoms))
 if(.not.(allocated(rec_ndx_remap%var)))  allocate(rec_ndx_remap%var(Natoms))

end subroutine very_first_pass_init

subroutine zero_initialize
use Ewald_data, only : Ewald_alpha, i_type_EWALD_CTRL,dfftx,dffty,dfftz
 use allocate_them, only : smpe_alloc
 use smpe_utility_pack_0, only : get_CMPLX_splines,get_CMPLX_splines_NEW
 use fft_3D
 use variables_smpe
 use cg_buffer, only : mask_qi,mask_di_xx,mask_di_yy,mask_di_zz, CG_TOLERANCE,cg_skip_fourier,l_try_picard_CTRL
 use math_constants, only : Pi,Pi2,sqrt_Pi
 use physical_constants, only : Red_Vacuum_EL_permitivity_4_Pi
 use sizes_data , only : N_pairs_14
 use ALL_atoms_data, only : all_DIPOLE_pol
 use field_constrain_data, only : ndx_remap
 use rolling_averages_data, only : RA_cg_iter
 use sizes_data, only : Natoms
 use cg_buffer, only : l_DO_CG_CTRL_Q,l_DO_CG_CTRL_DIP
implicit none

     call set_up_TAGS_and_NDX_arrays
     call set_up_intramol_list12
     if (N_pairs_14>0)call set_up_14_list12
! FOURIER EWALD  K non zero
if (i_type_EWALD_CTRL==2) then ! SLOW

else if(i_type_EWALD_CTRL==1) then !FAST
if (.not.cg_skip_fourier%lskip) then
     NFFT = nfftx*nffty*nfftz
     call smpe_alloc
     call dlpfft3_MOD(1,1,nfftx,nffty,nfftz,key1,key2,key3,ww1,ww2,ww3,qqq1)
     reciprocal_zz = Pi2/dble(nfftz)
     inv_rec_zz = 1.0d0/reciprocal_zz
     call get_CMPLX_splines_NEW
endif
else
print*, 'ERROR : in cg&veryfirstpass ewald method not defined i_type_EWALD_CTRL=',i_type_EWALD_CTRL
STOP
endif

     allocate(mask_qi(1:NV),mask_di_xx(1:NV),mask_di_yy(1:NV),mask_di_zz(1:NV))

 use_picard = TAG_SS==0.and.TAG_SP==0.and.TAG_PP/=0.and.l_try_picard_CTRL ! only for dipoles
 use_picard = use_picard.and.l_DO_CG_CTRL_DIP.and.(.not.l_DO_CG_CTRL_Q)

 if (use_picard) then
   allocate(vt(NVFC));
  vt(1:NVFC/3) = 1.0d0/all_DIPOLE_pol(ndx_remap%var(:))
  vt(1+NVFC/3:2*NVFC/3) = vt(1:NVFC/3)
  vt(1+2*NVFC/3:NVFC)   = vt(1:NVFC/3)
 endif


end subroutine zero_initialize
!-------------------------

subroutine set_up_14_list12
use ALL_atoms_data, only : Natoms
use cg_buffer, only : list1_14,list2_14,size_list1_14,size_list2_14
use connectivity_ALL_data, only : list_14,size_list_14
use field_constrain_data, only : ndx_remap,rec_ndx_remap
use sizes_data, only : N_pairs_14
use max_sizes_data, only : MX_in_list_14
implicit none
integer i,j,k,i1,j1,neightot
if (N_pairs_14>0)then
  allocate(list1_14(NVFC,MX_in_list_14),size_list1_14(NVFC))
  allocate(list2_14(NVFC,MX_in_list_14),size_list2_14(NVFC))
  size_list1_14=0
  size_list2_14=0
   do i = 1, Natoms
    i1 = rec_ndx_remap%var(i)
    neightot = size_list_14(i)
    do k = 1, neightot
      j = list_14(i,k)
      j1 = rec_ndx_remap%var(j)
      if (i1 < NV+1.and.j1>TAG_SP) then
                size_list2_14(i1) = size_list2_14(i1) + 1
                list2_14(i1, size_list2_14(i1)) = j
      endif
      if (i1 > TAG_SP.and.j1<NV+1) then
                size_list2_14(j1) = size_list2_14(j1) + 1
                list2_14(j1, size_list2_14(j1)) = i
      endif
      if (i1 < NV+1.and.j1<NV+1) then
                size_list1_14(i1) = size_list1_14(i1) + 1
!      size_list1(j1) = size_list1(j1) + 1 ! Now is no longer i>j
                list1_14(i1,size_list1_14(i1)) = j
!      list1(j1,size_list1(j1)) = i
     endif
    enddo ! k = 1, size_list_nonbonded(i)
    enddo  ! i
endif

end subroutine set_up_14_list12
!---------------------------------
subroutine set_up_intramol_list12
use ALL_atoms_data, only : Natoms
use connectivity_ALL_data, only : MX_excluded,&
                                  list_excluded_HALF_no_SFC,size_list_excluded_HALF_no_SFC
use cg_buffer, only : list1_ex,list2_ex,size_list1_ex,size_list2_ex
use field_constrain_data, only : ndx_remap,rec_ndx_remap

implicit none
integer i,j,k,i1,j1,neightot
  allocate(list1_ex(NVFC,MX_excluded),size_list1_ex(NVFC))
  allocate(list2_ex(NVFC,MX_excluded),size_list2_ex(NVFC))
  size_list1_ex = 0
  size_list2_ex = 0
  do i = 1, Natoms
    i1 = rec_ndx_remap%var(i)
    neightot = size_list_excluded_HALF_no_SFC(i)
    do k = 1, neightot
      j = list_excluded_HALF_no_SFC(i,k)
      j1 = rec_ndx_remap%var(j)
      if (i1 < NV+1.and.j1>TAG_SP) then
                size_list2_ex(i1) = size_list2_ex(i1) + 1
                list2_ex(i1, size_list2_ex(i1)) = j
      endif
      if (i1 > TAG_SP.and.j1<NV+1) then
                size_list2_ex(j1) = size_list2_ex(j1) + 1
                list2_ex(j1, size_list2_ex(j1)) = i
      endif
      if (i1 < NV+1.and.j1<NV+1) then
                size_list1_ex(i1) = size_list1_ex(i1) + 1
!      size_list1(j1) = size_list1(j1) + 1 ! Now is no longer i>j
                list1_ex(i1,size_list1_ex(i1)) = j
!      list1(j1,size_list1(j1)) = i
     endif
    enddo ! k = 1, size_list_nonbonded(i)
    enddo  ! i

end subroutine set_up_intramol_list12
! -----------------------
subroutine allocate_history
use cg_buffer, only : MAT_lsq_cp_predictor,cg_predict_restart_CTRL,MAT_lsq_cp_predictor,&
               order_lsq_cg_predictor,var_history
use field_constrain_data, only : N_variables_field_constrained

implicit none
  if (cg_predict_restart_CTRL == -1) then
    allocate(var_history(N_variables_field_constrained,2))
  elseif (cg_predict_restart_CTRL == 0) then   ! take the last iteration
    allocate(var_history(N_variables_field_constrained,2))
  elseif (cg_predict_restart_CTRL == 1) then
    allocate(var_history(N_variables_field_constrained,5))
  elseif (cg_predict_restart_CTRL == 2) then
    allocate(var_history(N_variables_field_constrained,order_lsq_cg_predictor+1))
    allocate(MAT_cg(order_lsq_cg_predictor,order_lsq_cg_predictor))
    allocate(MAT_lsq_cp_predictor(order_lsq_cg_predictor,order_lsq_cg_predictor))
    allocate(BB_cg(order_lsq_cg_predictor))
  elseif (cg_predict_restart_CTRL == 3) then  ! ALWAYS STABLE PREDICTOR CORRECTOR
    allocate(var_history(N_variables_field_constrained,0:5),X_predictor(N_variables_field_constrained))
  else
    print*, 'unknown method cg_predict_restart_CTRL to very first pass in cg',cg_predict_restart_CTRL
    STOP
  endif
end subroutine allocate_history
! -------------------
subroutine set_up_TAGS_and_NDX_arrays
use ALL_atoms_data, only : Natoms, is_sfield_constrained,is_dipole_polarizable
use field_constrain_data, only : ndx_remap,rec_ndx_remap,&
    N_atoms_field_constrained,N_atoms_variables,N_dipol_polarizable
use cg_buffer, only : l_DO_CG_CTRL_Q, l_DO_CG_CTRL_DIP
implicit none    
integer i,j,k,i1,j2
logical l1


  is_sfield(:) = is_sfield_constrained(:).and.l_DO_CG_CTRL_Q
  is_dip(:) = is_dipole_polarizable(:).and.l_DO_CG_CTRL_DIP

  NDP = 0
  NAFC = 0
  i1 = 0
  do i = 1, Natoms
   l1 = (is_sfield(i).or.is_dip(i))
   if (l1) i1=i1+1
   if (is_dip(i)) NDP = NDP + 1
   if (is_sfield(i)) NAFC = NAFC + 1
  enddo

! N_atoms_variables = i1
 NV = i1
 NVFC = NAFC + 3*NDP  
! N_atoms_field_constrained = NAFC


 i1 = 0
 TAG_SP = 1; TAG_SS = 1; TAG_PP = 1
  do i = 1, Natoms
   l1 = (is_sfield(i).and.(.not.is_dip(i)))
   if (l1) then
     i1 = i1 + 1
     ndx_remap%var(i1) = i
     rec_ndx_remap%var(i) = i1
!     l_remaped(i) = .true.
   endif
 enddo
 tag_SS = i1

  do i = 1, Natoms
   l1 = (is_sfield(i).and.(is_dip(i)))
   if (l1) then
     i1 = i1 + 1
     ndx_remap%var(i1) = i
     rec_ndx_remap%var(i) = i1
!     l_remaped(i) = .true.
   endif
 enddo
 TAG_SP = i1

 do i = 1, Natoms
   l1 = ((.not.is_sfield(i)).and.is_dip(i))
   if (l1) then
     i1 = i1 + 1
     ndx_remap%var(i1) = i
     rec_ndx_remap%var(i) = i1
   endif
 enddo
 TAG_PP = i1

 do i = 1, Natoms
   l1 = ((.not.is_sfield(i)).and.(.not.is_dip(i)))
   if (l1) then
     i1 = i1 + 1
     ndx_remap%var(i1) = i
     rec_ndx_remap%var(i) = i1
   endif
 enddo

end subroutine set_up_TAGS_and_NDX_arrays


! -------------------


subroutine finalize_history
 use cg_buffer, only : q, var_history
 use ALL_atoms_data, only : all_charges,all_dipoles_xx,all_dipoles_yy,all_dipoles_zz,Natoms
 use integrate_data, only : integration_step
 use field_constrain_data, only : ndx_remap
 implicit none
 integer i,j,k,i1,j1,i_adress
 select case (integration_step)
 case(1)
    q(1:TAG_SP) = all_charges(ndx_remap%s_field(1:TAG_SP))
    do i1 = TAG_SS+1,TAG_PP
      i_adress = i1 - TAG_SS + TAG_SP
      j = ndx_remap%var(i1)
      q(i_adress        ) = all_dipoles_xx(j)
      q(i_adress +   NDP) = all_dipoles_yy(j)
      q(i_adress + 2*NDP) = all_dipoles_zz(j)
    enddo
    var_history(:,2) = var_history(:,1)
    var_history(:,1) = q(:)

 case(2)
    var_history(:,3) = var_history(:,2)
    var_history(:,2) = var_history(:,1)
    var_history(:,1) = q(:)
 case(3)
    var_history(:,4) = var_history(:,3)
    var_history(:,3) = var_history(:,2)
    var_history(:,2) = var_history(:,1)
    var_history(:,1) = q(:)
 case(4)
    var_history(:,5) = var_history(:,4)
    var_history(:,4) = var_history(:,3)
    var_history(:,3) = var_history(:,2)
    var_history(:,2) = var_history(:,1)
    var_history(:,1) = q(:)
 case default  ! >= 3
    var_history(:,5) = var_history(:,4)
    var_history(:,4) = var_history(:,3)
    var_history(:,3) = var_history(:,2)
    var_history(:,2) = var_history(:,1)
    var_history(:,1) = q(:)
 end select
end subroutine finalize_history
!----------------------------------------------------
subroutine initialize
 use sizes_data, only :  Natoms,N_pairs_14,N_pairs_sfc_sfc_123
 use max_sizes_data, only : MX_in_list_14,MX_excluded
 use non_bonded_lists_data
 use ALL_atoms_data, only : all_p_charges, all_g_charges,xx,yy,zz,&
    xxx,yyy,zzz,is_charge_distributed, all_charges,&
    all_dipoles_xx,all_dipoles_yy,all_dipoles_zz
 use field_constrain_data
 use cg_buffer, only : q,BB0,BB,GG_0,GG_1,GG_2,list1,size_list1,list2,size_list2,NNX,NNY,NNZ,pre,&
                       Ih_on_grid,Ih_on_grid_dx,Ih_on_grid_dy,Ih_on_grid_dz, cg_skip_fourier, &
                       GG_0_THOLE,GG_1_THOLE,GG_0_14,GG_1_14,GG_2_14,&
                       l_do_FFT_in_inner_CG,l_DO_CG_CTRL_DIP,l_DO_CG_CTRL_Q,l_DO_CG_CTRL, &
                       list1_sfc_sfc,size_list1_sfc_sfc,list2_sfc_sfc,size_list2_sfc_sfc
 use sim_cel_data
 use boundaries, only : get_reciprocal_cut, periodic_images
 use cut_off_data
 use Ewald_data
 use spline_z_k0_module
 use variables_smpe
 use smpe_utility_pack_0, only : get_reduced_coordinates, get_pp_spline2_coef_REAL
 use integrate_data, only : integration_step
 use math, only : invert3
 use beta_spline_new

implicit none
 integer i,j,k,ii,jj,k_vct,i1,neightot,j1,N
 logical l_i,l_j
 real(8) tmp
 integer i_adress
 i_Area = 1.0d0/Area_xy
 if (cg_skip_fourier%lskip) then
  l_skip_fourier_here = cg_skip_fourier%lskip .and. mod(integration_step,cg_skip_fourier%how_often)==0
  l_do_fourier_here  = .not.l_skip_fourier_here
 else
  l_skip_fourier_here = .false.
  l_do_fourier_here = .true.
 endif
 if (.not.l_do_FFT_in_inner_CG) then
   if( l_DO_CG_CTRL_DIP.and..not.l_DO_CG_CTRL_Q) then
      l_skip_fourier_here = .true.
      l_do_fourier_here =.false.
   else
      l_skip_fourier_here = .false.
      l_do_fourier_here   =  .true.
   endif
 endif 
!l_do_fourier_here=.false.
 call initialize_arrays
 call re_assign_history
 call set_local_lists
 if (i_boundary_CTRL==1) then
   if (i_type_EWALD_CTRL == 1) then ! FAST
      call initialize_fourier_part_k0_2D
      if (l_do_fourier_here) call initialize_fourier_part_k_NON_0_2D
   endif
 else
   if (i_type_EWALD_CTRL == 1) then ! FAST
      if (l_do_fourier_here) call initialize_fourier_part_3D
   endif
 endif
 if (i_type_EWALD_CTRL /=1.and.(l_do_fourier_here)) call initialize_fourier_SLOW

 CONTAINS  
 subroutine initialize_arrays
 integer N
     allocate(q(NVFC))
     allocate(BB0(NVFC),BB(NVFC))
     N= NV   !N_atoms_variables
     allocate(GG_0(N,MX_list_nonbonded),GG_1(N,MX_list_nonbonded),GG_2(N,MX_list_nonbonded))
     if (N_pairs_14>0)&
     allocate(GG_0_14(N,MX_in_list_14),GG_1_14(N,MX_in_list_14),GG_2_14(N,MX_in_list_14))
     allocate(GG_0_THOLE(N,MX_list_nonbonded),GG_1_THOLE(N,MX_list_nonbonded))
     allocate(list1(N,MX_list_nonbonded))
     allocate(size_list1(N)) ; 
     size_list1=0
     allocate(list2(N,MX_list_nonbonded))
     allocate(size_list2(N))
     size_list2=0
     if (N_pairs_sfc_sfc_123>0)then
      allocate(size_list1_sfc_sfc(N))
      allocate(list1_sfc_sfc(N,MX_excluded))
      size_list1_sfc_sfc=0
      allocate(size_list2_sfc_sfc(N))
      allocate(list2_sfc_sfc(N,MX_excluded))
      size_list2_sfc_sfc=0
     endif
     
 end subroutine initialize_arrays

 subroutine initialize_fourier_part_k0_2D

      order = order_spline_zz_k0
      Ngrid = n_grid_zz_k0
      N_size_qq = N_grid_zz_k0+order_spline_zz_k0+1
      allocate(qq(1:N_size_qq), bv(1:N_grid_zz_k0+1,1:2),z_grid(1:N_grid_zz_k0))
      call get_z_grid(order_spline_zz_k0,N_grid_zz_k0,z_grid)
      call get_qq_coordinate(order_spline_zz_k0,N_grid_zz_k0,z_grid,qq)

 end subroutine initialize_fourier_part_k0_2D

 subroutine initialize_fourier_part_k_NON_0_2D
    integer ix,iy,iz,mx,my,mz,ix1,iy1,iz1,mx1,my1,mz1,nx0,ny0,nz0,jx,jy,jz,jx1,jy1,jz1,ooo,I_INDEX
    integer nx,ny,nz
    real(8) spline_product,exp_fct,vterm,pref,expfct,i4a2_2,i4a2
    real(8) d2,i_d2,rec_xx,rec_xy,rec_yx,rec_yy,rec_zz, z
    real(8) kx,ky,kz,KR
    integer i_adress
    real(8) icel(9),temp,axx,axy,ayx,ayy
    real(8) cox(order_spline_xx),coy(order_spline_yy),coz(order_spline_zz)
    real(8) cox_DD(order_spline_xx),coy_DD(order_spline_yy),coz_DD(order_spline_zz)
    real(8) tmpz,tmpdz,tmp_y_z,tmp_y_dz,tmp_dy_z,q_term,t_x,t_y,t_z
   

!     call invert3(sim_cel,icel,temp) ; i_sim_zz = icel(9)
!     axx= Inverse_cel(1) ; axy = Inverse_cel(2) 
!     ayx= Inverse_cel(4) ; ayy = Inverse_cel(5)

    call get_reciprocal_cut
    reciprocal_cut_sq = reciprocal_cut*reciprocal_cut
    ooo = order_spline_zz*order_spline_yy*order_spline_xx
    allocate(Ih_on_grid(Natoms,ooo),Ih_on_grid_dx(Natoms,ooo),Ih_on_grid_dy(Natoms,ooo),Ih_on_grid_dz(Natoms,ooo))
    allocate(NNX(Natoms),NNY(Natoms),NNZ(Natoms))
    nx0 = nfftx/2; ny0 = nffty/2 ; nz0 = nfftz/2
     k_vct = 0
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
             kz = rec_zz
             kx = rec_xx + rec_xy
             ky = rec_yx + rec_yy
             d2 = kx*kx + ky*ky + kz*kz
             if (jy1**2+jx1**2 > 0 .and. d2 < reciprocal_cut_sq) then
               k_vct = k_vct + 1
    ! need the k_vector
             endif     !  reciprocal_cutt within cut off
         enddo
    enddo
    enddo
    N_K_VCT = k_vct

    allocate(pre(N_K_VCT),k_vector(n_k_VCT,3))
     i4a2 = 0.25d0/(Ewald_alpha*Ewald_alpha)
     expfct = - i4a2
     i4a2_2 = 2.0d0 * i4a2

     k_vct = 0
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
          kz = rec_zz
          kx = rec_xx + rec_xy
          ky = rec_yx + rec_yy
          d2 = kx*kx + ky*ky + kz*kz
          if (jy1**2+jx1**2 > 0 .and. d2 < reciprocal_cut_sq) then
             k_vct = k_vct + 1
             i_d2 = 1.0d0/d2
             exp_fct = dexp(expfct*d2) * i_d2
             spline_product = spline2_CMPLX_xx(jx)*spline2_CMPLX_yy(jy)*spline2_CMPLX_zz(mz)
             vterm =  exp_fct / (Area_xy*spline_product) * reciprocal_zz
             pre(k_vct) = vterm
             k_vector(k_vct,1) = kx ; k_vector(k_vct,2) = ky ; k_vector(k_vct,3) = kz
          endif     !  reciprocal_cutt within cut off
        enddo
    enddo
    enddo

    allocate (spline2_REAL_pp_x(natoms,order_spline_xx),&
     spline2_REAL_pp_y(Natoms,order_spline_yy),&
      spline2_REAL_pp_z(Natoms,order_spline_zz),&
      spline2_REAL_dd_x(natoms,order_spline_xx),&
      spline2_REAL_dd_y(Natoms,order_spline_yy),&
      spline2_REAL_dd_z(Natoms,order_spline_zz))

    call get_reduced_coordinates
    do i = 1,Natoms
      z = zzz(i)
      nx = int(tx(i)) - order_spline_xx
      ny = int(ty(i)) - order_spline_yy
      nz = int(z) - order_spline_zz
      NNX(i) = nx
      NNY(i) = ny
      NNZ(i) = nz
    enddo
    call beta_spline_pp_dd

! Store Ih_on_grid
    do i = 1,Natoms
      cox(1:order_spline_xx) = spline2_REAL_pp_x(i,1:order_spline_xx)
      coy(1:order_spline_yy) = spline2_REAL_pp_y(i,1:order_spline_yy)
      coz(1:order_spline_zz) = spline2_REAL_pp_z(i,1:order_spline_zz)
      cox_DD(1:order_spline_xx) = spline2_REAL_dd_x(i,1:order_spline_xx)
      coy_DD(1:order_spline_yy) = spline2_REAL_dd_y(i,1:order_spline_yy)
      coz_DD(1:order_spline_zz) = spline2_REAL_dd_z(i,1:order_spline_zz)

      nx = NNX(i)  ! n = 0..nfftx-1-splines_xx
      ny = NNY(i)
      nz = NNZ(i)
      iz = nz
      I_INDEX = 0 
      do jz = 0, order_spline_zz-1
      iz = iz + 1
      kz = iz +  h_cut_z ! h_cut_z = nfftz/2
      if (kz >= nfftz) then  ! it cannot be nfftz
        write(6,*) 'error in set_q_2D kz >= nfftz; choose more nfftz points'
        write(6,*) 'You need to make nfftz at least ',int(sim_cel(9)) , &
        'or the first 2^N integer'
        write(6,*) 'kz boxz nfftz=',kz, sim_cel(9), nfftz
      STOP
      endif
      if (kz < 0) then
        write(6,*) 'error in set_q_2D kz < 0 : lower the splines order or increase the nfft',kz
        write(6,*) 'order spline = ',order_spline_xx,order_spline_yy,order_spline_zz
        write(6,*) 'nfft hcutz =',nfftx,nffty,nfftz,h_cut_z
        STOP
      endif
      tmpz =   coz(jz+1) ! iz = jz + 1
      tmpdz =  coz_DD(jz+1)
      iy = ny
      do jy = 0, order_spline_yy-1
        iy = iy + 1
        if (iy < 0) then
          ky = iy + nffty
        else
          ky = iy
        endif
        tmp_y_z  = coy(jy+1)     * tmpz
        tmp_dy_z = coy_DD(jy+1)  * tmpz;
        tmp_y_dz = coy(jy+1)     * tmpdz;
        ix = nx
        do jx = 0, order_spline_xx-1
          ix = ix + 1
          I_INDEX = I_INDEX + 1
          if (ix < 0) then
             kx = ix + nfftx
          else
             kx = ix
          endif
          i_adress = (ky+kz*nffty)*nfftx + kx + 1
!print*,i,jx,jy,jz,'k=',kx,ky,kz,i_adress
!read(*,*)

          t_x= cox_DD(jx+1) * tmp_y_z   * dfftx
          t_y= cox(jx+1)    * tmp_dy_z  * dffty
          t_z= cox(jx+1)    * tmp_y_dz  !* dble(nfftz)
          q_term = tmp_y_z*cox(jx+1)
          Ih_on_grid(i,I_INDEX) = q_term
          Ih_on_grid_dx(i,I_INDEX) = t_x
          Ih_on_grid_dy(i,I_INDEX) = t_y
          Ih_on_grid_dz(i,I_INDEX) = t_z

        enddo ! jx
        enddo ! jy
      enddo ! jz
    enddo ! i
  

 end subroutine initialize_fourier_part_k_NON_0_2D

!---
 subroutine initialize_fourier_part_3D
    use math_constants, only : Pi,Pi2
    integer ix,iy,iz,mx,my,mz,ix1,iy1,iz1,mx1,my1,mz1,nx0,ny0,nz0,jx,jy,jz,jx1,jy1,jz1,ooo,I_INDEX
    integer ii_xx,ii_yy,ii_zz
    integer nx,ny,nz
    real(8) spline_product,exp_fct,vterm,pref,expfct,i4a2_2,i4a2
    real(8) d2,i_d2,rec_xx,rec_xy,rec_yx,rec_yy,rec_zz,rec_xz,rec_zx,rec_yz,rec_zy , z
    real(8) kx,ky,kz,KR
    integer i_adress
    real(8) cox(order_spline_xx),coy(order_spline_yy),coz(order_spline_zz)
    real(8) cox_DD(order_spline_xx),coy_DD(order_spline_yy),coz_DD(order_spline_zz)
    real(8) tmp_x,tmp_y,tmp_z,tmp_y_z,tmp_y_dz,tmp_dy_z,q_term,t_x,t_y,t_z
    real(8) tmp_dx,tmp_dy,tmp_dz
     
    call get_reciprocal_cut
    reciprocal_cut_sq = reciprocal_cut*reciprocal_cut
    ooo = order_spline_zz*order_spline_yy*order_spline_xx
    allocate(Ih_on_grid(Natoms,ooo),Ih_on_grid_dx(Natoms,ooo),Ih_on_grid_dy(Natoms,ooo),Ih_on_grid_dz(Natoms,ooo))
    allocate(NNX(Natoms),NNY(Natoms),NNZ(Natoms))
    nx0 = nfftx/2; ny0 = nffty/2 ; nz0 = nfftz/2
    k_vct = 0
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
          do jx = 1, nfftx
           jx1 = jx -1
            if (jx > nx0) jx1 = jx1 - nfftx
            tmp = dble(jx1)
            rec_xx = tmp*Reciprocal_cel(1)
            rec_yx = tmp*Reciprocal_cel(4)
            rec_zx = tmp*Reciprocal_cel(7)
            kz = rec_zx + rec_zy + rec_zz
            kx = rec_xx + rec_xy + rec_xz
            ky = rec_yx + rec_yy + rec_yz
            d2 = kx*kx + ky*ky + kz*kz
 if (d2 < reciprocal_cut_sq.and.jz1**2+jy1**2+jx1**2 /= 0) then
             k_vct = k_vct + 1
 endif
          enddo
     enddo
    enddo
    N_K_VCT = k_vct
    allocate(pre(N_K_VCT),k_vector(n_k_VCT,3))
    i4a2 = 0.25d0/(Ewald_alpha*Ewald_alpha)
    expfct = - i4a2
    i4a2_2 = 2.0d0 * i4a2

     k_vct = 0
     do jz = 1,nfftz
       jz1 = jz-1
       if (jz > nz0) jz1 = jz1 - nfftz
       tmp = dble(jz1)
       rec_xz = tmp*Reciprocal_cel(3)
       rec_yz = tmp*Reciprocal_cel(6)
       rec_zz = tmp*Reciprocal_cel(9)      
       tmp_z = spline2_CMPLX_zz(jz)
       do jy = 1,nffty
          jy1 = jy - 1
          if (jy > ny0) jy1 = jy1 - nffty
          tmp = dble(jy1)
          rec_xy = tmp*Reciprocal_cel(2)
          rec_yy = tmp*Reciprocal_cel(5)
          rec_zy = tmp*Reciprocal_cel(8)
          tmp_y_z = spline2_CMPLX_yy(jy) * tmp_z
          do jx = 1, nfftx
           jx1 = jx -1
            if (jx > nx0) jx1 = jx1 - nfftx
            tmp = dble(jx1)
            rec_xx = tmp*Reciprocal_cel(1)
            rec_yx = tmp*Reciprocal_cel(4)
            rec_zx = tmp*Reciprocal_cel(7)
            kz = rec_zx + rec_zy + rec_zz
            kx = rec_xx + rec_xy + rec_xz
            ky = rec_yx + rec_yy + rec_yz
            d2 = kx*kx + ky*ky + kz*kz
 if (d2 < reciprocal_cut_sq.and.jz1**2+jy1**2+jx1**2 /= 0) then
             k_vct = k_vct + 1
             i_d2 = 1.0d0/d2
             exp_fct = dexp(expfct*d2) * i_d2
             spline_product = spline2_CMPLX_xx(jx) * tmp_y_z
             vterm =  exp_fct / (Volume*spline_product) * Pi2
             pre(k_vct) = vterm
             k_vector(k_vct,1) = kx ; k_vector(k_vct,2) = ky ; k_vector(k_vct,3) = kz
 endif
          enddo
     enddo
    enddo

    allocate (spline2_REAL_pp_x(natoms,order_spline_xx),&
     spline2_REAL_pp_y(Natoms,order_spline_yy),&
      spline2_REAL_pp_z(Natoms,order_spline_zz),&
      spline2_REAL_dd_x(natoms,order_spline_xx),&
      spline2_REAL_dd_y(Natoms,order_spline_yy),&
      spline2_REAL_dd_z(Natoms,order_spline_zz))

    call get_reduced_coordinates
    do i = 1,Natoms
      z = zzz(i)
      nx = int(tx(i)) - order_spline_xx
      ny = int(ty(i)) - order_spline_yy
      nz = int(tz(i)) - order_spline_zz
      NNX(i) = nx
      NNY(i) = ny
      NNZ(i) = nz
    enddo
    call beta_spline_pp_dd

    do i = 1,Natoms
      cox(1:order_spline_xx) = spline2_REAL_pp_x(i,1:order_spline_xx)
      coy(1:order_spline_yy) = spline2_REAL_pp_y(i,1:order_spline_yy)
      coz(1:order_spline_zz) = spline2_REAL_pp_z(i,1:order_spline_zz)
      cox_DD(1:order_spline_xx) = spline2_REAL_dd_x(i,1:order_spline_xx)
      coy_DD(1:order_spline_yy) = spline2_REAL_dd_y(i,1:order_spline_yy)
      coz_DD(1:order_spline_zz) = spline2_REAL_dd_z(i,1:order_spline_zz)

      nx = NNX(i)  ! n = 0..nfftx-1-splines_xx
      ny = NNY(i)
      nz = NNZ(i)
      ii_zz = nz
      I_INDEX = 0

        do iz=1,order_spline_zz
          tmp_z =   coz(iz)
          tmp_dz =  coz_DD(iz)
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
            tmp_y =  coy(iy)
            tmp_dy = coy_DD(iy)
            tmp_y_z  = tmp_y     * tmp_z;
            tmp_dy_z = tmp_dy * tmp_z;
            tmp_y_dz = tmp_y     * tmp_dz;

            ii_xx = nx
            do ix=1,order_spline_xx
              ii_xx = ii_xx + 1
              I_INDEX = I_INDEX + 1
              if (ii_xx < 0 ) then
                kx = ii_xx + nfftx +1
              else
                kx = ii_xx +1
              endif
              mx = kx !+ nfftx/2
              if (mx > nfftx) mx = mx - nfftx
                tmp_x    = cox(ix)
                tmp_dx   = cox_DD(ix)
                t_x= tmp_dx   * tmp_y_z   * dfftx
                t_y= tmp_x    * tmp_dy_z  * dffty
                t_z= tmp_x    * tmp_y_dz  * dfftz
                q_term = tmp_x * tmp_y* tmp_z
                Ih_on_grid(i,I_INDEX) = q_term
                Ih_on_grid_dx(i,I_INDEX) = t_x
                Ih_on_grid_dy(i,I_INDEX) = t_y
                Ih_on_grid_dz(i,I_INDEX) = t_z
            enddo
          enddo
        enddo

      enddo !i
 end subroutine initialize_fourier_part_3D
!----

 subroutine initialize_fourier_SLOW
 use cg_buffer, only : sns,css
 real(8) tmp,rec_xx,rec_xy,rec_xz,rec_yx,rec_yy,rec_yz,rec_zx,rec_zy,rec_zz
 real(8) kx,ky,kz,d2
 integer i,j,k,ix,iy,iz ,k_vct
 call get_reciprocal_cut
 if (i_boundary_CTRL == 1) then  ! 2D
    h_cut_off = h_cut_off2D *  Reciprocal_cel(9)
    h_step = h_cut_off/dble(K_MAX_Z)

    k_vct = 0
    do ix = -K_MAX_X,K_MAX_X
      tmp = dble(ix)
      rec_xx = tmp*Reciprocal_cel(1)
      rec_yx = tmp*Reciprocal_cel(4)
      do iy = -K_MAX_Y,K_MAX_Y
        tmp = dble(iy)
        rec_xy = tmp*Reciprocal_cel(2)
        rec_yy = tmp*Reciprocal_cel(5)
        if (ix**2 + iy**2 /= 0) then
        do iz = -K_MAX_Z,K_MAX_Z
          rec_zz = dble(iz) * h_step
          kz = rec_zz
          kx = rec_xx + rec_xy
          ky = rec_yx + rec_yy
          d2 = kx*kx + ky*ky + kz*kz
          if (d2 < reciprocal_cut_sq) then
             k_vct = k_vct + 1
          endif
        enddo
        endif
      enddo
     enddo
     N_k_vct = k_vct
 else   ! 3D SLOW I do not use symety for 3D case
    k_vct = 0
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
        do iz = -K_MAX_Z,K_MAX_Z
        if (ix*ix+iy*iy+iz*iz > 0) then
          tmp = dble(iz)
          rec_xz = tmp*Reciprocal_cel(3)
          rec_yz = tmp*Reciprocal_cel(6)
          rec_zz = tmp*Reciprocal_cel(9) 
          kx = rec_xx + rec_xy + rec_xz
          ky = rec_yx + rec_yy + rec_yz
          kz = rec_zx + rec_zy + rec_zz
          d2 = kx*kx + ky*ky + kz*kz
          if (d2 < reciprocal_cut_sq) then
             k_vct = k_vct + 1
          endif
         endif ! ix**2+iy**2+iz**2 > 0
        enddo
      enddo
     enddo
     N_k_vct = k_vct
 endif
 allocate(sns(Natoms,k_vct))
 allocate(css(Natoms,k_vct))
 allocate(pre(k_vct))
 allocate(k_vector(k_vct,3))

 end subroutine initialize_fourier_SLOW
!print*, 'exit initialize' 
end subroutine initialize

subroutine re_assign_history
 use cg_buffer, only : cg_predict_restart_CTRL,order_lsq_cg_predictor,MAT_lsq_cp_predictor,&
                       var_history,q, aspc_coars_Niters, aspc_update_4full_iters
 use integrate_data, only : integration_step
 use all_atoms_data, only : all_charges, all_dipoles_xx,all_dipoles_yy,all_dipoles_zz
 use field_constrain_data, only : ndx_remap
 use array_math, only : invmat
 
 implicit none
 integer i,j,k,i1,j1,i_adress
 real(8), allocatable :: XX_cg(:)
  if (cg_predict_restart_CTRL == -1) then ! it is simply zero 
     q(:) = 0.0d0
  else if (cg_predict_restart_CTRL == 0) then   ! take the last iteration

    select case (integration_step)
     case(1)
      var_history(1:TAG_SP,1) = all_charges(ndx_remap%var(1:TAG_SP))
            do i1 = TAG_SS+1,TAG_PP
               i_adress = i1 - TAG_SS + TAG_SP
               j = ndx_remap%var(i1)
               var_history(i_adress      ,1) = all_dipoles_xx(j)
               var_history(i_adress+  NDP,1) = all_dipoles_yy(j)
               var_history(i_adress+2*NDP,1) = all_dipoles_zz(j)
            enddo
            q(:) = var_history(:,1)
      case default
            q(:) = var_history(:,1)
    end select

  else if (cg_predict_restart_CTRL == 1) then

   select case (integration_step)
    case(1)
      var_history(1:TAG_SP,1) = all_charges(ndx_remap%var(1:TAG_SP))
      do i1 = TAG_SS+1,TAG_PP
        i_adress = i1 - TAG_SS + TAG_SP
        j = ndx_remap%var(i1)
        var_history(i_adress      ,1) = all_dipoles_xx(j)
        var_history(i_adress+  NDP,1) = all_dipoles_yy(j)
        var_history(i_adress+2*NDP,1) = all_dipoles_zz(j)
      enddo
       q(:) = var_history(:,1)
    case(2,3,4)
       q(:) = var_history(:,integration_step-1)
    case default  ! >= 3
       q(:) = 5.0d0*(var_history(:,1)-var_history(:,4))+10.0d0*(var_history(:,3)-var_history(:,2))+var_history(:,5)
    end select

  else if (cg_predict_restart_CTRL == 2) then
    allocate(XX_cg(order_lsq_cg_predictor))
      if (integration_step == 1) then
      var_history(1:TAG_SP,1) = all_charges(ndx_remap%var(1:TAG_SP))
      do i1 = TAG_SS+1,TAG_PP
         i_adress = i1 - TAG_SS + TAG_SP
         j = ndx_remap%var(i1)
         var_history(i_adress      ,1) = all_dipoles_xx(j)
         var_history(i_adress+  NDP,1) = all_dipoles_yy(j)
         var_history(i_adress+2*NDP,1) = all_dipoles_zz(j)
      enddo
      q(:) = var_history(:,1)
      else if (integration_step >1 .and. integration_step<order_lsq_cg_predictor+1) then
      q(:) = var_history(:,integration_step)
      else ! integration_step >=order_lsq_cg_predictor+1
       MAT_cg = MAT_lsq_cp_predictor
      call invmat(MAT_cg,order_lsq_cg_predictor,order_lsq_cg_predictor)
      do i = 2, order_lsq_cg_predictor+1
        BB_cg(i-1) = dot_product(var_history(:,i),var_history(:,1))
      enddo
      do i = 1, order_lsq_cg_predictor
        XX_cg(i) = dot_product(MAT_cg(i,:),BB_cg(:))
      enddo
      do i = 1, NAFC
        q(i) = dot_product(XX_cg(1:order_lsq_cg_predictor),var_history(i,2:order_lsq_cg_predictor+1))
      enddo
      MAT_cg = MAT_lsq_cp_predictor
      do i = 1, order_lsq_cg_predictor-1
      do j = 1, order_lsq_cg_predictor-1
       MAT_lsq_cp_predictor(i+1,j+1) = MAT_cg(i,j)
      enddo
      enddo
   deallocate(XX_cg)
   endif

  else if (cg_predict_restart_CTRL == 3) then ! ALWAYS STABLE PREDICTOR CORRECTOR

     if (integration_step <= 6) then
       var_history(1:TAG_SP,6-integration_step) = all_charges(ndx_remap%var(1:TAG_SP))
       do i1 = TAG_SS+1,TAG_PP
         i_adress = i1 - TAG_SS + TAG_SP
         j = ndx_remap%var(i1)
         var_history(i_adress      ,6-integration_step) = all_dipoles_xx(j)
         var_history(i_adress+  NDP,6-integration_step) = all_dipoles_yy(j)
         var_history(i_adress+2*NDP,6-integration_step) = all_dipoles_zz(j)
      enddo
       q(:) = var_history(:,6-integration_step)
       N_iterations = 200;
     else
       q(:) = (22.0d0/7.0d0)*var_history(:,0) + (-55.0d0/14.0d0)*var_history(:,1) + &
          (55.0d0/21.0d0)*var_history(:,2) + (-22.0d0/21.0d0)*var_history(:,3) + &
          (5.0d0/21.0d0)*var_history(:,4) + (-1.0d0/42.0d0)*var_history(:,5)
       X_predictor(:) = q(:)
       if (mod(integration_step, aspc_update_4full_iters)==0) then
          N_iterations = 200
       else
          N_iterations = aspc_coars_Niters
       endif
     endif
 
  else ! NO cg_predict_restart_CTRL defined

    print*, 'NO cg_predict_restart_CTRL defined in range in cg*%init',cg_predict_restart_CTRL ; STOP

  endif
end subroutine re_assign_history

 subroutine set_local_lists
 use cut_off_data, only : cut_off_sq
 use boundaries, only : periodic_images
 use field_constrain_data, only : ndx_remap, rec_ndx_remap
 use cg_buffer, only : list1,size_list1,list2,size_list2,list1_sfc_sfc,size_list1_sfc_sfc,&
                       list2_sfc_sfc,size_list2_sfc_sfc
 use max_sizes_data, only : MX_list_nonbonded,MX_excluded
 use ALL_atoms_data, only : Natoms, xxx,yyy,zzz
 use non_bonded_lists_data, only : list_nonbonded,size_list_nonbonded
 use connectivity_ALL_data, only : size_list_excluded_sfc_iANDj_HALF,list_excluded_sfc_iANDj_HALF
 use sizes_data, only : N_pairs_sfc_sfc_123
  implicit none
  integer, allocatable :: in_list(:)
  real(8), allocatable :: dx(:),dy(:),dz(:),dr_sq(:)
  integer i,j,k,i1,j1,neightot,N,ii,neightot1,neightot2
  N=MX_list_nonbonded
  allocate(in_list(N))
  allocate(dx(N),dy(N),dz(N),dr_sq(N))
  

    do i = 1, Natoms
    i1 = rec_ndx_remap%var(i)
    neightot1 = size_list_nonbonded(i)
    ii = 0
    do k = 1, neightot1
       ii = ii + 1
       j = list_nonbonded(i,k)
       dx(ii) = xxx(i) - xxx(j)
       dy(ii) = yyy(i) - yyy(j)
       dz(ii) = zzz(i) - zzz(j)
       in_list(ii) = j
    enddo
    neightot = neightot1 
    if (neightot > 0 ) then
       call periodic_images(dx(1:neightot),dy(1:neightot),dz(1:neightot))
       dr_sq(1:neightot) = dx(1:neightot)*dx(1:neightot) + dy(1:neightot)*dy(1:neightot) +&
                           dz(1:neightot)*dz(1:neightot)
    endif
    do k = 1, neightot
    if (dr_sq(k) < cut_off_sq) then
      j = in_list(k)
      j1 = rec_ndx_remap%var(j)
      if (i1 < NV+1.and.j1>TAG_SP) then
                size_list2(i1) = size_list2(i1) + 1
                list2(i1, size_list2(i1)) = j
      endif
      if (i1 > TAG_SP.and.j1<NV+1) then
                size_list2(j1) = size_list2(j1) + 1
                list2(j1, size_list2(j1)) = i
      endif
      if (i1 < NV+1.and.j1<NV+1) then
                size_list1(i1) = size_list1(i1) + 1
                list1(i1,size_list1(i1)) = j
     endif
     endif
    enddo ! k = 1, size_list_nonbonded(i)
    enddo  ! i
 deallocate(in_list)
 deallocate(dx,dy,dz,dr_sq)

! And the list sfc-sfc
    if (N_pairs_sfc_sfc_123>0)then
    N=MX_excluded
    allocate(in_list(N))
    allocate(dx(N),dy(N),dz(N),dr_sq(N))
    do i = 1, Natoms
    i1 = rec_ndx_remap%var(i)
    neightot1 = size_list_excluded_sfc_iANDj_HALF(i)
    ii = 0
    do k = 1, neightot1
       ii = ii + 1
       j = list_excluded_sfc_iANDj_HALF(i,k)
       dx(ii) = xxx(i) - xxx(j)
       dy(ii) = yyy(i) - yyy(j)
       dz(ii) = zzz(i) - zzz(j)
       in_list(ii) = j
    enddo
    neightot = neightot1
    if (neightot > 0 ) then
       call periodic_images(dx(1:neightot),dy(1:neightot),dz(1:neightot))
       dr_sq(1:neightot) = dx(1:neightot)*dx(1:neightot) + dy(1:neightot)*dy(1:neightot) +&
                           dz(1:neightot)*dz(1:neightot)
    endif
    do k = 1, neightot
    if (dr_sq(k) < cut_off_sq) then
      j = in_list(k)
      j1 = rec_ndx_remap%var(j)
      if (i1 < NV+1.and.j1>TAG_SP) then
                size_list2_sfc_sfc(i1) = size_list2_sfc_sfc(i1) + 1
                list2_sfc_sfc(i1, size_list2_sfc_sfc(i1)) = j
      endif
      if (i1 > TAG_SP.and.j1<NV+1) then
                size_list2_sfc_sfc(j1) = size_list2_sfc_sfc(j1) + 1
                list2_sfc_sfc(j1, size_list2_sfc_sfc(j1)) = i
      endif
      if (i1 < NV+1.and.j1<NV+1) then
                size_list1_sfc_sfc(i1) = size_list1_sfc_sfc(i1) + 1
                list1_sfc_sfc(i1,size_list1_sfc_sfc(i1)) = j
     endif
     else ! (dr_sq(k) < cut_off_sq)
        print*, 'ERROR in cg_Q_DIP when getting list2_sfc_sfc; dr_sq(k) < cut_off_sq'
        print*, 'The system is exploding or something ....' 
        STOP
     endif ! (dr_sq(k) < cut_off_sq)
    enddo ! k = 1, size_list_nonbonded(i)
    enddo  ! i
    deallocate(in_list)
    deallocate(dx,dy,dz,dr_sq)
    endif ! N_pairs_sfc_sfc_123>0

 end subroutine set_local_lists


subroutine set_GG_arrays
use boundaries, only : periodic_images
use Ewald_data, only : Ewald_alpha
use atom_type_data, only : which_atomStyle_pair
use max_sizes_data, only : MX_excluded,MX_list_nonbonded,MX_in_list_14
use sizes_data, only :  Natoms,N_pairs_14
use interpolate_data, only : MX_interpol_points,N_STYLE_ATOMS, vele_G,gele_G,vele2_G,RDR , iRDR, &
                             vele_THOLE,gele_THOLE,vele,gele,vele2
use cut_off_data
use field_constrain_data, only : ndx_remap,rec_ndx_remap
use cg_buffer, only : list1,size_list1,list1_ex,size_list1_ex, GG_0,GG_1,GG_2,&
                      GG_0_excluded,GG_1_excluded,GG_2_excluded, GG_0_THOLE,GG_1_THOLE,&
                      GG_0_14,GG_1_14,GG_2_14,list1_14,size_list1_14,list2_14,size_list2_14
use ALL_atoms_data, only : xxx,yyy,zzz, i_Style_atom

implicit none
real(8), allocatable :: dx(:),dy(:),dz(:),dr_sq(:)
integer, allocatable :: in_list(:)
real(8) r,r2,Inverse_r,Inverse_r2,Inverse_r3,Inverse_r5,ppp,En0,field ,t1,t2,vk1,vk2,vk
integer ndx,i_pair,i_type,j_type,neightot,i1,j1,i,j,N,k
real(8)  coef, EEE,x_x_x,local_cut
real(8) B0,B1,B2,B3, B0_THOLE, B1_THOLE,B00,B11,B22

local_cut = cut_off+displacement

N=MX_list_nonbonded
allocate(dx(N),dy(N),dz(N),dr_sq(N)) 
allocate(in_list(N))
do i1 = 1, NV
   i = ndx_remap%var(i1)
   i_type = i_Style_atom(i)
   neightot =  size_list1(i1)
   do k = 1, neightot
     j =  list1(i1,k)
     dx(k) = xxx(i) - xxx(j)
     dy(k) = yyy(i) - yyy(j)
     dz(k) = zzz(i) - zzz(j)
     in_list(k) = j
   enddo
   if (neightot > 0 ) then
     call periodic_images(dx(1:neightot),dy(1:neightot),dz(1:neightot))
     dr_sq(1:neightot) = dx(1:neightot)*dx(1:neightot) + dy(1:neightot)*dy(1:neightot) +&
                         dz(1:neightot)*dz(1:neightot)
   endif
   do k = 1,neightot ! l_j is now always true (see before )
     j = in_list(k)
     j_type = i_Style_atom(j)
     r2 = dr_sq(k)
     if ( r2 < cut_off_sq ) then
       r = dsqrt(r2)
       Inverse_r = 1.0d0/r
       i_pair = which_atomStyle_pair(i_type,j_type)
       NDX = max(1,int(r*irdr))
       ppp = (r*irdr) - dble(ndx)
       include 'interpolate_3.frg'
       if (is_dip(i).and.is_dip(j)) then
           include 'interpolate_THOLE.frg'
       else
           B0_THOLE=0.0d0
           B1_THOLE=0.0d0
       endif
       GG_0(i1,k) = B0
       GG_1(i1,k) = B1
       GG_2(i1,k) = B2
       GG_0_THOLE(i1,k) = B0_THOLE
       GG_1_THOLE(i1,k) = B1_THOLE
!       fi(i1) = fi(i1) + En0*qj
    else
       GG_0(i1,k) = 0.0d0
       GG_1(i1,k) = 0.0d0
       GG_2(i1,k) = 0.0d0
       GG_0_THOLE(i1,k) =0.0d0
       GG_1_THOLE(i1,k) = 0.0d0 
    endif ! within cut-off
  enddo
enddo
deallocate(in_list)
deallocate(dx,dy,dz,dr_sq)
! DO THE INTRAMOLECULAR PART:
if (MX_excluded>0) then
allocate(dx(MX_excluded),dy(MX_excluded),dz(MX_excluded),dr_sq(MX_excluded))
allocate(in_list(MX_excluded))
allocate(GG_0_excluded(NV,MX_excluded),GG_1_excluded(NV,MX_excluded),GG_2_excluded(NV,MX_excluded))
do i1 = 1, NV
   i = ndx_remap%var(i1)
   neightot =  size_list1_ex(i1)
   do k = 1, neightot
     j =  list1_ex(i1,k)
     dx(k) = xxx(i) - xxx(j)
     dy(k) = yyy(i) - yyy(j)
     dz(k) = zzz(i) - zzz(j)
     in_list(k) = j
   enddo
   if (neightot>0) then
    call periodic_images(dx(1:neightot),dy(1:neightot),dz(1:neightot))
    dr_sq(1:neightot) = dx(1:neightot)*dx(1:neightot) + dy(1:neightot)*dy(1:neightot) +&
       dz(1:neightot)*dz(1:neightot)
   endif
   do k = 1,neightot ! l_j is now always true (see before )
     j = in_list(k)
     r2 = dr_sq(k)
     r = dsqrt(r2)
     if (r<local_cut)then
       NDX = max(1,int(r*irdr))
       ppp = (r*irdr) - dble(ndx)
            vk  = vele(ndx)  ;  vk1 = vele(ndx+1) ; vk2 = vele(ndx+2)
            t1 = vk  + (vk1 - vk )*ppp
            t2 = vk1 + (vk2 - vk1)*(ppp - 1.0d0)
            B0 = (t1 + (t2-t1)*(ppp*0.5d0))
            vk  = gele(ndx)  ;  vk1 = gele(ndx+1) ; vk2 = gele(ndx+2)
            t1 = vk  + (vk1 - vk )*ppp
            t2 = vk1 + (vk2 - vk1)*(ppp - 1.0d0)
            B1 = (t1 + (t2-t1)*(ppp*0.5d0))
            vk  = vele2(ndx)  ;  vk1 = vele2(ndx+1) ; vk2 = vele2(ndx+2)
            t1 = vk  + (vk1 - vk )*ppp
            t2 = vk1 + (vk2 - vk1)*(ppp - 1.0d0)
            B2 = (t1 + (t2-t1)*(ppp*0.5d0))
       Inverse_r = 1.0d0/r
       Inverse_r2 = Inverse_r*Inverse_r
       Inverse_r3 = Inverse_r*Inverse_r2
       Inverse_r5 = Inverse_r2*Inverse_r3
!       x_x_x = Ewald_alpha * r
!       EEE = dexp(-x_x_x*x_x_x)
!       B0 = derfc(x_x_x)*Inverse_r
!       B1 = (B0 + ratio_B1*EEE) * Inverse_r2
!       B2 = (3.0d0*B1 + ratio_B2*EEE) * Inverse_r2
       GG_0_excluded(i1,k) = Inverse_r - B0
       GG_1_excluded(i1,k) = Inverse_r3 - B1
       GG_2_excluded(i1,k) = 3.0d0*Inverse_r5-B2
    else ! outside cut
      print*,'Error in cg_Q_DIP%set_GG_array : The bond distance latger than cut-off; the system is exploding or something ...'
      print*,'Atoms =',i,j
      print*,'xyz i = ',xxx(i),yyy(i),zzz(i)
      print*,'xyz j = ',xxx(j),yyy(j),zzz(j)
      print*,'r=',r

      STOP
    endif 
  enddo
enddo
deallocate(in_list)
deallocate(dx,dy,dz,dr_sq)
endif ! MX_excluded > 0

! 1-4 part
if (N_pairs_14>0) then
allocate(dx(MX_in_list_14),dy(MX_in_list_14),dz(MX_in_list_14),dr_sq(MX_in_list_14))
allocate(in_list(MX_in_list_14))
do i1 = 1, NV
   i = ndx_remap%var(i1)
   i_type = i_Style_atom(i)
   neightot =  size_list1_14(i1)
   do k = 1, neightot
     j =  list1_14(i1,k)
     dx(k) = xxx(i) - xxx(j)
     dy(k) = yyy(i) - yyy(j)
     dz(k) = zzz(i) - zzz(j)
     in_list(k) = j
   enddo
   if (neightot>0) then
    call periodic_images(dx(1:neightot),dy(1:neightot),dz(1:neightot))
    dr_sq(1:neightot) = dx(1:neightot)*dx(1:neightot) + dy(1:neightot)*dy(1:neightot) +&
       dz(1:neightot)*dz(1:neightot)
   endif
   do k = 1,neightot ! l_j is now always true (see before )
     j = in_list(k)
     j_type = i_Style_atom(j)
     r2 = dr_sq(k)
     if ( r2 < cut_off_sq ) then
       r = dsqrt(r2)
       Inverse_r = 1.0d0/r
       Inverse_r2 = Inverse_r*Inverse_r
       inverse_r3 = Inverse_r*Inverse_r2
       inverse_r5 = Inverse_r2*inverse_r3
       i_pair = which_atomStyle_pair(i_type,j_type)
       NDX = max(1,int(r*irdr))
       ppp = (r*irdr) - dble(ndx)
       include 'interpolate_3.frg'
            vk  = vele(ndx)  ;  vk1 = vele(ndx+1) ; vk2 = vele(ndx+2)
            t1 = vk  + (vk1 - vk )*ppp
            t2 = vk1 + (vk2 - vk1)*(ppp - 1.0d0)
            B00 = (t1 + (t2-t1)*(ppp*0.5d0))
            vk  = gele(ndx)  ;  vk1 = gele(ndx+1) ; vk2 = gele(ndx+2)
            t1 = vk  + (vk1 - vk )*ppp
            t2 = vk1 + (vk2 - vk1)*(ppp - 1.0d0)
            B11 = (t1 + (t2-t1)*(ppp*0.5d0))
            vk  = vele2(ndx)  ;  vk1 = vele2(ndx+1) ; vk2 = vele2(ndx+2)
            t1 = vk  + (vk1 - vk )*ppp
            t2 = vk1 + (vk2 - vk1)*(ppp - 1.0d0)
            B22 = (t1 + (t2-t1)*(ppp*0.5d0))

       GG_0_14(i1,k) = Inverse_r - (B0-B00)
       GG_1_14(i1,k) = inverse_r3 - (B1-B11)
       GG_2_14(i1,k) = 3.0d0*inverse_r5 - (B2-B22)
!       fi(i1) = fi(i1) + En0*qj
    else
       GG_0_14(i1,k) = 0.0d0
       GG_1_14(i1,k) = 0.0d0
       GG_2_14(i1,k) = 0.0d0
    endif ! within cut-off
  enddo
enddo

endif 

end subroutine  set_GG_arrays
 
subroutine first_iter_free_term_REAL_SPACE
use sizes_data, only :  Natoms
use non_bonded_lists_data
use field_constrain_data
use interpolate_data, only : MX_interpol_points,N_STYLE_ATOMS, vele_G,gele_G,vele2_G,RDR , iRDR,&
                             vele_THOLE,gele_THOLE
use cut_off_data
use ALL_atoms_data, only : all_p_charges,all_g_charges, xx,yy,zz,&
   i_Style_atom, is_charge_distributed, all_charges,xxx,yyy,zzz,&
   all_dipoles,all_dipoles_xx,all_dipoles_yy,all_dipoles_zz
use cg_buffer, only : BB0,list2,size_list2
use boundaries, only : periodic_images
use Ewald_data
use atom_type_data, only : which_atomStyle_pair
use sim_cel_data
use math_constants, only : two_per_sqrt_Pi,sqrt_Pi,Pi,Pi2
use physical_constants, only : Red_Vacuum_EL_permitivity_4_Pi, Volt_to_internal_field
 implicit none
 integer i,iii,k,j,kkk,jjj,ndx,i_pair,i_type,j_type,neightot,i1,N,j1
 integer i_1, j_1, kk,i_adress,j_adress
 real(8) qj,r,x,y,z ,qi, B0i,G1,G2,B0,B1,B2,B0i_2_xx,B0i_2_yy,B0i_2_zz,B0i_1, B0_THOLE
 real(8) r2,Inverse_r,ppp,En0,field ,t1,t2,vk1,vk2,vk
 logical l_i,l_j,l_in,l_out
 real(8) fii,CC1,x_x_x,zij,CC,CC2
 real(8) derf_x, dexp_x2
 real(8) di_xx,di_yy,di_zz,dj_xx,dj_yy,dj_zz,p_r,p_r_i, K_P, K_R, coef
 logical lgg(6)
 real(8), allocatable :: dx(:),dy(:),dz(:),dr_sq(:)
 integer, allocatable :: in_list(:)

allocate(dx(MX_list_nonbonded),dy(MX_list_nonbonded),dz(MX_list_nonbonded),&
         dr_sq(MX_list_nonbonded))
allocate(in_list(MX_list_nonbonded))
do i1 = 1,  TAG_PP
   i = ndx_remap%var(i1)
   i_type = i_Style_atom(i)
   neightot = size_list2(i1)
!print*,i1,i,neightot
!read(*,*)
   do k = 1, neightot
     j = list2(i1,k)
     dx(k) = xxx(i) - xxx(j)
     dy(k) = yyy(i) - yyy(j)
     dz(k) = zzz(i) - zzz(j)
     in_list(k) = j
   enddo
   if (neightot > 0 ) then
   call periodic_images(dx(1:neightot),dy(1:neightot),dz(1:neightot))
   dr_sq(1:neightot) = dx(1:neightot)*dx(1:neightot) + dy(1:neightot)*dy(1:neightot) +&
       dz(1:neightot)*dz(1:neightot)
   endif
   B0i = 0.0d0
   B0i_1=0.0d0
   B0i_2_xx = 0.0d0
   B0i_2_yy = 0.0d0
   B0i_2_zz = 0.0d0
if (i1 < TAG_SS + 1) then
   
   do k = 1,neightot ! l_j is now always true (see before )
    j = in_list(k)
    r2 = dr_sq(k)
    if ( r2 < cut_off_sq ) then
       j_type = i_Style_atom(j)
       r = dsqrt(r2)
       Inverse_r = 1.0d0/r
       NDX = max(1,int(r*irdr))
       ppp = (r*irdr) - dble(ndx)
       i_pair = which_atomStyle_pair(i_type,j_type)
       qj = all_charges(j)
      vk  = vele_G(ndx,i_pair)  ;  vk1 = vele_G(ndx+1,i_pair) ; vk2 = vele_G(ndx+2,i_pair)
      t1 = vk  + (vk1 - vk )*ppp
      t2 = vk1 + (vk2 - vk1)*(ppp - 1.0d0)
      B0 = (t1 + (t2-t1)*(ppp*0.5d0))
       j1 = rec_ndx_remap%var(j)
       if (j1 <  TAG_PP + 1) then 
         B0i_1 = B0i_1 + B0*qj 
!         lgg(1)=.true.
       else 
      vk  = gele_G(ndx,i_pair)  ;  vk1 = gele_G(ndx+1,i_pair) ; vk2 = gele_G(ndx+2,i_pair)
      t1 = vk  + (vk1 - vk )*ppp
      t2 = vk1 + (vk2 - vk1)*(ppp - 1.0d0)
      B1 = (t1 + (t2-t1)*(ppp*0.5d0))
         p_r = all_dipoles_xx(j)*dx(k) + all_dipoles_yy(j)*dy(k)+all_dipoles_zz(j)*dz(k)
         B0i_1 = B0i_1 + B0*qj + B1*p_r
       endif
    endif ! within cut-off
  enddo
  BB0(i1) =  - B0i_1
elseif (i1 < TAG_SP+1) then ! i1 > TAG_SS
     do k = 1,neightot ! l_j is now always true (see before )
     j = in_list(k)
     r2 = dr_sq(k)
     if ( r2 < cut_off_sq ) then
       j_type = i_Style_atom(j)
       r = dsqrt(r2)
       Inverse_r = 1.0d0/r
       NDX = max(1,int(r*irdr))
       ppp = (r*irdr) - dble(ndx)
       i_pair = which_atomStyle_pair(i_type,j_type)
     qj = all_charges(j)
     include 'interpolate_2.frg'
     j1 = rec_ndx_remap%var(j)
     if (j1 <  TAG_PP + 1) then
       B0i_1 = B0i_1 + B0*qj
       B0i_2_xx = B0i_2_xx + (- qj*dx(k) )*B1 
       B0i_2_yy = B0i_2_yy + (- qj*dy(k) )*B1 
       B0i_2_zz = B0i_2_zz + (- qj*dz(k) )*B1 
     else
      vk  = vele2_G(ndx,i_pair)  ;  vk1 = vele2_G(ndx+1,i_pair) ; vk2 = vele2_G(ndx+2,i_pair)
      t1 = vk  + (vk1 - vk )*ppp
      t2 = vk1 + (vk2 - vk1)*(ppp - 1.0d0)
      B2 = (t1 + (t2-t1)*(ppp*0.5d0))
       dj_xx = all_dipoles_xx(j); dj_yy = all_dipoles_yy(j) ; dj_zz = all_dipoles_zz(j)
       p_r = dj_xx*dx(k) + dj_yy*dy(k)+dj_zz*dz(k)
       B0i_1 = B0i_1 + B0*qj + B1*p_r
       B0i_2_xx = B0i_2_xx + (- qj*dx(k) + dj_xx)*B1 - dx(k)*(p_r*B2)
       B0i_2_yy = B0i_2_yy + (- qj*dy(k) + dj_yy)*B1 - dy(k)*(p_r*B2)
       B0i_2_zz = B0i_2_zz + (- qj*dz(k) + dj_zz)*B1 - dz(k)*(p_r*B2)
     endif
    endif ! within cut-off
  enddo
 i_adress = TAG_SP+i1-TAG_SS
 BB0(i1) = - B0i_1
 BB0(      i_adress) = - B0i_2_xx
 BB0(  NDP+i_adress) = - B0i_2_yy
 BB0(2*NDP+i_adress) = - B0i_2_zz
!print*,'case 2 BB=',BB0(i1), BB0(      i_adress),BB0(  NDP+i_adress),BB0(2*NDP+i_adress)
else  ! i1 > TAG_SP (dipol only)
     do k = 1,neightot ! l_j is now always true (see before )
     j = in_list(k)
     r2 = dr_sq(k)
     if ( r2 < cut_off_sq ) then
       j_type = i_Style_atom(j)
       r = dsqrt(r2)
       Inverse_r = 1.0d0/r
       NDX = max(1,int(r*irdr))
       ppp = (r*irdr) - dble(ndx)
       i_pair = which_atomStyle_pair(i_type,j_type)
       qj = all_charges(j)
      vk  = gele_G(ndx,i_pair)  ;  vk1 = gele_G(ndx+1,i_pair) ; vk2 = gele_G(ndx+2,i_pair)
      t1 = vk  + (vk1 - vk )*ppp
      t2 = vk1 + (vk2 - vk1)*(ppp - 1.0d0)
      B1 = (t1 + (t2-t1)*(ppp*0.5d0))
       j1 = rec_ndx_remap%var(j)
       if (j1 <  TAG_PP + 1) then
         B0i_2_xx = B0i_2_xx + (- qj*dx(k) )*B1 
         B0i_2_yy = B0i_2_yy + (- qj*dy(k) )*B1 
         B0i_2_zz = B0i_2_zz + (- qj*dz(k) )*B1 
!write(14,*)i,j,B1,r
       else
      vk  = vele2_G(ndx,i_pair)  ;  vk1 = vele2_G(ndx+1,i_pair) ; vk2 = vele2_G(ndx+2,i_pair)
      t1 = vk  + (vk1 - vk )*ppp
      t2 = vk1 + (vk2 - vk1)*(ppp - 1.0d0)
      B2 = (t1 + (t2-t1)*(ppp*0.5d0))

         dj_xx = all_dipoles_xx(j); dj_yy = all_dipoles_yy(j) ; dj_zz = all_dipoles_zz(j)
         p_r = dj_xx*dx(k) + dj_yy*dy(k) + dj_zz*dz(k)
         B0i_2_xx = B0i_2_xx + (- qj*dx(k) + dj_xx)*B1 - dx(k)*(p_r*B2)
         B0i_2_yy = B0i_2_yy + (- qj*dy(k) + dj_yy)*B1 - dy(k)*(p_r*B2)
         B0i_2_zz = B0i_2_zz + (- qj*dz(k) + dj_zz)*B1 - dz(k)*(p_r*B2)
!write(14,*)i,j,B1,r
       endif
    endif ! within cut-off

  enddo
 i_adress = TAG_SP+i1-TAG_SS
 BB0(      i_adress) = - B0i_2_xx
 BB0(  NDP+i_adress) = - B0i_2_yy
 BB0(2*NDP+i_adress) = - B0i_2_zz
!print*, 'case 3: ',BB0(      i_adress),BB0(  NDP+i_adress), BB0(2*NDP+i_adress)
endif
enddo   ! i1
!do i=1,Natoms
!if (is_dipole_polarizable(i))then
!i1= rec_ndx_remap%var(i)
!i_adress = TAG_SP+i1-TAG_SS
!write(14,*),i,BB0(i_adress),BB0(NDP+i_adress),BB0(2*NDP+i_adress)
!endif
!enddo
!close(14)
!STOP
deallocate(dx,dy,dz,dr_sq)
deallocate(in_list)

end subroutine first_iter_free_term_REAL_SPACE

subroutine first_iter_free_term_14
use sizes_data, only :  Natoms,N_pairs_14
use non_bonded_lists_data
use field_constrain_data
use interpolate_data, only : MX_interpol_points,N_STYLE_ATOMS, vele_G,gele_G,vele2_G,RDR , iRDR,&
                             vele_THOLE,gele_THOLE,vele,gele,vele2
use cut_off_data
use ALL_atoms_data, only : all_p_charges,all_g_charges, xx,yy,zz,&
   i_Style_atom, is_charge_distributed, all_charges,xxx,yyy,zzz,&
   all_dipoles,all_dipoles_xx,all_dipoles_yy,all_dipoles_zz
use cg_buffer, only : BB0,list2_14,size_list2_14
use boundaries, only : periodic_images
use Ewald_data
use atom_type_data, only : which_atomStyle_pair
use sim_cel_data
use math_constants, only : two_per_sqrt_Pi,sqrt_Pi,Pi,Pi2
use physical_constants, only : Red_Vacuum_EL_permitivity_4_Pi, Volt_to_internal_field
use connectivity_ALL_data, only : red_14_Q,red_14_Q_mu,red_14_mu_mu
use max_sizes_data, only : MX_in_list_14
 implicit none
 integer i,iii,k,j,kkk,jjj,ndx,i_pair,i_type,j_type,neightot,i1,N,j1
 integer i_1, j_1, kk,i_adress,j_adress
 real(8) qj,r,x,y,z ,qi, B0i,G1,G2,B0,B1,B2,B0i_2_xx,B0i_2_yy,B0i_2_zz,B0i_1, B0_THOLE
 real(8) B00,B11,B22,inv_r_B0,inv_r_B1,inv_r_B2,inv_r3,inv_r5,Inverse_r_squared
 real(8) r2,Inverse_r,ppp,En0,field ,t1,t2,vk1,vk2,vk
 logical l_i,l_j,l_in,l_out
 real(8) fii,CC1,x_x_x,zij,CC,CC2
 real(8) derf_x, dexp_x2
 real(8) di_xx,di_yy,di_zz,dj_xx,dj_yy,dj_zz,p_r,p_r_i, K_P, K_R, coef
 logical lgg(6)
 real(8), allocatable :: dx(:),dy(:),dz(:),dr_sq(:)
 integer, allocatable :: in_list(:)
allocate(dx(MX_in_list_14),dy(MX_in_list_14),dz(MX_in_list_14),dr_sq(MX_in_list_14))
allocate(in_list(MX_in_list_14))
do i1 = 1,  TAG_PP
   i = ndx_remap%var(i1)
   i_type = i_Style_atom(i)
   neightot = size_list2_14(i1)
!print*,i1,i,neightot
!read(*,*)
   do k = 1, neightot
     j = list2_14(i1,k)
     dx(k) = xxx(i) - xxx(j)
     dy(k) = yyy(i) - yyy(j)
     dz(k) = zzz(i) - zzz(j)
     in_list(k) = j
   enddo
   if (neightot > 0 ) then
   call periodic_images(dx(1:neightot),dy(1:neightot),dz(1:neightot))
   dr_sq(1:neightot) = dx(1:neightot)*dx(1:neightot) + dy(1:neightot)*dy(1:neightot) +&
       dz(1:neightot)*dz(1:neightot)
   endif
   B0i = 0.0d0
   B0i_1=0.0d0
   B0i_2_xx = 0.0d0
   B0i_2_yy = 0.0d0
   B0i_2_zz = 0.0d0
if (i1 < TAG_SS + 1) then

   do k = 1,neightot ! l_j is now always true (see before )
    j = in_list(k)
    r2 = dr_sq(k)
    if ( r2 < cut_off_sq ) then
       j_type = i_Style_atom(j)
       r = dsqrt(r2)
       Inverse_r = 1.0d0/r
       Inverse_r_squared = Inverse_r*Inverse_r
       NDX = max(1,int(r*irdr))
       ppp = (r*irdr) - dble(ndx)
       i_pair = which_atomStyle_pair(i_type,j_type)
       qj = all_charges(j)
            vk  = vele_G(ndx,i_pair)  ;  vk1 = vele_G(ndx+1,i_pair) ; vk2 = vele_G(ndx+2,i_pair)
       t1 = vk  + (vk1 - vk )*ppp
       t2 = vk1 + (vk2 - vk1)*(ppp - 1.0d0)
       B0 = (t1 + (t2-t1)*(ppp*0.5d0))
 
       vk  = vele(ndx)  ;  vk1 = vele(ndx+1) ; vk2 = vele(ndx+2)
       t1 = vk  + (vk1 - vk )*ppp
       t2 = vk1 + (vk2 - vk1)*(ppp - 1.0d0)
       B00 = (t1 + (t2-t1)*(ppp*0.5d0))
       inv_r_B0 = Inverse_r - (B0-B00)
       j1 = rec_ndx_remap%var(j)
       if (j1 <  TAG_PP + 1) then
         B0i_1 = B0i_1 + inv_r_B0 * qj * (-red_14_Q)  
!         lgg(1)=.true.
       else
      vk  = gele_G(ndx,i_pair)  ;  vk1 = gele_G(ndx+1,i_pair) ; vk2 = gele_G(ndx+2,i_pair)
       t1 = vk  + (vk1 - vk )*ppp
       t2 = vk1 + (vk2 - vk1)*(ppp - 1.0d0)
       B1 = (t1 + (t2-t1)*(ppp*0.5d0))

       vk  = gele(ndx)  ;  vk1 = gele(ndx+1) ; vk2 = gele(ndx+2)
       t1 = vk  + (vk1 - vk )*ppp
       t2 = vk1 + (vk2 - vk1)*(ppp - 1.0d0)
       B11 = (t1 + (t2-t1)*(ppp*0.5d0))
       inv_r3 = Inverse_r_squared * Inverse_r
       inv_r_B1 = inv_r3 - (B1-B11)
         p_r = all_dipoles_xx(j)*dx(k) + all_dipoles_yy(j)*dy(k)+all_dipoles_zz(j)*dz(k) 
         B0i_1 = B0i_1 + inv_r_B0*qj * (-red_14_Q) + inv_r_B1*p_r * (-red_14_Q_mu)
       endif
    endif ! within cut-off
  enddo
  BB0(i1) =   BB0(i1) - B0i_1
elseif (i1 < TAG_SP+1) then ! i1 > TAG_SS
     do k = 1,neightot ! l_j is now always true (see before )
     j = in_list(k)
     r2 = dr_sq(k)
     if ( r2 < cut_off_sq ) then
       j_type = i_Style_atom(j)
       r = dsqrt(r2)
       Inverse_r = 1.0d0/r
       Inverse_r_squared = Inverse_r*Inverse_r
       NDX = max(1,int(r*irdr))
       ppp = (r*irdr) - dble(ndx)
       i_pair = which_atomStyle_pair(i_type,j_type)
     qj = all_charges(j)
      vk  = vele_G(ndx,i_pair)  ;  vk1 = vele_G(ndx+1,i_pair) ; vk2 = vele_G(ndx+2,i_pair)
       t1 = vk  + (vk1 - vk )*ppp
       t2 = vk1 + (vk2 - vk1)*(ppp - 1.0d0)
       B0 = (t1 + (t2-t1)*(ppp*0.5d0))
      vk  = gele_G(ndx,i_pair)  ;  vk1 = gele_G(ndx+1,i_pair) ; vk2 = gele_G(ndx+2,i_pair)
       t1 = vk  + (vk1 - vk )*ppp
       t2 = vk1 + (vk2 - vk1)*(ppp - 1.0d0)
       B1 = (t1 + (t2-t1)*(ppp*0.5d0))

       vk  = vele(ndx)  ;  vk1 = vele(ndx+1) ; vk2 = vele(ndx+2)
       t1 = vk  + (vk1 - vk )*ppp
       t2 = vk1 + (vk2 - vk1)*(ppp - 1.0d0)
       B00 = (t1 + (t2-t1)*(ppp*0.5d0))
       vk  = gele(ndx)  ;  vk1 = gele(ndx+1) ; vk2 = gele(ndx+2)
       t1 = vk  + (vk1 - vk )*ppp
       t2 = vk1 + (vk2 - vk1)*(ppp - 1.0d0)
       B11 = (t1 + (t2-t1)*(ppp*0.5d0))
       inv_r3 = Inverse_r_squared * Inverse_r
       inv_r_B0 = Inverse_r - (B0-B00)
       inv_r_B1 = inv_r3 - (B1-B11)

     j1 = rec_ndx_remap%var(j)
     if (j1 <  TAG_PP + 1) then
       B0i_1 = B0i_1 + inv_r_B0*qj *(-red_14_Q)
       B0i_2_xx = B0i_2_xx + (- qj*dx(k) )*(inv_r_B1*(-red_14_Q_mu))
       B0i_2_yy = B0i_2_yy + (- qj*dy(k) )*(inv_r_B1*(-red_14_Q_mu))
       B0i_2_zz = B0i_2_zz + (- qj*dz(k) )*(inv_r_B1*(-red_14_Q_mu))
     else
      vk  = vele2_G(ndx,i_pair)  ;  vk1 = vele2_G(ndx+1,i_pair) ; vk2 = vele2_G(ndx+2,i_pair)
       t1 = vk  + (vk1 - vk )*ppp
       t2 = vk1 + (vk2 - vk1)*(ppp - 1.0d0)
       B2 = (t1 + (t2-t1)*(ppp*0.5d0))
       vk  = vele2(ndx)  ;  vk1 = vele2(ndx+1) ; vk2 = vele2(ndx+2)
       t1 = vk  + (vk1 - vk )*ppp
       t2 = vk1 + (vk2 - vk1)*(ppp - 1.0d0)
       B22 = (t1 + (t2-t1)*(ppp*0.5d0))
       inv_r5 = inv_r3 * Inverse_r_squared
       inv_r_B2 = 3.0d0*inv_r5-(B2-B22)
       dj_xx = all_dipoles_xx(j); dj_yy = all_dipoles_yy(j) ; dj_zz = all_dipoles_zz(j)
       p_r = dj_xx*dx(k) + dj_yy*dy(k)+dj_zz*dz(k)
       B0i_1 = B0i_1 + inv_r_B0*qj *(-red_14_Q) + inv_r_B1*p_r * (-red_14_Q_mu)
       B0i_2_xx = B0i_2_xx + (- qj*dx(k)*(-red_14_Q_mu) + dj_xx*(-red_14_mu_mu))*inv_r_B1 - dx(k)*(p_r*inv_r_B2)*(-red_14_mu_mu)
       B0i_2_yy = B0i_2_yy + (- qj*dy(k)*(-red_14_Q_mu) + dj_yy*(-red_14_mu_mu))*inv_r_B1 - dy(k)*(p_r*inv_r_B2)*(-red_14_mu_mu)
       B0i_2_zz = B0i_2_zz + (- qj*dz(k)*(-red_14_Q_mu) + dj_zz*(-red_14_mu_mu))*inv_r_B1 - dz(k)*(p_r*inv_r_B2)*(-red_14_mu_mu)
     endif
    endif ! within cut-off
  enddo
 i_adress = TAG_SP+i1-TAG_SS
 BB0(i1) = BB0(i1)  - B0i_1
 BB0(      i_adress) = BB0(      i_adress) - B0i_2_xx
 BB0(  NDP+i_adress) = BB0(  NDP+i_adress) - B0i_2_yy
 BB0(2*NDP+i_adress) = BB0(2*NDP+i_adress) - B0i_2_zz
else  ! i1 > TAG_SP (dipol only)
     do k = 1,neightot ! l_j is now always true (see before )
     j = in_list(k)
     r2 = dr_sq(k)
     if ( r2 < cut_off_sq ) then
       j_type = i_Style_atom(j)
       r = dsqrt(r2)
       Inverse_r = 1.0d0/r
       Inverse_r_squared = Inverse_r*Inverse_r
       NDX = max(1,int(r*irdr))
       ppp = (r*irdr) - dble(ndx)
       i_pair = which_atomStyle_pair(i_type,j_type)
       qj = all_charges(j)
       vk  = gele_G(ndx,i_pair)  ;  vk1 = gele_G(ndx+1,i_pair) ; vk2 = gele_G(ndx+2,i_pair)
       t1 = vk  + (vk1 - vk )*ppp
       t2 = vk1 + (vk2 - vk1)*(ppp - 1.0d0)
       B1 = (t1 + (t2-t1)*(ppp*0.5d0))
       vk  = gele(ndx)  ;  vk1 = gele(ndx+1) ; vk2 = gele(ndx+2)
       t1 = vk  + (vk1 - vk )*ppp
       t2 = vk1 + (vk2 - vk1)*(ppp - 1.0d0)
       B11 = (t1 + (t2-t1)*(ppp*0.5d0))
       inv_r3 = Inverse_r_squared * Inverse_r
       inv_r_B1 = inv_r3 - (B1-B11)

       j1 = rec_ndx_remap%var(j)
       if (j1 <  TAG_PP + 1) then
         B0i_2_xx = B0i_2_xx + (- qj*dx(k) )*inv_r_B1*(-red_14_Q_mu)
         B0i_2_yy = B0i_2_yy + (- qj*dy(k) )*inv_r_B1*(-red_14_Q_mu)
         B0i_2_zz = B0i_2_zz + (- qj*dz(k) )*inv_r_B1*(-red_14_Q_mu)
!write(14,*)i,j,B1,r
       else
      vk  = vele2_G(ndx,i_pair)  ;  vk1 = vele2_G(ndx+1,i_pair) ; vk2 = vele2_G(ndx+2,i_pair)
       t1 = vk  + (vk1 - vk )*ppp
       t2 = vk1 + (vk2 - vk1)*(ppp - 1.0d0)
       B2 = (t1 + (t2-t1)*(ppp*0.5d0))
       vk  = vele2(ndx)  ;  vk1 = vele2(ndx+1) ; vk2 = vele2(ndx+2)
       t1 = vk  + (vk1 - vk )*ppp
       t2 = vk1 + (vk2 - vk1)*(ppp - 1.0d0)
       B22 = (t1 + (t2-t1)*(ppp*0.5d0))
       inv_r5 = inv_r3 * Inverse_r_squared
       inv_r_B2 = 3.0d0*inv_r5-(B2-B22)

         dj_xx = all_dipoles_xx(j); dj_yy = all_dipoles_yy(j) ; dj_zz = all_dipoles_zz(j)
         p_r = dj_xx*dx(k) + dj_yy*dy(k) + dj_zz*dz(k)
         B0i_2_xx = B0i_2_xx + (- qj*dx(k)*(-red_14_Q_mu) + dj_xx*(-red_14_mu_mu))*inv_r_B1 - dx(k)*(p_r*inv_r_B2)*(-red_14_mu_mu)
         B0i_2_yy = B0i_2_yy + (- qj*dy(k)*(-red_14_Q_mu) + dj_yy*(-red_14_mu_mu))*inv_r_B1 - dy(k)*(p_r*inv_r_B2)*(-red_14_mu_mu)
         B0i_2_zz = B0i_2_zz + (- qj*dz(k)*(-red_14_Q_mu) + dj_zz*(-red_14_mu_mu))*inv_r_B1 - dz(k)*(p_r*inv_r_B2)*(-red_14_mu_mu)
!write(14,*)i,j,B1,r
       endif
    endif ! within cut-off

  enddo
 i_adress = TAG_SP+i1-TAG_SS
 BB0(      i_adress) = BB0(      i_adress) - B0i_2_xx
 BB0(  NDP+i_adress) = BB0(  NDP+i_adress) - B0i_2_yy
 BB0(2*NDP+i_adress) = BB0(2*NDP+i_adress) - B0i_2_zz
!print*, 'case 3: ',BB0(      i_adress),BB0(  NDP+i_adress), BB0(2*NDP+i_adress)
endif
enddo   ! i1
deallocate(dx,dy,dz,dr_sq)
deallocate(in_list)

end subroutine first_iter_free_term_14


subroutine first_iter_free_term_k0_2D
use spline_z_k0_module
 use array_math
 use ALL_atoms_data, only : zzz,zz,all_charges, &
    Natoms,all_charges,all_dipoles_zz
 use sim_cel_data
 use cg_buffer, only : BB0
 use Ewald_data, only : Ewald_alpha
 use field_constrain_data
 use math_constants, only : two_per_sqrt_Pi,sqrt_Pi,Pi,Pi2

 implicit none
 integer i,j,k,i1,om2,kkk,kk2,i_adress

 real(8) CC,CC2
 integer N
 real(8) sum_field_q,sum_field_miu,x_x_x,qi,di_zz,field,z,derf_x,dexp_x2
 real(8), allocatable :: alp_q(:),alp_miu(:),MAT(:,:),field_grid_q(:),field_grid_miu(:)

 allocate(alp_q(Ngrid),alp_miu(Ngrid))
 allocate(MAT(Ngrid,Ngrid))
 allocate(field_grid_q(Ngrid),field_grid_miu(Ngrid))

N = Ngrid - 1
CC2 = 4.0d0*Ewald_alpha*sqrt_Pi
CC = sqrt_Pi/Ewald_alpha

 do k = 1, Ngrid
 sum_field_q=0.0d0 ; sum_field_miu = 0.0d0
 do i1 = TAG_SP+1, TAG_PP  ! Only charges 
 i = ndx_remap%var(i1)
      z = z_grid(k)  - zzz(i)
      qi = all_charges(i)
      x_x_x = Ewald_alpha*z ; derf_x = derf(x_x_x); dexp_x2=dexp(-x_x_x*x_x_x)
      field = qi*(CC*dexp_x2+z*Pi*derf_x)
      sum_field_q = sum_field_q + field !
      field = (-Pi2*qi*derf_x )
      sum_field_miu = sum_field_miu  + field
 enddo
 do i1 = TAG_PP+1, Natoms  ! charges+dipoles
 i = ndx_remap%var(i1)
      z = z_grid(k) - zzz(i)
      qi = all_charges(i)
      di_zz = all_dipoles_zz(i)
      x_x_x = Ewald_alpha*z ; derf_x = derf(x_x_x); dexp_x2=dexp(-x_x_x*x_x_x)
      field = qi*(CC*dexp_x2+z*Pi*derf_x) - Pi*derf_x*di_zz
      sum_field_q = sum_field_q + field !
      field = (-Pi2*qi*derf_x + CC2*dexp_x2*di_zz)
      sum_field_miu = sum_field_miu  + field
 enddo
 field_grid_q(k) = sum_field_q
 field_grid_miu(k) = sum_field_miu
 enddo

 field_grid_q = field_grid_q * ( (-2.0d0) * i_area)
 field_grid_miu = field_grid_miu * (  i_area)

 kk2 = mod(order,2)+1
 do i = 0,n
     z = z_grid(i+1)
     call deboor_cox(order,Ngrid, order+1, kkk, qq, z, bv)
     MAT(i+1,1:n+1) = bv(1:n+1,kk2)
 enddo ! i=1,n
 call invmat(MAT,Ngrid,Ngrid)
 do i = 1, Ngrid
   alp_q(i) = dot_product(MAT(i,:),field_grid_q(:))
   alp_miu(i) = dot_product(MAT(i,:),field_grid_miu(:))
 enddo
 om2 = mod(order,2)+1

 do i1 = 1, TAG_PP
     i = ndx_remap%var(i1)
     z = zzz(i)
     call deboor_cox(order,Ngrid, order+1, kkk, qq, z, bv)
     j   = kkk - order;
     if (i1 < TAG_SP+1) BB0(i1) = BB0(i1) - dot_product(alp_q(j+1:kkk+1),bv(j+1:kkk+1,om2)) 
     if (i1 > TAG_SS) then
       i_adress = i1 - TAG_SS + TAG_SP
       BB0(i_adress+2*NDP) = BB0(i_adress+2*NDP) - dot_product(alp_miu(j+1:kkk+1),bv(j+1:kkk+1,om2))
     endif
 enddo

deallocate(MAT)
deallocate(field_grid_q,field_grid_miu)
deallocate(alp_miu,alp_q)
end subroutine first_iter_free_term_k0_2D


!---------------------------
subroutine first_iter_free_term_k0_2D_SLOW
use ALL_atoms_data, only : Natoms,zzz,zz,all_dipoles_zz,all_charges
use cg_buffer, only : BB0
use field_constrain_data, only : ndx_remap, rec_ndx_remap
use math_constants, only : Pi,sqrt_Pi, Pi2
use Ewald_data, only : ewald_alpha
implicit none
integer i,j,k,i1,j1,i_adress
real(8) derf_x,z,dexp_x2,di_zz,dj_zz,qi,qj,x_x_x, CC2,CC,B0i_1,B0i_2_zz
real(8), allocatable :: BB0_k0(:)

CC2 = 4.0d0*Ewald_alpha*sqrt_Pi
CC = CC_alpha
allocate(BB0_k0(NVFC))
BB0_k0 = 0.0d0
do i1 = 1,  TAG_PP
   i = ndx_remap%var(i1)
   B0i_1=0.0d0
   B0i_2_zz = 0.0d0
if (i1 < TAG_SS + 1) then
    do j1 = TAG_SP+1, Natoms
    j = ndx_remap%var(j1)
       if (i /= j) then
       z = zzz(i) - zzz(j)
       x_x_x = Ewald_alpha*z; derf_x = derf(x_x_x); dexp_x2=dexp(-x_x_x*x_x_x)
       qj = all_charges(j)
       if (j1 < TAG_PP+1) then
            B0i_1 = B0i_1 +  (qj*(CC*dexp_x2+z*Pi*derf_x) )
       else
            dj_zz = all_dipoles_zz(j)
            B0i_1 = B0i_1 +  (qj*(CC*dexp_x2+z*Pi*derf_x) - Pi*derf_x*dj_zz)
       endif
       endif ! (i/=j)
  enddo
  BB0_k0(i1) =  - B0i_1 * 2.0d0
elseif (i1 < TAG_SP+1) then ! i1 > TAG_SS
     do j1 = TAG_SP+1,Natoms  ! l_j is now always true (see before )
       j=ndx_remap%var(j1)
       if (i /=j) then
       z = zzz(i) - zzz(j)
       x_x_x = Ewald_alpha*z; derf_x = derf(x_x_x); dexp_x2=dexp(-x_x_x*x_x_x)
       qj = all_charges(j)
     if (j1 <  TAG_PP + 1) then
       B0i_1 = B0i_1 + qj*(CC*dexp_x2+z*Pi*derf_x)
       B0i_2_zz = B0i_2_zz - Pi2*qj*derf_x
     else
       dj_zz = all_dipoles_zz(j)
       B0i_1 = B0i_1 + (qj*(CC*dexp_x2+z*Pi*derf_x) - Pi*derf_x*dj_zz)
       B0i_2_zz = B0i_2_zz + (-Pi2*qj*derf_x + CC2*dexp_x2*dj_zz)
     endif
      endif!i/=j
  enddo
 i_adress = TAG_SP+i1-TAG_SS
 BB0_k0(i1) = - B0i_1 * 2.0d0
 BB0_k0(2*NDP+i_adress) =  B0i_2_zz
else  ! i1 > TAG_SP (dipol only)
     do j1 = TAG_SP+1,Natoms  ! l_j is now always true (see before )
       j=ndx_remap%var(j1)
       if (i/=j) then
       z = zzz(i) - zzz(j)
       x_x_x = Ewald_alpha*z; derf_x = derf(x_x_x); dexp_x2=dexp(-x_x_x*x_x_x)
       qj = all_charges(j)
       dj_zz = all_dipoles_zz(j)
       if (j1 <  TAG_PP + 1) then
         B0i_2_zz = B0i_2_zz - Pi2*qj*derf_x
       else
         B0i_2_zz = B0i_2_zz - Pi2*qj*derf_x + CC2*dexp_x2*dj_zz
       endif
       endif ! i/=j
  enddo
 i_adress = TAG_SP+i1-TAG_SS
 BB0_k0(2*NDP+i_adress) =  B0i_2_zz
endif
enddo   ! i1

BB0 = BB0 - BB0_k0*i_Area

deallocate(BB0_k0)

end subroutine first_iter_free_term_k0_2D_SLOW
!----------------------------
subroutine first_iter_free_term_k_NON_0_2D_SLOW
use ALL_atoms_data, only : Natoms,all_dipoles_xx,all_dipoles_yy,all_dipoles_zz,all_charges,&
                          xxx,yyy,zzz
use cg_buffer, only : BB0, pre,sns,css
use field_constrain_data, only : ndx_remap, rec_ndx_remap
use math_constants, only : Pi,sqrt_Pi, Pi2
use Ewald_data, only : ewald_alpha,k_max_x,k_max_y,k_max_z
use sim_cel_data, only : Reciprocal_cel
use cut_off_data,only : reciprocal_cut_sq
implicit none
real(8)kx,ky,kz,i4a2,tmp,d2,GG0,GG_00,KR,K_P,Sum_Re,Sum_Im,Sum_Re_DIP,Sum_Im_DIP,coef
integer ix,iy,iz,k_vct,i,j,k,i1,j1,i_adress
real(8) rec_xx,rec_xy,rec_yx,rec_yy,rec_zz , qi, di_xx,di_yy,di_zz
real(8) , allocatable :: BB0_Fourier(:)

allocate(BB0_Fourier(NVFC)) ; BB0_Fourier = 0.0d0
     k_vct = 0
     i4a2 =  1.0d0/(4.0d0*Ewald_alpha**2)
     do ix = -K_MAX_X,K_MAX_X
      tmp = dble(ix)
      rec_xx = tmp*Reciprocal_cel(1)
      rec_yx = tmp*Reciprocal_cel(4)
      do iy = -K_MAX_Y,K_MAX_Y
        tmp = dble(iy)
        rec_xy = tmp*Reciprocal_cel(2)
        rec_yy = tmp*Reciprocal_cel(5)
        if (ix**2 + iy**2 /= 0) then
        do iz = -K_MAX_Z,K_MAX_Z
          rec_zz = dble(iz) * h_step
          kz = rec_zz
          kx = rec_xx + rec_xy
          ky = rec_yx + rec_yy
          d2 = kx*kx + ky*ky + kz*kz
          if (d2 < reciprocal_cut_sq) then
             k_vct = k_vct + 1
             GG0 = dexp(-(i4a2)*d2)/d2
             GG_00 = GG0   ! it is 0.5 at exponent here!!!! and not 0.25
             pre(k_vct) = GG0
             Sum_Re = 0.0d0 ; Sum_Im = 0.0d0
             Sum_Re_DIP = 0.0d0 ; Sum_Im_DIP = 0.0d0
             k_vector(k_vct,1) = kx
             k_vector(k_vct,2) = ky
             k_vector(k_vct,3) = kz
             do i = 1, Natoms
              KR = kx*xxx(i)+ky*yyy(i)+kz*zzz(i)
              K_P = all_dipoles_xx(i)*kx + all_dipoles_yy(i)*ky+all_dipoles_zz(i)*kz
              sns(i,k_vct) = dsin(KR)
              css(i,k_vct) = dcos(KR)
              if (.not.is_sfield(i)) then
                 qi = all_charges(i)
                 Sum_Re = Sum_Re +  css(i,k_vct)*qi
                 Sum_Im = Sum_Im +  sns(i,k_vct)*qi
              endif
              if (.not.is_dip(i)) then
                 Sum_Re = Sum_Re +  ( - K_P*sns(i,k_vct))
                 Sum_Im = Sum_Im +  ( + K_P*css(i,k_vct))
              endif
            enddo ! i = 1, Natoms
             Sum_Re = Sum_Re * GG0
             Sum_Im = Sum_Im * GG0

             do i1 = 1,TAG_SP
                 i = ndx_remap%var(i1)
                 BB0_Fourier(i1) = BB0_Fourier(i1) + (css(i,k_vct) * Sum_Re + sns(i,k_vct) * Sum_Im)
             enddo  ! i = 1, Natoms
             do i1 = TAG_SS+1,TAG_PP
                 i_adress = TAG_SP+i1-TAG_SS
                 i = ndx_remap%var(i1)
                 coef = (css(i,k_vct) * Sum_Im - sns(i,k_vct) * Sum_Re)
                 BB0_Fourier(i_adress      ) = BB0_Fourier(i_adress      ) + kx * coef
                 BB0_Fourier(i_adress+  NDP) = BB0_Fourier(i_adress+  NDP) + ky * coef
                 BB0_Fourier(i_adress+2*NDP) = BB0_Fourier(i_adress+2*NDP) + kz * coef
             enddo  ! i = 1,
          endif ! (d2 < reciprocal_cut_sq)
        enddo   ! iz
        endif   ! (ix**2 + iy**2 /= 0)
      enddo     ! iy
     enddo      ! ix

    BB0 = BB0 - BB0_Fourier*(2.0d0*i_Area*h_step)  ! minus because we have AX-B = 0; 
deallocate(BB0_Fourier)
end subroutine first_iter_free_term_k_NON_0_2D_SLOW
!---------------------------

subroutine first_iter_free_term_FOURIER_3D_SLOW
use ALL_atoms_data, only : Natoms,all_dipoles_xx,all_dipoles_yy,all_dipoles_zz,all_charges,&
                          xxx,yyy,zzz
use cg_buffer, only : BB0, pre,sns,css
use field_constrain_data, only : ndx_remap, rec_ndx_remap
use math_constants, only : Pi,sqrt_Pi, Pi2
use Ewald_data, only : ewald_alpha,k_max_x,k_max_y,k_max_z
use sim_cel_data, only : Reciprocal_cel,Volume
use cut_off_data,only : reciprocal_cut_sq
implicit none
real(8)kx,ky,kz,i4a2,tmp,d2,GG0,GG_00,KR,K_P,Sum_Re,Sum_Im,Sum_Re_DIP,Sum_Im_DIP,coef
integer ix,iy,iz,k_vct,i,j,k,i1,j1,i_adress
real(8) rec_xx,rec_xy,rec_yx,rec_yy,rec_zz,rec_xz,rec_zx,rec_yz,rec_zy,qi,di_xx,di_yy,di_zz
real(8) , allocatable :: BB0_Fourier(:)


allocate(BB0_Fourier(NVFC)) ; BB0_Fourier = 0.0d0
     k_vct = 0
     i4a2 =  1.0d0/(4.0d0*Ewald_alpha*Ewald_alpha)
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
        do iz = -K_MAX_Z,K_MAX_Z
        if (ix*ix+iy*iy+iz*iz > 0) then
          tmp = dble(iz)
          rec_xz = tmp*Reciprocal_cel(3)
          rec_yz = tmp*Reciprocal_cel(6)
          rec_zz = tmp*Reciprocal_cel(9)
          kx = rec_xx + rec_xy + rec_xz
          ky = rec_yx + rec_yy + rec_yz
          kz = rec_zx + rec_zy + rec_zz
          d2 = kx*kx + ky*ky + kz*kz
          if (d2 < reciprocal_cut_sq) then
             k_vct = k_vct + 1
             GG0 = dexp(-(i4a2)*d2)/d2
             GG_00 = GG0   ! it is 0.5 at exponent here!!!! and not 0.25
             pre(k_vct) = GG0
             Sum_Re = 0.0d0 ; Sum_Im = 0.0d0
             Sum_Re_DIP = 0.0d0 ; Sum_Im_DIP = 0.0d0
             k_vector(k_vct,1) = kx
             k_vector(k_vct,2) = ky
             k_vector(k_vct,3) = kz
             do i = 1, Natoms
              KR = kx*xxx(i)+ky*yyy(i)+kz*zzz(i)
              K_P = all_dipoles_xx(i)*kx + all_dipoles_yy(i)*ky+all_dipoles_zz(i)*kz
              sns(i,k_vct) = dsin(KR)
              css(i,k_vct) = dcos(KR)
              if (.not.is_sfield(i)) then
                 qi = all_charges(i)
                 Sum_Re = Sum_Re +  css(i,k_vct)*qi
                 Sum_Im = Sum_Im +  sns(i,k_vct)*qi
              endif
              if (.not.is_dip(i)) then
                 Sum_Re = Sum_Re +  ( - K_P*sns(i,k_vct))
                 Sum_Im = Sum_Im +  ( + K_P*css(i,k_vct))
              endif
            enddo ! i = 1, Natoms
             Sum_Re = Sum_Re * GG0
             Sum_Im = Sum_Im * GG0

             do i1 = 1,TAG_SP
                 i = ndx_remap%var(i1)
                 BB0_Fourier(i1) = BB0_Fourier(i1) + (css(i,k_vct) * Sum_Re + sns(i,k_vct) * Sum_Im)
             enddo  ! i = 1, Natoms
             do i1 = TAG_SS+1,TAG_PP
                 i_adress = TAG_SP+i1-TAG_SS
                 i = ndx_remap%var(i1)
                 coef = (css(i,k_vct) * Sum_Im - sns(i,k_vct) * Sum_Re)
                 BB0_Fourier(i_adress      ) = BB0_Fourier(i_adress      ) + kx * coef
                 BB0_Fourier(i_adress+  NDP) = BB0_Fourier(i_adress+  NDP) + ky * coef
                 BB0_Fourier(i_adress+2*NDP) = BB0_Fourier(i_adress+2*NDP) + kz * coef
             enddo  ! i = 1,
          endif ! (d2 < reciprocal_cut_sq)
          endif ! ix**2+iy**2+iz**2 > 0
        enddo   ! iz
      enddo     ! iy
     enddo      ! ix

    BB0 = BB0 - BB0_Fourier*(2.0d0*Pi2/Volume)   ! minus because we have AX-B = 0; The 2 coef appear from math when swith from
deallocate(BB0_Fourier)
end subroutine first_iter_free_term_FOURIER_3D_SLOW
!----------------------------
subroutine first_iter_free_term_k_NON_0_2D
! get the free term
    use fft_3D
    use cg_buffer, only : BB0,Ih_on_grid_FREE
    use ALL_atoms_data, only : Natoms
    use Ewald_data, only : order_spline_zz,order_spline_yy,order_spline_xx, &
                           nfftx,nffty,nfftz
    use variables_smpe, only : qqq1,qqq2,key1,key2,key3,ww1,ww2,ww3
    implicit none

    call set_grid_Q_DIP_init_2D
    call dlpfft3_MOD(0,1,nfftx,nffty,nfftz,key1,key2,key3,ww1,ww2,ww3,qqq1)
    call smpe_eval1_Q_DIP_2D   !  potential and stresses
    call dlpfft3_MOD(0,-1,nfftx,nffty,nfftz,key1,key2,key3,ww1,ww2,ww3,qqq1)
    call smpe_eval2_Q_DIP_init_2D ! get free term


end subroutine first_iter_free_term_k_NON_0_2D

subroutine first_iter_free_term_Fourier_3D
! get the free term
    use fft_3D
    use cg_buffer, only : BB0,Ih_on_grid_FREE
    use ALL_atoms_data, only : Natoms
    use Ewald_data, only : order_spline_zz,order_spline_yy,order_spline_xx, &
                           nfftx,nffty,nfftz
    use variables_smpe, only : qqq1,qqq2,key1,key2,key3,ww1,ww2,ww3
    implicit none

    call set_grid_Q_DIP_init_3D
    call dlpfft3_MOD(0,1,nfftx,nffty,nfftz,key1,key2,key3,ww1,ww2,ww3,qqq1)
    call smpe_eval1_Q_DIP_3D   !  potential and stresses
    call dlpfft3_MOD(0,-1,nfftx,nffty,nfftz,key1,key2,key3,ww1,ww2,ww3,qqq1)
    call smpe_eval2_Q_DIP_init_3D ! get free term


end subroutine first_iter_free_term_Fourier_3D


subroutine get_AX_k_NON_0_2D
! get AX
    use fft_3D
    use Ewald_data, only : order_spline_zz,order_spline_yy,order_spline_xx, &
                           nfftx,nffty,nfftz
    use variables_smpe, only : qqq1,qqq2,key1,key2,key3,ww1,ww2,ww3

    implicit none
    integer i,j,k
    call set_grid_Q_DIP_cycle_2D
    call dlpfft3_MOD(0,1,nfftx,nffty,nfftz,key1,key2,key3,ww1,ww2,ww3,qqq1)
    call smpe_eval1_Q_DIP_2D   !  potential and stresses
    call dlpfft3_MOD(0,-1,nfftx,nffty,nfftz,key1,key2,key3,ww1,ww2,ww3,qqq1)
    call smpe_eval2_Q_DIP_cycle_2D

end subroutine get_AX_k_NON_0_2D

subroutine get_AX_Fourier_3D
! get AX
    use fft_3D
    use Ewald_data, only : order_spline_zz,order_spline_yy,order_spline_xx, &
                           nfftx,nffty,nfftz
    use variables_smpe, only : qqq1,qqq2,key1,key2,key3,ww1,ww2,ww3

    implicit none
    integer i,j,k
    call set_grid_Q_DIP_cycle_3D
    call dlpfft3_MOD(0,1,nfftx,nffty,nfftz,key1,key2,key3,ww1,ww2,ww3,qqq1)
    call smpe_eval1_Q_DIP_3D   !  potential and stresses
    call dlpfft3_MOD(0,-1,nfftx,nffty,nfftz,key1,key2,key3,ww1,ww2,ww3,qqq1)
    call smpe_eval2_Q_DIP_cycle_3D
end subroutine get_AX_Fourier_3D


subroutine set_grid_Q_DIP_init_2D
    use variables_smpe
    use Ewald_data
    use sim_cel_data
    use sizes_data, only : Natoms
    use all_atoms_data , only : all_charges,l_WALL, zz,is_charge_distributed,all_dipoles,&
        all_dipoles_xx,all_dipoles_yy,all_dipoles_zz
    use cg_buffer, only : NNX,NNY,NNZ,Ih_on_grid, Ih_on_grid_dx,Ih_on_grid_dy,Ih_on_grid_dz
    use field_constrain_data, only : ndx_remap

    implicit none
    integer i,j,k,nx,ny,nz,ix,iy,iz,jx,jy,jz,kx,ky,kz,I_INDEX
    integer ii_xx,ii_yy,ii_zz,i_adress,ii,jj,kk,i1
    integer kx1,ky1,kz1
    real(8) ci,di_xx,di_yy,di_zz
    real(8) t_x,t_y,t_z
    real(8) dipole_term,q_term,p_nabla_ux,p_nabla_uy,p_nabla_uz
    real(8) axx,axy,ayx,ayy,azz

    axx=Inverse_cel(1) ; axy = Inverse_cel(2)
    ayx =Inverse_cel(4); ayy = Inverse_cel(5)

    qqq1_Re=0.0d0
    do i1 = TAG_SP+1,Natoms
      i=ndx_remap%var(i1)
      ci = all_charges(i)
      if (i1 > TAG_PP) then
      di_xx = all_dipoles_xx(i); di_yy = all_dipoles_yy(i);di_zz=all_dipoles_zz(i)
      else
      di_xx=0.0d0;di_yy=0.0d0;di_zz=0.0d0
      endif
      p_nabla_ux = di_xx*axx + di_yy*axy
      p_nabla_uy = di_xx*ayx + di_yy*ayy
      p_nabla_uz = di_zz
      nx = NNX(i)
      ny = NNY(i)
      nz = NNZ(i)
      iz = nz
      I_INDEX = 0
      iz = nz
      do jz = 0, order_spline_zz-1
      iz = iz + 1
      kz = iz +  h_cut_z ! h_cut_z = nfftz/2
      if (kz >= nfftz) then  ! it cannot be nfftz
        write(6,*) 'error in cg%set_q_2D kz >= nfftz; choose more nfftz points'
        write(6,*) 'You need to make nfftz at least ',int(sim_cel(9)) , &
        'or the first 2^N integer'
        write(6,*) 'kz boxz nfftz=',kz, sim_cel(9), nfftz
      STOP
      endif
      if (kz < 0) then
        write(6,*) 'error in cg%set_q_dip_2D kz < 0 : lower the splines order or increase the nfft',kz
        write(6,*) 'order spline = ',order_spline_xx,order_spline_yy,order_spline_zz
        write(6,*) 'nfft hcutz =',nfftx,nffty,nfftz,h_cut_z
        STOP
      endif
      iy = ny
      iy = ny
      do jy = 0, order_spline_yy-1
        iy = iy + 1
        if (iy < 0) then
          ky = iy + nffty
        else
          ky = iy
        endif
        ix = nx
        do jx = 0, order_spline_xx-1
          ix = ix + 1
          I_INDEX = I_INDEX + 1
          if (ix < 0) then
             kx = ix + nfftx
          else
             kx = ix
          endif
          i_adress = (ky+kz*nffty)*nfftx + kx + 1
          t_x= Ih_on_grid_dx(i,I_INDEX)
          t_y= Ih_on_grid_dy(i,I_INDEX)
          t_z= Ih_on_grid_dz(i,I_INDEX)
          dipole_term = t_x * p_nabla_ux + t_y * p_nabla_uy + t_z * p_nabla_uz
          q_term = ci*Ih_on_grid(i,I_INDEX)
!print*, t_x,t_y,t_z,q_term,ci,dipole_term,'\\\'
          qqq1_Re(i_adress) = qqq1_Re(i_adress) + q_term + dipole_term! potential POINT CHARGE
        enddo ! jx
!read(*,*)
        enddo ! jy
      enddo ! jz
    enddo ! i
    qqq1 = cmplx(qqq1_Re,0.0d0,kind=8)


end subroutine set_grid_Q_DIP_init_2D


!---------------

subroutine set_grid_Q_DIP_init_3D
    use variables_smpe
    use Ewald_data
    use sim_cel_data
    use sizes_data, only : Natoms
    use all_atoms_data , only : all_charges,l_WALL, zz,is_charge_distributed,all_dipoles,&
        all_dipoles_xx,all_dipoles_yy,all_dipoles_zz
    use cg_buffer, only : NNX,NNY,NNZ,Ih_on_grid, Ih_on_grid_dx,Ih_on_grid_dy,Ih_on_grid_dz
    use field_constrain_data, only : ndx_remap

    implicit none
    integer i,j,k,nx,ny,nz,ix,iy,iz,jx,jy,jz,kx,ky,kz,I_INDEX
    integer ii_xx,ii_yy,ii_zz,i_adress,ii,jj,kk,i1
    integer kx1,ky1,kz1
    real(8) ci,di_xx,di_yy,di_zz
    real(8) t_x,t_y,t_z
    real(8) dipole_term,q_term,p_nabla_ux,p_nabla_uy,p_nabla_uz
    real(8) axx,axy,ayx,ayy,azz,axz,azx,ayz,azy

    axx =Inverse_cel(1); axy = Inverse_cel(2); axz = Inverse_cel(2)
    ayx =Inverse_cel(4); ayy = Inverse_cel(5); ayz = Inverse_cel(6)
    azx =Inverse_cel(7); azy = Inverse_cel(8); azz = Inverse_cel(9)

    qqq1_Re=0.0d0
    
      do i1 = TAG_SP+1,Natoms
      i=ndx_remap%var(i1)
      ci = all_charges(i)
      if (i1 > TAG_PP) then
      di_xx = all_dipoles_xx(i); di_yy = all_dipoles_yy(i);di_zz=all_dipoles_zz(i)
      else
      di_xx=0.0d0;di_yy=0.0d0;di_zz=0.0d0
      endif
      p_nabla_ux = di_xx*axx + di_yy*axy + di_zz*axz
      p_nabla_uy = di_xx*ayx + di_yy*ayy + di_zz*ayz
      p_nabla_uz = di_xx*azx + di_yy*azy + di_zz*azz

      nx = NNX(i)
      ny = NNY(i)
      nz = NNZ(i)
      I_INDEX = 0
      iz = nz
      do jz = 0, order_spline_zz-1
      iz = iz + 1
      if (iz < 0) then
         kz = iz + nfftz
      else
         kz = iz
      endif

      iy = ny
      do jy = 0, order_spline_yy-1
        iy = iy + 1
        if (iy < 0) then
          ky = iy + nffty
        else
          ky = iy
        endif

        ix = nx
        do jx = 0, order_spline_xx-1
          ix = ix + 1
          I_INDEX = I_INDEX + 1
          if (ix < 0) then
             kx = ix + nfftx
          else
             kx = ix
          endif

          i_adress = (ky+kz*nffty)*nfftx + kx + 1
          t_x= Ih_on_grid_dx(i,I_INDEX)
          t_y= Ih_on_grid_dy(i,I_INDEX)
          t_z= Ih_on_grid_dz(i,I_INDEX)
          
          dipole_term = t_x * p_nabla_ux + t_y * p_nabla_uy + t_z * p_nabla_uz
          q_term = ci * Ih_on_grid(i,I_INDEX)
          qqq1_Re(i_adress) = qqq1_Re(i_adress) + q_term + dipole_term! potential POINT CHARGE
          
        enddo ! jx
        enddo ! jy
      enddo ! jz
    enddo ! i
    qqq1 = cmplx(qqq1_Re,0.0d0,kind=8)

 end subroutine set_grid_Q_DIP_init_3D

! ----

subroutine set_grid_Q_DIP_cycle_2D
    use variables_smpe
    use Ewald_data
    use sim_cel_data
    use sizes_data, only : Natoms
    use all_atoms_data , only : all_charges,l_WALL, zz,is_charge_distributed,all_dipoles,&
        all_dipoles_xx,all_dipoles_yy,all_dipoles_zz
    use cg_buffer, only : NNX,NNY,NNZ,Ih_on_grid, Ih_on_grid_dx,Ih_on_grid_dy,Ih_on_grid_dz,&
                          mask_qi,mask_di_xx,mask_di_yy,mask_di_zz
    use field_constrain_data, only : ndx_remap

    implicit none
    integer i,j,k,nx,ny,nz,ix,iy,iz,jx,jy,jz,kx,ky,kz,I_INDEX
    integer ii_xx,ii_yy,ii_zz,i_adress,ii,jj,kk,i1
    integer kx1,ky1,kz1
    real(8) inv_box(3),box(3), pref
    real(8) ci,di_xx,di_yy,di_zz
    real(8) t_x,t_y,t_z
    real(8) dipole_term,q_term,p_nabla_ux,p_nabla_uy,p_nabla_uz
    real(8) axx,axy,ayx,ayy,azz

    axx=Inverse_cel(1) ; axy = Inverse_cel(2)
    ayx =Inverse_cel(4); ayy = Inverse_cel(5)

    qqq1_Re=0.0d0

    do i1 = 1,TAG_PP
      i=ndx_remap%var(i1)
      ci = mask_qi(i1)
      di_xx = mask_di_xx(i1); di_yy = mask_di_yy(i1);di_zz=mask_di_zz(i1)
      p_nabla_ux = di_xx*axx + di_yy*axy
      p_nabla_uy = di_xx*ayx + di_yy*ayy
      p_nabla_uz = di_zz
      nx = NNX(i)
      ny = NNY(i)
      nz = NNZ(i)
      iz = nz
      I_INDEX = 0
      iz = nz
      do jz = 0, order_spline_zz-1
      iz = iz + 1
      kz = iz +  h_cut_z ! h_cut_z = nfftz/2
      if (kz >= nfftz) then  ! it cannot be nfftz
        write(6,*) 'error in cg%set_q_2D kz >= nfftz; choose more nfftz points'
        write(6,*) 'You need to make nfftz at least ',int(sim_cel(9)) , &
        'or the first 2^N integer'
        write(6,*) 'kz boxz nfftz=',kz, sim_cel(9), nfftz
      STOP
      endif
      if (kz < 0) then
        write(6,*) 'error in cg%set_q_dip_2D kz < 0 : lower the splines order or increase the nfft',kz
        write(6,*) 'order spline = ',order_spline_xx,order_spline_yy,order_spline_zz
        write(6,*) 'nfft hcutz =',nfftx,nffty,nfftz,h_cut_z
        STOP
      endif
      iy = ny
      iy = ny
      do jy = 0, order_spline_yy-1
        iy = iy + 1
        if (iy < 0) then
          ky = iy + nffty
        else
          ky = iy
        endif
        ix = nx
        do jx = 0, order_spline_xx-1
          ix = ix + 1
          I_INDEX = I_INDEX + 1
          if (ix < 0) then
             kx = ix + nfftx
          else
             kx = ix
          endif
          i_adress = (ky+kz*nffty)*nfftx + kx + 1
!print*,i,jx,jy,jz,'k=',kx,ky,kz,i_adress
!read(*,*)

          t_x= Ih_on_grid_dx(i,I_INDEX)
          t_y= Ih_on_grid_dy(i,I_INDEX)
          t_z= Ih_on_grid_dz(i,I_INDEX)
          dipole_term = t_x * p_nabla_ux + t_y * p_nabla_uy + t_z * p_nabla_uz
          q_term = Ih_on_grid(i,I_INDEX) * ci
          qqq1_Re(i_adress) = qqq1_Re(i_adress) + q_term + dipole_term! potential POINT CHARGE

        enddo ! jx
        enddo ! jy
      enddo ! jz
    enddo ! i

    qqq1 = cmplx(qqq1_Re,0.0d0,kind=8)


end subroutine set_grid_Q_DIP_cycle_2D

!----------------------
subroutine set_grid_Q_DIP_cycle_3D
    use variables_smpe
    use Ewald_data
    use sim_cel_data
    use sizes_data, only : Natoms
    use all_atoms_data , only : all_charges,l_WALL, zz,is_charge_distributed,all_dipoles,&
        all_dipoles_xx,all_dipoles_yy,all_dipoles_zz
    use cg_buffer, only : NNX,NNY,NNZ,Ih_on_grid, Ih_on_grid_dx,Ih_on_grid_dy,Ih_on_grid_dz,&
                         mask_qi,mask_di_xx,mask_di_yy,mask_di_zz
    use field_constrain_data, only : ndx_remap

    implicit none
    integer i,j,k,nx,ny,nz,ix,iy,iz,jx,jy,jz,kx,ky,kz,I_INDEX
    integer ii_xx,ii_yy,ii_zz,i_adress,ii,jj,kk,i1
    integer kx1,ky1,kz1
    real(8) ci, di_xx,di_yy,di_zz
    real(8) t_x,t_y,t_z
    real(8) dipole_term,q_term,p_nabla_ux,p_nabla_uy,p_nabla_uz
    real(8) axx,axy,ayx,ayy,azz,axz,azx,ayz,azy

    axx =Inverse_cel(1); axy = Inverse_cel(2); axz = Inverse_cel(2)
    ayx =Inverse_cel(4); ayy = Inverse_cel(5); ayz = Inverse_cel(6)
    azx =Inverse_cel(7); azy = Inverse_cel(8); azz = Inverse_cel(9)

    qqq1_Re=0.0d0
    
    do i1 = 1,TAG_PP
      i=ndx_remap%var(i1)
      ci = mask_qi(i1)
      di_xx = mask_di_xx(i1); di_yy = mask_di_yy(i1);di_zz=mask_di_zz(i1)
      p_nabla_ux = di_xx*axx + di_yy*axy + di_zz*axz
      p_nabla_uy = di_xx*ayx + di_yy*ayy + di_zz*ayz
      p_nabla_uz = di_xx*azx + di_yy*azy + di_zz*azz

      nx = NNX(i)
      ny = NNY(i)
      nz = NNZ(i)
      I_INDEX = 0
      iz = nz
      do jz = 0, order_spline_zz-1
      iz = iz + 1
      if (iz < 0) then
         kz = iz + nfftz
      else
         kz = iz
      endif

      iy = ny
      do jy = 0, order_spline_yy-1
        iy = iy + 1
        if (iy < 0) then
          ky = iy + nffty
        else
          ky = iy
        endif

        ix = nx
        do jx = 0, order_spline_xx-1
          ix = ix + 1
          I_INDEX = I_INDEX + 1
          if (ix < 0) then
             kx = ix + nfftx
          else
             kx = ix
          endif

          i_adress = (ky+kz*nffty)*nfftx + kx + 1
          t_x= Ih_on_grid_dx(i,I_INDEX)
          t_y= Ih_on_grid_dy(i,I_INDEX)
          t_z= Ih_on_grid_dz(i,I_INDEX)
          
          dipole_term = t_x * p_nabla_ux + t_y * p_nabla_uy + t_z * p_nabla_uz
          q_term = ci * Ih_on_grid(i,I_INDEX)
          qqq1_Re(i_adress) = qqq1_Re(i_adress) + q_term + dipole_term! potential POINT CHARGE
          
        enddo ! jx
        enddo ! jy
      enddo ! jz
    enddo ! i

    qqq1 = cmplx(qqq1_Re,0.0d0,kind=8)

  end subroutine set_grid_Q_DIP_cycle_3D

! ---------

subroutine smpe_eval1_Q_DIP_2D
     use variables_smpe, only : qqq1
     use cg_buffer, only : pre
     use Ewald_data, only : nfftx,nffty,nfftz
     use sim_cel_data
     use cut_off_data, only : reciprocal_cut_sq
     use variables_smpe, only : reciprocal_zz
     implicit none
     real(8), parameter :: four_pi_sq = 39.4784176043574d0
     real(8), parameter :: Pi2 = 6.28318530717959d0
     integer i,j,k,ix,iy,iz,jx,jy,jz,nx,ny,nz,nx0,ny0,nz0
     integer m_iy1,m_iy2,m_ix1,m_ix2,m_iz1,m_iz2,m_ix,m_iy,m_iz
     integer ii_xx,ii_yy,ii_zz
     integer jy1,jx1,jz1,mz,mz1,i_index,k_vct
     real(8) rec_xx,rec_yy,rec_zz,rec_xy,rec_xz,rec_yx,rec_zx,rec_zy,tmp,d2
     real(8) kx , ky, kz

     nx0 = nfftx/2 ; ny0 = nffty/2 ; nz0 = nfftz/2
     k_vct = 0
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
              k_vct = k_vct + 1
              qqq1(i_index) = qqq1(i_index) * pre(k_vct)
 else
              qqq1(i_index) = (0.0d0,0.0d0)
 endif     !  reciprocal_cutt within cut off
        enddo
     enddo
    enddo
end subroutine smpe_eval1_Q_DIP_2D

! -------------------------
   subroutine smpe_eval1_Q_DIP_3D
! Eval stresses and energy for Q_POINT charges only
     use variables_smpe, only : qqq1
     use cg_buffer, only : pre
     use Ewald_data, only : nfftx,nffty,nfftz
     use sim_cel_data
     use cut_off_data, only : reciprocal_cut_sq
     implicit none
     integer i,j,k,ix,iy,iz,jx,jy,jz,nx,ny,nz,nx0,ny0,nz0
     integer m_iy1,m_iy2,m_ix1,m_ix2,m_iz1,m_iz2,m_ix,m_iy,m_iz
     integer ii_xx,ii_yy,ii_zz
     integer jy1,jx1,jz1,i_index,k_vct
     real(8) tmp
     real(8) rec_xx,rec_xy,rec_xz,rec_yx,rec_yy,rec_yz,rec_zx,rec_zy,rec_zz
     real(8) kx,ky,kz,d2

     nx0 = nfftx/2 ; ny0 = nffty/2 ; nz0 = nfftz/2
     k_vct = 0
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
              k_vct = k_vct + 1
!print*,k_vct,qqq1(i_index),pre(k_vct)
              qqq1(i_index) = qqq1(i_index) * pre(k_vct)

 else
              qqq1(i_index) = 0.0d0
 endif     !  reciprocal_cutt within cut off
        enddo
     enddo
    enddo
   

end subroutine smpe_eval1_Q_DIP_3D
! ------------------------

subroutine smpe_eval2_Q_DIP_init_2D

     use sizes_data, only : Natoms
     use sim_cel_data
     use ALL_atoms_data, only : all_p_charges,all_g_charges, fxx, fyy, fzz, xx, yy, zz,&
                                zzz,is_charge_distributed, all_charges,&
                                all_dipoles,all_dipoles_xx,all_dipoles_yy,all_dipoles_zz
     use profiles_data, only : atom_profile, l_need_2nd_profile
     use variables_smpe
     use cg_buffer, only : NNX,NNY,NNZ,Ih_on_grid, Ih_on_grid_dx,Ih_on_grid_dy,Ih_on_grid_dz, BB0
     use field_constrain_data, only : ndx_remap
     implicit none
     integer i,j,k,ix,iy,iz,kx,ky,kz,nx,ny,nz,mx,my,mz, i_index, i1,i_adress
     integer ii_xx, ii_yy , ii_zz
     real(8) qsum, spline_product, Eni0
     real(8) t, t_x, t_y, t_z
     real(8) axx,axy,ayx,ayy
     real(8), save :: eta
     logical l_i

     axx=Inverse_cel(1) ; axy = Inverse_cel(2)
     ayx =Inverse_cel(4); ayy = Inverse_cel(5)

     do i1 = 1,TAG_PP
     i = ndx_remap%var(i1)
     nx = NNX(i)
     ny = NNY(i)
     nz = NNZ(i)

        t = 0.0d0
        ii_zz = nz
        I_INDEX = 0
        do iz=1,order_spline_zz
          ii_zz = ii_zz + 1
          kz = ii_zz + nfftz/2 + 1
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
            ii_xx = nx
            do ix=1,order_spline_xx
              ii_xx = ii_xx + 1
              I_INDEX = I_INDEX + 1
              if (ii_xx < 0 ) then
                kx = ii_xx + nfftx +1
              else
                kx = ii_xx +1
              endif
              mx = kx !+ nfftx/2
              if (mx > nfftx) mx = mx - nfftx
              if (mx**2+my**2 /= 0 ) then  ! skip (0,0)
                i_adress = ((my-1)+(mz-1)*nffty)*nfftx + mx
!                dipole_term = t_x * p_nabla_ux + t_y * p_nabla_uy + t_z * p_nabla_uz
                if (i1<TAG_SP+1) then 
                   spline_product =  Ih_on_grid(i,I_INDEX)
                   Eni0 =  real(qqq1 (i_adress), kind=8) * spline_product
                   BB0(i1)=BB0(i1) -  Eni0 * 2.0d0  ! sign minus because we have Ax-B=0
                endif
                if (i1>TAG_SS) then
                   t_x= Ih_on_grid_dx(i,I_INDEX)
                   t_y= Ih_on_grid_dy(i,I_INDEX)
                   t_z= Ih_on_grid_dz(i,I_INDEX)
                   qsum = real(qqq1 (i_adress), kind=8) * 2.0d0
                   i_adress = i1 - TAG_SS + TAG_SP
                   BB0(i_adress      )=BB0(i_adress      ) - qsum*(t_x*axx+t_y*axy)
                   BB0(i_adress+  NDP)=BB0(i_adress+  NDP) - qsum*(t_x*ayx+t_y*ayy)
                   BB0(i_adress+2*NDP)=BB0(i_adress+2*NDP) - qsum*t_z 
                endif
            endif
            enddo
          enddo
        enddo

      enddo !i

end subroutine smpe_eval2_Q_DIP_init_2D

!------------------
subroutine smpe_eval2_Q_DIP_init_3D

     use sizes_data, only : Natoms
     use sim_cel_data
     use ALL_atoms_data, only : is_charge_distributed, all_charges,&
                                all_dipoles_xx,all_dipoles_yy,all_dipoles_zz
     use variables_smpe
     use cg_buffer, only : NNX,NNY,NNZ,Ih_on_grid,Ih_on_grid_dx,Ih_on_grid_dy,Ih_on_grid_dz,BB0
     use field_constrain_data, only : ndx_remap
     implicit none
     integer i,j,k,ix,iy,iz,kx,ky,kz,nx,ny,nz,mx,my,mz, i_index, i1
     integer ii_xx, ii_yy , ii_zz
     real(8) qsum, spline_product
     real(8) z, Eni0
     real(8) t, t_x, t_y, t_z
     real(8) axx,axy,axz,ayx,ayy,ayz,azx,azy,azz
     integer i_adress

     axx = Inverse_cel(1) ; axy = Inverse_cel(2); axz=Inverse_cel(3)
     ayx = Inverse_cel(4) ; ayy = Inverse_cel(5); ayz=Inverse_cel(6)
     azx = Inverse_cel(7) ; azy = Inverse_cel(8); azz=Inverse_cel(9)
     

     do i1 = 1,TAG_PP
     i = ndx_remap%var(i1)
     nx = NNX(i)
     ny = NNY(i)
     nz = NNZ(i)
        t = 0.0d0
        ii_zz = nz
        I_INDEX = 0
        do iz=1,order_spline_zz
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
            ii_xx = nx
            do ix=1,order_spline_xx
              ii_xx = ii_xx + 1
              I_INDEX = I_INDEX + 1
              if (ii_xx < 0 ) then
                kx = ii_xx + nfftx +1
              else
                kx = ii_xx +1
              endif
              mx = kx !+ nfftx/2
              if (mx > nfftx) mx = mx - nfftx
                i_adress = ((my-1)+(mz-1)*nffty)*nfftx + mx
                if (i1<TAG_SP+1) then
                   spline_product =  Ih_on_grid(i,I_INDEX)
                   Eni0 =  real(qqq1 (i_adress), kind=8) * spline_product
                   BB0(i1)=BB0(i1) -  Eni0 * 2.0d0  ! sign minus because we have Ax-B=0
                endif
                if (i1>TAG_SS) then
                   t_x= Ih_on_grid_dx(i,I_INDEX)
                   t_y= Ih_on_grid_dy(i,I_INDEX)
                   t_z= Ih_on_grid_dz(i,I_INDEX)
                   qsum = real(qqq1 (i_adress), kind=8) * 2.0d0
                   i_adress = i1 - TAG_SS + TAG_SP
                   BB0(i_adress      )=BB0(i_adress      ) - qsum*(t_x*axx+t_y*axy+t_z*axz)
                   BB0(i_adress+  NDP)=BB0(i_adress+  NDP) - qsum*(t_x*ayx+t_y*ayy+t_z*ayz)
                   BB0(i_adress+2*NDP)=BB0(i_adress+2*NDP) - qsum*(t_x*azx+t_y*azy+t_z*azz)
                endif
            enddo
          enddo
        enddo

      enddo !i

end subroutine smpe_eval2_Q_DIP_init_3D

subroutine smpe_eval2_Q_DIP_cycle_2D

     use sizes_data, only : Natoms
     use sim_cel_data
     use ALL_atoms_data, only : all_p_charges,all_g_charges, fxx, fyy, fzz, xx, yy, zz,&
                                zzz,is_charge_distributed, all_charges,&
                                all_dipoles,all_dipoles_xx,all_dipoles_yy,all_dipoles_zz
     use profiles_data, only : atom_profile, l_need_2nd_profile
     use variables_smpe
     use cg_buffer, only : NNX,NNY,NNZ,Ih_on_grid, Ih_on_grid_dx,Ih_on_grid_dy,Ih_on_grid_dz
     use field_constrain_data, only : ndx_remap
     implicit none
     integer i,j,k,ix,iy,iz,kx,ky,kz,nx,ny,nz,mx,my,mz, i_index, i1,i_adress
     integer ii_xx, ii_yy , ii_zz
     real(8) qsum, spline_product, Eni0
     real(8) t, t_x, t_y, t_z
     real(8) axx,axy,ayx,ayy
     real(8), save :: eta
     logical l_i

    axx=Inverse_cel(1) ; axy = Inverse_cel(2)
    ayx =Inverse_cel(4); ayy = Inverse_cel(5)

     do i1 = 1,TAG_PP
     i = ndx_remap%var(i1)
     nx = NNX(i)
     ny = NNY(i)
     nz = NNZ(i)

        t = 0.0d0
        ii_zz = nz
        I_INDEX = 0
        do iz=1,order_spline_zz
          ii_zz = ii_zz + 1
          kz = ii_zz + nfftz/2 + 1
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
            ii_xx = nx
            do ix=1,order_spline_xx
              ii_xx = ii_xx + 1
              I_INDEX = I_INDEX + 1
              if (ii_xx < 0 ) then
                kx = ii_xx + nfftx +1
              else
                kx = ii_xx +1
              endif
              mx = kx !+ nfftx/2
              if (mx > nfftx) mx = mx - nfftx
              if (mx**2+my**2 /= 0 ) then  ! skip (0,0)
                i_adress = ((my-1)+(mz-1)*nffty)*nfftx + mx
!                dipole_term = t_x * p_nabla_ux + t_y * p_nabla_uy + t_z * p_nabla_uz
                if (i1<TAG_SP+1) then
                   spline_product =  Ih_on_grid(i,I_INDEX)
                   Eni0 =  real(qqq1 (i_adress), kind=8) * spline_product
                   Ax(i1)=Ax(i1) +  Eni0 * 2.0d0  ! sign minus because we have Ax-B=0
                endif
                if (i1>TAG_SS) then
                   t_x= Ih_on_grid_dx(i,I_INDEX)
                   t_y= Ih_on_grid_dy(i,I_INDEX)
                   t_z= Ih_on_grid_dz(i,I_INDEX)
                   qsum = real(qqq1 (i_adress), kind=8) * 2.0d0
                   i_adress = i1 - TAG_SS + TAG_SP
                   Ax(i_adress      )=Ax(i_adress      ) + qsum*(t_x*axx+t_y*axy)
                   Ax(i_adress+  NDP)=Ax(i_adress+  NDP) + qsum*(t_x*ayx+t_y*ayy)
                   Ax(i_adress+2*NDP)=Ax(i_adress+2*NDP) + qsum*t_z
                endif
            endif
            enddo
          enddo
        enddo

      enddo !i

end subroutine smpe_eval2_Q_DIP_cycle_2D
 
! ---------------------
subroutine smpe_eval2_Q_DIP_cycle_3D
     use sizes_data, only : Natoms
     use sim_cel_data
     use ALL_atoms_data, only : is_charge_distributed, all_charges,&
                                all_dipoles_xx,all_dipoles_yy,all_dipoles_zz
     use variables_smpe
     use cg_buffer, only : NNX,NNY,NNZ,Ih_on_grid,Ih_on_grid_dx,Ih_on_grid_dy,Ih_on_grid_dz
     use field_constrain_data, only : ndx_remap
     implicit none
     integer i,j,k,ix,iy,iz,kx,ky,kz,nx,ny,nz,mx,my,mz, i_index, i1
     integer ii_xx, ii_yy , ii_zz
     real(8) qsum, spline_product
     real(8) z, Eni0
     real(8) t, t_x, t_y, t_z
     real(8) axx,axy,axz,ayx,ayy,ayz,azx,azy,azz
     integer i_adress


     axx = Inverse_cel(1) ; axy = Inverse_cel(2); axz=Inverse_cel(3)
     ayx = Inverse_cel(4) ; ayy = Inverse_cel(5); ayz=Inverse_cel(6)
     azx = Inverse_cel(7) ; azy = Inverse_cel(8); azz=Inverse_cel(9)
     

     do i1 = 1,TAG_PP
     i = ndx_remap%var(i1)
     nx = NNX(i)
     ny = NNY(i)
     nz = NNZ(i)
        t = 0.0d0
        ii_zz = nz
        I_INDEX = 0
        do iz=1,order_spline_zz
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
            ii_xx = nx
            do ix=1,order_spline_xx
              ii_xx = ii_xx + 1
              I_INDEX = I_INDEX + 1
              if (ii_xx < 0 ) then
                kx = ii_xx + nfftx +1
              else
                kx = ii_xx +1
              endif
              mx = kx !+ nfftx/2
              if (mx > nfftx) mx = mx - nfftx
                i_adress = ((my-1)+(mz-1)*nffty)*nfftx + mx
                if (i1<TAG_SP+1) then
                   spline_product =  Ih_on_grid(i,I_INDEX)
                   Eni0 =  real(qqq1 (i_adress), kind=8) * spline_product
                   AX(i1)=AX(i1) +  Eni0 * 2.0d0  ! sign minus because we have Ax-B=0
                endif
                if (i1>TAG_SS) then
                   t_x= Ih_on_grid_dx(i,I_INDEX)
                   t_y= Ih_on_grid_dy(i,I_INDEX)
                   t_z= Ih_on_grid_dz(i,I_INDEX)
                   qsum = real(qqq1 (i_adress), kind=8) * 2.0d0
                   i_adress = i1 - TAG_SS + TAG_SP
                   AX(i_adress      )=AX(i_adress      ) + qsum*(t_x*axx+t_y*axy+t_z*axz)
                   AX(i_adress+  NDP)=AX(i_adress+  NDP) + qsum*(t_x*ayx+t_y*ayy+t_z*ayz)
                   AX(i_adress+2*NDP)=AX(i_adress+2*NDP) + qsum*(t_x*azx+t_y*azy+t_z*azz)
                endif
            enddo
          enddo
        enddo

      enddo !i

end subroutine smpe_eval2_Q_DIP_cycle_3D

! -----------------------


subroutine first_iter_intra_correct_Fourier
use sizes_data, only :  Natoms
use field_constrain_data
use cut_off_data
use ALL_atoms_data, only : all_p_charges,all_g_charges, xx,yy,zz,&
   i_Style_atom, is_charge_distributed, all_charges,xxx,yyy,zzz,&
   all_dipoles,all_dipoles_xx,all_dipoles_yy,all_dipoles_zz
use cg_buffer, only : BB0,list2_ex,size_list2_ex
use boundaries, only : periodic_images
use Ewald_data
use atom_type_data, only : which_atomStyle_pair
use sim_cel_data
use max_sizes_data , only : MX_excluded
use math_constants, only : two_per_sqrt_Pi,sqrt_Pi,Pi,Pi2
use interpolate_data, only : rdr,irdr,vele,gele,vele2
 implicit none
 integer i,iii,k,j,kkk,jjj,ndx,i_pair,i_type,j_type,neightot,i1,N,j1
 integer i_1, j_1, kk,i_adress,j_adress
 real(8) qj,r,x,y,z ,qi, B0i,G1,G2,B0,B1,B2,B0i_2_xx,B0i_2_yy,B0i_2_zz,B0i_1
 real(8) r2,Inverse_r,ppp,En0,field ,t1,t2,vk1,vk2,vk
 logical l_i,l_j,l_in,l_out
 logical is_sfc_i,is_sfc_j,l_proceed,is_dip_i
 real(8) fii,CC1,x_x_x,zij,CC,CC2,CC0
 real(8) derf_x, dexp_x2
 real(8) di_xx,di_yy,di_zz,dj_xx,dj_yy,dj_zz,p_r,p_r_i, K_P, K_R, coef
 real(8), allocatable :: dx(:),dy(:),dz(:),dr_sq(:)
 integer, allocatable :: in_list(:)
 real(8) EEE,Inverse_r2,Inverse_r3,Inverse_r5,inv_r_B0,inv_r_B1,inv_r_B2
 real(8) inv_r3,inv_r5,Inverse_r_squared

allocate(dx(MX_excluded),dy(MX_excluded),dz(MX_excluded),dr_sq(MX_excluded))
allocate(in_list(MX_excluded))
  CC1 = 2.0d0*CC_alpha
  CC0 = CC_alpha

do i1 = 1,  TAG_PP
   i = ndx_remap%var(i1)
   is_sfc_i = is_sfield(i)
   is_dip_i = is_dip(i)
   i_type = i_Style_atom(i)
   neightot = size_list2_ex(i1)
!print*,i1,i,neightot
!read(*,*)
   do k = 1, neightot
     j = list2_ex(i1,k)
     dx(k) = xxx(i) - xxx(j)
     dy(k) = yyy(i) - yyy(j)
     dz(k) = zzz(i) - zzz(j)
     in_list(k) = j
   enddo
   if (neightot>0) then
   call periodic_images(dx(1:neightot),dy(1:neightot),dz(1:neightot))
   dr_sq(1:neightot) = dx(1:neightot)*dx(1:neightot) + dy(1:neightot)*dy(1:neightot) +&
       dz(1:neightot)*dz(1:neightot)
   endif
   B0i = 0.0d0
   B0i_1=0.0d0
   B0i_2_xx = 0.0d0
   B0i_2_yy = 0.0d0
   B0i_2_zz = 0.0d0
if (i1 < TAG_SS + 1) then

   do k = 1,neightot ! l_j is now always true (see before )
    j = in_list(k)
    is_sfc_j = is_sfield(j)
!    l_proceed = .not.(is_sfc_i.and.is_sfc_j)
!if(l_proceed) then
    r2 = dr_sq(k)
       j_type = i_Style_atom(j)
       r = dsqrt(r2)
!       x_x_x=Ewald_alpha*r
       Inverse_r = 1.0d0/r
       NDX = max(1,int(r*irdr))
       ppp = (r*irdr) - dble(ndx)
        vk  = vele(ndx)  ;  vk1 = vele(ndx+1) ; vk2 = vele(ndx+2)
        t1 = vk  + (vk1 - vk )*ppp
        t2 = vk1 + (vk2 - vk1)*(ppp - 1.0d0)
        B0 = (t1 + (t2-t1)*(ppp*0.5d0))
        inv_r_B0 = Inverse_r - B0

!       B0 = derfc(x_x_x)*Inverse_r
!       inv_r_B0 = Inverse_r-B0
       qj = all_charges(j)
       j1 = rec_ndx_remap%var(j)
       if (j1 <  TAG_PP + 1) then
         B0i_1 = B0i_1 + inv_r_B0 * qj 
       else
         vk  = gele(ndx)  ;  vk1 = gele(ndx+1) ; vk2 = gele(ndx+2)
         t1 = vk  + (vk1 - vk )*ppp
         t2 = vk1 + (vk2 - vk1)*(ppp - 1.0d0)
         B1 = (t1 + (t2-t1)*(ppp*0.5d0))
        Inverse_r_squared = Inverse_r*Inverse_r
        inv_r_B1 = Inverse_r*Inverse_r_squared - B1

!         Inverse_r2 = Inverse_r*Inverse_r; Inverse_r3 = Inverse_r2*Inverse_r; 
!         EEE = dexp(-x_x_x*x_x_x)
!         B1 =       (B0 + ratio_B1*EEE) * Inverse_r2
!         inv_r_B1 = Inverse_r3-B1
         p_r = all_dipoles_xx(j)*dx(k) + all_dipoles_yy(j)*dy(k)+all_dipoles_zz(j)*dz(k)
         B0i_1 = B0i_1 + inv_r_B0 * qj + inv_r_B1 * p_r
       endif
!   endif ! l_proceed
  enddo
  BB0(i1) =   BB0(i1) + B0i_1
elseif (i1 < TAG_SP+1) then ! i1 > TAG_SS
     do k = 1,neightot ! l_j is now always true (see before )
     j = in_list(k)
    is_sfc_j = is_sfield(j)
!    l_proceed = .not.(is_sfc_i.and.is_sfc_j)
!if(l_proceed) then
     r2 = dr_sq(k)
       j_type = i_Style_atom(j)
       r = dsqrt(r2)
       Inverse_r = 1.0d0/r
       Inverse_r_squared = Inverse_r*Inverse_r
       inv_r3 = Inverse_r * Inverse_r_squared
      NDX = max(1,int(r*irdr))
       ppp = (r*irdr) - dble(ndx)
        vk  = vele(ndx)  ;  vk1 = vele(ndx+1) ; vk2 = vele(ndx+2)
        t1 = vk  + (vk1 - vk )*ppp
        t2 = vk1 + (vk2 - vk1)*(ppp - 1.0d0)
        B0 = (t1 + (t2-t1)*(ppp*0.5d0))
        vk  = gele(ndx)  ;  vk1 = gele(ndx+1) ; vk2 = gele(ndx+2)
        t1 = vk  + (vk1 - vk )*ppp
        t2 = vk1 + (vk2 - vk1)*(ppp - 1.0d0)
        B1 = (t1 + (t2-t1)*(ppp*0.5d0))
        inv_r_B0 = Inverse_r - B0
        inv_r_B1 = inv_r3 - B1
!       x_x_x=Ewald_alpha*r ; EEE = dexp(-x_x_x*x_x_x)
!       B0 = derfc(x_x_x)*Inverse_r
!       inv_r_B0 = Inverse_r-B0
       qj = all_charges(j)
       j1 = rec_ndx_remap%var(j)
     if (j1 <  TAG_PP + 1) then
!       Inverse_r2= Inverse_r*Inverse_r; Inverse_r3=Inverse_r2*Inverse_r
!       B1 = (B0 + ratio_B1*EEE) * Inverse_r2
!       inv_r_B1 = Inverse_r3-B1
       B0i_1 = B0i_1 + inv_r_B0 * qj
       B0i_2_xx = B0i_2_xx + (- dx(k) )*(qj*inv_r_B1)
       B0i_2_yy = B0i_2_yy + (- dy(k) )*(qj*inv_r_B1)
       B0i_2_zz = B0i_2_zz + (- dz(k) )*(qj*inv_r_B1)
     else
       dj_xx = all_dipoles_xx(j); dj_yy = all_dipoles_yy(j) ; dj_zz = all_dipoles_zz(j)
       p_r = dj_xx*dx(k) + dj_yy*dy(k)+dj_zz*dz(k)
!       Inverse_r2= Inverse_r*Inverse_r; Inverse_r3=Inverse_r2*Inverse_r
!       B1 = (B0 + ratio_B1*EEE) * Inverse_r2
!       inv_r_B1 = Inverse_r3-B1
!       Inverse_r5 = Inverse_r3 * Inverse_r2
!       B2 = (3.0d0*B1 + ratio_B2*EEE) * Inverse_r2
!       inv_r_B2 = 3.0d0*Inverse_r5-B2
        vk  = vele2(ndx)  ;  vk1 = vele2(ndx+1) ; vk2 = vele2(ndx+2)
        t1 = vk  + (vk1 - vk )*ppp
        t2 = vk1 + (vk2 - vk1)*(ppp - 1.0d0)
        B2 = (t1 + (t2-t1)*(ppp*0.5d0))
       inv_r5 = inv_r3 * Inverse_r_squared
       inv_r_B2 = 3.0d0*inv_r5-B2
       B0i_1 = B0i_1 + inv_r_B0 * qj + inv_r_B1 * p_r
       B0i_2_xx = B0i_2_xx + (- qj*dx(k) + dj_xx)*inv_r_B1 - dx(k)*(p_r*inv_r_B2)
       B0i_2_yy = B0i_2_yy + (- qj*dy(k) + dj_yy)*inv_r_B1 - dy(k)*(p_r*inv_r_B2)
       B0i_2_zz = B0i_2_zz + (- qj*dz(k) + dj_zz)*inv_r_B1 - dz(k)*(p_r*inv_r_B2)
     endif
!endif ! l_proceed
  enddo
 i_adress = TAG_SP+i1-TAG_SS
 BB0(i1) =  BB0(i1) + B0i_1
 BB0(      i_adress) =  BB0(      i_adress) + B0i_2_xx
 BB0(  NDP+i_adress) =  BB0(  NDP+i_adress) + B0i_2_yy
 BB0(2*NDP+i_adress) =  BB0(2*NDP+i_adress) + B0i_2_zz
else  ! i1 > TAG_SP (dipol only)
     do k = 1,neightot ! l_j is now always true (see before )
     j = in_list(k)
    is_sfc_j = is_sfield(j)
!    l_proceed = .not.(is_sfc_i.and.is_sfc_j)
!if(l_proceed) then
     r2 = dr_sq(k)
       j_type = i_Style_atom(j)
       r = dsqrt(r2)
       Inverse_r = 1.0d0/r
       Inverse_r_squared = Inverse_r*Inverse_r
       inv_r3 = Inverse_r * Inverse_r_squared
      NDX = max(1,int(r*irdr))
       ppp = (r*irdr) - dble(ndx)
        vk  = gele(ndx)  ;  vk1 = gele(ndx+1) ; vk2 = gele(ndx+2)
        t1 = vk  + (vk1 - vk )*ppp
        t2 = vk1 + (vk2 - vk1)*(ppp - 1.0d0)
        B1 = (t1 + (t2-t1)*(ppp*0.5d0))
        inv_r_B1 = inv_r3 - B1

!       x_x_x=Ewald_alpha*r ; EEE = dexp(-x_x_x*x_x_x)
!       Inverse_r = 1.0d0/r
!       B0 = derfc(x_x_x)*Inverse_r
!       inv_r_B0 = Inverse_r-B0
       qj = all_charges(j)
       j1 = rec_ndx_remap%var(j)
       if (j1 <  TAG_PP + 1) then
!         Inverse_r2= Inverse_r*Inverse_r; Inverse_r3=Inverse_r2*Inverse_r
!         B1 = (B0 + ratio_B1*EEE) * Inverse_r2
!         inv_r_B1 = Inverse_r3-B1
         B0i_2_xx = B0i_2_xx + (- qj*dx(k) )*inv_r_B1
         B0i_2_yy = B0i_2_yy + (- qj*dy(k) )*inv_r_B1
         B0i_2_zz = B0i_2_zz + (- qj*dz(k) )*inv_r_B1
       else
!         Inverse_r2= Inverse_r*Inverse_r; Inverse_r3=Inverse_r2*Inverse_r
!         B1 = (B0 + ratio_B1*EEE) * Inverse_r2
!         inv_r_B1 = Inverse_r3-B1
!         Inverse_r5 = Inverse_r3 * Inverse_r2
!         B2 = (3.0d0*B1 + ratio_B2*EEE) * Inverse_r2
!         inv_r_B2 = 3.0d0*Inverse_r5-B2
        vk  = vele2(ndx)  ;  vk1 = vele2(ndx+1) ; vk2 = vele2(ndx+2)
        t1 = vk  + (vk1 - vk )*ppp
        t2 = vk1 + (vk2 - vk1)*(ppp - 1.0d0)
        B2 = (t1 + (t2-t1)*(ppp*0.5d0))
       inv_r5 = inv_r3 * Inverse_r_squared
       inv_r_B2 = 3.0d0*inv_r5-B2

         dj_xx = all_dipoles_xx(j); dj_yy = all_dipoles_yy(j) ; dj_zz = all_dipoles_zz(j)
         p_r = dj_xx*dx(k) + dj_yy*dy(k) + dj_zz*dz(k)
         B0i_2_xx = B0i_2_xx + (- qj*dx(k) + dj_xx)*inv_r_B1 - dx(k)*(p_r*inv_r_B2)
         B0i_2_yy = B0i_2_yy + (- qj*dy(k) + dj_yy)*inv_r_B1 - dy(k)*(p_r*inv_r_B2)
         B0i_2_zz = B0i_2_zz + (- qj*dz(k) + dj_zz)*inv_r_B1 - dz(k)*(p_r*inv_r_B2)
       endif
!endif ! l_proceed
  enddo
 i_adress = TAG_SP+i1-TAG_SS
 BB0(      i_adress) =  BB0(      i_adress) + B0i_2_xx
 BB0(  NDP+i_adress) =  BB0(  NDP+i_adress) + B0i_2_yy
 BB0(2*NDP+i_adress) =  BB0(2*NDP+i_adress) + B0i_2_zz
!print*, 'case 3: ',BB0(      i_adress),BB0(  NDP+i_adress), BB0(2*NDP+i_adress)
!read(*,*)
endif
enddo   ! i1

deallocate(dx,dy,dz,dr_sq)
deallocate(in_list)
end subroutine first_iter_intra_correct_fourier

subroutine first_iter_intra_correct_K_0
use sizes_data, only :  Natoms
use ALL_atoms_data, only :  xx,yy,zz, &
                           is_charge_distributed, all_charges,xxx,yyy,zzz,&
                           all_dipoles,all_dipoles_xx,all_dipoles_yy,all_dipoles_zz
use cg_buffer, only : size_list2_ex,list2_ex, BB0
use Ewald_data, only : Ewald_alpha
use field_constrain_data, only : ndx_remap, rec_ndx_remap
use math_constants, only : two_per_sqrt_Pi,sqrt_Pi,Pi,Pi2
 implicit none
 integer i,iii,k,j,kkk,jjj,neightot,i1,j1
 integer i_1, j_1, kk,i_adress,j_adress
 real(8) qj,r,x,y,z ,qi, B0i,G1,G2,B0,B1,B2,B0i_2_xx,B0i_2_yy,B0i_2_zz,B0i_1
 real(8) r2,Inverse_r,ppp,En0,field ,t1,t2,vk1,vk2,vk
 logical l_i,l_j,l_in,l_out,is_sfc_i,is_sfc_j,l_proceed
 real(8) fii,CC1,x_x_x,zij,CC,CC2
 real(8) di_xx,di_yy,di_zz,dj_xx,dj_yy,dj_zz,dexp_x2,derf_x


CC2 = 4.0d0*Ewald_alpha*sqrt_Pi
CC = sqrt_Pi/Ewald_alpha

do i1 = 1,  TAG_PP
   i = ndx_remap%var(i1)
   is_sfc_i = is_sfield(i)
   B0i_1=0.0d0
   B0i_2_zz = 0.0d0
   neightot = size_list2_ex(i1)
if (i1 < TAG_SS + 1) then
    do k = 1,neightot
    j = list2_ex(i1,k)
    is_sfc_j = is_sfield(j)
!    l_proceed = .not.(is_sfc_i.and.is_sfc_j)
!if(l_proceed) then
    j1 = rec_ndx_remap%var(j)
       z = zzz(i) - zzz(j)
       x_x_x = Ewald_alpha*z; derf_x = derf(x_x_x); dexp_x2=dexp(-x_x_x*x_x_x)
       qj = all_charges(j)
       if (j1 < TAG_PP+1) then
            B0i_1 = B0i_1 +  (qj*(CC*dexp_x2+z*Pi*derf_x) )
       else
            dj_zz = all_dipoles_zz(j)
            B0i_1 = B0i_1 +  (qj*(CC*dexp_x2+z*Pi*derf_x) - Pi*derf_x*dj_zz)
       endif
!endif ! l_proceed
  enddo
  BB0(i1) =   BB0(i1) - B0i_1 * (2.0d0*i_Area)
elseif (i1 < TAG_SP+1) then ! i1 > TAG_SS
     do k = 1,neightot ! l_j is now always true (see before )
       j = list2_ex(i1,k)
    is_sfc_j = is_sfield(j)
!    l_proceed = .not.(is_sfc_i.and.is_sfc_j)
!if(l_proceed) then
       j1=rec_ndx_remap%var(j)
       z = zzz(i) - zzz(j)
       x_x_x = Ewald_alpha*z; derf_x = derf(x_x_x); dexp_x2=dexp(-x_x_x*x_x_x)
       qj = all_charges(j)
     if (j1 <  TAG_PP + 1) then
       B0i_1 = B0i_1 + qj*(CC*dexp_x2+z*Pi*derf_x)
       B0i_2_zz = B0i_2_zz - Pi2*qj*derf_x
     else
       dj_zz = all_dipoles_zz(j)
       B0i_1 = B0i_1 + (qj*(CC*dexp_x2+z*Pi*derf_x) - Pi*derf_x*dj_zz)
       B0i_2_zz = B0i_2_zz + (-Pi2*qj*derf_x + CC2*dexp_x2*dj_zz)
     endif
!endif ! l_proceed
  enddo
 i_adress = TAG_SP+i1-TAG_SS
 BB0(i1) =  BB0(i1) - B0i_1 * (2.0d0*i_Area)
 BB0(2*NDP+i_adress) = BB0(2*NDP+i_adress) + B0i_2_zz * i_Area
else  ! i1 > TAG_SP (dipol only)
     do k = 1,neightot  ! l_j is now always true (see before )
       j = list2_ex(i1,k)
    is_sfc_j = is_sfield(j)
!    l_proceed = .not.(is_sfc_i.and.is_sfc_j)
!if(l_proceed) then
       j1=rec_ndx_remap%var(j)
       z = zzz(i) - zzz(j)
       x_x_x = Ewald_alpha*z; derf_x = derf(x_x_x); dexp_x2=dexp(-x_x_x*x_x_x)
       qj = all_charges(j)
       dj_zz = all_dipoles_zz(j)
       if (j1 <  TAG_PP + 1) then
         B0i_2_zz = B0i_2_zz - Pi2*qj*derf_x
       else
         B0i_2_zz = B0i_2_zz - Pi2*qj*derf_x + CC2*dexp_x2*dj_zz
       endif
!endif ! l_proceed
  enddo
 i_adress = TAG_SP+i1-TAG_SS
 BB0(2*NDP+i_adress) =  BB0(2*NDP+i_adress) + B0i_2_zz*i_Area
endif
enddo   ! i1

end subroutine first_iter_intra_correct_K_0


! -----------------------------------------------------


subroutine get_AX_intra_correct_fourier
use sizes_data, only :  Natoms
use non_bonded_lists_data
use field_constrain_data
use cut_off_data
use ALL_atoms_data, only : all_p_charges,all_g_charges, xx,yy,zz,&
   i_Style_atom, is_charge_distributed, all_charges,xxx,yyy,zzz,&
   all_dipoles,all_dipoles_xx,all_dipoles_yy,all_dipoles_zz
use cg_buffer, only : GG_0_excluded,GG_1_excluded,GG_2_excluded,ndx_remap,rec_ndx_remap,&
                      mask_qi,mask_di_xx,mask_di_yy,mask_di_zz,list1_ex,list2_ex,&
                      size_list1_ex,size_list2_ex
use boundaries, only : periodic_images
use Ewald_data
use atom_type_data, only : which_atomStyle_pair
use sim_cel_data
use max_sizes_data, only : MX_excluded
use math_constants, only : two_per_sqrt_Pi,sqrt_Pi,Pi,Pi2
 implicit none
 integer i,iii,k,j,kkk,jjj,ndx,i_pair,i_type,j_type,neightot,i1,N,j1
 integer i_1, j_1, kk,i_adress,j_adress
 real(8) qj,r,x,y,z ,qi, B0i,G1,G2,B0,B1,B2,B0i_2_xx,B0i_2_yy,B0i_2_zz,B0i_1
 logical l_i,l_j,l_in,l_out,is_sfc_i,is_sfc_j,l_proceed,is_dip_i
 real(8) CC1,CC,CC2,CC0
 real(8) di_xx,di_yy,di_zz,dj_xx,dj_yy,dj_zz,p_r,p_r_i
 real(8), allocatable :: dx(:),dy(:),dz(:),dr_sq(:)
 integer, allocatable :: in_list(:)
 real(8) EEE,xr,yr,zr
 real(8) Axii, Axii_1,Axii_2,Axii_3

allocate(dx(MX_excluded),dy(MX_excluded),dz(MX_excluded),dr_sq(MX_excluded))
allocate(in_list(MX_excluded))

  CC1 = 2.0d0*CC_alpha
  CC0 = CC_alpha
 do i1 = 1, NV
 i = ndx_remap%var(i1)
 is_sfc_i = is_sfield(i)
 is_dip_i = is_dip(i)
 i_adress = i1-TAG_SS+TAG_SP
  qi = mask_qi(i1)
  di_xx = mask_di_xx(i1)
  di_yy = mask_di_yy(i1)
  di_zz = mask_di_zz(i1)
   neightot = size_list1_ex(i1)
   do k = 1, neightot
     j = list1_ex(i1,k)
     dx(k) = xxx(i)-xxx(j)
     dy(k) = yyy(i)-yyy(j)
     dz(k) = zzz(i)-zzz(j)
     in_list(k) = j
   enddo
   if (neightot>0) then
   call periodic_images(dx(1:neightot),dy(1:neightot),dz(1:neightot))
   dr_sq(1:neightot)=dx(1:neightot)*dx(1:neightot)+dy(1:neightot)*dy(1:neightot)+ &
                    dz(1:neightot)*dz(1:neightot)
   endif
   Axii=0.0d0; Axii_1=0.d0; Axii_2=0.0d0; Axii_3=0.0d0
   do k = 1, neightot
          j = in_list(k)
          is_sfc_j = is_sfield(j)
!          l_proceed = .not.(is_sfc_i.and.is_sfc_j)
!if(l_proceed)then
          j1 = rec_ndx_remap%var(j)
          j_adress = j1-TAG_SS+TAG_SP
          qj = mask_qi(j1)
          dj_xx = mask_di_xx(j1)
          dj_yy = mask_di_yy(j1)
          dj_zz = mask_di_zz(j1)
          xr=dx(k);yr=dy(k);zr=dz(k)
          p_r = dj_xx*xr + dj_yy*yr + dj_zz*zr
          p_r_i = di_xx*xr+di_yy*yr+di_zz*zr
          B0 = GG_0_excluded(i1,k)
          B1 = GG_1_excluded(i1,k)
          B2 = GG_2_excluded(i1,k)
          if (is_sfc_i) then
          AXii             = AXii - ( B0*qj + B1*p_r) ! minus because take out its contribution
          endif
          if (is_dip_i) then
          AXii_1       = AXii_1 -      ( (-xr*qj+dj_xx)*B1 - xr*(p_r*B2))
          AXii_2       = AXii_2 -      ( (-yr*qj+dj_yy)*B1 - yr*(p_r*B2))
          AXii_3       = AXii_3 -      ( (-zr*qj+dj_zz)*B1 - zr*(p_r*B2))
          endif
       if (is_sfield(j)) then
          AX(j1)             = AX(j1) - ( B0*qi - B1*p_r_i)
       endif
       if (is_dip(j)) then
          AX(j_adress)       = AX(j_adress) -      ( (xr*qi+di_xx)*B1 - xr*(p_r_i*B2) )
          AX(j_adress+NDP)   = AX(j_adress+NDP) -  ( (yr*qi+di_yy)*B1 - yr*(p_r_i*B2) )
          AX(j_adress+2*NDP) = AX(j_adress+2*NDP) -( (zr*qi+di_zz)*B1 - zr*(p_r_i*B2) )
       endif
!endif !l_proceed
   enddo
   if (is_sfc_i) then
   Ax(i1) = AX(i1) + Axii
   endif
   if (is_dip_i) then
   AX(i_adress)       = AX(i_adress)       + Axii_1
   AX(i_adress+NDP)   = AX(i_adress+NDP)   + Axii_2
   AX(i_adress+2*NDP) = AX(i_adress+2*NDP) + Axii_3
   endif
 enddo ! i

deallocate(dx,dy,dz,dr_sq)
deallocate(in_list)

end subroutine get_AX_intra_correct_fourier


 subroutine get_AX_REAL  

 use cg_buffer, only : list1,size_list1, safe_mode_get_AX, mask_qi,mask_di_xx,mask_di_yy,mask_di_zz,GG_0,GG_1,GG_2,&
                       GG_0_THOLE,GG_1_THOLE
 use ALL_atoms_data, only : i_type_atom, all_p_charges,all_G_charges,zz,&
                            is_charge_distributed,xxx,yyy,zzz,&
                            all_charges, all_dipoles_xx,all_dipoles_yy,all_dipoles_zz,i_Style_atom
 use sizes_data, only : Natoms
 use max_sizes_data, only : MX_list_nonbonded
 use non_bonded_lists_data, only : list_nonbonded,size_list_nonbonded
 use field_constrain_data , only :  rec_ndx_remap,ndx_remap
 use Ewald_data
 use sim_cel_data
 use physical_constants, only : Volt_to_internal_field
 use boundaries, only : periodic_images
 use cut_off_data, only : cut_off
 use interpolate_data
 use math_constants, only : two_per_sqrt_Pi,sqrt_Pi,Pi,Pi2
 use atom_type_data, only : which_atomStyle_pair

 implicit none
 real(8) t,Axi,xi,AXi_xx,AXi_yy,AXi_zz
 integer i,iii,j,jjj,k, i1,j1,jj1,itype,jtype,i_pair,k_vct
 real(8) CC,CC1,CC2, x_x_x,zij,CC_eta
 real(8) Sum_Re,Sum_Im
 real(8), allocatable :: AAx(:)
 integer neightot,j_adress,i_adress
 real(8) xr,yr,zr,dj_xx,dj_yy,dj_zz,t0,t1,t2,p_r,qj
 real(8) eta3,ew3,eta_sq,ew2
 real(8) z
 real(8), allocatable :: dx(:),dy(:),dz(:)
 real(8) derf_x,dexp_x2,qi_t1,qj_t1,qi,di_xx,di_yy,di_zz,p_r_i
 real(8) buff3(100,3) ! a real buffer
 real(8) B1,B2,B3,B0,r2
 real(8) t1t,t2t,t1_thole,t2_thole
 real(8) vk,vk1,vk2,ppp, r, inverse_r
 real(8) cf_dip,Ewald_alpha_3,CC_eta_3,term, K_P,K_R, fct
 real(8) sn,cs,pref
 integer i_type,j_type,ndx
 integer, allocatable :: in_list(:)

 logical lgg(9)
 lgg=.false.
 allocate(dx(MX_list_nonbonded),dy(MX_list_nonbonded),dz(MX_list_nonbonded),in_list(MX_list_nonbonded))

! the charge in s-field
   CC = CC_alpha
   cf_dip = two_per_sqrt_Pi * 2.0d0/3.0d0
   Ewald_alpha_3 = Ewald_alpha * Ewald_alpha * Ewald_alpha


 if (safe_mode_get_AX) then

 do i1 = 1, NV
 i = ndx_remap%var(i1)
 i_adress = i1-TAG_SS+TAG_SP
  qi = mask_qi(i1)
  di_xx = mask_di_xx(i1)
  di_yy = mask_di_yy(i1)
  di_zz = mask_di_zz(i1)
   neightot = size_list1(i1)
   do k = 1, neightot
     j = list1(i1,k)
     dx(k) = xxx(i)-xxx(j)
     dy(k) = yyy(i)-yyy(j)
     dz(k) = zzz(i)-zzz(j)
     in_list(k) = j
   enddo
   call periodic_images(dx(1:neightot),dy(1:neightot),dz(1:neightot))

   do k = 1, neightot
          j = in_list(k) 
          j1 = rec_ndx_remap%var(j)
          j_adress = j1-TAG_SS+TAG_SP       
          qj = mask_qi(j1)
          dj_xx = mask_di_xx(j1)
          dj_yy = mask_di_yy(j1)
          dj_zz = mask_di_zz(j1) 
          xr=dx(k);yr=dy(k);zr=dz(k)
          p_r = dj_xx*xr + dj_yy*yr + dj_zz*zr
          p_r_i = di_xx*xr+di_yy*yr+di_zz*zr
          t0 = GG_0(i1,k)
          t1 = GG_1(i1,k)
          t2 = GG_2(i1,k)
          t1_thole = GG_0_THOLE(i1,k)
          t2_thole = GG_1_THOLE(i1,k)
          t1t = t1+t1_thole
          t2t = t2+t2_thole
          if (is_sfield(i)) then
          AX(i1)             = AX(i1) + t0*qj+t1*p_r
          endif
          if (is_dip(i)) then
          AX(i_adress)       = AX(i_adress)      -   xr*(qj*t1)  +  dj_xx*t1t - xr*(p_r*t2t)
          AX(i_adress+NDP)   = AX(i_adress+NDP)  -   yr*(qj*t1)  +  dj_yy*t1t - yr*(p_r*t2t)
          AX(i_adress+2*NDP) = AX(i_adress+2*NDP)-   zr*(qj*t1)  +  dj_zz*t1t - zr*(p_r*t2t)
          endif
       if (is_sfield(j)) then
          AX(j1)             = AX(j1) + t0*qi - t1*p_r_i
       endif
       if (is_dip(j)) then
          AX(j_adress)       = AX(j_adress)       +  xr*(qi*t1)  + di_xx*t1t - xr*(p_r_i*t2t)
          AX(j_adress+NDP)   = AX(j_adress+NDP)   +  yr*(qi*t1)  + di_yy*t1t - yr*(p_r_i*t2t)
          AX(j_adress+2*NDP) = AX(j_adress+2*NDP) +  zr*(qi*t1)  + di_zz*t1t - zr*(p_r_i*t2t)
       endif
   enddo
 enddo ! i
else  ! safe_mode_get_AX

 do i1 = 1, NV

!  THOLE NOT IMPLEMENTED

   i = ndx_remap%var(i1)
   Axi = 0.0d0
   neightot = size_list1(i1)
   do k = 1, neightot
     j = list1(i1,k)
     dx(k) = xxx(i)-xxx(j)
     dy(k) = yyy(i)-yyy(j)
     dz(k) = zzz(i)-zzz(j)
   enddo
   call periodic_images(dx(1:neightot),dy(1:neightot),dz(1:neightot))
   i_adress = i1 - TAG_SS+TAG_SP
if (i1 < TAG_SS + 1) then
     qi = X(i1)
     do k =1,  neightot
        j = list1(i1,k)
        j1 = rec_ndx_remap%var(j)
        if (j1 < TAG_SS + 1) then
          AXi = AXi + GG_0(i1,k)*X(j1)
          AX(j1) = AX(j1) + GG_0(i1,k)*qi
!lgg(1)=.true.
!if (i==601) write(13,*) j,GG_0(i1,k)*X(j1),' c1'!,GG_0(i1,k),X(j1)
!if (j==601) write(13,*) i,GG_0(i1,k)*qi, ' c1'!,GG_0(i1,k),qi
        else if (j1 < TAG_SP+1) then
          xr = dx(k) ; yr = dy(k) ; zr = dz(k)
          j_adress = j1-TAG_SS+TAG_SP
          dj_xx = X(j_adress); dj_yy = X(j_adress+NDP); dj_zz = X(j_adress+2*NDP) ;
          p_r   = dj_xx*xr + dj_yy*yr + dj_xx*zr
          t0 =  GG_0(i1,k)
          t1 =  GG_1(i1,k)
          qi_t1 = qi*t1
          AXi = AXi + t0*X(j1) + p_r*t1 ! dU/dq_i
          AX(j1) = AX(j1) + t0*qi       ! dU/dq_j
          AX(j_adress    )   = AX(j_adress)        +  xr*qi_t1 ! dU/dp_j
          AX(j_adress+  NDP) = AX(j_adress+NDP)    +  yr*qi_t1
          AX(j_adress+2*NDP) = AX(j_adress+2*NDP)  +  zr*qi_t1
!lgg(2)=.true.
!if (i==601) write(13,*) j,t0*X(j1) + p_r*t1,' c2'
!if (j==601) write(13,*) i,t0*qi,' c2'
         else  ! j > TAG_SP (dipole only)
          xr = dx(k) ; yr = dy(k) ; zr = dz(k)
          j_adress = j1-TAG_SS+TAG_SP
          dj_xx = X(j_adress); dj_yy = X(j_adress+NDP); dj_zz = X(j_adress+2*NDP) ; 
          p_r   = dj_xx*xr + dj_yy*yr + dj_xx*zr
          t0 =  GG_0(i1,k)
          t1 =  GG_1(i1,k)
          qi_t1 = qi*t1
          AXi = AXi + p_r*t1
          AX(j_adress    )   = AX(j_adress)        +  xr*qi_t1 ! dU/dp_j
          AX(j_adress+  NDP) = AX(j_adress+NDP)    +  yr*qi_t1
          AX(j_adress+2*NDP) = AX(j_adress+2*NDP)  +  zr*qi_t1
!lgg(3)=.true.
!if (i==601) write(13,*) j,p_r*t1,' c3'
!if (j==601) write(13,*) i,'OUCH',' c3'
         endif
     enddo
AX(i1) = AX(i1) + Axi
else if (i1 < TAG_SP + 1) then 
     AXi_xx = 0.0d0 ; AXi_yy=0.0d0 ; AXi_zz=0.0d0 ; 
     di_xx = X(i_adress); di_yy = X(i_adress+NDP); di_zz = X(i_adress+2*NDP)
     qi = X(i1)
     do k =1, neightot 
        j = list1(i1,k)
        j1 = rec_ndx_remap%var(j)
        xr = dx(k) ; yr = dy(k) ; zr = dz(k)
        qj = X(j1)
        t0 =  GG_0(i1,k)
        t1 =  GG_1(i1,k)
        if (j1 < TAG_SS + 1) then
          AXi = AXi + t0*qj
          qj_t1 = qj*t1
          AX(j1) = AX(j1) + t0*qi - (di_xx*xr+di_yy*yr+di_zz*zr)*t1
          AXi_xx = AXi_xx  - xr*qj_t1
          AXi_yy = AXi_yy  - yr*qj_t1
          AXi_zz = AXi_zz  - zr*qj_t1
!lgg(4)=.true.
!if (i==601) write(13,*) j,t0*qj,' c4'
!if (j==601) write(13,*) i, t0*qi - (di_xx*xr+di_yy*yr+di_zz*zr)*t1, ' c4'
        else if (j1 < TAG_SP + 1) then
          j_adress = j1-TAG_SS+TAG_SP
          dj_xx = X(j_adress); dj_yy = X(j_adress+NDP); dj_zz = X(j_adress+2*NDP) ; 
          p_r = dj_xx*xr + dj_yy*yr + dj_zz*zr
          p_r_i = di_xx*xr+di_yy*yr+di_zz*zr
          t2 =  GG_2(i1,k)
          AXi = AXi + t0*qj+t1*p_r
          AX(j1) = AX(j1) + t0*qi - p_r_i*t1
          AXi_xx = AXi_xx + (-xr*qj+dj_xx)*t1 - xr*(p_r*t2)
          AXi_yy = AXi_yy + (-yr*qj+dj_yy)*t1 - yr*(p_r*t2)
          AXi_zz = AXi_zz + (-zr*qj+dj_zz)*t1 - zr*(p_r*t2)
          AX(j_adress)       = AX(j_adress)       + ( xr*qi+di_xx)*t1 - xr*(p_r_i*t2)
          AX(j_adress+NDP)   = AX(j_adress+NDP)   + ( yr*qi+di_yy)*t1 - yr*(p_r_i*t2)
          AX(j_adress+2*NDP) = AX(j_adress+2*NDP) + ( zr*qi+di_zz)*t1 - zr*(p_r_i*t2)
!lgg(5)=.true.
!if (i==601) write(13,*) j,t0*qj+t1*p_r,' c5'!,t0*qj,t1*p_r,t0,qj,t1,p_r,i,i1,'i',TAG_SP
!if (j==601) write(13,*) i,t0*qi - p_r_i*t1, ' c5'!,t0*qi,- p_r_i*t1,t0,qi,t1,p_r_i,j,j1,'j',TAG_SP

         else
          j_adress = j1-TAG_SS+TAG_SP
          dj_xx = X(j_adress); dj_yy = X(j_adress+NDP); dj_zz = X(j_adress+2*NDP) ;
          p_r = dj_xx*xr + dj_yy*yr + dj_xx*zr
          p_r_i = di_xx*xr+di_yy*yr+di_zz*zr
          t2 =  GG_2(i1,k)
          AXi = AXi + t1*p_r
          AXi_xx = AXi_xx + (dj_xx)*t1 - xr*(p_r*t2)
          AXi_yy = AXi_yy + (dj_yy)*t1 - yr*(p_r*t2)
          AXi_zz = AXi_zz + (dj_zz)*t1 - zr*(p_r*t2)
          AX(j_adress)       = AX(j_adress)       + ( xr*qi+di_xx)*t1 - xr*(p_r_i*t2)
          AX(j_adress+NDP)   = AX(j_adress+NDP)   + ( yr*qi+di_yy)*t1 - yr*(p_r_i*t2)
          AX(j_adress+2*NDP) = AX(j_adress+2*NDP) + ( zr*qi+di_zz)*t1 - zr*(p_r_i*t2)
!lgg(6)=.true.
!if (i==601) write(13,*) j,t1*p_r,' c6'
!if (j==601) write(13,*) i,'OUCH', ' c6'

         endif
     enddo
AX(i1) = AX(i1) + Axi
AX(i_adress       ) = AX(i_adress       ) + AXi_xx
AX(i_adress +  NDP) = AX(i_adress +  NDP) + AXi_yy
AX(i_adress +2*NDP) = AX(i_adress +2*NDP) + AXi_zz
else  ! i1 > TAG_SP (dipole only)
     AXi_xx = 0.0d0 ; AXi_yy=0.0d0 ; AXi_zz=0.0d0
     di_xx = X(i_adress); di_yy = X(i_adress+NDP); di_zz = X(i_adress+2*NDP)
     do k =1,  neightot
        j = list1(i1,k)
        j1 = rec_ndx_remap%var(j)
        xr = dx(k) ; yr = dy(k) ; zr = dz(k)
        qj = X(j1)
        t1 =  GG_1(i1,k)
        if (j1 < TAG_SS + 1) then
          p_r_i = di_xx*xr + di_yy*yr + di_zz*zr
          AX(j1) = AX(j1) - p_r_i*t1
          qj_t1  = qj*t1
          AXi_xx = AXi_xx  - xr*qj_t1 
          AXi_yy = AXi_yy  - yr*qj_t1 
          AXi_zz = AXi_zz  - zr*qj_t1
!lgg(7)=.true.
!if (i==601) write(13,*) j,'OUCH',' c7'
!if (j==601) write(13,*) i,- p_r_i*t1, ' c7'

        else if (j1 < TAG_SP + 1) then
          j_adress = j1-TAG_SS+TAG_SP
          dj_xx = X(j_adress); dj_yy = X(j_adress+NDP); dj_zz = X(j_adress+2*NDP) ; 
          p_r = dj_xx*xr + dj_yy*yr + dj_zz*zr
          p_r_i = di_xx*xr + di_yy*yr + di_zz*zr
          t2 =  GG_2(i1,k)
          AXi_xx = AXi_xx + ( xr*qj+dj_xx)*t1 - xr*(p_r*t2)
          AXi_yy = AXi_yy + ( yr*qj+dj_yy)*t1 - yr*(p_r*t2)
          AXi_zz = AXi_zz + ( zr*qj+dj_zz)*t1 - zr*(p_r*t2)
          AX(j1) = AX(j1) - p_r_i*t1
          AX(j_adress      ) = AX(j_adress      ) + di_xx*t1-xr*(p_r_i*t2)
          AX(j_adress+NDP  ) = AX(j_adress+NDP  ) + di_yy*t1-yr*(p_r_i*t2)
          AX(j_adress+2*NDP) = AX(j_adress+2*NDP) + di_zz*t1-zr*(p_r_i*t2)
!lgg(8)=.true.
!if (i==601) write(13,*) j,'OUCH',' c8'
!if (j==601) write(13,*) i,- p_r_i*t1, ' c8'

        else
          j_adress = j1-TAG_SS+TAG_SP
          dj_xx = X(j_adress); dj_yy = X(j_adress+NDP); dj_zz = X(j_adress+2*NDP) ;
          p_r = dj_xx*xr + dj_yy*yr + dj_xx*zr
          p_r_i = di_xx*xr + di_yy*yr + di_zz*zr
          t2 =  GG_2(i1,k)
          AXi_xx = AXi_xx + (dj_xx)*t1 - xr*(p_r*t2)
          AXi_yy = AXi_yy + (dj_yy)*t1 - yr*(p_r*t2)
          AXi_zz = AXi_zz + (dj_zz)*t1 - zr*(p_r*t2)
          AX(j_adress      ) = AX(j_adress      ) + di_xx*t1-xr*(p_r_i*t2)
          AX(j_adress+NDP  ) = AX(j_adress+NDP  ) + di_yy*t1-yr*(p_r_i*t2)
          AX(j_adress+2*NDP) = AX(j_adress+2*NDP) + di_zz*t1-zr*(p_r_i*t2)
!lgg(9)=.true.
!if (i==601) write(13,*) j,'OUCH',' c9'
!if (j==601) write(13,*) i,'OUCH', ' c9'

        endif
     enddo
i_adress = i1-TAG_SS+TAG_SP
AX(i_adress       ) = AX(i_adress       ) + AXi_xx
AX(i_adress +  NDP) = AX(i_adress +  NDP) + AXi_yy
AX(i_adress +2*NDP) = AX(i_adress +2*NDP) + AXi_zz
endif

enddo ! i1
! \\\End REAL PART
!print*,'lgg=',lgg

endif ! safe_mode_get_AX

end subroutine get_AX_REAL


 subroutine get_AX_14

 use cg_buffer, only : list1_14,size_list1_14, safe_mode_get_AX, mask_qi,mask_di_xx,mask_di_yy,mask_di_zz,&
                       GG_0_14,GG_1_14,GG_2_14
 use ALL_atoms_data, only : i_type_atom, all_p_charges,all_G_charges,zz,&
                            is_charge_distributed,xxx,yyy,zzz,&
                            all_charges, all_dipoles_xx,all_dipoles_yy,all_dipoles_zz,i_Style_atom
 use sizes_data, only : Natoms
 use max_sizes_data, only : MX_in_list_14
 use field_constrain_data , only :   rec_ndx_remap,ndx_remap
 use Ewald_data
 use sim_cel_data
 use physical_constants, only : Volt_to_internal_field
 use boundaries, only : periodic_images
 use cut_off_data, only : cut_off
 use interpolate_data
 use math_constants, only : two_per_sqrt_Pi,sqrt_Pi,Pi,Pi2
 use atom_type_data, only : which_atomStyle_pair
 use connectivity_ALL_data, only : l_red_14_Q_CTRL,l_red_14_Q_mu_CTRL,l_red_14_mu_mu_CTRL,&
                                  red_14_Q,red_14_Q_mu,red_14_mu_mu
 implicit none
 real(8) t,Axi,xi,AXi_xx,AXi_yy,AXi_zz
 integer i,iii,j,jjj,k, i1,j1,jj1,itype,jtype,i_pair,k_vct
 real(8) CC,CC1,CC2, x_x_x,zij,CC_eta
 real(8) Sum_Re,Sum_Im
 real(8), allocatable :: AAx(:)
 integer neightot,j_adress,i_adress
 real(8) xr,yr,zr,dj_xx,dj_yy,dj_zz,t0,t1,t2,p_r,qj
 real(8) eta3,ew3,eta_sq,ew2
 real(8) z
 real(8), allocatable :: dx(:),dy(:),dz(:)
 real(8) derf_x,dexp_x2,qi_t1,qj_t1,qi,di_xx,di_yy,di_zz,p_r_i
 real(8) buff3(100,3) ! a real buffer
 real(8) B1,B2,B3,B0,r2
 real(8) t1t,t2t,t1_thole,t2_thole
 real(8) vk,vk1,vk2,ppp, r, inverse_r
 real(8) cf_dip,Ewald_alpha_3,CC_eta_3,term, K_P,K_R, fct
 real(8) sn,cs,pref
 integer i_type,j_type,ndx
 integer, allocatable :: in_list(:)

 allocate(dx(MX_in_list_14),dy(MX_in_list_14),dz(MX_in_list_14),in_list(MX_in_list_14))

   CC = CC_alpha
   cf_dip = two_per_sqrt_Pi * 2.0d0/3.0d0
   Ewald_alpha_3 = Ewald_alpha * Ewald_alpha * Ewald_alpha


 if (safe_mode_get_AX) then

 do i1 = 1, NV
 i = ndx_remap%var(i1)
 i_adress = i1-TAG_SS+TAG_SP
  qi = mask_qi(i1)
  di_xx = mask_di_xx(i1)
  di_yy = mask_di_yy(i1)
  di_zz = mask_di_zz(i1)
   neightot = size_list1_14(i1)
   do k = 1, neightot
     j = list1_14(i1,k)
     dx(k) = xxx(i)-xxx(j)
     dy(k) = yyy(i)-yyy(j)
     dz(k) = zzz(i)-zzz(j)
     in_list(k) = j
   enddo
   call periodic_images(dx(1:neightot),dy(1:neightot),dz(1:neightot))
   do k = 1, neightot
          j = in_list(k)
          j1 = rec_ndx_remap%var(j)
          j_adress = j1-TAG_SS+TAG_SP
          qj = mask_qi(j1)
          dj_xx = mask_di_xx(j1)
          dj_yy = mask_di_yy(j1)
          dj_zz = mask_di_zz(j1)
          xr=dx(k);yr=dy(k);zr=dz(k)
          p_r = dj_xx*xr + dj_yy*yr + dj_zz*zr
          p_r_i = di_xx*xr+di_yy*yr+di_zz*zr
          t0 = GG_0_14(i1,k)
          t1 = GG_1_14(i1,k)
          t2 = GG_2_14(i1,k)
          if (is_sfield(i)) then
          AX(i1)             = AX(i1) + t0*qj*(-red_14_Q)+t1*p_r*(-red_14_Q_mu)
          endif
          if (is_dip(i)) then
          AX(i_adress)       = AX(i_adress)      -   xr*(qj*t1*(-red_14_Q_mu) ) +  (dj_xx*t1 - xr*(p_r*t2))*(-red_14_mu_mu)
          AX(i_adress+NDP)   = AX(i_adress+NDP)  -   yr*(qj*t1*(-red_14_Q_mu) ) +  (dj_yy*t1 - yr*(p_r*t2))*(-red_14_mu_mu)
          AX(i_adress+2*NDP) = AX(i_adress+2*NDP)-   zr*(qj*t1*(-red_14_Q_mu) ) +  (dj_zz*t1 - zr*(p_r*t2))*(-red_14_mu_mu)
          endif
       if (is_sfield(j)) then
          AX(j1)             = AX(j1) + t0*qi*(-red_14_Q) - t1*p_r_i*(-red_14_Q_mu)
       endif
       if (is_dip(j)) then
          AX(j_adress)       = AX(j_adress)       +  xr*(qi*t1*(-red_14_Q_mu) ) + (di_xx*t1 - xr*(p_r_i*t2))*(-red_14_mu_mu)
          AX(j_adress+NDP)   = AX(j_adress+NDP)   +  yr*(qi*t1*(-red_14_Q_mu) ) + (di_yy*t1 - yr*(p_r_i*t2))*(-red_14_mu_mu)
          AX(j_adress+2*NDP) = AX(j_adress+2*NDP) +  zr*(qi*t1*(-red_14_Q_mu) ) + (di_zz*t1 - zr*(p_r_i*t2))*(-red_14_mu_mu)
       endif
   enddo
 enddo ! i
else  ! safe_mode_get_AX
  print*,'NOT.safe_mode_eval in get_AX_14 not implemented'
  STOP
!  I would not bother with it.....
endif
end subroutine get_AX_14


subroutine get_AX_self_interact
 use ALL_atoms_data, only : i_Style_atom, is_charge_distributed
 use Ewald_data
 use field_constrain_data,only : ndx_remap
 use math_constants, only : two_per_sqrt_Pi,sqrt_Pi
 implicit none
 real(8), parameter :: cf_dip = two_per_sqrt_Pi * 2.0d0/3.0d0
 integer i,j,k , i1, i_adress,itype
 real(8) CC_eta,CC_eta_3,term, Ewald_alpha_3, CC
 real(8) di_xx,di_yy,di_zz

! DOING SELF INTERACTIONS

    Ewald_alpha_3=Ewald_alpha*Ewald_alpha*Ewald_alpha
    CC = CC_alpha

    do i1 = 1, TAG_SP  ! charges first
     i = ndx_remap%var(i1)
     if (is_charge_distributed(i)) then
      itype = i_Style_atom(i)
      CC_eta = Ewald_eta(itype,itype)
    else
      CC_eta = 0.0d0
    endif
     AX(i1) = AX(i1) + two_per_sqrt_Pi * X(i1) * (CC_eta - Ewald_alpha)   ! self interactions added here
   enddo

   do i1 = TAG_SS+1,TAG_PP
     i_adress = i1-TAG_SS+TAG_SP
     i = ndx_remap%var(i1)
     if (is_charge_distributed(i)) then
      itype = i_Style_atom(i)
      CC_eta = Ewald_eta(itype,itype)
    else
      CC_eta = 0.0d0
    endif
    CC_eta_3 = CC_eta * CC_eta * CC_eta
    term = cf_dip * (cc_eta_3 - Ewald_alpha_3)
    di_xx = X(i_adress) ; di_yy=X(i_adress+NDP) ; di_zz = X(i_adress+2*NDP)
    AX(i_adress      ) = AX(i_adress      ) + term * di_xx
    AX(i_adress+  NDP) = AX(i_adress+  NDP) + term * di_yy
    AX(i_adress+2*NDP) = AX(i_adress+2*NDP) + term * di_zz
!print*,i1,'self AX=',term * di_xx
   enddo

! DONE WITH SELF INTERACTIONS

end subroutine get_AX_self_interact


subroutine get_Ax_at_k0_2D

use spline_z_k0_module
 use array_math
 use ALL_atoms_data, only : zzz,zz,all_charges,&
                            Natoms,all_charges,all_dipoles_zz
 use sim_cel_data
 use cg_buffer, only : BB0,mask_qi,mask_di_xx,mask_di_yy,mask_di_zz
 use Ewald_data, only : Ewald_alpha
 use field_constrain_data
 use math_constants, only : two_per_sqrt_Pi,sqrt_Pi,Pi,Pi2
 implicit none
 integer i,j,k,i1,om2,i_adress,kk0,kkk,kk2
 real(8) CC,CC2
 integer N
 real(8) sum_field_q,sum_field_miu,x_x_x,qi,di_zz,field,z,derf_x,dexp_x2
 real(8), allocatable :: field_grid_q(:),alp_q(:),alp_miu(:),field_grid_miu(:),MAT(:,:)

 allocate(alp_q(Ngrid),alp_miu(Ngrid))
 allocate(MAT(Ngrid,Ngrid))
 allocate(field_grid_q(Ngrid),field_grid_miu(Ngrid))

N = Ngrid - 1
CC2 = 4.0d0*Ewald_alpha*sqrt_Pi
CC = sqrt_Pi/Ewald_alpha
 do k = 1, Ngrid
 sum_field_q=0.0d0 ; sum_field_miu = 0.0d0
 do i1 = 1, TAG_PP  ! Only charges
 i = ndx_remap%var(i1)
      z = z_grid(k) - zzz(i)
      qi = mask_qi(i1)
      di_zz = mask_di_zz(i1)
      x_x_x = Ewald_alpha*z ; derf_x = derf(x_x_x); dexp_x2=dexp(-x_x_x*x_x_x)
      field = qi*(CC*dexp_x2+z*Pi*derf_x) - Pi*derf_x*di_zz
      sum_field_q = sum_field_q + field !
 enddo
 do i1 = 1, TAG_PP  ! charges+dipoles
 i = ndx_remap%var(i1)
      z = z_grid(k) - zzz(i)
      qi = mask_qi(i1)
      di_zz = mask_di_zz(i1)
      x_x_x = Ewald_alpha*z ; derf_x = derf(x_x_x); dexp_x2=dexp(-x_x_x*x_x_x)
      field = (-Pi2*qi*derf_x + CC2*dexp_x2*di_zz)
      sum_field_miu = sum_field_miu  + field
 enddo
 field_grid_q(k) = sum_field_q
 field_grid_miu(k) = sum_field_miu
 enddo
 field_grid_q = field_grid_q * ( (-2.0d0) * i_area)
 field_grid_miu = field_grid_miu * (   i_area)


 kk2 = mod(order,2)+1
 do i = 0,n
     z = z_grid(i+1)
     call deboor_cox(order,Ngrid, order+1, kkk, qq, z, bv)
     MAT(i+1,1:n+1) = bv(1:n+1,kk2)
 enddo ! i=1,n
 call invmat(MAT,Ngrid,Ngrid)
 do i = 1, Ngrid
   alp_q(i) = dot_product(MAT(i,:),field_grid_q(:))
   alp_miu(i) = dot_product(MAT(i,:),field_grid_miu(:))
 enddo
 om2 = mod(order,2)+1

 do i1 = 1, TAG_PP
     i = ndx_remap%var(i1)
     z = zzz(i)
     call deboor_cox(order,Ngrid, order+1, kkk, qq, z, bv)
     j   = kkk - order;
     if (i1 < TAG_SP + 1) Ax(i1) = Ax(i1) + dot_product(alp_q(j+1:kkk+1),bv(j+1:kkk+1,om2)) 
     if (i1 > TAG_SS ) then
       i_adress = i1 - TAG_SS + TAG_SP
       Ax(i_adress+2*NDP) = Ax(i_adress+2*NDP) + dot_product(alp_miu(j+1:kkk+1),bv(j+1:kkk+1,om2))
     endif
 enddo

deallocate(MAT)
deallocate(field_grid_q,field_grid_miu)
deallocate(alp_miu,alp_q)
end subroutine get_Ax_at_k0_2D

! -------------------------------
subroutine get_Ax_at_k0_2D_SLOW
use ALL_atoms_data, only : zz,zzz
use math_constants, only : Pi, Pi2,sqrt_Pi
use cg_buffer, only : mask_qi,mask_di_zz
use field_constrain_data, only : ndx_remap
use ewald_data, only : Ewald_alpha

implicit none
integer i,j,k,i1,j1, i_adress,j_adress
real(8) qi,qj,di_xx,di_yy,di_zz,dj_xx,dj_yy,dj_zz,derf_x,dexp_x2,x_x_x,z,CC,CC2

 CC = CC_alpha
 CC2 = 4.0d0*Ewald_alpha*sqrt_Pi

 do i1 = 1, NV
 i = ndx_remap%var(i1)
 i_adress = i1-TAG_SS+TAG_SP
 qi = mask_qi(i1)
 di_zz = mask_di_zz(i1)
  do j1 = i1+1,NV
   j = ndx_remap%var(j1)
   if (i/=j) then
   j_adress = j1-TAG_SS+TAG_SP
   qj = mask_qi(j1)
   dj_zz = mask_di_zz(j1)
   z = zzz(i) - zzz(j)
   x_x_x = Ewald_alpha*z;
   derf_x = derf(x_x_x); dexp_x2=dexp(-x_x_x*x_x_x)
   if (is_sfield(i)) then
     Ax(i1) = Ax(i1) - (qj*(CC*dexp_x2+z*Pi*derf_x) - Pi*derf_x*dj_zz) * (2.0d0 * i_Area)
   endif
   if (is_dip(i)) then
     Ax(i_adress+2*NDP) = Ax(i_adress+2*NDP) + (-Pi2*qj*derf_x + CC2*dexp_x2*dj_zz) * ( i_Area)
   endif
   if (is_sfield(j)) then
     Ax(j1) = Ax(j1) - (qi*(CC*dexp_x2+z*Pi*derf_x) + Pi*derf_x*di_zz) * (2.0d0 * i_Area)
   endif
   if (is_dip(j)) then
     Ax(j_adress+2*NDP) = Ax(j_adress+2*NDP) + ( Pi2*qi*derf_x + CC2*dexp_x2*di_zz) * ( i_Area)
   endif
   endif ! i/=j
  enddo
  enddo
! add the self
 do i1 = 1, NV
    i = ndx_remap%var(i1)
    i_adress = i1-TAG_SS+TAG_SP
    qi = mask_qi(i1)
    di_zz = mask_di_zz(i1)
    if (is_sfield(i)) AX(i1) = AX(i1) - qi*CC  * (2.0d0 * i_Area)
    if (is_dip(i)) &
       Ax(i_adress+2*NDP) = Ax(i_adress+2*NDP) + ( CC2*di_zz) * ( i_Area)
 enddo

end subroutine get_Ax_at_k0_2D_SLOW
! -------------------------------

subroutine get_AX_k_NON_0_2D_SLOW
use cg_buffer, only : mask_qi,mask_di_xx,mask_di_yy,mask_di_zz,pre,sns,css
use field_constrain_data, only : ndx_remap
implicit none
integer i,j,k,i_adress,i1,j1,k_vct
real(8) Sum_Im,Sum_Re,pref,cs,sn,fct,qi,di_xx,di_yy,di_zz,K_R,KP
real(8), allocatable :: AAX(:)

   allocate(AAx(NVFC))
   AAx = 0.0d0
   do k_vct = 1, N_K_vct
   Sum_Re = 0.0d0 ; Sum_Im = 0.0d0
   do j1 = 1, NV
      j = ndx_remap%var(j1)
      qi = mask_qi(j1)
      K_R = mask_di_xx(j1)*k_vector(k_vct,1)+mask_di_yy(j1)*k_vector(k_vct,2)+mask_di_zz(j1)*k_vector(k_vct,3)
      Sum_Re = Sum_Re + css(j,k_vct) * qi - sns(j,k_vct) * K_R
      Sum_Im = Sum_Im + sns(j,k_vct) * qi + css(j,k_vct) * K_R
   enddo ! j1

   do i1 = 1, NV
     i = ndx_remap%var(i1)
     i_adress = i1 - TAG_SS + TAG_SP
     cs = css(i,k_vct) ; sn = sns(i,k_vct) ; pref = pre(k_vct)
     fct = ( cs*Sum_Re+sn*Sum_Im ) * pref
     if (i1 < TAG_SP+1) AAx(i1) = AAx(i1) + fct
     fct = ( cs*Sum_Im-sn*Sum_Re ) * pref
     if (i1 > TAG_SS) then
       AAx(i_adress        ) = AAx(i_adress        ) + fct * k_vector(k_vct,1)
       AAx(i_adress +   NDP) = AAx(i_adress +   NDP) + fct * k_vector(k_vct,2)
       AAx(i_adress + 2*NDP) = AAx(i_adress + 2*NDP) + fct * k_vector(k_vct,3)
     endif
   enddo
   enddo ! k_vct

   Ax(:) = Ax(:) + AAx(:) * ( 2.0d0 * i_Area * h_step )

 deallocate(AAx)


end subroutine get_AX_k_NON_0_2D_SLOW
! ------------------------------

subroutine get_Ax_fourier_3D_SLOW
use cg_buffer, only : mask_qi,mask_di_xx,mask_di_yy,mask_di_zz,pre,sns,css
use sim_cel_data, only : Volume
use math_constants, only : Pi2
use field_constrain_data, only : ndx_remap
implicit none
integer i,j,k,i_adress,i1,j1,k_vct
real(8) Sum_Im,Sum_Re,pref,cs,sn,fct,qi,di_xx,di_yy,di_zz,K_R,KP
real(8), allocatable :: AAX(:)

   allocate(AAx(NVFC))
   AAx = 0.0d0
   do k_vct = 1, N_K_vct
   Sum_Re = 0.0d0 ; Sum_Im = 0.0d0
   do j1 = 1, NV
      j = ndx_remap%var(j1)
      qi = mask_qi(j1)
      K_R = mask_di_xx(j1)*k_vector(k_vct,1)+mask_di_yy(j1)*k_vector(k_vct,2)+mask_di_zz(j1)*k_vector(k_vct,3)
      Sum_Re = Sum_Re + css(j,k_vct) * qi - sns(j,k_vct) * K_R
      Sum_Im = Sum_Im + sns(j,k_vct) * qi + css(j,k_vct) * K_R
   enddo ! j1

   do i1 = 1, NV
     i = ndx_remap%var(i1)
     i_adress = i1 - TAG_SS + TAG_SP
     cs = css(i,k_vct) ; sn = sns(i,k_vct) ; pref = pre(k_vct)
     fct = ( cs*Sum_Re+sn*Sum_Im ) * pref
     if (i1 < TAG_SP+1) AAx(i1) = AAx(i1) + fct
     fct = ( cs*Sum_Im-sn*Sum_Re ) * pref
     if (i1 > TAG_SS) then
       AAx(i_adress        ) = AAx(i_adress        ) + fct * k_vector(k_vct,1)
       AAx(i_adress +   NDP) = AAx(i_adress +   NDP) + fct * k_vector(k_vct,2)
       AAx(i_adress + 2*NDP) = AAx(i_adress + 2*NDP) + fct * k_vector(k_vct,3)
     endif
   enddo
   enddo ! k_vct

   Ax(:) = Ax(:) + AAx(:) * ( 2.0d0 * Pi2 /  Volume )

 deallocate(AAx)


end subroutine get_Ax_fourier_3D_SLOW


! ---------------------------------------------
subroutine get_AX_intra_correct_k_0
use sizes_data, only :  Natoms
use ALL_atoms_data, only :  xx,yy,zz, &
                           is_charge_distributed, all_charges,xxx,yyy,zzz,&
                           all_dipoles,all_dipoles_xx,all_dipoles_yy,all_dipoles_zz
use cg_buffer, only : size_list2_ex,list2_ex,mask_qi,mask_di_xx,mask_di_yy,mask_di_zz,&
                      size_list1_ex,list1_ex
use Ewald_data, only : Ewald_alpha
use field_constrain_data, only : ndx_remap, rec_ndx_remap
use math_constants, only : two_per_sqrt_Pi,sqrt_Pi,Pi,Pi2
 implicit none
 integer i,iii,k,j,kkk,jjj,neightot,i1,j1
 integer i_1, j_1, kk,i_adress,j_adress
 real(8) qj,r,x,y,z ,qi, B0i,G1,G2,B0,B1,B2,B0i_2_xx,B0i_2_yy,B0i_2_zz,B0i_1
 real(8) r2,Inverse_r,ppp,En0,field ,t1,t2,vk1,vk2,vk
 logical l_i,l_j,l_in,l_out, is_sfc_i,is_dip_i,is_sfc_j,l_proceed
 real(8) fii,CC1,x_x_x,zij,CC,CC2
 real(8) di_xx,di_yy,di_zz,dj_xx,dj_yy,dj_zz,dexp_x2,derf_x
 real(8) Axii,Axii_1


CC2 = 4.0d0*Ewald_alpha*sqrt_Pi
CC = sqrt_Pi/Ewald_alpha

 do i1 = 1, NV
 i = ndx_remap%var(i1)
 is_sfc_i = is_sfield(i)
 is_dip_i = is_dip(i)
 i_adress = i1-TAG_SS+TAG_SP
 qi = mask_qi(i1)
 di_zz = mask_di_zz(i1)
 Axii = 0.0d0; Axii_1 = 0.0d0
  do k=1,size_list1_ex(i1)
   j = list1_ex(i1,k)
   is_sfc_j = is_sfield(j)
!   l_proceed = .not.(is_sfc_i.and.is_sfc_j)
!if(l_proceed) then
   j1 = rec_ndx_remap%var(j)
   j_adress = j1-TAG_SS+TAG_SP
   qj = mask_qi(j1)
   dj_zz = mask_di_zz(j1)
   z = zzz(i) - zzz(j)
   x_x_x = Ewald_alpha*z;
   derf_x = derf(x_x_x); dexp_x2=dexp(-x_x_x*x_x_x)
   if (is_sfc_i) then
     Axii = Axii + (qj*(CC*dexp_x2+z*Pi*derf_x) - Pi*derf_x*dj_zz) * (2.0d0 * i_Area)
   endif
   if (is_dip_i) then
     Axii_1 = Axii_1 - (-Pi2*qj*derf_x + CC2*dexp_x2*dj_zz) * ( i_Area)
   endif
   if (is_sfield(j)) then
     Ax(j1) = Ax(j1) + (qi*(CC*dexp_x2+z*Pi*derf_x) + Pi*derf_x*di_zz) * (2.0d0 * i_Area)
   endif
   if (is_dip(j)) then
     Ax(j_adress+2*NDP) = Ax(j_adress+2*NDP) - ( Pi2*qi*derf_x + CC2*dexp_x2*di_zz) * ( i_Area)
   endif
!endif ! l_proceed
  enddo
  if (is_sfc_i) then
     Ax(i1) = Ax(i1) + Axii
  endif
  if (is_dip_i) then
     Ax(i_adress+2*NDP) = Ax(i_adress+2*NDP) + Axii_1
  endif

  enddo

end subroutine get_AX_intra_correct_k_0

subroutine mask_vars  !(X)
use cg_buffer, only : mask_di_xx,mask_di_yy,mask_di_zz,mask_qi
use field_constrain_data, only : ndx_remap
 implicit none

 integer i,j,k,i1,i_adress
 
  do i1 = 1, NV
   i = ndx_remap%var(i1)
    i_adress = i1-TAG_SS+TAG_SP
    if (is_sfield(i)) then
     mask_qi(i1) = X(i1)
    else
     mask_qi(i1) = 0.0d0
    endif
    if (is_dip(i)) then
      mask_di_xx(i1) = X(i_adress)
      mask_di_yy(i1) = X(i_adress+NDP)
      mask_di_zz(i1) = X(i_adress+2*NDP)
    else
      mask_di_xx(i1) = 0.0d0
      mask_di_yy(i1) = 0.0d0
      mask_di_zz(i1) = 0.0d0
    endif
  enddo
end subroutine mask_vars



subroutine get_sfield_free_term(N,B)
 use field_constrain_data
 use ALL_atoms_data, only : external_sfield_CONSTR, all_DIPOLE_pol,&
                            all_dipoles_xx,all_dipoles_yy,all_dipoles_zz
 use cg_buffer, only :  BB0, ndx_remap
 implicit none
 integer,  intent(IN):: N
 real(8), intent(OUT) :: B(N)
 integer i,i1,i2
 real(8) di_xx,di_yy,di_zz

 B=BB0
  do i = 1, TAG_SP ! N_atoms_sfield_constrained
     B(i) = B(i) + external_sfield_CONSTR(ndx_remap%var(i))   ! + because I have AX - B = 0
  enddo
 BB0=B
end subroutine get_sfield_free_term

subroutine get_dipol_free_term(XX,AAx)
 use field_constrain_data
 use ALL_atoms_data, only : external_sfield_CONSTR, all_DIPOLE_pol,&
                            all_dipoles_xx,all_dipoles_yy,all_dipoles_zz
 use cg_buffer, only :   ndx_remap
 implicit none
 integer i,i1,i2
 real(8) di_xx,di_yy,di_zz
 real(8), intent(IN) :: XX(:)
 real(8), intent(INOUT) :: AAx(:) 
!real(8) AX0(N)
! AX0=0.0d0


  do i1 = TAG_SS+1,TAG_PP
     i2 = i1 - TAG_SS + TAG_SP
     i = ndx_remap%var(i1)
     di_xx = XX(i2       )
     di_yy = XX(i2+   NDP)
     di_zz = XX(i2+ 2*NDP)
     AAX(i2       ) =  AAX(i2      ) + di_xx*all_DIPOLE_pol(i)
     AAX(i2+   NDP) =  AAX(i2+  NDP) + di_yy*all_DIPOLE_pol(i)
     AAX(i2+ 2*NDP) =  AAX(i2+2*NDP) + di_zz*all_DIPOLE_pol(i)
  enddo

! AX = AX + AX0

end subroutine get_dipol_free_term

end module cg_Q_DIP_module
