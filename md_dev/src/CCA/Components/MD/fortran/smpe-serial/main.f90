
program main ! finally the main program
use sim_cel_data ! this contains the initializations that we use in this test
use IO_module                    ! input (read config) and output 
use energies_data
use smpe_driver
use Ewald_data
use boundaries
use slow_fourier_ewald_3D
use ALL_atoms_data
implicit none
integer i,iseed,N_trials
character(250) nf_config

! <local initializations>
iseed = -89675
i_boundary_CTRL = 2 ! 3D periodic boundaries
nf_config = 'config.small' ! or 'config.large'
N_trials = 1 ! when do tinming change N_trials to larger number (say 100) for longer running times and better stats on running times.
! </local_initializations>

call read_parameters('in.in')
call read_config(trim(nf_config))  ! here the ALL_atoms arrays are allocated as well and then read from cfg and storred
call local_init();

print*,'start'
do i = 1, N_trials
if (i_type_EWALD_CTRL == 1) then
  call smpe_Q ! this is the actual main smpe subroutine
else if (i_type_EWALD_CTRL == 2) then
  call Ew_Q_SLOW ! this is for testing smpe; impractical for large systems.
else
  print*, 'ERROR: i_type_EWALD_CTRL can be either 1(smpe) or 2(slow) '
  STOP
endif
enddo
print*, 'smpe Ewald en = ',En_Q_cmplx  ! lets print oout the energy 
print*,'f1=', fxx(1),fyy(1),fzz(1)     ! and lets print out the force on first atom

contains 
 subroutine local_init
 use random_generator_module, only : randomize_config
   call cel_properties(.true.)
   call randomize_config(0.25d0,iseed,xxx,yyy,zzz) ! if I dont randomize it then fft fart coukd be zero due to crystal symmetry
   xx=xxx;yy=yyy;zz=zzz;
   call periodic_images(xx,yy,zz)
   call get_reciprocal_cut
   En_Q_cmplx=0.0d0 ! set energy to zero before computing it
   fxx=0.0d0;fyy=0.0d0;fzz=0.0d0 ! set forces to zero before computing
   dfftx=nfftx;dffty=nffty;dfftz=nfftz
 end subroutine local_init
end program main

