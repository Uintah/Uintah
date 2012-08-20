!===============================================================================
module data_module
!==============================================================================
implicit none
integer N
real(8) , parameter :: fq = 1.0d0
integer , parameter :: MAX_NEIGH = 2000  ! The maximum number of atoms expected to neighbour a given atom within a short ranged cut off. 
real(8) cut,cut_sq  ! The actual cut off with  which interactions are computed
real(8) totalE, en_vdw, box(3)
real(8) ff_A12, ff_B6 ! force field parameters
real(8), allocatable :: xyz(:,:) !, qq(:), q(:),V(:,:)
real(8), allocatable :: fxx(:) , fyy(:) , fzz(:) ! x y z component of forces
integer, allocatable :: list(:,:) ! list (i, :) contains the idensity of all atoms located within a short ranged cut off from atom "i"
integer, allocatable ::  size_list(:) ! size_list(i) is the number of atoms within cut off from a central atom "i"
end module data_module


!==============================================================================
module local_initializations_module
!==============================================================================
contains
subroutine local_init
use data_module
implicit none
  cut = 10.0d0    ! This is the short ranged cut off within which we compute the short ranged interations. It is in Angstroms
  cut_sq=cut*cut
  ff_A12 = 1.0d5  ! this is the v.d.w. repulsive parameter
  ff_B6 = 1.0d3   ! this is the v.d.w. dispersion
  en_vdw=0.0d0 ! set initial energy to zero
  allocate (fxx(N),fyy(N),fzz(N))  ! allocate memory for forces array
  fxx=0.0d0; fyy=0.0d0;fzz=0.0d0 ! initialize forces to zero
  allocate(size_list(N),list(N,MAX_NEIGH))  ! allocate memory for neighbours list arrays
  size_list=0; ! initialize the size_list array to zero
end subroutine local_init
end module local_initializations_module


!==============================================================================
! file IO
!==============================================================================
module IO_module
implicit none
contains
subroutine read_config(file_name)
use data_module, only : xyz, N, box
implicit none
character(*),intent(IN) :: file_name
integer i
  open (unit=10, file=trim(file_name))  ! open the input file
  read(10,*) N ! read the number of atoms
  read(10,*) box ! read the size of simulation cell
  allocate (xyz(N,3))  ! allocate memory for cordinates  arrays.
  do i=1,N
    read(10,*) xyz(i,1:3)  ! read the coordinates of atoms
  enddo
  close(10)
end subroutine read_config


!==============================================================================
! output results
!==============================================================================
subroutine out_results ! output results
use data_module
implicit none
  print*, 'energy vdw = ',en_vdw
  print*, 'forces     = 0? ',sum(fxx),sum(fyy),sum(fzz)
end subroutine out_results
end module IO_module


!==============================================================================
! start generate list. This is slow
!==============================================================================
module list_generator_module
contains 
subroutine build_list
use data_module
implicit none
integer i,j
real(8) r2,t(3)
do i = 1,N-1
  if(mod(i,1000)==0)print*,i,N
    do j = i+1, N  ! start building the list of neighbours around atom i
      if(i/=j)then ! i != j
        t = xyz(i,:) - xyz(j,:)  ! the vector distance between atom i and j
        t = t - ANINT(t/box)*box   ! this is required for periodic boudary conditions
        if(dabs(t(3)) < cut) then  ! this is for speed
          if(dabs(t(2)) < cut) then
            if(dabs(t(1)) < cut) then
              r2=dsqrt(t(1)**2 + t(2)**2 + t(3)**2)
                if (r2 < cut_sq ) then  ! select only atoms "j" within spherical cut-off around atom "i"
                  size_list(i) = size_list(i) + 1 ! count one more atom around i
                  if (size_list(i)  > MAX_NEIGH ) then ! If array overflow STOP
                    print*,'ERROR size_list(i)  > MAX_NEIGH: Increase MAX_NEIGH'
                    stop
                  endif
                  list(i,size_list(i)) = j  ! store in the array list(i,:) the identity of a new atom
                endif
             endif
           endif
        endif
      endif ! i/=j
    enddo   ! just finished the loop
enddo   ! just finished the list builder
!end generate list
print*, 'LIST GENERATED '
end subroutine build_list
end module list_generator_module


!==============================================================================
! compute the energy and forces:
!    once list is generated (which is slow), this is faster
!==============================================================================
! en_vdw=0.0d0
module vdw_forces_modules
contains
subroutine vdw_forces_serial 
use data_module
implicit none
integer i,j,k
real(8) en,r2,t(3), ir2,ir6,ir12,ffxx,ffyy,ffzz,fxx_i,fyy_i,fzz_i,T12,T6,ff
fxx=0.0d0; fyy=0.0d0; fzz=0.0d0;
do i = 1,N
  fxx_i=0.0d0; fyy_i=0.0d0; fzz_i=0.0d0
  do k = 1, size_list(i)  ! loop over all neighbours of "i"
    j = list(i,k)  ! extract the identity of neighbour
    t = xyz(i,:) - xyz(j,:)
    t = t - ANINT(t/box)*box   ! this is required for periodic boudary conditions
    r2=(t(1)**2+t(2)**2+t(3)**2)
    ir2 = 1.0d0 / r2   ! 1/r^2
    ir6 = ir2*ir2*ir2     ! 1/r^6
    ir12 = ir6*ir6       ! 1/r^12
    T12 = ff_A12 * ir12
    T6  =  ff_B6* ir6
    En = T12 - T6  ! Energy
    en_vdw = en_vdw + En ! count the energy
    ff = (12.0d0 * T12 - 6.0d0*T6)*ir2  ! the force term
    ffxx = ff*t(1) ; ffyy = ff*t(2) ; ffzz = ff*t(3)
    fxx(j) = fxx(j) - ffxx  ! the force on atom j
    fyy(j) = fyy(j) - ffyy
    fzz(j) = fzz(j) - ffzz
    fxx_i = fxx_i + ffxx ! the contribution of force on atom i
    fyy_i = fyy_i + ffyy
    fzz_i = fzz_i + ffzz
  enddo
  fxx(i) = fxx(i) + fxx_i ! sum up contributions to force for atom i
  fyy(i) = fyy(i) + fyy_i
  fzz(i) = fzz(i) + fzz_i
enddo
totalE= en_vdw
end subroutine vdw_forces_serial


!==============================================================================
! setup to compute forces; the main driver subroutine to get forces
!==============================================================================
subroutine forces(update_list)
use list_generator_module, only : build_list
logical, intent(IN) :: update_list
if (update_list) then
  call build_list
  call vdw_forces_serial
else
  call vdw_forces_serial
endif
end subroutine forces
end module vdw_forces_modules


!==============================================================================
! main driver, program entry point
!==============================================================================
program main                     ! finally the main program
use local_initializations_module ! this contains the initializations that we use in this test
use IO_module                    ! input (read config) and output 
use list_generator_module        ! generate the list ; for each atom we store in an array how many other atoms are within cut off 
use vdw_forces_modules           ! evaluate energy forces 
implicit none

call read_config('input.medium') ! get the configuration with atom coordinates; change 'input.small' with whatever file i.e. input.large (if needed)
call local_init ! initialize the local variablibels used in this test program 
call forces(.true.) ! compute forces ! this is the main thing
call out_results
 
end program main

