module data_module
implicit none
integer N
real(8) , parameter :: fq = 1.0d0
integer , parameter :: MAX_NEIGH = 2000  ! The maximum number of atoms expected to neighbour a given atom within a short ranged cut off. 
real(8) cut,cut_sq  ! The actual cut off wihith  which interactions are computed 
real(8) totalE, en_vdw, box(3)
real(8) ff_A12, ff_B6 ! force field parameters
real(8), allocatable :: xyz(:,:), qq(:), q(:),V(:,:)
real(8), allocatable :: fxx(:) , fyy(:) , fzz(:) ! x y z component of forces
integer, allocatable :: list(:,:) ! list (i, :) contains the idensity of all atoms located within a short ranged cut off from atom "i"
integer, allocatable ::  size_list(:) ! size_list(i) is the number of atoms within cut off from a central atom "i"
character(250) my_code_location
end module data_module



module local_initializations_module
contains
subroutine init ! these are global initializations
use comunications, only : COMM_init,COMM
use data_module,  only : cut,cut_sq,ff_A12, ff_B6,en_vdw
implicit none
   call COMM_init()  ! Initialzie the paralel environment
   cut = 10.0d0    ! This is the short ranged cut off within which we compute the short ranged interations . It is in Angstroms
   cut_sq=cut*cut
   ff_A12 = 1.0d5  ! this is the v.d.w. repulsive parameter
   ff_B6 = 1.0d3   ! this is the v.d.w. dispersion
   en_vdw=0.0d0 ! set initial energy to zero
end subroutine init

subroutine local_init
use data_module
implicit none
 allocate (fxx(N),fyy(N),fzz(N))  ! allocate memory for forces array
 fxx=0.0d0; fyy=0.0d0;fzz=0.0d0 ! initialize forces to zero
 allocate(size_list(N),list(N,MAX_NEIGH))  ! allocate memory for neighbours list arrays
 size_list=0; ! initialize the size_list array to zero
end subroutine local_init
end module local_initializations_module

subroutine finalize
use data_module
use comunications, only : COMM_exit
 deallocate(xyz); deallocate(fxx,fyy,fzz)
 call COMM_exit   ! close the paralel environment 
end subroutine finalize

module IO_module
implicit none
contains
subroutine read_config(file_name)
use data_module, only : xyz, N, box
use comunications, only :  COMM
implicit none
character(*),intent(IN) :: file_name
integer i
if (COMM%is_master) then ! if I am in master node read input data
 open (unit=10, file=trim(file_name))  ! open the input file
 read(10,*) N ! read the number of atoms 
 read(10,*) box ! read the size of simulation cell
 allocate (xyz(N,3))  ! allocate memory for cordinates  arrays. 
 do i=1,N
   read(10,*) xyz(i,1:3)  ! read the coordinates of atoms
 enddo
 close(10)
endif
end subroutine read_config

subroutine bcast_config
use data_module, only : xyz, N, box
use comunications, only : COMM_bcast, COMM, COMM_syncronize
  call COMM_bcast(N)
  call COMM_bcast(box)
  call COMM_syncronize()
  if (COMM%is_slave)then
   allocate (xyz(N,3)) ! need syncronization before allocation to make sure I have the N broadcased to slaves
  endif
  call COMM_syncronize()
  call COMM_bcast(xyz)
  call COMM_syncronize() ! however I am not quite sure I need so many COMM_syncronize() .... 
end subroutine bcast_config

subroutine out_results ! output results
use data_module
use comunications, only : COMM
use error_handler_module, only : error_code
implicit none
if(COMM%is_master)then
 print*, 'energy vdw=',en_vdw
 print*, 'forces=0? ',sum(fxx),sum(fyy),sum(fzz)
 if (error_code==0) then
   print*, 'Finished OK'
 else
   print*, 'WARNING: Finished with error_code=',error_code
 endif
endif
end subroutine out_results

end module IO_module




