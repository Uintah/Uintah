
module list_generator_module
! start generate list. This is slow
contains 
subroutine build_list_serial ! 
use data_module
use error_handler_module

implicit none
integer i,j,k,rank,Ncpus
real(8) r2,t(3)
my_code_location='list_generator_module->build_list_serial'
rank = 0  ! the id of my CPUS
Ncpus = 1   ! The number of CPU
do i = 1+rank,N,Ncpus   ! this is how I split the computations between CPUs
do j = i+1, N  ! start building the list of neighbours around atom i 
if(i/=j)then
t = xyz(i,:) - xyz(j,:)  ! the vector distance between atom i and j
t = t - ANINT(t/box)*box   ! this is required for periodic boudary conditions 
if(dabs(t(3)) < cut) then  ! this is for speed
if(dabs(t(2)) < cut) then
if(dabs(t(1)) < cut) then
        r2=dsqrt(t(1)**2+t(2)**2+t(3)**2)
        if (r2 < cut_sq ) then  ! select only atoms "j" within spherical cut-off around atom "i"
             size_list(i) = size_list(i) + 1 ! count one more atom around i
             if (size_list(i)  > MAX_NEIGH ) then ! If array overflow STOP
                     error_code=100
                     call error_handler(my_code_location,'')
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

end subroutine build_list_serial

subroutine build_list_paralel_v0 !
use data_module
use comunications, only : COMM,COMM_syncronize
use error_handler_module
implicit none
integer i,j,k,rank,Ncpus
real(8) r2,t(3)
my_code_location='list_generator_module->build_list_paralel_v0'
rank = COMM%my_rank  ! the id of my CPUS
Ncpus = COMM%Ncpus   ! The number of CPU
do i = 1+rank,N,Ncpus   ! this is how I split the computations between CPUs
!print*,i,rank
do j = i+1, N  ! start building the list of neighbours around atom i
if(i/=j)then
t = xyz(i,:) - xyz(j,:)  ! the vector distance between atom i and j
t = t - ANINT(t/box)*box   ! this is required for periodic boudary conditions
if(dabs(t(3)) < cut) then  ! this is for speed
if(dabs(t(2)) < cut) then
if(dabs(t(1)) < cut) then
        r2=dsqrt(t(1)**2+t(2)**2+t(3)**2)
        if (r2 < cut_sq ) then  ! select only atoms "j" within spherical cut-off around atom "i"
             size_list(i) = size_list(i) + 1 ! count one more atom around i
             if (size_list(i)  > MAX_NEIGH ) then ! If array overflow STOP
                     error_code=100
                     call error_handler(my_code_location,'')
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
call COMM_syncronize()
end subroutine build_list_paralel_v0


end module list_generator_module



