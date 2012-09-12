! compute the energy and forces ; once I have the list (which are slow to compute), this is faster
! en_vdw=0.0d0

module vdw_forces_modules
contains
subroutine vdw_forces_serial 
use data_module
implicit none
integer i,j,k,rank,Ncpus
real(8) en,r,r2,t(3),qi,qij, r6,r12, ir2,ir6,ir12,ffxx,ffyy,ffzz,fxx_i,fyy_i,fzz_i,T12,T6,ff
my_code_location='vdw_forces_modules->vdw_forces_serial'
rank = 0  ! the id of my CPUS
Ncpus = 1   ! The number of CPU
fxx=0.0d0; fyy=0.0d0; fzz=0.0d0;
do i = 1+rank,N,Ncpus 
   fxx_i=0.0d0; fyy_i=0.0d0; fzz_i=0.0d0
   do k = 1, size_list(i)  ! loop over all neighbours of "i"
     j = list(i,k)  ! extract the identity of neoghbour 
     t = xyz(i,:) - xyz(j,:)
     t = t - ANINT(t/box)*box   ! this is required for periodic boudary conditions
     r2=(t(1)**2+t(2)**2+t(3)**2)
     ir2 = 1.0d0 / r2   ! 1/r^2
     ir6 = ir2*ir2*ir2     ! 1/r^6
     ir12 = ir6*ir6       ! 1/r^12
     T12 = ff_A12 * ir12  ; T6  =  ff_B6* ir6
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

subroutine vdw_forces_paralel_v0 
use data_module
use comunications, only : COMM,COMM_syncronize
implicit none
integer i,j,k, rank,Ncpus
real(8) en,r,r2,t(3),qi,qij, r6,r12, ir2,ir6,ir12,ffxx,ffyy,ffzz,fxx_i,fyy_i,fzz_i,T12,T6,ff
my_code_location='vdw_forces_modules->vdw_forces_paralel_v0'
fxx=0.0d0; fyy=0.0d0; fzz=0.0d0;
rank = COMM%my_rank  ! the id of my CPUS
Ncpus = COMM%Ncpus   ! The number of CPU
do i = 1+rank,N, Ncpus
   fxx_i=0.0d0; fyy_i=0.0d0; fzz_i=0.0d0
   do k = 1, size_list(i)  ! loop over all neighbours of "i"
     j = list(i,k)  ! extract the identity of neoghbour
     t = xyz(i,:) - xyz(j,:)
     t = t - ANINT(t/box)*box   ! this is required for periodic boudary conditions
     r2=(t(1)**2+t(2)**2+t(3)**2)
     ir2 = 1.0d0 / r2   ! 1/r^2
     ir6 = ir2*ir2*ir2     ! 1/r^6
     ir12 = ir6*ir6       ! 1/r^12
     T12 = ff_A12 * ir12  ; T6  =  ff_B6* ir6
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
call COMM_syncronize();
end subroutine vdw_forces_paralel_v0

subroutine forces(update_list) ! the main driver subroutine to get forces
use list_generator_module
use comunications, only : COMM
use data_module, only : my_code_location
logical, intent(IN) :: update_list
my_code_location='vdw_forces_modules->forces'
if (COMM%Ncpus==1)then  ! this is serial
 if (update_list) then
  call build_list_serial
  call vdw_forces_serial
 else
  call vdw_forces_serial
 endif 
else ! COMM%Ncpus>1 ; this will be parralel
 if (update_list) then
  call build_list_paralel_v0
  call vdw_forces_paralel_v0
 else
  call vdw_forces_paralel_v0
 endif
 call reduce_forces   
endif
end subroutine forces

subroutine reduce_forces  ! colect in master all forces from all nodes. Master contain a sum of what is on all nodes
use comunications
use data_module, only : fxx,fyy,fzz,en_vdw,N,my_code_location
implicit none
real(8), allocatable :: buffer(:)
integer i,j,k
my_code_location='vdw_forces_modules->reduce_forces'
allocate(buffer(3*N+1))
do i = 1, N
 buffer(i) = fxx(i) 
 buffer(N+i) = fyy(i)
 buffer(2*N+i) = fzz(i)
enddo 
buffer(3*N+1) = en_vdw
call COMM_gsum(buffer) !  collect forces from all nodes and sum up the computed  contrinbution on each node
if (COMM%is_master) then ! only in master I need the correct forces
 do i = 1, N
   fxx(i) = buffer(i)
   fyy(i) = buffer(N+i)
   fzz(i) = buffer(2*N+i)
 enddo
en_vdw = buffer(3*N+1)
endif
deallocate(buffer)
call COMM_syncronize()
end subroutine reduce_forces

end module vdw_forces_modules



