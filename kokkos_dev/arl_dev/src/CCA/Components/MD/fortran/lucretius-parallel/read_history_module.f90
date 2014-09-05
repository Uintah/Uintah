

module read_history_module
implicit none
real(8), allocatable :: xyz_history(:,:,:)
contains

subroutine read_history(ifile)
use sim_cel_data
use ALL_atoms_data
integer, intent(IN):: ifile


logical, save :: first_time=.true.
integer, save :: icel,ix,iv,ifor
integer i,j,k,Na,Nm,Nta,Ntm
integer ibla
real(8) bla
real(8),allocatable :: liv(:)



if (first_time) then
  first_time=.false.
  read(ifile) 
  read(ifile) !liv(1:Ntm); deallocate(liv)

  read(ifile) sim_cel,i_boundary_CTRL
  print*,'sim_cel=',sim_cel
  read(ifile) i,icel,ix,iv,ifor
  print*, 'i icel ix iv ifor=',i,icel,ix,iv,ifor
  read(ifile) 
endif
!allocate(xyz_history(Nrecords, Natoms, 3))

  read(ifile) 
!  if (icel==1) read(33) sim_cel,i_boundary_CTRL
   if (ix==1)    read(ifile) xxx(1:Natoms),yyy(1:Natoms),zzz(1:Natoms)
!  if (iv==1)    read(33) vxx,vyy,vzz
!  if (ifor==1)    read(33) fxx,fyy,fzz
!  xyz_history(i,:,1) = xxx; xyz_history(i,:,2)=yyy; xyz_history(i,:,3)=zzz
end subroutine read_history


end module read_history_module



