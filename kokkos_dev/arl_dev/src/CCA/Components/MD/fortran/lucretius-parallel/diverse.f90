
 module diverse

 implicit none
 contains
 subroutine Re_Center_forces
 use sizes_data, only : Natoms
 use ALL_atoms_data, only : fxx,fyy,fzz
 implicit none
 real(8) sx,sy,sz
 integer i
  sx = sum(fxx(1:Natoms))/dble(Natoms)
  sy = sum(fyy(1:Natoms))/dble(Natoms)
  sz = sum(fzz(1:Natoms))/dble(Natoms)
  do i = 1, Natoms
     fxx(i) = fxx(i) - sx
     fyy(i) = fyy(i) - sy
     fzz(i) = fzz(i) - sz
  enddo
 !print*, 'momentum conservation', sx*dble(Natoms), sy*dble(Natoms),sz*dble(Natoms) 
 end subroutine Re_Center_forces
 end module diverse
