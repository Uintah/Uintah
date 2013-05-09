 integer, parameter :: NNN = 153
 real(8) a,b,c
 integer i,j,k
 real(8), allocatable :: V1(:),V(:)
 integer, allocatable :: ndx(:),NDX1(:)


 allocate(V1(NNN))
 allocate(ndx1(NNN))
 allocate(ndx(NNN))
 allocate(V(NNN)) 
 do i = 1, NNN
   read(15,*) ndx1(i),V1(i)
 enddo
 do i = 1, NNN
   read(14,*) ndx(i),V(i)
 enddo

 do i = 1, NNN
 do j = 1, NNN
 if (ndx(i)==ndx1(j))then
 if (dabs(V(i)-V1(j))>1.0d-12)then
   print*,i,j,ndx(i),ndx1(j),V(i),V1(j),V(i)-V1(j)
read(*,*)
 endif
write(111,*) V(i),V1(j),V(i)-V1(j)
 endif
 enddo;enddo
  
 end

