 integer Natoms,NNN,TAG_SP,NDP
 real(8) a,b,c
 integer i,j,k
 real(8), allocatable :: fi(:),V(:),di(:,:)
 integer, allocatable :: ndx(:)


open(unit=888,file='fort.888',recl=1000)

 read(666,*) NNN, TAG_SP,NDP
 read(77,*) Natoms
 allocate(fi(Natoms),di(Natoms,3))

 allocate(ndx(NNN))
 allocate(V(NNN)) 
 do i = 1, NNN
   read(666,*) ndx(i),V(i)
 enddo
 do i = 1, Natoms
   read(77,*) 
   read(77,*) fi(i)
   read(77,*) di(i,:)
 enddo

 do i = 1, NNN
   j = ndx(i)
   if (i<TAG_SP+1) b=fi(j)
   if (i>TAG_SP.and.i<TAG_SP+NDP+1) b=di(j,1)
   if (i>TAG_SP+NDP.and.i<TAG_SP+2*NDP+1) b=di(j,2)
   if (i>TAG_SP+2*NDP) b=di(j,3)
   write(888,*)i,j,V(i),b,V(i)-b
   if (dabs(V(i)-b)>9.2d-12) print*,i,j,V(i),b,V(i)-b
  enddo
  
 end

