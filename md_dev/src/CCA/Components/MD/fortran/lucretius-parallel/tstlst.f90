
 implicit none
 real(8) box(3),bla
 integer ,parameter:: Natoms = 11040
 integer, parameter :: NDIM = 2
 real(8) :: cut_off = 10.0d0
 real(8) :: displacement = 1.0d0
 real(8) xyz(Natoms,3) 
 real(8) t(3),r,r2
 integer i,j,k,imol
 integer aim(Natoms)
 integer size_list(Natoms)
 integer , allocatable :: list(:,:)

 allocate( list(Natoms,1300))

 open(unit=10,file='./runs/config.config')
 read(10,*); read(10,*)

 read(10,*) box(1)
 read(10,*) bla,box(2)
 read(10,*) bla,bla,box(3)
print*, 'box=',box
 close(10)
  open(unit=10,file='./runs/config.config')
 do i = 1, 21
  read(10,*)
 enddo
 do i = 1, Natoms
   read(10,*) xyz(i,:)
   xyz(i,1:NDIM) = xyz(i,1:NDIM) - ANINT(xyz(i,1:NDIM)/box(1:NDIM))*box(1:NDIM)
 enddo
 print*, 'extreme z=',minval(xyz(:,3)),maxval(xyz(:,3)), box(3)/2-(cut_off-displacement)
 imol = 1
 do i = 1, Natoms
 aim(i) = imol
 if (mod(i,6) == 0 ) imol = imol+1
 enddo 


 size_list=0 
 do i = 1, Natoms
 do j = i+1, Natoms
   if (aim(i) /= aim(j)) then
    t = xyz(i,:)-xyz(j,:)
    t(1:NDIM) = t(1:NDIM) - ANINT(t(1:NDIM)/box(1:NDIM))*box(1:NDIM)
    r2 = dot_product(t,t)
    r = dsqrt(r2)
    if (r < (cut_off+displacement)) then
     size_list(i) = size_list(i)+1
     list(i,size_list(i)) = j
    endif
   endif
 enddo
 enddo 

 print* , 'sum size_list=',sum(size_list)
 do i = 1, Natoms
 write(77,*) i, size_list(i)
 write(77,*) list(i,1:size_list(i))
 write(77,*) '-----'
 enddo


end
