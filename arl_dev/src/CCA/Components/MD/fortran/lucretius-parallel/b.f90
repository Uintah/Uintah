
 integer, parameter :: MX14 = 499
 integer, parameter :: MX15 =
 integer iv14(MX14),iv15(MX15),i,j,k
 logical l_1

 open(unit=15,file='../lucretius_integrate/fort.15')
 do i = 1, MX
  read(14,*) ibla,iv14(i)
  read(15,*)  ibla,iv15(i)
 enddo


 do i = 1, MX14
 l_1=.false.
  do j = 1, MX15
   if (iv14(i)==iv15(j)) l_1 = .true.
  enddo
 if (.not.l_1) then
   print* , ' I=',i,iv14(i)
  read(*,*)
 endif

 enddo

print*,':'
 do j = 1, MX15
 l_1=.false.
  do i = 1, MX14
   if (iv14(i)==iv15(j)) l_1 = .true.
  enddo
 if (.not.l_1) then
   print* , ' I=',j,iv15(j)
  read(*,*)
 endif
 
 enddo

  

end
