
 integer i,j,k,MX,i1
 real(8) a,b,a1,b1

 MX = 19200

 i1 = 0
 do i = 1, MX
  read(14,*) ibla, a,b
  read(15,*) ibla, a1,b1
  if (dabs(a-a1) > 5.0d-12) then
   i1 = i1 + 1
   print*, i1,i,a,a1,a-a1
!   read(*,*)
  endif
 enddo

 end

