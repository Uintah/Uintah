integer i,j,k,N,i1
real(8) a,b,a1,b1
do k = 1, 10000
  read(13,*) i,a,b
  read(14,*) i1,a1,b1
if (dabs(a-a1)>1.0d-12) then
  print*,k,a,a1,a-a1, 'i i1=',i,i1
   read(*,*) 
endif
enddo

end
