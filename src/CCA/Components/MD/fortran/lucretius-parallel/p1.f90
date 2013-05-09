 integer i,j,k

 do i = 1, 100000
 read(13,*) j
 read(14,*) k
 print*,i,j,k,j-k
 if(mod(i,20)==0)read(*,*)
 enddo

 end 
