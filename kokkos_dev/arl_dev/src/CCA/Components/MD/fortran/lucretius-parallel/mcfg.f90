 integer N,Nmols,i,j
 real(8) box(3)
 real(8) , allocatable :: xyz(:,:),q(:)
 integer, allocatable :: aim(:)

 open(unit=10,file='a')
 read(10,*)N
 allocate(xyz(N,3),q(N),aim(N))
 read(10,*) box
 Nmols = N/6

 do i = 1, N
 read(10,*) xyz(i,:),q(i),aim(i)
 enddo

 close(10)

 open(unit=10,file='config.config')
 write(10,*)
 write(10,*) 'SIM_BOX'
 write(10,*) box(1),0.0d0,0.0d0
 write(10,*) 0.0d0 ,box(2),0.0d0
 write(10,*) 0.0d0, 0.0d0  ,box(3)
 write(10,*) 1
 write(10,*) 
 write(10,*) 'INDEX_CONTINUE_JOB'
 write(10,*) '0 0 0'
 write(10,*)
 write(10,*) 'MOLECULES_AND_ATOMS'
 write(10,*) Nmols,N
 write(10,*)
 write(10,*) 'ENSAMBLE'
 write(10,*) 0
 write(10,*) 
 write(10,*) 'THERMOSTAT'
 write(10,*)
 write(10,*) 'BAROSTAT'
 write(10,*)

 write(10,*) 'XYZ       : atom positions'
 do i = 1, N
 write(10,*) xyz(i,:) 
 enddo 

 write(10,*) 'GAUSS_CHARGE   :Qg '
 do i = 1, N
 write(10,*) 0.0d0,'   .false.'
 enddo

close(10) 
end
