
 module functii
 implicit none
 contains
 subroutine func(N,x,f)
 integer N
 real(8) x(N)
 real(8) f(N)
  f(1) = 2.0d0*x(1)+3.0d0*x(2)-5.0d0
  f(2) = 5.0d0*x(1)-x(2) -4.0d0
 end subroutine func
 
 subroutine jacoby(N,x,Jac,iJac)
 integer N
 real(8)jac(n,n),x(n),iJac(n,n)
 real(8) det
  jac(1,1) = 2.0d0 ; jac(1,2) = 3.0d0
  jac(2,1) = 5.0d0 ; jac(2,2) = -1.0d0

  det = jac(1,1)*jac(2,2) - jac(2,1)*jac(2,1)
  iJac(1,1) = jac(2,2)/det;  iJac(1,2) = - jac(2,1)/det
  iJac(2,1) = - jac(1,2)/det ; iJac(2,2) = Jac(1,1)/det
 end subroutine jacoby


 end module functii

 program main
 use functii
 implicit none
 integer, parameter :: N=2
 real(8) x(N), x_new(N), x_mixt(N), x_old(N), f(N), Jac(N,N),iJac(N,N)
 real(8) :: alpha = 0.1d0
 integer i

 x_old(1) = 0.1d0
 x_old(2) = 0.1d0

 do i = 1, 1000
  call func(N,x_old,f)
  call jacoby(N,x,Jac,iJac); !iJac = 1.0d0; iJac(1,2) = 0.0d0; iJac(2,1) = 0.0d0
  x_new(1) = iJac(1,1)*f(1)+iJac(1,2)*f(2)
  x_new(2) = iJac(2,1)*f(1)+iJac(2,2)*f(2)
  x_new = x_old - x_new
  x_mixt = alpha*x_old + (1.0d0-alpha)*x_new
  print*, i,'x_old x_mixt=',x_old,x_mixt
read(*,*)
  x_old = x_mixt
 enddo

 end program main
