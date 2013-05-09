module spline_z_k0_module
! it is doing the interpolation at k=0 for a 2D-Ewald
implicit none
public :: get_z_grid
public :: get_qq_coordinate
public :: deboor_cox
public :: spline_k0_deriv

contains

 subroutine get_z_grid(order,Ngrid,z_grid)
 use sim_cel_data
 implicit none
 integer, intent(IN) :: order,Ngrid
 real(8), intent(OUT) :: z_grid(0:Ngrid-1)
 real(8) dg
 integer k

 dg = sim_cel(9) /dble(ngrid-1)
 do k = 0, Ngrid-1
   z_grid(k) = -sim_cel(9)*0.5d0 + dg *dble(k)
 enddo
 end subroutine get_z_grid 

 subroutine get_qq_coordinate(order,Ngrid,z_grid,qq)
! Ngrid must be > order ; otherwise nonsense
 implicit none
 integer, intent(IN) :: order,Ngrid
 real(8), intent(IN) :: z_grid(0:Ngrid-1)
 real(8), intent(OUT) :: qq(0:Order+Ngrid)
 integer i,j,k
 integer ioo1

    ioo1 = Order+1
    do i = 0, order
      qq(i) = z_grid(0)
    enddo
    do i = 0, Ngrid-ioo1-1
      qq(i+ioo1) = ( z_grid(i)+z_grid(i+ioo1) ) * 0.5d0
    enddo
    do i = 0, order
      qq(i+Ngrid) = z_grid(Ngrid-1)
    enddo

 end subroutine get_qq_coordinate

 subroutine deboor_cox(order,Ngrid,kai,kkk,qq,x,bv)
 real(8), parameter :: Thin = 1.0d-6
 integer,intent(IN) ::  order,Ngrid
 integer kai,kkk
 real(8), intent(IN) ::  qq(0:Order+Ngrid)
 real(8), intent(IN) :: x
 real(8), intent(OUT) :: bv(0:Ngrid,0:1)
 integer i,j,k,Ngrid1,ij1,jm2,j1m2
 real(8) x1,x2
 logical l_x_is_out
 

!  bv(0:Ngrid,0:Ngrid)  
  bv(:,0) = 0.0d0

  Ngrid1 = Ngrid-1
  l_x_is_out=.true.
  do i = 0, Ngrid1
!  print*, i, qq(i),qq(i+1),x, qq(i) <= x .and. x < qq(i+1)
     if (qq(i) <= x .and. x < qq(i+1)) then
        bv(i,0) = 1.0d0
        kkk = i
        l_x_is_out=.false.
     endif
  enddo

!print*, 'bv=',bv(:,0)
!read(*,*)

  if (qq(Ngrid1) <= x .and. x <= qq(Ngrid)+ Thin) then
     bv(Ngrid1,0) = 1.0d0
     kkk = Ngrid1
     l_x_is_out=.false.
  endif   

  if (l_x_is_out) then
print*, 'ERROR in deboor_cox : the x is not in inteval: '
print*,' qq=',qq
print*, 'x=',x
STOP
  endif

  do j = 1, kai-1
     jm2 = mod(j,2)
     j1m2 = mod(j-1,2)
     bv(:,jm2) = 0.0d0
     k = kkk - j
     if (k<0) k=0
     do i = k, kkk
        x1 = 0.0d0
        x2 = 0.0d0
        ij1 = i + j + 1
        if (qq(i+1) /= qq(ij1)) then
          x1 = ( qq(ij1)-x ) * bv(i+1,j1m2) / ( qq(ij1) - qq(i+1) );
!print*, j,'Case1 : ',qq(ij1)-x ,qq(ij1) - qq(i+1),bv(i+1,j1m2)
!read(*,*)
        endif
        if( qq(i) /= qq(i+j) ) then
          x2 = ( x-qq(i) ) * bv(i,j1m2) /  ( qq(i+j) - qq(i) );
!print*, j,'case2 :', x-qq(i),qq(i+j) - qq(i),bv(i,j1m2)
!read(*,*)
        endif
        bv(i,jm2) = x1 + x2;
!print*,'j x2 x2 = ',j,x1,x2
!read(*,*)

     enddo ! i = k , kkk
  enddo ! j 

!do i = 0 , Ngrid-1
!print*,i, 'bv=',bv(:,i)
!read(*,*)
!enddo

 end subroutine deboor_cox

! The derivatives:

real(8) function spline_k0_deriv(order, Ngrid, kkk, x, alp, qq,db) 
   implicit none
   integer order,Ngrid,kkk
   real(8) x
   real(8) qq(0:Ngrid+order), alp(0:Ngrid-1)
   real(8) db(0:Ngrid,0:1)

   integer   i,j,k,lnk,m, m1_mod2, k_mod2,j_mod2, imj
   real(8) x1,x2,x_n,delta,dy;

   k  = 1;   ! k is the order of derivate; we only need first order derivatives
   m = order + 1 - k;
   lnk = kkk - m;

   call deboor_cox(order,Ngrid, m, kkk,qq,x,db);
       !deboor_cox(order,Ngrid,kai,kkk,qq,x,bv)   

   m1_mod2 = mod(m-1,2) 
   k_mod2 = mod(k,2)
   if (m1_mod2 /= 0 ) then
      db(0:Ngrid-1,0) = db(0:Ngrid-1,m1_mod2) 
   endif
     
   if(k == 0) then
      spline_k0_deriv = 0.e0;
      RETURN
   endif


   db(:,1) = 0.0;

   j = 0
     j_mod2 = mod(j,2)
     do i = lnk-j, kkk
         x1 = 0.0d0 ; x2 = 0.0d0
         imj = i+m+j
         if (qq(imj) /= qq(i)) then
            x1 = db(i,j_mod2) / (qq(imj)-qq(i)) 
         endif
         if (qq(imj+1) /= qq(i+1)) then
            x2 = - db(i+1,j_mod2)/(qq(imj+1)-qq(i+1))
         endif
!print*, i,x1,x2,dble(m+j)*(x1+x2)
         db(i,j_mod2)=dble(m+j)*(x1+x2);
     enddo
   

   spline_k0_deriv = dot_product(alp(lnk:kkk),db(lnk:kkk,j_mod2))

!do i = 0, Ngrid
! print*, 'db=',db(i,:)
!if (mod(i,30)==0) read(*,*)
!enddo
!
!print*, ' in derivatives :'
!print*,'alp=',alp(lnk-1:kkk+1)
!print*, 'db=',db(lnk-1:kkk+1,j_mod2)
! read(*,*) 
  end function spline_k0_deriv

end module spline_z_k0_module

