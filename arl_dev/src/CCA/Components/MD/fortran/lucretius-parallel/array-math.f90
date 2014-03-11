
module array_math
implicit none

interface order_vect
  module procedure order_R8vect
  module procedure order_ivect
end interface order_vect

private :: order_R8vect,order_ivect
contains

      subroutine fit_a_line(x,y,slope,intercept)
        real(8), intent(IN) :: x(:),y(:)
        real(8), intent(OUT) :: slope,intercept
        real(8) tmp, a11,a12,a21,a22,b1,b2,det
        integer i,j,k,N
! V(1) is slope ; V(2) in intercept
        N=ubound(x,dim=1)-lbound(x,dim=1)+1

        a11 = sum(x(:)**2) ; a12 = sum(x(:)) ; b1 = sum(x(:)*y(:)) ; b2 = sum(y(:))
        a21 = a12 ;          a22 = dble(N)

        det = a11*a22-a12*a21 ;

        if (det ==0) then
           slope=0.0d0; intercept=0.0d0
          RETURN
        endif

         det = 1.0d0/det
         tmp = a11 ; a11 = a22*det ; a12 = -a12*det ; a21=a12 ; a22 = tmp * det
         slope = a11*b1+a12*b2 ; intercept = a21*b1+a22*b2

      end subroutine fit_a_line



subroutine order_R8vect(V)
real(8), intent(INOUT) :: V(:)
integer up,down,first,i,j
real(8) temp

up = ubound(V,dim=1)
down = lbound(V,dim=1)

  do i =  up,down,-1  !N,1,-1
    first = down      !1
    do j = down+1,i   !2, i
      if (V(j) > V(first)) then
       first = j
      endif
    enddo ! j
    temp = V(first)
    V(first) = V(i)
    V(i) = temp
  enddo  ! i

end subroutine order_R8vect

subroutine order_ivect(V)
integer, intent(INOUT) :: V(:)
integer up,down,first,i,j
integer temp

up = ubound(V,dim=1)
down = lbound(V,dim=1)

  do i =  up,down,-1  !N,1,-1
    first = down      !1
    do j = down+1,i   !2, i
      if (V(j) > V(first)) then
       first = j
      endif
    enddo ! j
    temp = V(first)
    V(first) = V(i)
    V(i) = temp
  enddo  ! i

end subroutine order_ivect
!****************
subroutine order_vect_iv(V,iv)
real(8), intent(INOUT) :: V(:)
integer, intent(INOUT) :: iv(:)
integer up,down,first,i,j
real(8) temp,itemp

up = ubound(V,dim=1)
down = lbound(V,dim=1)

  do i =  up,down,-1  !N,1,-1
    first = down      !1
    do j = down+1,i   !2, i
      if (V(j) > V(first)) then
       first = j
      endif
    enddo ! j
    temp = V(first)
    V(first) = V(i)
    V(i) = temp
        itemp = iV(first)
    iV(first) = iV(i)
    iV(i) = itemp

  enddo  ! i



end subroutine order_vect_iv
!**************************************************************

subroutine polifit(order,X,Y,V)
real(8), intent(IN) :: X(:),Y(:)
real(8), intent(INOUT) :: V(:)
integer, intent(IN) :: order
integer i,j,k,N
real(8) , allocatable :: a(:,:),b(:)
print*,'in polifit',allocated(a),allocated(b)
  allocate(a(order,order),b(order))
  N=ubound(Y,dim=1)
  if (ubound(X,dim=1)/=N) then 
    print*,'error in array-math polifit; sizes of X Y arrays are not equal'
    STOP
  endif
  if (order > N) then
     print*,'error in array-math polifit; too few experimental points'
    STOP
  endif
  a = 0.0d0; b=0.0d0
  do i = 1, order
  do j = 1, order
     a(i,j) = a(i,j) + sum(X(:)**(i-1+j-1))
  enddo
  enddo 
  a(1,1) = dble(N)
  do i = 1, order
     b(i) = dot_product(Y(:),X(:)**(i-1))
  enddo
  call invmat(a,order,order)
  do i = 1, order
    V(i) = dot_product(b(:), a(i,:))
  enddo
 deallocate(a,b)
end subroutine polifit
!**************************************************************
       subroutine trapez(Npct,Xlow,Xupp,Y,Rezult)
       integer Npct
       real(8) Xlow,Xupp
       real(8) Y(Npct)
       real(8) Rezult
       real(8) h,s,s1,s2
       integer i

        h=(Xupp-Xlow)/dble(Npct-1)

        S=0.0D0
        do i=2,Npct-1
         S=S+Y(i)
        enddo
        print*,'suma pura=',S
        S1=Y(1)+Y(npct)+2.0D0*S
        print*, 'S1=',S1
        S2=h/2.0D0*S1
        Rezult=S2*1.0D0

       end subroutine trapez

!**********************************************************************

       subroutine ludcmp(a,n,np,indx,d)
       integer n,np
       integer indx(np)
       real(8) a(np,np)
       real(8) d
       real(8),allocatable :: vv(:)
       real(8) , parameter:: hups=1.0d-90
       integer i,j,k,imax
       real(8) aamax,hops,suma,dum

       hops=hups

       allocate(vv(max(5000,np)))
       d=1.0d0
       do 12 i=1,n
         aamax=0.0d0
         do 11 j=1,n
           if (dabs(a(i,j)).gt.aamax) then
                aamax=dabs(a(i,j))
           endif
   11      enddo
         if (aamax.eq.0.0d0) then
              pause 'singular matrix!!!!!'
         endif
         vv(i)=1.0d0/aamax
   12    enddo
     

       do 19 j=1,n
          do 14 i=1,j-1
              suma=a(i,j)-dot_product(a(i,1:i-1),a(1:i-1,j))
              a(i,j)=suma
   14      enddo
          aamax=0.0d0
          do 16 i=j,n
            suma=a(i,j) - dot_product(a(i,1:j-1),a(1:j-1,j))
            a(i,j)=suma
            dum=vv(i)*dabs(suma)
            if (dum.ge.aamax) then
              imax=i
              aamax=dum
            endif
   16     enddo
          if (j.ne.imax) then
             do 17 k=1,n
               dum=a(imax,k)
               a(imax,k)=a(j,k)
               a(j,k)=dum
   17        enddo
             d=-1.0d0*d
             vv(imax)=vv(j)
          endif
          indx(j)=imax
          if (a(j,j).eq.(0.0d0)) then
            a(j,j)=hops
          endif
          if (j.ne.n) then
              dum=1.0d0/a(j,j)
              do 18 i=j+1,n
                a(i,j)=a(i,j)*dum
   18        enddo
          endif
   19   enddo

        deallocate(vv)
        end subroutine ludcmp
!**************************************************** 

       subroutine ludksb(a,n,np,indx,b)
       integer n,np
       integer indx(np)
       real(8) a(np,np), b(np)
       integer i,j,k,ii,ll
       real(8) suma

       ii=0
       do 12 i=1,n
          ll=indx(i)
          suma=b(ll)
          b(ll)=b(i)
          if (ii.ne.0) then
             do 11 j=ii,i-1
                suma=suma-a(i,j)*b(j)
   11        enddo
          else if (suma.ne.0.0d0) then
              ii=i
          endif
          b(i)=suma
   12  enddo
       do 14 i=n,1,-1
          suma=b(i)
          do 13 j=i+1,n
             suma=suma-a(i,j)*b(j)
   13     enddo
        b(i)=suma/a(i,i)
   14   enddo

        end subroutine ludksb


!**************************************************************
  
       subroutine invmat(a,n,np)
       integer n,np
       real(8) a(np,np)
       real(8), allocatable :: y(:,:)
       integer indx(np)
       integer i,j,k
       real(8) d
       allocate(y(np,np)) 
       do i=1,n
          do j=1,n
           y(i,j)=0.0d0
        enddo
       enddo
       do i=1,n
        y(i,i)=1.0d0
       enddo
       call ludcmp(a,n,np,indx,d)
       do j=1,n
         call ludksb(a,n,np,indx,y(1,j))
       enddo

       do i=1,n
         do j=1,n
          a(i,j)=y(i,j)
         enddo
       enddo
       deallocate(y)
       end subroutine invmat
!***********************************
       subroutine det(a,n,np,d)
       integer n,np
       real(8) d
       real(8) a(np,np)
       integer indx(np)
       integer i,j
       call ludcmp(a,n,np,indx,d)
       do j=1,n
        d=d*a(j,j)
       enddo

       end subroutine det


!*********************************
!1C Utils Newton-Solving

        subroutine lnsrch(n,xold,fold,g,p,x,f,stpmax,check,func)
        integer n 
        real(8) stpmax
        real(8) fold
        logical check
        real(8) x(n),xold(n),g(n),p(n)
        real(8) , parameter :: alf=1d-4 
        real(8) , parameter :: tolx=1d-7
        external func
        integer i,j,k,imax
        real(8) f,alam,test,alamin,temp,a,b,alam2,f2,rhs1,rhs2,disc,slope,tmplam
        real(8) suma, func, fold2
        
        check=.true.
        suma=dsqrt(dot_product(p(1:n),p(1:n)))
        
        if(suma.gt.stpmax) then
          do i=1,n
            p(i)=p(i)*stpmax/suma
          enddo
        endif
        slope = dot_product(g(1:n),p(1:n))
        test=0.0d0
        do i=1,n
          temp=abs(p(i))/dmax1(dabs(xold(i)),1.0d0) 
          if (temp.gt.test) then
             temp=test
          endif
         enddo
        alamin=TOLX/test
        alam=1.0d0
  11  continue 
        x(1:n)=xold(1:n)+alam*p(1:n)
        f=func(x)
        if (alam.lt.alamin) then
         x(1:n) = xold(1:n)
         check=.true.
         return
        else if (f.le.(fold+ALF*alam*slope)) then
         return
        else if (alam.eq.1.0d0) then 
            tmplam=-slope/(2.0d0*(f-fold-slope))
          else
           rhs1=f-fold-alam*slope
           rhs2=f2-fold2-alam2*slope
           a=(rhs1/alam**2-rhs2/alam2**2)/(alam-alam2)
           b=-alam2*rhs1/alam**2+alam*rhs2/alam2**2
           b=b/(alam-alam2)
           if (a.eq.0) then
            tmplam=-slope/(2.0d0*b)
           else
             disc=b*b-3.0d0*a*slope
             tmplam=(-b+dsqrt(disc))/(3.0*a)
           endif

           if(tmplam.gt.(0.50d0*alam)) then
               tmplam=0.5d0*tmplam
           endif 
          endif
        alam2=alam
        f2=f
        fold2=fold
        alam=dmax1(tmplam,0.1d0*alam)
       goto 11

      END subroutine lnsrch ! lnrscr

end module array_math

