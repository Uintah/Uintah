
module minimize 

       

       implicit none
       real(8), parameter,private  :: TOLX=1.0d-8   ! coordinate tolerance
       real(8), parameter,private  :: GTOL=1.0d-8   ! gradient tolerance
       real(8), parameter,private  :: STPMX=0.1d0   ! maximun Newton step
       real(8), parameter,private  :: error_func=1.0d-4 ! the error in the variables
       logical,  parameter,private :: l_all_var_same_error =.true.

!all variables have the same error_func
       logical , parameter,private :: xcheck=.true. ! check for coordinate convergence
       logical , parameter,private :: gcheck=.true. ! check for gradient convergence

       public :: opt
       public :: func_QN
       private :: dfpmin
       private :: lnsrch1
       private :: dfunc
       private :: dfridr
!       private :: func 

contains
       real(8) function func_QN(X0,X,Npar,Nvar,XC,p)
       implicit none
       integer Npar,Nvar
       real(8) X(Npar,3), XC(3),qn(Npar), X0(Npar,3), X1(Npar,3),p(Npar)
       func_QN = func(X0,X,Npar,Nvar,XC,p)
       end function func_QN

       subroutine opt(X0,X,Npar,Nvar,XC,parset,fret) ! Geometry Optimization Subroutine
	implicit real*8 (a-h,o-z)
        real(8) X(Npar,3), XC(3),qn(Npar), X0(Npar,3), X1(Npar,3)
	integer iter,n,nvar
        real*8 parset(nvar)             ! an array of input parameters
        real*8 finpar(nvar)             ! an array of input parameters
        integer npar                 ! number of parameters
	real*8 p(nvar), fret, fpp!, func, dfunc
        integer i,iat


	do iat=1,nvar                 ! maxat has been changed to npar
	   p(iat) = parset(iat)
	end do
!        write(6,*) 'inside opt,initial parameters'
!        write(6,'(50F10.4)') p 
	call dfpmin(X0,X,Npar,Nvar,XC,p,iter,fret)	!Calls the BFGS routine
!       write(6,*)'convg in iter=',iter , 'fret=',fret
!       write(6,*) 'The min value of the function is: ',fret
!write(6,*) 'The optimized parameters are: '
!do i=1,nvar
   !write(6,*) i,p(i)
!end do
!        print*, 'optimized value is :',p,fret
        do i=1,nvar
          parset(i)=p(i)
        end do
	return
	end subroutine opt

!C	Subroutine dfpmin (Actual Optimization [BFGS] Routine)
	Subroutine dfpmin(X0,X,N,Nvar,XC,p,iter,fret)

!C
!C	----------------------------------------------------------------
!C	Declarations and stuff:
!C	----------------------------------------------------------------
!C	iter: is number of iterations max number is set by ITMAX
!C	fret: is the minimum value of the function
!C	func: and dfunc are subroutines that evaluate the func and the 
!C	      derivative vector of the function
!C	----------------------------------------------------------------	
	
	implicit real*8 (a-e,g-h,o-z)
	integer iter,N,Nvar !NMAX:  max anticipated value of n
	real(8) fret,p(n)
        real(8) X(N,3), XC(3),qn(Nvar), X0(N,3), X1(N,3) 
	integer , parameter :: NMAX=150
        integer , parameter ::  ITMAX=1000
	real(8) , parameter :: EPS=3.0d-8 
	integer i, its, j
	logical check
	real*8 den, fac, fad, fae, fp, stpmax, sum, sumdg, sumxi, temp1, &
     	 testx,testg, dg(NMAX), g(NMAX), hdg(NMAX), hessin(NMAX,NMAX), pnew(NMAX), xi(NMAX)

	fp = func(X0,X,N,Nvar,XC,p) !subroutine that calculates the value of pot func
!       print*,'initial value of the objective func.',fp
	call dfunc(X0,X,N,Nvar,XC,p,g) !g(1:n): the gradient vector returned by dfunc
	sum = 0.d0
	do i=1,nvar
	   do j=1,nvar
	      hessin(i,j) = 0.0d0	!Initialize all elements to 0
	   end do
	   hessin(i,i) = 1.0d0	 !This makes it a unit vector
	   xi(i) = -g(i)	 !xi() Initial line direction
	   sum = sum + p(i)*p(i) !Adding up the squares of all coordinates 
	end do
	stpmax=STPMX*max(dsqrt(sum),dble(nvar)) 
!C	stpmax:is max step size that limits the size of steps in lnsrch
!C	STPMX: is the scaled max step length allowed in line searches
!C
	do its=1,ITMAX		!Main loop over all the iterations
	   iter = its		!Iteration number
	   call lnsrch1(nvar,p,fp,g,xi,pnew,fret,stpmax,check,func,TOLX,N,X0,X,XC)
!           print*,'Iteration: ',its,' Func value: ', fret
           fp = fret!New func evaluation happens in lnsrch. Save the func
           do i=1,nvar
               xi(i)=pnew(i)-p(i)!Update line direction and current pnt
               p(i)=pnew(i)
           end do
!C	-----Convergence block------------------------------------------
!C
           if (check) then
            write(6,*) 'you requested the program to stop'
            return
           endif
	   testx = 0.0d0!Test for coordinate convergence 
	   do i=1,nvar
	      temp1=dabs(xi(i))/max(dabs(p(i)),1.d0)
	      if (temp1.gt.testx) then
	         testx=temp1
	      end if
	   end do
	   do i=1,nvar
	      dg(i) = g(i)	!Save the old gradient
	   end do
!C
	   call dfunc(X0,X,N,Nvar,XC,p,g)	!and get the new gradient
!C
	   testg = 0.0d0		!Test for convergence on zero gradient
	   den=max(dabs(fret),1.d0)
	   do i=1,nvar
	      temp1=dabs(g(i))*max(dabs(p(i)),1.d0)/den
	      if(temp1.gt.testg) testg = temp1
 	   end do
           if (xcheck.and.gcheck) then    ! check for both coord. and grad conv.
            if ((testg.lt.GTOL).and.(testx.lt.TOLX)) then
             print*, 'Coord. and grad. convergence'
             return
            endif
           else
            if (xcheck.and.(.not.gcheck)) then
	      if (testx.lt.TOLX) then
	        print*,'Coordinates based convergence'
	        return	
              endif
            endif
            if (gcheck.and.(.not.xcheck)) then
	     if (testg.lt.GTOL) then
	      print*,'Zero gradient convergence'
	      return
             endif
            endif
            if ((.not.xcheck).and.(.not.gcheck)) then
              print*, 'You did not ask for minimization'
            endif
	   end if
!C
!C	-----Convergence block ends-------------------------------------
!C
       do i=1,nvar
          dg(i)=g(i)-dg(i)!compute difference of gradients
       end do
       do i=1,nvar!and difference times current matrix
          hdg(i)=0
          do j=1,nvar
             hdg(i)=hdg(i)+hessin(i,j)*dg(j)
          end do
       end do
       fac=0.d0!calculate dot products for the denominators
       fae=0.d0      
       sumdg=0.d0
       sumxi=0.d0
       do i=1,nvar
          fac=fac+dg(i)*xi(i)
          fae=fae+dg(i)*hdg(i)
          sumdg=sumdg+dg(i)**2
          sumxi=sumxi+xi(i)**2
       end do
       if (fac**2.gt.EPS*sumdg*sumxi) then !skip update if fac is not
          fac=1./fac !sufficiently positive
          fad=1./fae
          do i=1,nvar !The vector that makes BFGS
             dg(i)=fac*xi(i)-fad*hdg(i)	!different from DFP
          end do
          do i=1,nvar !The BFGS updating formula
             do j=1,nvar
                hessin(i,j)=hessin(i,j) + fac*xi(i)*xi(j) &
                        -fad*hdg(i)*hdg(j)+fae*dg(i)*dg(j)
             end do
          end do
       end if
       do i=1,nvar !Calculate the next direction
          xi(i) = 0
          do j=1,nvar
             xi(i)=xi(i)-hessin(i,j)*g(j)
          end do
       end do
    end do        ! next its
 print*,'Iteration: ',its,' Func value: ', fret
    print*, 'too many iterations in dfpmin'
    END Subroutine dfpmin
!C
!C	****************************************************************
!C	Subroutine lnsrch (Does the 1-D minimization)
!C	****************************************************************
!C
       Subroutine lnsrch1(n,xold,fold,g,p,xgopt,fgopt,stpmax,check,func,TOLX,N1,X0,X,XC)
!C
!C	----------------------------------------------------------------
!C	Given an n-dimensional point xold(1:n), the value of the function
!C	and the gradient there, fold and g(1:n), and a direction p(1:n),
!C	finds a new point xgopt(1:n) along the direction p from xold where
!C	the function func has decreased "sufficiently". The new function
!C	value is returned in fgopt. stpmax is an input quantity that limits
!C	the length of steps . P is usually the Newton direction.
!C
!C	Parameters:
!C	ALF ensures sufficient decrease in function value
!C	TOLX is the convergence criterion on delx
!C
	implicit real*8 (a-h,o-z)
	integer n, n1
	logical check
        real*8 fgopt,fold,stpmax,g(n),p(n),xgopt(n),xold(n),func,ALF,TOLX
	parameter (ALF=1.0d-4)
	integer i
	real*8 a,alam,alam2,alamin,b,disc,f2,fold2,rhs1,rhs2,slope,sum,temp1, test, tmplam
        real(8) X0(N1,3),X(N1,3),XC(3)
        integer iflag
        data iflag /0/
	check=.false.	!check: logical var which is ignored for minimization


	sum=0.d0
	do i=1,n
	   sum=sum+p(i)*p(i)
	end do
	sum=dsqrt(sum)
	if (sum.gt.stpmax) then		!Scale if attempted step is too big
!         print*,'Attempted step too big. Scaling step size',stpmax
	   do  i=1,n
	      p(i)=p(i)*stpmax/sum
	   end do
	end if
	slope=0.d0
	do i=1,n
	   slope=slope+g(i)*p(i)
	end do
	test = 0.0d0	! Compute alamin (lamda-min)
	do i=1,n
	   temp1 = dabs(p(i))/max(dabs(xold(i)),1.0d0)
	   if (temp1.gt.test) test = temp1
	end do

	alamin = TOLX/test	!*Careful*: If test goes to zero it's trouble
	alam=1.0d0              !full step. Always try full step first.
 1	continue		!Infinite loop begins
	   do i=1,n
	      xgopt(i)=xold(i)+alam*p(i)
	   end do
	   fgopt=func(X0,X,N1,N,XC,xgopt)		!Call to subroutine that evaluates func
	   if (alam.lt.0.1*alamin) then	!Convergence on Delx
            if (iflag.ne.2) then
             write(6,*) 'a very small step in the line search'
!             write(6,*) '1=contunue, 0=exit,2=always continue.'
!             read(5,*) iflag
               iflag=2
            endif
             if (iflag.eq.0) then
	      do i=1,n
	         xgopt(i) = xold(i)
	      end do
	      check = .true.
             endif
	      return
	   else if (fgopt.le.fold+ALF*alam*slope) then	!Sufficient func dec
	      return
	   else
	      if(alam.eq.1.) then			!Backtrack 1st time
	         tmplam= -slope/(2.d0*(fgopt-fold-slope)) 	!tmplam is prob temp-lamda
	      else
	         rhs1=fgopt-fold-alam*slope
	         rhs2=f2-fold2-alam2*slope
	         a=(rhs1/alam**2-rhs2/alam2**2)/(alam-alam2)
	         b=(-alam2*rhs1/alam**2+alam*rhs2/alam2**2)/(alam-alam2)
	         if (a.eq.0.d0) then
	            tmplam=-slope/(2.d0*b)
	         else
	            disc=b*b-3.d0*a*slope	!The abs prevents the code from
	            if (b*b-3.d0*a*slope.lt.0) then 
!C       In this case cubic interpolation is inadequed 
!C       set lamba(new)=0.5*lamda(old)
                     tmplam=0.5d0*alam
                    else
	             tmplam=(-b+dsqrt(disc))/(3.0d0*a) 	
                    endif
	         end if
	         if (tmplam.gt.0.5d0*alam) tmplam=0.5d0*alam
	      end if
	   end if
	   alam2=alam
	   f2=fgopt
	   fold2=fold
	   alam=max(tmplam,0.1d0*alam)
	go to 1
        print*, 'EXIT from lnsrch1'
        read(*,*)
	END  Subroutine lnsrch1

!C	****************************************************************
!C	subroutine dfunc (Returns the vector of derivatives, g)
!C	****************************************************************

        Subroutine dfunc(X0,X,Npar,Nvar,XC,p,g)
        implicit real*8 (a-e,g-h,o-z)
        integer, intent(IN) :: Npar,Nvar
	real(8) p(nvar),g(nvar) 
        real(8), allocatable :: h(:)
        real(8), allocatable , save :: hfactor(:)
        real(8) X0(Npar,3),X(Npar,3),XC(3)
        logical, save :: l_very_first_pass
        data l_very_first_pass / .true./
        integer ncall
        data ncall/0/
        integer i

        allocate(h(Nvar))

        ncall=ncall+1
!C  Oleg
        if (l_very_first_pass) then
         allocate(hfactor(Nvar))
          if (l_all_var_same_error) then
             hfactor = error_func
          else 
           open(81,file='/home/jenel/minim/h.inp',status='old')
           do i=1,nvar
            read(81,*) hfactor(i)
           end do
          endif  ! (l_all_var_same_error).and.(nvar.ge.2)
          close(81)
         l_very_first_pass = .false.
        endif

        h(:) = hfactor(:) ! h(:) = p(:)*h_factor(:)
	do i=1,nvar
	   g(i) = dfridr(X0,X,Npar,Nvar,XC,p,i,h(i))	
	end do
        deallocate(h)
	end Subroutine dfunc
	
!C	****************************************************************
!C	Function that returns derivatives of any function using Ridders
!C	method of polynomial extrapolation
!C	****************************************************************	
!C	----------------------------------------------------------------
!C	This function returns the derivative of a function func at a 
!C	point x by Ridders method of polynomial extrapolation. The value
!C	h is input as an estimated stepsize; it need not be small, but
!C	rather should be an increment in x over which func changes
!C	substantially. An estimate of the error in the derivative is
!C	returned as err.
!C
!!C	Parameters:
!C	Step size is decreased by CON at each iteration
!C	Max size of tableau is set by NTAB
!C	Return when error is SAFE worse than the best so far
!C	
!C	Raman R. Tallamraju
!C	Aug 18 1999
!C	----------------------------------------------------------------

	real*8 function dfridr(X0,X,Npar,Nvar,XC,p,arrnum,h)

	implicit real*8 (a-e,g-h,o-z)
        integer, intent(IN) :: Npar,Nvar
	integer NTAB, arrnum
	real*8 err, h, xgopt, CON, CON2, BIG, SAFE,p(npar)
	parameter (CON=4.0d0, CON2=CON*CON,BIG=1.0D30,NTAB=10,SAFE=2.0d0)
	real*8 funxph, funxmh, temploc
	integer i,j
	real*8 errt, fac, hh, a(NTAB,NTAB)
        real(8) X0(Npar,3),X(Npar,3),XC(3)

	if (h.eq.0) then
         print*,  'h must be nonzero in dfridr stop in dfridr'
         stop
        endif

	hh = h
	temploc = p(arrnum)	!Temporary location for actual array element
	p(arrnum)=temploc - hh
	funxmh = func(X0,X,Npar,Nvar,XC,p)
	p(arrnum) =temploc + hh
	funxph = func(X0,X,Npar,Nvar,XC,p)
	a(1,1) = (funxph-funxmh)/(2.d0*hh)
	p(arrnum) =temploc	!Gets the original value back into array
	err = BIG
	do i=2,NTAB	!Successive columns in Neville Table will go to smaller
	   hh=hh/CON	!step sizes and higher order of extrapolation
	   p(arrnum)=temploc - hh
	   funxmh = func(X0,X,Npar,Nvar,XC,p)
	   p(arrnum) =temploc + hh
	   funxph = func(X0,X,Npar,Nvar,XC,p)
	   a(1,i)=(funxph-funxmh)/(2.d0*hh)	!Try new, smaller stepsize
	   p(arrnum) = temploc	!Gets the original value back into array
	   fac=CON2
	   do j=2,i
	      a(j,i)=(a(j-1,i)*fac-a(j-1,i-1))/(fac-1.d0)
	      fac=CON2*fac
	      errt=max(dabs(a(j,i)-a(j-1,i)),dabs(a(j,i)-a(j-1,i-1)))
	      if (errt.le.err) then
	         err=errt
	         dfridr=a(j,i)
	      end if
	   end do
	   if(dabs(a(i,i)-a(i-1,i-1)).ge.SAFE*err) return
	end do
	END function dfridr 
	   
!C	****************************************************************
!C	function that evaluates the pot function
!C	****************************************************************

!        real*8 function func(p,n) !function that evaluates the pot func
!       implicit real*8 (a-h,o-z)
!       include '/home/jenel/minim/qfit.par'
!	real*8 p(n)
!       real*8 parset(2*maxcharges)
!
!        do i=1,n
!          parset(i)=(p(i))
!        end do
!
!        sum=0.0d0
!        do i=2,n-1
!          sum=sum+(dsin(p(i))-10.0d0)**2*(p(i-1)-1.0d0)
!        enddo
!         sum=sum+p(1)*(p(n)-2)**2
!	func = sum ! ofunc(parset)
!
!        return
!        end

        real(8) function func(X0,X,Npar,Nvar,XC,p)
        implicit real(8) (a-h,o-z)
        integer , intent(IN) :: Nvar,Npar
        real(8) X(Npar,3), XC(3),qn(4), X0(Npar,3)
        real(8) X1(Npar,3),  suma, p(Npar)
!        real(8) func
        real(8) orient(9)
        integer i,j,k,N
        N=Npar
        if (Nvar.ne.4) then
          print*, 'This func is doing quaterniones only so Nvar must be Nvar=4',&
           'your vaelue is actually Nvar=',Nvar
         stop
        endif
        !X0 comes with respect to mass centra
        do k=1,N
         X(k,:)=X(k,:)-XC(:)
        enddo
        d_n=1.0d0/dsqrt(p(1)**2+p(2)**2+p(3)**2+p(4)**2)
!        p(1:4)=p(1:4)*d_n
        qn(1)=p(1)
        qn(2)=p(2)
        qn(3)=p(3)
        qn(4)=p(4)
       qn(1:4)=qn(1:4)*d_n


        orient(1)=-qn(1)**2+qn(2)**2-qn(3)**2+qn(4)**2
        orient(2)=-2.0d0*(qn(1)*qn(2)+qn(3)*qn(4))
        orient(3)=2.0d0*(qn(2)*qn(3)-qn(1)*qn(4))
        orient(4)=2.0d0*(qn(3)*qn(4)-qn(1)*qn(2))
        orient(5)=qn(1)**2-qn(2)**2-qn(3)**2+qn(4)**2
        orient(6)=-2.0d0*(qn(1)*qn(3)+qn(2)*qn(4))
        orient(7)=2.0d0*(qn(2)*qn(3)+qn(1)*qn(4))
        orient(8)=2.0d0*(qn(2)*qn(4)-qn(1)*qn(3))
        orient(9)=-qn(1)**2-qn(2)**2+qn(3)**2+qn(4)**2

!       print*, 'orient=',orient

!        print*,'qn=',qn 

        x1(1:N,1)=orient(1)*X0(1:N,1) + orient(2)*X0(1:N,2) + orient(3)*X0(1:N,3)
        x1(1:N,2)=orient(4)*X0(1:N,1) + orient(5)*X0(1:N,2) + orient(6)*X0(1:N,3)
        x1(1:N,3)=orient(7)*X0(1:N,1) + orient(8)*X0(1:N,2) + orient(9)*X0(1:N,3)

        suma=0.0d0
        do i=1,N
         do j=1,3
          suma=suma+(X(i,j)-X1(i,j))**2
         enddo
if (i.eq.3) then
!         print*,'X=',X(i,:)-X1(i,:)
!         print*, 'x1=',X1(i,:)
!         print*, i,'dist dist1=',dsqrt(dot_product(x(i,:),x(i,:)) )
endif
!     : dsqrt(dot_product(x1(i,:),x1(i,:))),
!     : dsqrt(dot_product(x0(i,:),x0(i,:)))
        enddo

        !suma=suma+p(1)**2+p(2)*p(2)+p(3)**2+p(4)**2-1.0d0
        func=suma 
        
!        print*, 'func=',func
!        print*, '-----'

        do i = 1,N
             ax1=dot_product(X(i,:),X(i,:))
             ax2=dot_product(X1(i,:),X1(i,:))
              if (dabs(ax1-ax2).gt.1.0d-7) then
print*, 'error in func they are not centered weell', ax1,ax2
stop            
               endif
        enddo 

        do k=1,N
         X(k,:)=X(k,:)+XC(:)
        enddo


        end function func


end module minimize
