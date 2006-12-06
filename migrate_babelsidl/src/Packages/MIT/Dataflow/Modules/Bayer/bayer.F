c      double precision k(6), y(7), tf
c	
c	k(1)=0.0562
c	k(2)=0.0164
c	k(3)=0.0001
c	k(6)=0.0049
c	k(4)=0.0249
c	k(5)=0.0842
c
c	tf=200.0
c
c	call bayerproblem(tf,k,y)
c	write(*,*) tf,y
c	end

	
	
	subroutine bayer(tf,k,y)
c	
c	for a given final time tf, and values of the log of kinetic constants, it returns 
c	the vector y of concentrations mol/kg
c
c
	double precision k1,k2,k21,k22,k31,k32, k(6),tf,tend
C
      integer iwork
      integer neq, itol, istate, itask, liw, lrw
      parameter(neq=7)
      external fex, fjac
      double precision y, t, tout, atol, rtol, tfinal
      double precision tinit, ybig
      double precision rwork
c
c   y is dimensioned to be neq long.  You can dimension
c  y longer than neq so that you can pass extra parameters to your 
c  differential equation subroutine.  
c
c  rwork is 70 long and iwork is 23 long.  This is sufficient for neq = 1.
c  For bigger systems, you may have to increase these.  See the docs.
c rwork  = real work array of length at least..
c             22 + neq * max(16, neq + 9).=22+7*16=134
c          see also paragraph e below.
c lrw    = declared length of rwork (in user-s dimension).
c iwork  = integer work array of length at least  20 + neq.=27

c
      dimension y(neq), rwork(134), iwork(27)

      common /ks/ k1,k2,k21,k22,k31,k32

c	write(*,*) 'tf = ', tf
c	write(*,*) 'k = ', k
c	write(*,*) 'y = ', y
c	write(*,*)

	k1=exp(k(1))
	k21=exp(k(2))
	k22=exp(k(3))
	k31=exp(k(4))
	k32=exp(k(5))
	k2=exp(k(6))

	
	
      T    = 0.0
      Y(1) = 0.0
      Y(2) = 2.8111
      Y(3) = 2.5148
	Y(4) = 0.0
	Y(5) = 0.0
	Y(6) = 0.0
	Y(7) = 0.0048
      TOL  = 1.0E-6
C                                 Write title
      
C                                 Integrate ODE
      
	tend=tf

c
c  atol and rtol describe the error tolerance of the problem.  atol is 
c  the absolute error tolerance and rtol is relative error tolerance.
c
      atol = 1.d-6
      rtol = 1.d-6
      tfinal = tend

c     Use itol = 1 if atol is a scalar and itol = 2 if atol is an array
      itol = 1
c     Since this is the first call to lsoda, set istate = 1.
      istate = 1
c     We want to take individual steps through the integration.  Set itask = 2.
c	No individual steps. Set itask=1
      itask = 1
c     Use iopt = 0 if no optional input is used.
      iopt = 0

c lrw    = declared length of rwork (in user-s dimension).
c liw    = declared length of iwork (in user-s dimension).

      lrw = 134
      liw = 27
c
c  Values of y(L) for L > NEQ are considered as parameters to the problem,
c  such as alpha.  This is a good way to pass this parameter to the 'fex' 
c  routine that calculates the differential equation (system).
c
      tinit = 0.d0
      t = 0.d0
      tout = tfinal
c
c  jt = 1 means use an analytic Jacobian for the computation.  This is 
c  easy to find for this 1-equation "system", so we'll use it.  If you 
c  want lsoda to compute the Jacobian for you, you would set jt = 2.
c
      jt = 1

c
c      write(*,*)atol, rtol, tfinal, y

	call lsoda (fex, neq, y, t, tout, itol, rtol, atol, itask, 
     1	        istate, iopt, rwork, lrw, iwork, liw, fjac, jt)
c  
c  This line writes several items of interest to the screen.  t and 
c  y(1) are, of course, the independent and dependent variables.  
c  iwork(19) = what method was used last?  1 = adams (nonstiff), 2 = bdf
c     (stiff)
c  iwork(11) = Number of computational steps taken in the problem so far.
c  iwork(12) = Number of f evaluations in the problem so far 
c  iwork(13) = Number of Jacobian evaluations in the problem so far.
c
c	write(*,*) iwork(19), iwork(11), iwork(12), iwork(13), 
c     1              y
c
c  istate .lt. 0 means an error occured.
c  y(1) .ge. ybig means ignition.
c
	if (istate .ge. 0) go to 70

      write(*,*)'istate = ', istate
 70   continue

c	write(*,*) 'lsoda: ', y
      end

c
c ************************************************************************
c
      subroutine fex (neq, t, y, ydot)
c  
c  This subroutine defines your differential equation system.  If you 
c  had more than one equation, you would define the others similarly.
c  Remember that y(2) = alpha in this model.
c
      double precision t, y, ydot
      double precision k1,k2,k21,k22,k31,k32
      dimension y(neq), ydot(neq)

      common /ks/ k1,k2,k21,k22,k31,k32

      ydot(1) = k1*y(2)*y(3)-k21*y(1)+k22*y(4)*y(7)-k2*y(2)*y(1)
      ydot(2) =-k1*y(2)*y(3)-2.0*k31*y(2)*y(2)+
     1               2.0*k32*y(5)*y(7)-k2*y(2)*y(1)
      ydot(3) =-k1*y(2)*y(3)
      ydot(4) = k21*y(1)-k22*y(4)*y(7)
      ydot(5) = k31*y(2)*y(2)-k32*y(5)*y(7)
      ydot(6) = k2*y(2)*y(1)
      ydot(7) =k21*y(1)-k22*y(4)*y(7)+k31*y(2)*y(2)-k32*y(5)*y(7)

      return
      end

c
c ************************************************************************
c
      subroutine fjac (neq, t, y, ml, mu, pd, nrowpd)
c
c  Since this is a one-equation "system", it's easy to determine the 
c  Jacobian for the problem.  In general, whenever it is possible to provide
c  the analytic Jacobian to LSODA, you should do so.  Elements of the 
c  Jacobian are provided by pd(i,j) = dy_i/dx_j.
c
      integer neq, ml, mu, nrowpd
      double precision y, pd
      double precision k1,k2,k21,k22,k31,k32
      dimension y(neq), pd(neq,neq)

      common /ks/ k1,k2,k21,k22,k31,k32



      pd(1,1) = -k21-k2*y(2) 
      pd(1,2) = k1*y(3)-k2*y(1)
      pd(1,3) = k1*y(2)
      pd(1,4) = k22*y(7)
      pd(1,5) = 0.0
      pd(1,6) = 0.0
      pd(1,7) = k22*y(4)
      pd(2,1) = -k2*y(2)
      pd(2,2) = -k1*y(3)-0.4E1*k31*y(2)-k2*y(1)
      pd(2,3) = -k1*y(2)
      pd(2,4) = 0.0
      pd(2,5) = 0.2E1*k32*y(7)
      pd(2,6) = 0.0
      pd(2,7) = 0.2E1*k32*y(5)
      pd(3,1) = 0.0
      pd(3,2) = -k1*y(3)
      pd(3,3) = -k1*y(2)
      pd(3,4) = 0.0
      pd(3,5) = 0.0
      pd(3,6) = 0.0
      pd(3,7) = 0.0
      pd(4,1) = k21
      pd(4,2) = 0.0
      pd(4,3) = 0.0
      pd(4,4) = -k22*y(7)
      pd(4,5) = 0.0
      pd(4,6) = 0.0
      pd(4,7) = -k22*y(4)
      pd(5,1) = 0.0
      pd(5,2) = 2*k31*y(2)
      pd(5,3) = 0.0
      pd(5,4) = 0.0
      pd(5,5) = -k32*y(7)
      pd(5,6) = 0.0
      pd(5,7) = -k32*y(5)
      pd(6,1) = k2*y(2)
      pd(6,2) = k2*y(1)
      pd(6,3) = 0.0
      pd(6,4) = 0.0
      pd(6,5) = 0.0
      pd(6,6) = 0.0
      pd(6,7) = 0.0
      pd(7,1) = k21
      pd(7,2) = 2*k31*y(2)
      pd(7,3) = 0.0
      pd(7,4) = -k22*y(7)
      pd(7,5) = -k32*y(7)
      pd(7,6) = 0.0
      pd(7,7) = -k22*y(4)-k32*y(5)


      return
      end
