c
c $Id$
c

      SUBROUTINE scalcoef(domLo, domHi, idxLo, idxHi,
     $     den,vis,
     &     ae,aw,an,as,at,ab,
     $     cesav,cwsav,cnsav,cssav,ctsav,cbsav,
     &     domLoU, domHiU, UU,
     &     domLoV, domHiV, VV,
     &     domLoW, domHiW, WW,
c    &     prinv,crn,
c    $     VOL,RV,RA,RONE,
     &     sew,sns,stb,
     $     cee,cwe,cww,cnn,csn,css,ctt,cbt,cbb,
     $     efac,wfac,enfac,sfac,tfac,bfac,
     $     dxpw,dyps,dzpb)
c    $     L2UP)
C-----------------------------------------------------------------------
C     WRITTEN BY SESHADRI KUMAR, OCTOBER 14, 1996
C
C     COEFS calculates the seven diagonals of the seven-point finite
C     difference equation for any scalar equation (e.g., F,G, TKE,
C     TED, ETA, etc.) 
C
C     PASSED VARIABLES:
C
C	den	Density field (i/p)
C	vis	Viscosity field (i/p)
C	u	X-directed velocity (i/p)
C	v	Y- or r-directed velocity (i/p)
C	w 	z- or theta-directed velocity (i/p)
C	ae,aw,an,as,at,ab,ap
C		Coefficients of the seven-diagonal finite difference
C		equation (o/p)
C	cesav,cwsav,cnsav,cssav,ctsav,cbsaV
C		Convection fluxes on the east,west,north,south,top,
C		and bottom faces of the P-cell (o/p)
C	prinv	Inverse of the "Prandtl number" for this scalar
C		(e.g., PRF, PRG, etc.) (i/p)
C	crn	Average Cell Reynolds number (o/p)
C	vol	Array: Volume of P-Cell (o/p)
C	rv	Radius of V-velocity node point (i/p)
C	ra	Radius of P-cell center (i/p)
C	sew	Width of P-cell in x-direction (i/p)
C	sns	Width of P-cell in y-direction (i/p)
C	stb	Width of P-cell in z-direction (i/p)
C	rone	Dummy radius array, = 1.0 (i/p)
C	cee,cwe,cww
C		Interpolation factors for interpolating from
C		E- and W-faces of regular P-cell to artificial
C		cell for accurate second order finite differences (i/p)
C	cnn,csn,css
C		Interpolation factors for interpolating from
C		N- and S-faces of regular P-cell to artificial
C		cell for accurate second order finite differences (i/p)
C	ctt,cbt,cbb
C		Interpolation factors for interpolating from
C		T- and B-faces of regular P-cell to artificial
C		cell for accurate second order finite differences (i/p)
C	efac,wfac	
C		Interpolation factors for calculating value at
C		P-node from values at its east and west cell faces
C		respectively (i/p)
C	enfac,sfac	
C		Interpolation factors for calculating value at
C		P-node from values at its north and south cell faces
C		respectively (i/p)
C	tfac,bfac	
C		Interpolation factors for calculating value at
C		P-node from values at its top and bottom cell faces
C		respectively (i/p)
C	dxpw	Distance between successive P-nodes in the x-direction
C		(see cellg.f) (i/p)
C	dyps	Distance between successive P-nodes in the y-direction
C		(i/p)
C	dzpb	Distance between successive P-nodes in the z-direction
C		(i/p)
C	ist,jst,kst
C		Starting indices for the calculation of the coefficients
C		(i/p)
C	iend,jend,kend
C		Ending indices for calculation (i/p)
C	l2up	Logical, =T implies that the 2nd upwind differencing
C		scheme of Roache is used
C
C-----------------------------------------------------------------------
      implicit none
      INCLUDE 'param4.h'
*-----------------------------------------------------------------------
*     passed variables
*
      integer domLo(3), domHi(3), idxLo(3), idxHi(3)

      double precision den(
     $     domLo(1):domHi(1),
     $     domLo(2):domHi(2),
     $     domLo(3):domHi(3))

      double precision vis(
     $     domLo(1):domHi(1),
     $     domLo(2):domHi(2),
     $     domLo(3):domHi(3))

      double precision ae(
     $     domLo(1):domHi(1),
     $     domLo(2):domHi(2),
     $     domLo(3):domHi(3))

      double precision aw(
     $     domLo(1):domHi(1),
     $     domLo(2):domHi(2),
     $     domLo(3):domHi(3))

      double precision an(
     $     domLo(1):domHi(1),
     $     domLo(2):domHi(2),
     $     domLo(3):domHi(3))

      double precision as(
     $     domLo(1):domHi(1),
     $     domLo(2):domHi(2),
     $     domLo(3):domHi(3))

      double precision at(
     $     domLo(1):domHi(1),
     $     domLo(2):domHi(2),
     $     domLo(3):domHi(3))

      double precision ab(
     $     domLo(1):domHi(1),
     $     domLo(2):domHi(2),
     $     domLo(3):domHi(3))

      double precision cesav(
     $     domLo(1):domHi(1),
     $     domLo(2):domHi(2),
     $     domLo(3):domHi(3))

      double precision cwsav(
     $     domLo(1):domHi(1),
     $     domLo(2):domHi(2),
     $     domLo(3):domHi(3))

      double precision cnsav(
     $     domLo(1):domHi(1),
     $     domLo(2):domHi(2),
     $     domLo(3):domHi(3))

      double precision cssav(
     $     domLo(1):domHi(1),
     $     domLo(2):domHi(2),
     $     domLo(3):domHi(3))

      double precision ctsav(
     $     domLo(1):domHi(1),
     $     domLo(2):domHi(2),
     $     domLo(3):domHi(3))

      double precision cbsav(
     $     domLo(1):domHi(1),
     $     domLo(2):domHi(2),
     $     domLo(3):domHi(3))

      integer domLoU(3), domHiU(3)
      double precision uu(
     $     domLoU(1):domHiU(1),
     $     domLoU(2):domHiU(2),
     $     domLoU(3):domHiU(3))

      integer domLoV(3), domHiV(3)
      double precision vv(
     $     domLoV(1):domHiV(1),
     $     domLoV(2):domHiV(2),
     $     domLoV(3):domHiV(3))

      integer domLoW(3), domHiW(3)
      double precision ww(
     $     domLoW(1):domHiW(1),
     $     domLoW(2):domHiW(2),
     $     domLoW(3):domHiW(3))

c     double precision prinv
c     double precision crn

c     double precision vol(
c    $     domLo(1):domHi(1),
c    $     domLo(2):domHi(2),
c    $     domLo(3):domHi(3))

c     double precision rv(domLo(2):domHi(2))
c     double precision ra(domLo(2):domHi(2))
c     double precision rone(domLo(2):domHi(2))
      
      double precision sew(domLo(1):domHi(1))
      double precision sns(domLo(2):domHi(2))
      double precision stb(domLo(3):domHi(3))

      double precision cee(domLo(1):domHi(1))
      double precision cwe(domLo(1):domHi(1))
      double precision cww(domLo(1):domHi(1))

      double precision cnn(domLo(2):domHi(2))
      double precision csn(domLo(2):domHi(2))
      double precision css(domLo(2):domHi(2))

      double precision ctt(domLo(3):domHi(3))
      double precision cbt(domLo(3):domHi(3))
      double precision cbb(domLo(3):domHi(3))

      double precision efac(domLo(1):domHi(1))
      double precision wfac(domLo(1):domHi(1))

      double precision enfac(domLo(2):domHi(2))
      double precision sfac(domLo(2):domHi(2))

      double precision tfac(domLo(3):domHi(3))
      double precision bfac(domLo(3):domHi(3))

      double precision dxpw(domLo(1):domHi(1))
      double precision dyps(domLo(2):domHi(2))
      double precision dzpb(domLo(3):domHi(3))


*-----------------------------------------------------------------------
*     local variables
*
      double precision prinv
      double precision crn
      logical l2up
      double precision vol

      double precision crn0

      integer i, j, k
      integer IST, JST, KST
      integer IEND, JEND, KEND
      integer ioff, joff, koff
      integer indexflo, indext1, indext2
*-----------------------------------------------------------------------

      prinv = 0.01
      crn = zero
      l2up = .true.
      vol = zero

      IST = idxLo(1)
      JST = idxLo(2)
      KST = idxLo(3)
      IEND = idxHi(1)
      JEND = idxHi(2)
      KEND = idxHi(3)

      do 220 k = KST, KEND
         do 210 j = JST, JEND
            do 203 i = IST, IEND

c-----compute areas and volume------------------------------------------

               crn0 = zero

c              vol(i,j,k) = ra(j)*sns(j)*sew(i)*stb(k)
               vol = sns(j)*sew(i)*stb(k)

c-----calculate e-w coefficients----------------------------------------
               ioff = 1
               joff = 0
               koff = 0
               indexflo = 1
               indext1 = 2
               indext2 = 3
               call coeffb(domLoU, domHiU, uu,
     &              domLo, domHi, den,vis,ae,aw,cesav,cwsav,
     $              cee,cwe,cww,efac,wfac,sns,stb,
c    &              ra,
     &              vol,
     $              dxpw,
c    &              rone,
     &              prinv,crn0,i,j,k,ioff,joff,koff,i,j,k,
     $              indexflo, indext1, indext2,
     $              l2up)

c-----calculate n-s coefficients----------------------------------------
               ioff = 0
               joff = 1
               koff = 0
               indexflo = 2
               indext1 = 3
               indext2 = 1

               call coeffb(domLoV, domHiV, vv,
     &              domLo, domHi, den,vis,an,as,cnsav,cssav,
     $              cnn,csn,css,enfac,sfac,stb,sew,
c    &              rv,
     &              vol,
     $              dyps,
c    &              rone,
     &              prinv,crn0,j,k,i,ioff,joff,koff,i,j,k,
     $              indexflo, indext1, indext2,
     $              l2up)

c-----calculate t-b coefficients----------------------------------------
               ioff = 0
               joff = 0
               koff = 1
               indexflo = 3
               indext1 = 1
               indext2 = 2

               call coeffb(domLoW, domHiW, ww,
     &              den,vis,at,ab,ctsav,cbsav,
     $              ctt,cbt,cbb,tfac,bfac,sew,sns,
c    &              rone,
     &              vol,
     $              dzpb,
c    &              ra,
     &              prinv,crn0,k,i,j,ioff,joff,koff,i,j,k,
     $              indexflo, indext1, indext2,
     $              l2up)

               crn = crn + crn0/six
c-----------------------------------------------------------------------
 203        continue
 210     continue
 220  continue

      crn = crn/((IEND-IST+1)*(JEND-JST+1)*(KEND-KST+1))

      return
      end

c
c $Log$
c Revision 1.2  2000/08/08 23:34:20  rawat
c fixed some bugs in profv.F and Properties.cc
c
c Revision 1.1  2000/07/14 05:23:52  bbanerje
c Added scalcoef.F and updated related stuff in C++. scalcoef ==> coefs.f
c in Kumar's code.
c
c
