c
c

#if 1
#include <Packages/Uintah/CCA/Components/Arches/fortran/coeffb_fort.h>
#else
      SUBROUTINE coeffb(domLoVel, domHiVel, VEL,
     $     domLo, domHi, 
     &     DEN,VIS,
     &     AF,AB,
     &     CFSAV,CBSAV,
     $     CFF,CBF,CBB,
     &     FACF,FACB,
     &     ST1,ST2,
c    &     RA,
     &     VOL,
     &     DFB,
c    &     RD,
     &     PRINV,CRN,
     $     IDF,IDT1,IDT2,IOFF,JOFF,KOFF,I,J,K, 
     $     indexflo, indext1, indext2,
     $     L2UP)
C-----------------------------------------------------------------------
C     WRITTEN BY SESHADRI KUMAR, OCTOBER 14, 1996
c     Modified by Seshadri Kumar for complete modularity, June 11, 2000
C
C     PASS:
C     E-W COEFFICIENTS:
C     vel=u,cfsav=cesav,cbsav=cwsav,af=ae,ab=aw,
C     cff=cee,cbf=cwe,cbb=cww,
C     facf=efac,facb=wfac,st1=sns,st2=stb,
C     ra=ra,dfb=dxpw,rd=rone,
C     idf=i,idt1=j,idt2=k,ioff=1,joff=0,koff=0
C
C     N-S COEFFICIENTS:
C     vel=v,cfsav=cnsav,cbsav=cssav,af=an,ab=as,
C     cff=cnn,cbf=csn,cbb=css,
C     facf=enfac,facb=sfac,st1=stb,st2=sew,
C     ra=rv,dfb=dyps,rd=rone,
C     idf=j,idt1=k,idt2=i,ioff=0,joff=1,koff=0
C
C     T-B COEFFICIENTS:
C     vel=w,cfsav=ctsav,cbsav=cbsav,af=at,ab=ab,
C     cff=ctt,cbf=cbt,cbb=cbb,
C     facf=tfac,facb=bfac,st1=sew,st2=sns,
C     ra=rone,dfb=dzpb,rd=ra,
C     idf=k,idt1=i,idt2=j,ioff=0,joff=0,koff=1
C
C     PASSED VARIABLES:
C     
C	VEL	Velocity in flow direction (i/p)
C	DEN	Density field (i/p)
C	VIS	Viscosity field (i/p)
C	AF,AB	Front and Back coefficients in the flow direction of
C		the finite difference equation (o/p)
C	CFSAV,CBSAV
C		Convection fluxes on the front and back faces of 
C		the P-cell (o/p) in the flow direction
C	CFF,CBF,CBB
C		Interpolation factors for interpolating from
C		F- and B-faces of regular P-cell to artificial
C		cell for accurate second order finite differences (i/p)
C	FACF,FACB	
C		Interpolation factors for calculating value at
C		P-node from values at its front and back cell faces
C		respectively (i/p)
C	ST1	Width of P-cell in T1-direction (i/p)
C	ST2	Width of P-cell in T2-direction (i/p)
C	RA	Radius needed for calculation of front and back areas
C		of P-cell in flow (F) direction
C	VOL	Array: Volume of P-Cell (o/p)
C	DFB	Distance between successive P-nodes in the flow 
C		direction (i/p)
C	RD	Radius needed for calculation of distance between P-cells
C		in flow direction
C	PRINV	Inverse of the "Prandtl number" for this scalar
C		(e.g., PRF, PRG, etc.) (i/p)
C	CRN	Average Cell Reynolds number (o/p)
C	IDF	Index of the flow direction (e.g. = J if the y-direction
C		is the flow direction) (i/p)
C	IDT1	Index of the first transverse direction (e.g., = K if 
C		the y-direction is the flow direction) (i/p)
C	IDT2 	Index of the second transverse direction (e.g., = I if 
C		the y-direction is the flow direction) (i/p)
C	IOFF,JOFF,KOFF
C		Offsets in the i-,j-, and k- directions.  These determine
C		which direction is the "flow" direction.  Only one of these
C		can be unity and the other two must be zero.  For example,
C		if KOFF=1, then the z-direction is the flow direction.
C		This subroutine (coeffb) is called three times for
C		each (i,j,k) in coefs.f, once for each direction, with
C		different offsets each time, thus
C		calculating coefficients for all three directions
C	I,J,K	Co-ordinates of the point for which coefficients are
C		being calculated
C	L2UP	Logical, =T implies that the 2nd upwind differencing
C		scheme of Roache is used
C
C      SUBROUTINE coeffb(VEL,DEN,VIS,AF,AB,CFSAV,CBSAV,
C     $     CFF,CBF,CBB,FACF,FACB,ST1,ST2,RA,VOL,DFB,RD,PRINV,CRN,
C     $     IDF,IDT1,IDT2,IOFF,JOFF,KOFF,I,J,K,L2UP)
C-----------------------------------------------------------------------
      implicit none

      INCLUDE 'param4.h'
C-----------------------------------------------------------------------

      integer domLoVel(3), domHiVel(3)
      double precision vel(
     $     domLoVel(1):domHiVel(1),
     $     domLoVel(2):domHiVel(2),
     $     domLoVel(3):domHiVel(3))

      integer domLo(3), domHi(3)

      integer indexflo
      integer indext1
      integer indext2

      double precision den(
     $     domLo(1):domHi(1),
     $     domLo(2):domHi(2),
     $     domLo(3):domHi(3))

      double precision vis(
     $     domLo(1):domHi(1),
     $     domLo(2):domHi(2),
     $     domLo(3):domHi(3))

      double precision af(
     $     domLo(1):domHi(1),
     $     domLo(2):domHi(2),
     $     domLo(3):domHi(3))

      double precision ab(
     $     domLo(1):domHi(1),
     $     domLo(2):domHi(2),
     $     domLo(3):domHi(3))

      double precision cfsav(
     $     domLo(1):domHi(1),
     $     domLo(2):domHi(2),
     $     domLo(3):domHi(3))

      double precision cbsav(
     $     domLo(1):domHi(1),
     $     domLo(2):domHi(2),
     $     domLo(3):domHi(3))

      double precision cff(domLo(indexflo):domHi(indexflo))
      double precision cbf(domLo(indexflo):domHi(indexflo))
      double precision cbb(domLo(indexflo):domHi(indexflo))

      double precision facf(domLo(indexflo):domHi(indexflo))
      double precision facb(domLo(indexflo):domHi(indexflo))

      double precision st1(domLo(indext1):domHi(indext1))
      double precision st2(domLo(indext2):domHi(indext2))

c     double precision ra(domLo(2):domHi(2))

c     double precision vol(
c    $     domLo(1):domHi(1),
c    $     domLo(2):domHi(2),
c    $     domLo(3):domHi(3))
      double precision vol

      double precision dfb(domLo(indexflo):domHi(indexflo))
c     double precision rd(domLo(2):domHi(2))

      double precision prinv
      double precision crn

      integer idf
      integer idt1
      integer idt2
      integer ioff
      integer joff
      integer koff
      integer i
      integer j
      integer k

      LOGICAL L2UP

*     local variables

      double precision areaf
      double precision areab
      double precision distf
      double precision distb
      double precision denf
      double precision denb
      double precision cfo
      double precision cbo
      double precision gvp
      double precision cpf
      double precision cpb
      double precision gamf
      double precision gamb
      double precision dfo
      double precision dbo
      double precision df
      double precision db
      double precision cf
      double precision cb
      double precision coeff
      double precision coefb
      double precision tfb
      double precision afc
      double precision abc     

*     externals
      
      double precision rmean

C-----CALCULATE AREAS AND DISTANCES-------------------------------------

c     AREAF = RA(J+JOFF)*ST2(IDT2)*ST1(IDT1)
c     AREAB = RA(J)*ST2(IDT2)*ST1(IDT1)
c     DISTF = RD(J)*DFB(IDF+1)
c     DISTB = RD(J)*DFB(IDF)
      AREAF = ST2(IDT2)*ST1(IDT1)
      AREAB = ST2(IDT2)*ST1(IDT1)
      DISTF = DFB(IDF+1)
      DISTB = DFB(IDF)

C-----CALCULATE CONVECTION COEFFICIENTS---------------------------------

      DENF = RMEAN(DEN,I+IOFF,J+JOFF,K+KOFF,IOFF,JOFF,KOFF,
     $     domLo, domHi)
      DENB = RMEAN(DEN,I,J,K,IOFF,JOFF,KOFF,
     $     domLo, domHi)
      CFO = DENF*VEL(I+IOFF,J+JOFF,K+KOFF)*AREAF
      CBO = DENB*VEL(I,J,K)*AREAB

C------UPWIND DIFFERENCING STUFF----------------------------------------

      GVP = DEN(I,J,K)*(FACF(IDF)*VEL(I+IOFF,J+JOFF,K+KOFF)+
     $     FACB(IDF)*VEL(I,J,K))
      CPF = GVP*VOL/DISTF
      CPB = GVP*VOL/DISTB

C-----CALCULATE DIFFUSION COEFFICIENTS-------------------------------

      GAMF = RMEAN(VIS,I+IOFF,J+JOFF,K+KOFF,IOFF,JOFF,KOFF,
     $     domLo, domHi)*PRINV
      GAMB = RMEAN(VIS,I,J,K,IOFF,JOFF,KOFF,
     $     domLo, domHi)*PRINV
      DFO = GAMF*AREAF/DISTF
      DBO = GAMB*AREAB/DISTB

C-----DIFFUSION COEFFICIENTS FOR ACCURATE 2ND ORDER DIFFERENCES---------

      DF = CFF(IDF)*DFO + CBF(IDF)*DBO*DISTB/DISTF
      DB = CBB(IDF)*DBO - CBF(IDF)*DFO*DISTF/DISTB

C-----CONVECTION COEFFICIENTS FOR ACCURATE 2ND ORDER DIFFERENCES--------

      CF = CFF(IDF)*CFO + CBF(IDF)*CBO
      CB = CBB(IDF)*CBO - CBF(IDF)*CFO

C-----SAVE CONVECTION COEFFICIENTS FOR MASS BALANCE---------------------

      CFSAV(I,J,K) = CF
      CBSAV(I,J,K) = CB

C-----CONVECTION DIFFERENCING SCHEME SELECTION--------------------------

      COEFF = DF - PT5*ABS(CF)
      COEFB = DB - PT5*ABS(CB)
      TFB = ONE
      IF (COEFF.LT.ZERO .OR. COEFB.LT.ZERO) TFB = ZERO
      AFC = -PT5*CF*TFB + MAX(ZERO,-CPF)*(ONE-TFB)
      ABC = PT5*CB*TFB + MAX(ZERO,CPB)*(ONE-TFB)

C-----------------------------------------------------------------------

      IF (L2UP) THEN
         AF(I,J,K) = DF*(ONE-MAX(ZERO,SIGN(ONE,COEFF)))
     &        + MAX(ZERO,COEFF) + MAX(ZERO,-CF)
         AB(I,J,K) = DB*(ONE-MAX(ZERO,SIGN(ONE,COEFB)))
     &        + MAX(ZERO,COEFB) + MAX(ZERO,CB)
      ELSE
         AF(I,J,K) = AFC + DF
         AB(I,J,K) = ABC + DB
      END IF

C-----------------------------------------------------------------------

      CRN = CRN + ABS(CF/DF) + ABS(CB/DB)

C-----------------------------------------------------------------------
      RETURN
      END

c
c Revision 1.1  2000/07/14 05:23:52  bbanerje
c Added scalcoef.F and updated related stuff in C++. scalcoef ==> coefs.f
c in Kumar's code.
c
c
