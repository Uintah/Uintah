c*********************************************************************
c
c
c*********************************************************************

#include <Packages/Uintah/CCA/Components/Arches/fortran/rescal_fort.h>
C-----------------------------------------------------------------------
C     WRITTEN BY: SESHADRI KUMAR, OCTOBER 14, 1996
C     Modified, December 11, 1996, to isolate TRNC calculation from
C     RES calculation, and to use matvec call for streamlining.
C
C     rescal calculates the pointwise residual of the finite difference
C     equation for any variable, between the limits (IST,JST,KST) and
C     (IEND,JEND,KEND) for that variable, and stores the residual in 
C     the array RES.  It also calculates the L1 norm of vector RES,
C     RESPHI.
C
C     EXPLANATION OF SYMBOLS USED:
C
C	AB 	Bottom coefficient (i/p)
C	AE	East coefficient in the seven-point stencil
C		equation for variable V (i/p)
C	AN	North coefficient (i/p)
C	AP	Diagonal coefficient (i/p)
C	AS	South coefficient (i/p)
C	AT	Top coefficient (i/p)
C	AW	West coefficient array (i/p)
C	IST,JST,KST
C		Starting indices for the calculation (i/p)
C	IEND,JEND,KEND
C		Ending indices for calculation (i/p)
C	PHI	Variable for which equation is being solved
C	RES	Array containing the residual of the PHI-equation (o/p)
C	RESPHI	L1 norm of the residual of the PHI-equation (o/p)
c       TRUNCPHI L1 norm of order of magnitude terms in the PHI-equation
C	SU	Source term for the PHI-momentum equation,
C		excluding the pressure gradient (i/p)
C-----------------------------------------------------------------------
c*********************************************************************
c     Local Variables :
c*********************************************************************
      double precision RESAVE, TRUNC

      integer i,j,k

C-----------------------------------------------------------------------
C     EXECUTABLE STATEMENTS
C-----------------------------------------------------------------------
      truncphi = 0.0D0

      do 300 k = idxLo(3), idxHi(3)
         do 200 j = idxLo(2), idxHi(2)
            do 100 i = idxLo(1), idxHi(1)

c-----calculate residual------------------------------------------------

               res(i,j,k) = 
     $              ap(i,j,k)*phi(i,j,k) - (
     $              ae(i,j,k)*phi(i+1,j,k)+
     &              aw(i,j,k)*phi(i-1,j,k)+
     &              an(i,j,k)*phi(i,j+1,k)+
     &              as(i,j,k)*phi(i,j-1,k)+
     &              at(i,j,k)*phi(i,j,k+1)+
     &              ab(i,j,k)*phi(i,j,k-1))
               res(i,j,k) = su(i,j,k) - res(i,j,k)
               trunc = dmax1(dabs(ap(i,j,k)*phi(i,j,k)),
     $                           dabs(su(i,j,k)))
               truncphi = truncphi + trunc
               

 100        continue
 200     continue
 300  continue

      call arrl1(resphi,resave,
     & PASS3A(res)
     & PASS3B(res),
     $     idxLo, idxHi)

C-----------------------------------------------------------------------
      RETURN
      END

c*********************************************************************
c
c Revision 1.4  2000/09/28 22:16:47  bigler
c Changed certain functions to explicity be double precision functions.
c
c Revision 1.3  2000/08/11 21:26:41  rawat
c added linear solver for pressure eqn
c
c Revision 1.2  2000/08/01 23:28:46  skumar
c Added residual calculation procedure and modified templates in linear
c solver.  Added template for order-of-magnitude term calculation.
c
c Revision 1.1  2000/04/12 20:31:50  rawat
c modified PressureSolver and added fortran subroutines
c
c
c*********************************************************************
