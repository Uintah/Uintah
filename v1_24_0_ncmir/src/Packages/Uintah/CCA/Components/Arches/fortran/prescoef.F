c*********************************************************************
c
c
c*********************************************************************

#include <Packages/Uintah/CCA/Components/Arches/fortran/prescoef_fort.h>
#include "param4.h"
*----------------------------------------------------------------------- 
*     Written by Seshadri Kumar, August 4, 1995
*     Last modified, September 16, 1998
*
*     Passed variables (partial list):
*
*	ILCNT	Total number of calls to subroutine linegs in the
*		entire calculation (i/p and o/p)
*-----------------------------------------------------------------------

c*********************************************************************
c     Locals :
c*********************************************************************
      double precision AREAN,AREAS,AREATB,AREAEW,EPSG,VOL 
      integer i,j,k
      integer IST, JST, KST, IEND, JEND, KEND

c*********************************************************************
c     Start :
c*********************************************************************
      IST = idxLo(1)
      JST = idxLo(2)
      KST = idxLo(3)
      IEND = idxHi(1)
      JEND = idxHi(2)
      KEND = idxHi(3)
C-----------------------------------------------------------------------
C     ASSEMBLY OF COEFFICIENTS  
C-----------------------------------------------------------------------
      DO 120 k = KST,KEND
         DO 110 j = JST,JEND
            DO 100 i = IST,IEND
C-----------------------------------------------------------------------
C     COMPUTE AREAS AND VOLUME
C-----------------------------------------------------------------------
               AREAN = SEW(i)*STB(k)
               AREAS = SEW(i)*STB(k)
               AREAEW = SNS(j)*STB(k)
               AREATB = SNS(j)*SEW(i)
               VOL = SNS(j)*SEW(i)*STB(k)
               EPSG = ONE
C-----------------------------------------------------------------------
C     CALCULATE COEFFICIENTS
C-----------------------------------------------------------------------

               AE(i,j,k) = AREAEW/DXEP(i)*EPSG

               AW(i,j,k) = AREAEW/DXPW(i)*EPSG

               AN(i,j,k) = AREAN/DYNP(j)*EPSG

               AS(i,j,k) = AREAS/DYPS(j)*EPSG

               AT(i,j,k) = AREATB/DZTP(k)*EPSG

               AB(i,j,k) = AREATB/DZPB(k)*EPSG
 100        CONTINUE
 110     CONTINUE
 120  CONTINUE

      RETURN
      END
c*********************************************************************
c
c Revision 1.4  2000/09/26 04:35:34  rawat
c added some more multi-patch support
c
c Revision 1.3  2000/07/28 02:31:04  rawat
c moved all the labels in ArchesLabel. fixed some bugs and added matrix_dw to store matrix
c coeffecients
c
c Revision 1.2  2000/07/17 22:07:02  rawat
c modified momentum source
c
c Revision 1.1  2000/07/12 22:15:05  bbanerje
c Added pressure Coef .. will do until Kumar's code is up and running
c
c
c*********************************************************************

