c*********************************************************************
c
c
c*********************************************************************
      SUBROUTINE prescoef(domLo, domHi, domLong, domHing, 
     $     idxLo, idxHi, DEN,
     &     AE, AW, AN, AS, AT, AB,
     &     domLoU, domHiU, AUP,
     &     domLoV, domHiV, AVP,
     &     domLoW, domHiW, AWP,
     &     SEW, SNS, STB,
     &     SEWU, DXEP, DXPW, 
     &     SNSV, DYNP, DYPS, 
     &     STBW, DZTP, DZPB)

*----------------------------------------------------------------------- 
*     Written by Seshadri Kumar, August 4, 1995
*     Last modified, September 16, 1998
*
*     Passed variables (partial list):
*
*	ILCNT	Total number of calls to subroutine linegs in the
*		entire calculation (i/p and o/p)
*-----------------------------------------------------------------------
      implicit none

#include "param4.h"

c*********************************************************************
c     Arguments :
c*********************************************************************
      integer domLo(3), domHi(3)
      integer domLong(3), domHing(3)
      integer idxLo(3), idxHi(3)
      double precision DEN(domLo(1):domHi(1),
     &     domLo(2):domHi(2),domLo(3):domHi(3))
      double precision AE(domLong(1):domHing(1),
     &     domLong(2):domHing(2),domLong(3):domHing(3))
      double precision AW(domLong(1):domHing(1),
     &     domLong(2):domHing(2),domLong(3):domHing(3))
      double precision AN(domLong(1):domHing(1),
     &     domLong(2):domHing(2),domLong(3):domHing(3))
      double precision AS(domLong(1):domHing(1),
     &     domLong(2):domHing(2),domLong(3):domHing(3))
      double precision AT(domLong(1):domHing(1),
     &     domLong(2):domHing(2),domLong(3):domHing(3))
      double precision AB(domLong(1):domHing(1),
     &     domLong(2):domHing(2),domLong(3):domHing(3))
      integer domLoU(3), domHiU(3)
      double precision AUP(domLoU(1):domHiU(1),
     &     domLoU(2):domHiU(2),domLoU(3):domHiU(3))
      integer domLoV(3), domHiV(3)
      double precision AVP(domLoV(1):domHiV(1),
     &     domLoV(2):domHiV(2),domLoV(3):domHiV(3))
      integer domLoW(3), domHiW(3)
      double precision AWP(domLoW(1):domHiW(1),
     &     domLoW(2):domHiW(2),domLoW(3):domHiW(3))
      double precision SEWU(domLo(1):domHi(1))
      double precision SEW(domLo(1):domHi(1))
      double precision SNSV(domLo(2):domHi(2))
      double precision SNS(domLo(2):domHi(2))
      double precision STBW(domLo(3):domHi(3))
      double precision STB(domLo(3):domHi(3))
      double precision DXEP(domLo(1):domHi(1))
      double precision DXPW(domLo(1):domHi(1))
      double precision DYNP(domLo(2):domHi(2))
      double precision DYPS(domLo(2):domHi(2))
      double precision DZTP(domLo(3):domHi(3))
      double precision DZPB(domLo(3):domHi(3))

c*********************************************************************
c     Locals :
c*********************************************************************
      double precision AREAN,AREAS,AREATB,AREAEW,EPSG,VOL 
      double precision DENE,DENW,DENN,DENS,DENT,DENB
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
c              IF (LCALP .AND. .NOT.LP) THEN
c                 CALL CALEPS(domLo, domHi, 
c    &                 DENP,PDEN,I,J,K,EPSG)
c              ENDIF
C-----------------------------------------------------------------------
C     CALCULATE COEFFICIENTS
C-----------------------------------------------------------------------
               DENE = PT5*(DEN(i,j,k)+DEN(i+1,j,k))
               DENW = PT5*(DEN(i,j,k)+DEN(i-1,j,k))
               DENN = PT5*(DEN(i,j,k)+DEN(i,j+1,k))
               DENS = PT5*(DEN(i,j,k)+DEN(i,j-1,k))
               DENT = PT5*(DEN(i,j,k)+DEN(i,j,k+1))
               DENB = PT5*(DEN(i,j,k)+DEN(i,j,k-1))

               AE(i,j,k) = DENE*AREAEW*AREAEW*SEWU(i+1)/
     &              (DXEP(i)*AUP(i+1,j,k))*EPSG

               AW(i,j,k) = DENW*AREAEW*AREAEW*SEWU(i)/
     &              (DXPW(i)*AUP(i,j,k))*EPSG

               AN(i,j,k) = DENN*AREAN*SEW(i)*SNSV(j+1)*
     &              STB(k)/(DYNP(j)*AVP(i,j+1,k))*EPSG

               AS(i,j,k) = DENS*AREAS*SEW(i)*SNSV(j)*
     &              STB(k)/(DYPS(j)*AVP(i,j,k))*EPSG

               AT(i,j,k) = DENT*AREATB*AREATB*STBW(k+1)/
     &              (DZTP(k)*AWP(i,j,k+1))*EPSG

               AB(i,j,k) = DENB*AREATB*AREATB*STBW(k)/
     &              (DZPB(k)*AWP(i,j,k)) *EPSG
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

