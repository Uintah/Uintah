c*********************************************************************
c
c $Id$
c
c*********************************************************************

      SUBROUTINE cellg(indexLow, indexHigh, 
     &     SEW, SNS, STB,
     &     SEWU, SNSV, STBW,
     &     DXEP, DYNP, DZTP,
     &     DXEPU, DYNPV, DZTPW,
     &     DXPW, DYPS, DZPB,
     &     DXPWU, DYPSV, DZPBW,
     &     CEE, CWE, CWW,
     &     CEEU, CWEU, CWWU,
     &     CNN, CSN, CSS,
     &     CNNV, CSNV, CSSV,
     &     CTT, CBT, CBB,
     &     CTTW, CBTW, CBBW,
c     &     RR, RA, RV, RONE,
c     &     RCV, RCVA,
     &     XX, XU, YY, YV, ZZ, ZW,
     &     EFAC, WFAC, NFAC, SFAC, TFAC, BFAC,
     &     FAC1U, FAC2U, FAC3U, FAC4U,
     &     FAC1V, FAC2V, FAC3V, FAC4V,
     &     FAC1W, FAC2W, FAC3W, FAC4W,
     &     IESDU, IWSDU, JNSDV, JSSDV, KTSDW, KBSDW,
c     &     LCORD,
     &     ierr)
C-----------------------------------------------------------------------
C     THREE-DIMENSIONAL VERSION
C     CALCULATION OF CELL GEOMETRY PARAMETERS
C-----------------------------------------------------------------------

      implicit none

#include "param4.h"

c*********************************************************************
c     Arguments :
c*********************************************************************
      integer indexLow(3)
      integer indexHigh(3)
      double precision SEW(indexLow(1):indexHigh(1))
      double precision SNS(indexLow(2):indexHigh(2))
      double precision STB(indexLow(3):indexHigh(3))
      double precision SEWU(indexLow(1):indexHigh(1))
      double precision SNSV(indexLow(2):indexHigh(2))
      double precision STBW(indexLow(3):indexHigh(3))
      double precision DXEP(indexLow(1):indexHigh(1))
      double precision DYNP(indexLow(2):indexHigh(2))
      double precision DZTP(indexLow(3):indexHigh(3))
      double precision DXEPU(indexLow(1):indexHigh(1))
      double precision DYNPV(indexLow(2):indexHigh(2))
      double precision DZTPW(indexLow(3):indexHigh(3))
      double precision DXPW(indexLow(1):indexHigh(1))
      double precision DYPS(indexLow(2):indexHigh(2))
      double precision DZPB(indexLow(3):indexHigh(3))
      double precision DXPWU(indexLow(1):indexHigh(1))
      double precision DYPSV(indexLow(2):indexHigh(2))
      double precision DZPBW(indexLow(3):indexHigh(3))
      double precision CEE(indexLow(1):indexHigh(1))
      double precision CWE(indexLow(1):indexHigh(1))
      double precision CWW(indexLow(1):indexHigh(1))
      double precision CEEU(indexLow(1):indexHigh(1))
      double precision CWEU(indexLow(1):indexHigh(1))
      double precision CWWU(indexLow(1):indexHigh(1))
      double precision CNN(indexLow(2):indexHigh(2))
      double precision CSN(indexLow(2):indexHigh(2))
      double precision CSS(indexLow(2):indexHigh(2))
      double precision CNNV(indexLow(2):indexHigh(2))
      double precision CSNV(indexLow(2):indexHigh(2))
      double precision CSSV(indexLow(2):indexHigh(2))
      double precision CTT(indexLow(3):indexHigh(3))
      double precision CBT(indexLow(3):indexHigh(3))
      double precision CBB(indexLow(3):indexHigh(3))
      double precision CTTW(indexLow(3):indexHigh(3))
      double precision CBTW(indexLow(3):indexHigh(3))
      double precision CBBW(indexLow(3):indexHigh(3))
c      double precision RR(indexLow(2):indexHigh(2))
c      double precision RA(indexLow(2):indexHigh(2))
c      double precision RV(indexLow(2):indexHigh(2))
c      double precision RONE(indexLow(2):indexHigh(2))
c      double precision RCV(indexLow(2):indexHigh(2))
c      double precision RCVA(indexLow(2):indexHigh(2))
      double precision XX(indexLow(1):indexHigh(1))
      double precision XU(indexLow(1):indexHigh(1))
      double precision YY(indexLow(2):indexHigh(2))
      double precision YV(indexLow(2):indexHigh(2))
      double precision ZZ(indexLow(3):indexHigh(3))
      double precision ZW(indexLow(3):indexHigh(3))
      double precision EFAC(indexLow(1):indexHigh(1))
      double precision WFAC(indexLow(1):indexHigh(1))
      double precision NFAC(indexLow(2):indexHigh(2))
      double precision SFAC(indexLow(2):indexHigh(2))
      double precision TFAC(indexLow(3):indexHigh(3))
      double precision BFAC(indexLow(3):indexHigh(3))
      double precision FAC1U(indexLow(1):indexHigh(1))
      double precision FAC2U(indexLow(1):indexHigh(1))
      double precision FAC3U(indexLow(1):indexHigh(1))
      double precision FAC4U(indexLow(1):indexHigh(1))
      double precision FAC1V(indexLow(2):indexHigh(2))
      double precision FAC2V(indexLow(2):indexHigh(2))
      double precision FAC3V(indexLow(2):indexHigh(2))
      double precision FAC4V(indexLow(2):indexHigh(2))
      double precision FAC1W(indexLow(3):indexHigh(3))
      double precision FAC2W(indexLow(3):indexHigh(3))
      double precision FAC3W(indexLow(3):indexHigh(3))
      double precision FAC4W(indexLow(3):indexHigh(3))
      integer IESDU(indexLow(1):indexHigh(1))
      integer IWSDU(indexLow(1):indexHigh(1))
      integer JNSDV(indexLow(2):indexHigh(2))
      integer JSSDV(indexLow(2):indexHigh(2))
      integer KTSDW(indexLow(3):indexHigh(3))
      integer KBSDW(indexLow(3):indexHigh(3))
c      logical LCORD
      integer ierr

c*********************************************************************
c     Local Variables :
c*********************************************************************
      integer iBegGhost, jBegGhost, kBegGhost
      integer iEndGhost, jEndGhost, kEndGhost
      integer IST, JST, KST, IEND, JEND, KEND
      integer i, j, k
      double precision ESD, PESD, WSD, WWSD
      double precision SSD, PNSD, RNSD, SSSD
      double precision TSD, BSD, PTSD, BBSD

c*********************************************************************
c     Start :
c*********************************************************************
      iBegGhost = indexLow(1)
      jBegGhost = indexLow(2)
      kBegGhost = indexLow(3)
      iEndGhost = indexHigh(1)
      jEndGhost = indexHigh(2)
      kEndGhost = indexHigh(3)
      IST = iBegGhost+1
      JST = jBegGhost+1
      KST = kBegGhost+1
      IEND = iEndGhost-1
      JEND = jEndGhost-1
      KEND = kEndGhost-1

C-----------------------------------------------------------------------
C-----------------------------------------------------------------------

C-----------------------------------------------------------------------
C     CYLINDRICAL METRICS
C-----------------------------------------------------------------------

c      DO 10 J=jBegGhost,jEndGhost
c         RONE(J) = ONE
c 10   CONTINUE
c      DO 100 J = jBegGhost,jEndGhost
c       RR(J) = YY(J)
c        IF (.NOT. LCORD) RR(J) = ONE
c  100 CONTINUE
c      IF (LCORD) THEN
c        RV(jBegGhost) = (THREE*RR(jBegGhost)-RR(JST))/TWO
c      ELSE
c        RV(jBegGhost) = ONE
c      END IF
c      DO 600 J = JST,jEndGhost
c        RV(J) = PT5*(RR(J)+RR(J-1))
c        RCV(J) = PT5*(RV(J)+RV(J-1))
c       RA(J-1) = RCV(J)
c  600 CONTINUE
c      RCV(jBegGhost) = (THREE*RV(jBegGhost)-RV(JST))/TWO
c      RA(jEndGhost) = RR(jEndGhost)
c
c      DO 610 J = JST,JEND
c         RCVA(J) = PT5*(RCV(J)+RCV(J+1))
c 610  CONTINUE
c      

C-----------------------------------------------------------------------
C     X-DIRECTION
C-----------------------------------------------------------------------

*     p-cell distances and co-ordinates

      DO 150 I = iBegGhost,IEND
        DXEP(I) = XX(I+1)-XX(I)
        DXPW(I+1) = DXEP(I)
  150 CONTINUE
      DXPW(iBegGhost) = DXPW(IST)
      DXEP(iEndGhost) = DXEP(IEND)
      DO 300 I = iBegGhost,iEndGhost
        SEW(I) = PT5*(DXEP(I)+DXPW(I))
  300 CONTINUE

*     u-cell distances and co-ordinates

      DO 450 I = IST,iEndGhost
        XU(I) = PT5*(XX(I)+XX(I-1))
  450 CONTINUE
      XU(iBegGhost) = (THREE*XX(iBegGhost)-XX(IST))/TWO
      DO 500 I = iBegGhost,IEND
        DXEPU(I) = XU(I+1)-XU(I)
        DXPWU(I+1) = DXEPU(I)
  500 CONTINUE
      DXPWU(iBegGhost) = DXEPU(iBegGhost)
      DXEPU(iEndGhost) = DXPWU(iEndGhost)
      DO 550 I = iBegGhost,iEndGhost
        SEWU(I) = PT5*(DXEPU(I)+DXPWU(I))
  550 CONTINUE

*     factors for differencing schemes: p-cell

      DO 780 I = iBegGhost,iEndGhost
         CEE(I) = DXPW(I)/SEW(I)
         CWW(I) = DXEP(I)/SEW(I)
         CWE(I) = PT5*(CWW(I)-CEE(I))
         CEEU(I) = DXPWU(I)/SEWU(I)
         CWWU(I) = DXEPU(I)/SEWU(I)
         CWEU(I) = PT5*(CWWU(I)-CEEU(I))
 780  CONTINUE
      DO 781 I = iBegGhost,iEndGhost
         EFAC(I) = (XX(I)-XU(I))/SEW(I)
 781  CONTINUE
      DO 782 I = iBegGhost,IEND
         WFAC(I) = (XU(I+1)-XX(I))/SEW(I)
 782  CONTINUE
      WFAC(iEndGhost) = WFAC(IEND)

*     factors for differencing schemes: u-cell

      DO 800 I = IST+1,IEND
        ESD = PT5*(XU(I+1)+XU(I))
        WSD = PT5*(XU(I)+XU(I-1))
        PESD = XX(I)-ESD
        WWSD = XX(I-1)-WSD
        IF (PESD.LE.ZERO) THEN
          FAC1U(I) = (XX(I+1)-ESD)/DXPW(I+1)
          FAC2U(I) = (ESD-XX(I))/DXPW(I+1)
          IESDU(I) = 1
        ELSE
          FAC1U(I) = (ESD-XX(I-1))/DXPW(I)
          FAC2U(I) = (XX(I)-ESD)/DXPW(I)
          IESDU(I) = -1
        END IF
        IF (WWSD.LE.ZERO) THEN
          FAC3U(I) = (XX(I)-WSD)/DXPW(I)
          FAC4U(I) = (WSD-XX(I-1))/DXPW(I)
          IWSDU(I) = 0
        ELSE
          FAC3U(I) = (WSD-XX(I-2))/DXPW(I-1)
          FAC4U(I) = (XX(I-1)-WSD)/DXPW(I-1)
          IWSDU(I) = -2
        END IF
  800 CONTINUE

C-----------------------------------------------------------------------
C     Y-DIRECTION
C-----------------------------------------------------------------------

*     p-cell distances and co-ordinates

      DO 200 J = jBegGhost,JEND
        DYNP(J) = YY(J+1)-YY(J)
        DYPS(J+1) = DYNP(J)
  200 CONTINUE
      DYPS(jBegGhost) = DYPS(JST)
      DYNP(jEndGhost) = DYNP(JEND)
      DO 350 J = jBegGhost,jEndGhost
        SNS(J) = PT5*(DYNP(J)+DYPS(J))
  350 CONTINUE

*     v-cell distances and co-ordinates

      DO 601 J = JST,jEndGhost
         YV(J) = PT5*(YY(J)+YY(J-1))
 601  CONTINUE
      YV(jBegGhost) = (THREE*YY(jBegGhost)-YY(JST))/TWO
      DO 650 J = jBegGhost,JEND
        DYNPV(J) = YV(J+1)-YV(J)
        DYPSV(J+1) = DYNPV(J)
  650 CONTINUE
      DYPSV(jBegGhost) = DYNPV(jBegGhost)
      DYNPV(jEndGhost) = DYPSV(jEndGhost)
      DO 700 J = jBegGhost,jEndGhost
        SNSV(J) = PT5*(DYNPV(J)+DYPSV(J))
  700 CONTINUE

*     factors for differencing schemes: p-cell

      DO 785 J = jBegGhost,jEndGhost
         CNN(J) = DYPS(J)/SNS(J)
         CSS(J) = DYNP(J)/SNS(J)
         CSN(J) = PT5*(CSS(J)-CNN(J))
         CNNV(J) = DYPSV(J)/SNSV(J)
         CSSV(J) = DYNPV(J)/SNSV(J)
         CSNV(J) = PT5*(CSSV(J)-CNNV(J))
  785 CONTINUE
      DO 786 J = jBegGhost,jEndGhost
         NFAC(J) = (YY(J)-YV(J))/SNS(J)
 786  CONTINUE
      DO 787 J = jBegGhost,JEND
         SFAC(J) = (YV(J+1)-YY(J))/SNS(J)
 787  CONTINUE
      SFAC(jEndGhost) = SFAC(JEND)

*     factors for differencing schemes: v-cell

      DO 850 J = JST+1,JEND
        RNSD = PT5*(YV(J+1)+YV(J))
        SSD = PT5*(YV(J)+YV(J-1))
        PNSD = YY(J)-RNSD
        SSSD = YY(J-1)-SSD
        IF (PNSD.LE.ZERO) THEN
          FAC1V(J) = (YY(J+1)-RNSD)/DYPS(J+1)
          FAC2V(J) = (RNSD-YY(J))/DYPS(J+1)
          JNSDV(J) = 1
        ELSE
          FAC1V(J) = (RNSD-YY(J-1))/DYPS(J)
          FAC2V(J) = (YY(J)-RNSD)/DYPS(J)
          JNSDV(J) = -1
        END IF
        IF (SSSD.LE.ZERO) THEN
          FAC3V(J) = (YY(J)-SSD)/DYPS(J)
          FAC4V(J) = (SSD-YY(J-1))/DYPS(J)
          JSSDV(J) = 0
        ELSE
          FAC3V(J) = (SSD-YY(J-2))/DYPS(J-1)
          FAC4V(J) = (YY(J-1)-SSD)/DYPS(J-1)
          JSSDV(J) = -2
        END IF
  850 CONTINUE

C-----------------------------------------------------------------------
C     Z-DIRECTION
C-----------------------------------------------------------------------

*     p-cell distances and co-ordinates

      DO 250 K = kBegGhost,KEND
        DZTP(K) = ZZ(K+1)-ZZ(K)
        DZPB(K+1) = DZTP(K)
  250 CONTINUE
      DZPB(kBegGhost) = DZPB(KST) 
      DZTP(kEndGhost) = DZTP(KEND) 
c      IF (LCORD) DZPB(kBegGhost) = DZPB(KEND)
c      IF (LCORD) DZTP(kEndGhost) = DZTP(KST)
      DO 400 K = kBegGhost,kEndGhost
        STB(K) = PT5*(DZTP(K)+DZPB(K))
  400 CONTINUE

*     w-cell distances and co-ordinates

      DO 725 K = KST,kEndGhost
         ZW(K) = PT5*(ZZ(K)+ZZ(K-1))
  725 CONTINUE
      ZW(kBegGhost) = (THREE*ZZ(kBegGhost)-ZZ(KST))/TWO
      DO 750 K = kBegGhost,KEND
        DZTPW(K) = ZW(K+1)-ZW(K)
        DZPBW(K+1) = DZTPW(K)
  750 CONTINUE
      DZPBW(kBegGhost) = DZTPW(kBegGhost)
      DZTPW(kEndGhost) = DZPBW(kEndGhost)
c      IF (LCORD) DZPBW(kBegGhost) = DZPBW(KEND)
c      IF (LCORD) DZTPW(kEndGhost) = DZTPW(KST)
      DO 775 K = kBegGhost,kEndGhost
        STBW(K) = PT5*(DZTPW(K)+DZPBW(K))
  775 CONTINUE
c      IF (LCORD) STBW(kBegGhost) = STBW(KEND)
c      IF (LCORD) STBW(kEndGhost) = STBW(KST)

*     factors for differencing schemes: p-cell

      DO 790 K = kBegGhost,kEndGhost
         CTT(K) = DZPB(K)/STB(K)
         CBB(K) = DZTP(K)/STB(K)
         CBT(K) = PT5*(CBB(K)-CTT(K))
         CTTW(K) = DZPBW(K)/STBW(K)
         CBBW(K) = DZTPW(K)/STBW(K)
         CBTW(K) = PT5*(CBBW(K)-CTTW(K))
  790 CONTINUE
      DO 791 K = kBegGhost,kEndGhost
         TFAC(K) = (ZZ(K)-ZW(K))/STB(K)
 791  CONTINUE
      DO 792 K = kBegGhost,KEND
         BFAC(K) = (ZW(K+1)-ZZ(K))/STB(K)
 792  CONTINUE
      BFAC(kEndGhost) = BFAC(KEND)

*     factors for differencing schemes: w-cell

C$DIR NO_RECURRENCE
      DO 900 K = KST,KEND
        TSD = PT5*(ZW(K+1)+ZW(K))
        BSD = PT5*(ZW(K)+ZW(K-1))
        PTSD = ZZ(K)-TSD
        BBSD = ZZ(K-1)-BSD
        IF (PTSD.LE.ZERO) THEN
          FAC1W(K) = (ZZ(K+1)-TSD)/DZPB(K+1)
          FAC2W(K) = (TSD-ZZ(K))/DZPB(K+1)
          KTSDW(K) = 1
        ELSE
          FAC1W(K) = (TSD-ZZ(K-1))/DZPB(K)
          FAC2W(K) = (ZZ(K)-TSD)/DZPB(K)
          KTSDW(K) = -1
        END IF
        IF (K.NE.KST) THEN
          IF (BBSD.LE.ZERO) THEN
            FAC3W(K) = (ZZ(K)-BSD)/DZPB(K)
            FAC4W(K) = (BSD-ZZ(K-1))/DZPB(K)
            KBSDW(K) = 0
          ELSE
            FAC3W(K) = (BSD-ZZ(K-2))/DZPB(K-1)
            FAC4W(K) = (ZZ(K-1)-BSD)/DZPB(K-1)
            KBSDW(K) = -2
          END IF
        END IF
  900 CONTINUE
      FAC3W(KST) = FAC1W(KEND)
      FAC4W(KST) = FAC2W(KEND)
      RETURN
      END

c*********************************************************************
c
c $Log$
c Revision 1.2  2000/06/14 20:40:53  rawat
c modified boundarycondition for physical boundaries and
c added CellInformation class
c
c Revision 1.1  2000/04/12 20:31:44  rawat
c modified PressureSolver and added fortran subroutines
c
c
c*********************************************************************
