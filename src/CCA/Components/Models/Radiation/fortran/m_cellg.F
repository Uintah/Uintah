C
C Copyright (c) 1997-2012 The University of Utah
C
C Permission is hereby granted, free of charge, to any person obtaining a copy
C of this software and associated documentation files (the \"Software\"), to
C deal in the Software without restriction, including without limitation the
C rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
C sell copies of the Software, and to permit persons to whom the Software is
C furnished to do so, subject to the following conditions:
C
C The above copyright notice and this permission notice shall be included in
C all copies or substantial portions of the Software.
C
C THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
C IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
C FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
C AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
C LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
C FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
C IN THE SOFTWARE.
C
C 
#include <CCA/Components/Models/Radiation/fortran/m_cellg_fort.h>
#include "m_param4.h"

c*********************************************************************
c
c
c*********************************************************************
C-----------------------------------------------------------------------
C     THREE-DIMENSIONAL VERSION
C     CALCULATION OF CELL GEOMETRY PARAMETERS
C-----------------------------------------------------------------------
c*********************************************************************
c     Local Variables :
c*********************************************************************
      integer iBegGhost, jBegGhost, kBegGhost
      integer iEndGhost, jEndGhost, kEndGhost
      integer IST, JST, KST, IEND, JEND, KEND
      integer ISTU, JSTV, KSTW, IENDU, JENDV, KENDW
      integer i, j, k, ii, jj, kk
      double precision ESD, PESD, WSD, WWSD
      double precision SSD, PNSD, RNSD, SSSD
      double precision TSD, BSD, PTSD, BBSD
      double precision local_small

c*********************************************************************
c     Start :
c*********************************************************************
      iBegGhost = domainLow(1)
      jBegGhost = domainLow(2)
      kBegGhost = domainLow(3)
      iEndGhost = domainHigh(1)
      jEndGhost = domainHigh(2)
      kEndGhost = domainHigh(3)
      IST = indexLow(1)
      JST = indexLow(2)
      KST = indexLow(3)
      IEND = indexHigh(1)
      JEND = indexHigh(2)
      KEND = indexHigh(3)
      ISTU = indexLowU(1)
      JSTV = indexLowV(2)
      KSTW = indexLowW(3)
      IENDU = indexHighU(1)
      JENDV = indexHighV(2)
      KENDW = indexHighW(3)
c WARNING: set the following to remove the roundoff error. Needs to be changed
c if abs values of variables can be that small
      local_small=1.0D-12

C-----------------------------------------------------------------------
c     INIT to ZERO
C-----------------------------------------------------------------------
      do 10 ii = iBegGhost, iEndGhost
         dxep(ii) = 0.0D0
         dxpw(ii) = 0.0D0
         sew(ii) = 0.0D0
         xu(ii) = 0.0D0
         dxpwu(ii) = 0.0D0
         dxepu(ii) = 0.0D0
         sewu(ii) = 0.0D0
         cee(ii) = 0.0D0
         cww(ii) = 0.0D0
         cwe(ii) = 0.0D0
         ceeu(ii) = 0.0D0
         cwwu(ii) = 0.0D0
         cweu(ii) = 0.0D0
         efac(ii) = 0.0D0
         wfac(ii) = 0.0D0
         fac1u(ii) = 0.0D0
         fac2u(ii) = 0.0D0
         iesdu(ii) = 0
         fac3u(ii) = 0.0D0
         fac4u(ii) = 0.0D0
         iwsdu(ii) = 0
 10   continue
      do 20 jj = jBegGhost, jEndGhost
         dynp(jj) = 0.0D0
         dyps(jj) = 0.0D0
         sns(jj) = 0.0D0
         yv(jj) = 0.0D0
         dynpv(jj) = 0.0D0
         dypsv(jj) = 0.0D0
         snsv(jj) = 0.0D0
         cnn(jj) = 0.0D0
         css(jj) = 0.0D0
         csn(jj) = 0.0D0
         cnnv(jj) = 0.0D0
         cssv(jj) = 0.0D0
         csnv(jj) = 0.0D0
         nfac(jj) = 0.0D0
         sfac(jj) = 0.0D0
         fac1v(jj) = 0.0D0
         fac2v(jj) = 0.0D0
         jnsdv(jj) = 0
         fac3v(jj) = 0.0D0
         fac4v(jj) = 0.0D0
         jssdv(jj) = 0
 20   continue
      do 30 kk = kBegGhost, kEndGhost
         dztp(kk) = 0.0D0
         dzpb(kk) = 0.0D0
         stb(kk) = 0.0D0
         zw(kk) = 0.0D0
         dztpw(kk) = 0.0D0
         dzpbw(kk) = 0.0D0
         stbw(kk) = 0.0D0
         ctt(kk) = 0.0D0
         cbb(kk) = 0.0D0
         cbt(kk) = 0.0D0
         cttw(kk) = 0.0D0
         cbbw(kk) = 0.0D0
         cbtw(kk) = 0.0D0
         tfac(kk) = 0.0D0
         bfac(kk) = 0.0D0
         fac1w(kk) = 0.0D0
         fac2w(kk) = 0.0D0
         ktsdw(kk) = 0
         fac3w(kk) = 0.0D0
         fac4w(kk) = 0.0D0
         kbsdw(kk) = 0
 30   continue

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

      DO 150 I = iBegGhost,iEndGhost-1
        DXEP(I) = XX(I+1)-XX(I)
        DXPW(I+1) = DXEP(I)
  150 CONTINUE
      DXPW(iBegGhost) = DXPW(IST)
      DXEP(iEndGhost) = DXEP(IEND)
      DO 300 I = iBegGhost,iEndGhost
        SEW(I) = PT5*(DXEP(I)+DXPW(I))
  300 CONTINUE

*     u-cell distances and co-ordinates

      DO 450 I = iBegGhost+1,iEndGhost
        XU(I) = PT5*(XX(I)+XX(I-1))
  450 CONTINUE
      XU(iBegGhost) = (THREE*XX(iBegGhost)-XX(iBegGhost+1))/TWO
      DO 500 I = iBegGhost,iEndGhost-1
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
         if (abs(cwe(i)).lt.local_small) cwe(i)=0.0D0
         if (abs(cweu(i)).lt.local_small) cweu(i)=0.0D0
 780  CONTINUE
      DO 781 I = iBegGhost,iEndGhost
         EFAC(I) = (XX(I)-XU(I))/SEW(I)
 781  CONTINUE
      DO 782 I = iBegGhost,IEND
         WFAC(I) = (XU(I+1)-XX(I))/SEW(I)
 782  CONTINUE
      WFAC(iEndGhost) = WFAC(IEND)

*     factors for differencing schemes: u-cell

      DO 800 I = ISTU,IENDU
        ESD = PT5*(XU(I+1)+XU(I))
        WSD = PT5*(XU(I)+XU(I-1))
        PESD = XX(I)-ESD
        WWSD = XX(I-1)-WSD
        if (abs(pesd).lt.local_small) pesd=0.0D0
        if (abs(wwsd).lt.local_small) wwsd=0.0D0
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
        if (abs(fac1u(i)).lt.local_small) fac1u(i)=0.0D0
        if (abs(fac2u(i)).lt.local_small) fac2u(i)=0.0D0
        if (abs(fac3u(i)).lt.local_small) fac3u(i)=0.0D0
        if (abs(fac4u(i)).lt.local_small) fac4u(i)=0.0D0
  800 CONTINUE
c      iwsdu(ist) = 0
c      iesdu(ist) = 0
C-----------------------------------------------------------------------
C     Y-DIRECTION
C-----------------------------------------------------------------------

*     p-cell distances and co-ordinates

      DO 200 J = jBegGhost,jEndGhost-1
        DYNP(J) = YY(J+1)-YY(J)
        DYPS(J+1) = DYNP(J)
  200 CONTINUE
      DYPS(jBegGhost) = DYPS(JST)
      DYNP(jEndGhost) = DYNP(JEND)
      DO 350 J = jBegGhost,jEndGhost
        SNS(J) = PT5*(DYNP(J)+DYPS(J))
  350 CONTINUE

*     v-cell distances and co-ordinates

      DO 601 J = jBegGhost+1,jEndGhost
         YV(J) = PT5*(YY(J)+YY(J-1))
 601  CONTINUE
      YV(jBegGhost) = (THREE*YY(jBegGhost)-YY(jBegGhost+1))/TWO
      DO 650 J = jBegGhost,jEndGhost-1
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
         if (abs(csn(j)).lt.local_small) csn(j)=0.0D0
         if (abs(csnv(j)).lt.local_small) csnv(j)=0.0D0
  785 CONTINUE
      DO 786 J = jBegGhost,jEndGhost
         NFAC(J) = (YY(J)-YV(J))/SNS(J)
 786  CONTINUE
      DO 787 J = jBegGhost,JEND
         SFAC(J) = (YV(J+1)-YY(J))/SNS(J)
 787  CONTINUE
      SFAC(jEndGhost) = SFAC(JEND)

*     factors for differencing schemes: v-cell

      DO 850 J = JSTV,JENDV
        RNSD = PT5*(YV(J+1)+YV(J))
        SSD = PT5*(YV(J)+YV(J-1))
        PNSD = YY(J)-RNSD
        SSSD = YY(J-1)-SSD
        if (abs(pnsd).lt.local_small) pnsd=0.0D0
        if (abs(sssd).lt.local_small) sssd=0.0D0
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
        if (abs(fac1v(j)).lt.local_small) fac1v(j)=0.0D0
        if (abs(fac2v(j)).lt.local_small) fac2v(j)=0.0D0
        if (abs(fac3v(j)).lt.local_small) fac3v(j)=0.0D0
        if (abs(fac4v(j)).lt.local_small) fac4v(j)=0.0D0
  850 CONTINUE
c        jssdv(jst) = 0
c        jnsdv(jst) = 0

C-----------------------------------------------------------------------
C     Z-DIRECTION
C-----------------------------------------------------------------------

*     p-cell distances and co-ordinates

      DO 250 K = kBegGhost,kEndGhost-1
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

      DO 725 K = kBegGhost+1,kEndGhost
         ZW(K) = PT5*(ZZ(K)+ZZ(K-1))
  725 CONTINUE
      ZW(kBegGhost) = (THREE*ZZ(kBegGhost)-ZZ(kBegGhost+1))/TWO
      DO 750 K = kBegGhost,kEndGhost-1
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
         if (abs(cbt(k)).lt.local_small) cbt(k)=0.0D0
         if (abs(cbtw(k)).lt.local_small) cbtw(k)=0.0D0
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
      DO 900 K = KSTW,KENDW
        TSD = PT5*(ZW(K+1)+ZW(K))
        BSD = PT5*(ZW(K)+ZW(K-1))
        PTSD = ZZ(K)-TSD
        BBSD = ZZ(K-1)-BSD
        if (abs(ptsd).lt.local_small) ptsd=0.0D0
        if (abs(bbsd).lt.local_small) bbsd=0.0D0
        IF (PTSD.LE.ZERO) THEN
          FAC1W(K) = (ZZ(K+1)-TSD)/DZPB(K+1)
          FAC2W(K) = (TSD-ZZ(K))/DZPB(K+1)
          KTSDW(K) = 1
        ELSE
          FAC1W(K) = (TSD-ZZ(K-1))/DZPB(K)
          FAC2W(K) = (ZZ(K)-TSD)/DZPB(K)
          KTSDW(K) = -1
        END IF
        IF (K.NE.KSTW) THEN
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
        if (abs(fac1w(k)).lt.local_small) fac1w(k)=0.0D0
        if (abs(fac2w(k)).lt.local_small) fac2w(k)=0.0D0
        if (abs(fac3w(k)).lt.local_small) fac3w(k)=0.0D0
        if (abs(fac4w(k)).lt.local_small) fac4w(k)=0.0D0
c        KBSDW(KST) = 0
c        ktsdw(kst) = 0
  900 CONTINUE
      FAC3W(KST) = FAC1W(KEND)
      FAC4W(KST) = FAC2W(KEND)
      RETURN
      END

c*********************************************************************
c
c Revision 1.8  2000/10/04 16:46:29  rawat
c Parallel solver for pressure is working
c
c Revision 1.7  2000/10/02 16:40:30  rawat
c updated cellinformation for multi-patch
c
c Revision 1.6  2000/08/04 03:02:02  bbanerje
c Add some inits.
c
c Revision 1.5  2000/07/28 02:31:04  rawat
c moved all the labels in ArchesLabel. fixed some bugs and added matrix_dw to store matrix
c coeffecients
c
c Revision 1.4  2000/07/01 05:21:01  bbanerje
c Changed CellInformation calcs for Turbulence model requirements ..
c CellInformation still needs work.
c
c Revision 1.3  2000/06/19 18:00:38  rawat
c added function to compute velocity and density profiles and inlet bc.
c Fixed bugs in CellInformation.cc
c
c Revision 1.2  2000/06/14 20:40:53  rawat
c modified boundarycondition for physical boundaries and
c added CellInformation class
c
c Revision 1.1  2000/04/12 20:31:44  rawat
c modified PressureSolver and added fortran subroutines
c
c
c*********************************************************************
