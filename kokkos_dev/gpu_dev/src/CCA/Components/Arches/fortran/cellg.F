C 
C 
C The MIT License
C 
C Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
C Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
C University of Utah.
C 
C License for the specific language governing rights and limitations under
C Permission is hereby granted, free of charge, to any person obtaining a 
C copy of this software and associated documentation files (the "Software"),
C to deal in the Software without restriction, including without limitation 
C the rights to use, copy, modify, merge, publish, distribute, sublicense, 
C and/or sell copies of the Software, and to permit persons to whom the 
C Software is furnished to do so, subject to the following conditions:
C 
C The above copyright notice and this permission notice shall be included 
C in all copies or substantial portions of the Software.
C 
C THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
C OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
C FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
C THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
C LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
C FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
C DEALINGS IN THE SOFTWARE.
C 
C
C 
C 
#include <CCA/Components/Arches/fortran/cellg_fort.h>
#include "param4.h"

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
      integer ISTU, JSTV, KSTW, IENDU, JENDV, KENDW
      integer i, j, k
      double precision ESD, PESD, WSD, WWSD
      double precision SSD, PNSD, RNSD, SSSD
      double precision TSD, BSD, PTSD, BBSD
      double precision local_small
      double precision E_face, W_face, E_face_to_E, W_face_to_W
      double precision N_face, S_face, N_face_to_N, S_face_to_S
      double precision T_face, B_face, T_face_to_T, B_face_to_B
      double precision XUxplus, YVyplus, ZWzplus

c*********************************************************************
c     Start :
c*********************************************************************
      iBegGhost = domainLow(1)
      jBegGhost = domainLow(2)
      kBegGhost = domainLow(3)
      iEndGhost = domainHigh(1)
      jEndGhost = domainHigh(2)
      kEndGhost = domainHigh(3)
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
      do 10 i = iBegGhost, iEndGhost
         dxep(i) = 0.0D0
         dxpw(i) = 0.0D0
         sew(i) = 0.0D0
         xu(i) = 0.0D0
         dxpwu(i) = 0.0D0
         dxepu(i) = 0.0D0
         sewu(i) = 0.0D0
         cee(i) = 0.0D0
         cww(i) = 0.0D0
         cwe(i) = 0.0D0
         ceeu(i) = 0.0D0
         cwwu(i) = 0.0D0
         cweu(i) = 0.0D0
         efac(i) = 0.0D0
         wfac(i) = 0.0D0
         fac1u(i) = 0.0D0
         fac2u(i) = 0.0D0
         iesdu(i) = 0
         fac3u(i) = 0.0D0
         fac4u(i) = 0.0D0
         iwsdu(i) = 0
 10   continue
      do 20 j = jBegGhost, jEndGhost
         dynp(j) = 0.0D0
         dyps(j) = 0.0D0
         sns(j) = 0.0D0
         yv(j) = 0.0D0
         dynpv(j) = 0.0D0
         dypsv(j) = 0.0D0
         snsv(j) = 0.0D0
         cnn(j) = 0.0D0
         css(j) = 0.0D0
         csn(j) = 0.0D0
         cnnv(j) = 0.0D0
         cssv(j) = 0.0D0
         csnv(j) = 0.0D0
         nfac(j) = 0.0D0
         sfac(j) = 0.0D0
         fac1v(j) = 0.0D0
         fac2v(j) = 0.0D0
         jnsdv(j) = 0
         fac3v(j) = 0.0D0
         fac4v(j) = 0.0D0
         jssdv(j) = 0
 20   continue
      do 30 k = kBegGhost, kEndGhost
         dztp(k) = 0.0D0
         dzpb(k) = 0.0D0
         stb(k) = 0.0D0
         zw(k) = 0.0D0
         dztpw(k) = 0.0D0
         dzpbw(k) = 0.0D0
         stbw(k) = 0.0D0
         ctt(k) = 0.0D0
         cbb(k) = 0.0D0
         cbt(k) = 0.0D0
         cttw(k) = 0.0D0
         cbbw(k) = 0.0D0
         cbtw(k) = 0.0D0
         tfac(k) = 0.0D0
         bfac(k) = 0.0D0
         fac1w(k) = 0.0D0
         fac2w(k) = 0.0D0
         ktsdw(k) = 0
         fac3w(k) = 0.0D0
         fac4w(k) = 0.0D0
         kbsdw(k) = 0
 30   continue


C-----------------------------------------------------------------------
C     X-DIRECTION
C-----------------------------------------------------------------------

*     p-cell distances and co-ordinates

      DO 150 I = iBegGhost,iEndGhost-1
        DXEP(I) = XX(I+1)-XX(I)
        DXPW(I+1) = DXEP(I)
  150 CONTINUE
      if (xminus) then
        DXPW(iBegGhost) = DXPW(iBegGhost+1)
      else
        DXPW(iBegGhost) = XX(iBegGhost)-XX(iBegGhost-1)
      end if
      if (xplus) then
        DXEP(iEndGhost) = DXEP(iEndGhost-1)
      else
        DXEP(iEndGhost) = XX(iEndGhost+1)-XX(iEndGhost)
      end if

      DO 300 I = iBegGhost,iEndGhost
        SEW(I) = PT5*(DXEP(I)+DXPW(I))
  300 CONTINUE

*     u-cell distances and co-ordinates

      DO 450 I = iBegGhost+1,iEndGhost
        XU(I) = PT5*(XX(I)+XX(I-1))
  450 CONTINUE
      if (xminus) then
        XU(iBegGhost) = (THREE*XX(iBegGhost)-XX(iBegGhost+1))/TWO
      else
        XU(iBegGhost) = PT5*(XX(iBegGhost)+XX(iBegGhost-1))
      end if
      if (xplus) then
        XUxplus = (THREE*XX(iEndGhost)-XX(iEndGhost-1))/TWO
      end if

      DO 500 I = iBegGhost,iEndGhost-1
        DXEPU(I) = XU(I+1)-XU(I)
        DXPWU(I+1) = DXEPU(I)
  500 CONTINUE
      if (xminus) then
        DXPWU(iBegGhost) = DXEPU(iBegGhost)
      else
        DXPWU(iBegGhost) = XU(iBegGhost)-
     -                     PT5*(XX(iBegGhost-1)+XX(iBegGhost-2))
      end if
      if (xplus) then
c because indexing used here is for CC variables
c this fix is needed to compute sewu correctly on xplus boundary face
        DXEPU(iEndGhost) = XUxplus-XU(iEndGhost)
      else
        DXEPU(iEndGhost) = PT5*(XX(iEndGhost+1)+XX(iEndGhost))-
     -                     XU(iEndGhost)
      end if

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
      DO 782 I = iBegGhost,iEndGhost-1
         WFAC(I) = (XU(I+1)-XX(I))/SEW(I)
 782  CONTINUE
      if (xplus) then
        WFAC(iEndGhost) = WFAC(iEndGhost-1)
      else
        WFAC(iEndGhost) = (PT5*(XX(iEndGhost+1)+XX(iEndGhost))-
     -                     XX(iEndGhost))/SEW(iEndGhost)
      end if

*     factors for CC variable interpolation: p-cell

      DO I = indexLow(1),indexHigh(1)
        E_face = XX(I) + PT5*SEW(I)
        W_face = XX(I) - PT5*SEW(I)
        E_face_to_E = XX(I+1)-E_face
        W_face_to_W = W_face-XX(I-1)
        if (abs(E_face_to_E).lt.local_small) E_face_to_E=0.0D0
        if (abs(W_face_to_W).lt.local_small) W_face_to_W=0.0D0
        IF ((E_face_to_E.LE.ZERO).and.
     .      (i.eq.indexHigh(1)).and.(xplus)) then
          write(6,*) "grid stretching is too high near xplus boundary"
          call exit(0)
        ELSE IF (E_face_to_E.LE.ZERO) THEN
          FAC1EW(I) = (XX(I+2)-E_face)/DXEP(I+1)
          FAC2EW(I) = (E_face-XX(I+1))/DXEP(I+1)
          E_shift(I) = 2
        ELSE
          FAC1EW(I) = (E_face-XX(I))/DXEP(I)
          FAC2EW(I) = (XX(I+1)-E_face)/DXEP(I)
          E_shift(I) = 0
        END IF
        IF ((W_face_to_W.LE.ZERO).and.
     .      (i.eq.indexLow(1)).and.(xminus)) then
          write(6,*) "grid stretching is too high near xminus boundary"
          call exit(0)
        ELSE IF (W_face_to_W.LE.ZERO) THEN
          FAC3EW(I) = (W_face-XX(I-2))/DXPW(I-1)
          FAC4EW(I) = (XX(I-1)-W_face)/DXPW(I-1)
          W_shift(I) = -2
        ELSE
          FAC3EW(I) = (XX(I)-W_face)/DXPW(I)
          FAC4EW(I) = (W_face-XX(I-1))/DXPW(I)
          W_shift(I) = 0
        END IF
        if (abs(fac1ew(i)).lt.local_small) fac1ew(i)=0.0D0
        if (abs(fac2ew(i)).lt.local_small) fac2ew(i)=0.0D0
        if (abs(fac3ew(i)).lt.local_small) fac3ew(i)=0.0D0
        if (abs(fac4ew(i)).lt.local_small) fac4ew(i)=0.0D0
      END DO

*     factors for differencing schemes: u-cell

      DO 800 I = ISTU,IENDU
        ESD = XU(I) + PT5*SEWU(I)
        WSD = XU(I) - PT5*SEWU(I)
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
C-----------------------------------------------------------------------
C     Y-DIRECTION
C-----------------------------------------------------------------------

*     p-cell distances and co-ordinates

      DO 200 J = jBegGhost,jEndGhost-1
        DYNP(J) = YY(J+1)-YY(J)
        DYPS(J+1) = DYNP(J)
  200 CONTINUE
      if (yminus) then
        DYPS(jBegGhost) = DYPS(jBegGhost+1)
      else
        DYPS(jBegGhost) = YY(jBegGhost)-YY(jBegGhost-1)
      end if
      if (yplus) then
        DYNP(jEndGhost) = DYNP(jEndGhost-1)
      else
        DYNP(jEndGhost) = YY(jEndGhost+1)-YY(jEndGhost)
      end if

      DO 350 J = jBegGhost,jEndGhost
        SNS(J) = PT5*(DYNP(J)+DYPS(J))
  350 CONTINUE

*     v-cell distances and co-ordinates

      DO 601 J = jBegGhost+1,jEndGhost
         YV(J) = PT5*(YY(J)+YY(J-1))
 601  CONTINUE
      if (yminus) then
        YV(jBegGhost) = (THREE*YY(jBegGhost)-YY(jBegGhost+1))/TWO
      else
        YV(jBegGhost) = PT5*(YY(jBegGhost)+YY(jBegGhost-1))
      end if
      if (yplus) then
        YVyplus = (THREE*YY(jEndGhost)-YY(jEndGhost-1))/TWO
      end if

      DO 650 J = jBegGhost,jEndGhost-1
        DYNPV(J) = YV(J+1)-YV(J)
        DYPSV(J+1) = DYNPV(J)
  650 CONTINUE
      if (yminus) then
        DYPSV(jBegGhost) = DYNPV(jBegGhost)
      else
        DYPSV(jBegGhost) = YV(jBegGhost)-
     -                     PT5*(YY(jBegGhost-1)+YY(jBegGhost-2))
      end if
      if (yplus) then
c because indexing used here is for CC variables
c this fix is needed to compute snsv correctly on yplus boundary face
        DYNPV(jEndGhost) = YVyplus-YV(jEndGhost)
      else
        DYNPV(jEndGhost) = PT5*(YY(jEndGhost+1)+YY(jEndGhost))-
     -                     YV(jEndGhost)
      end if

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
      DO 787 J = jBegGhost,jEndGhost-1
         SFAC(J) = (YV(J+1)-YY(J))/SNS(J)
 787  CONTINUE
      if (yplus) then
        SFAC(jEndGhost) = SFAC(jEndGhost-1)
      else
        SFAC(jEndGhost) = (PT5*(YY(jEndGhost+1)+YY(jEndGhost))-
     -                     YY(jEndGhost))/SNS(jEndGhost)
      end if

*     factors for CC variable interpolation: p-cell

      DO J = indexLow(2),indexHigh(2)
        N_face = YY(J) + PT5*SNS(J)
        S_face = YY(J) - PT5*SNS(J)
        N_face_to_N = YY(J+1)-N_face
        S_face_to_S = S_face-YY(J-1)
        if (abs(N_face_to_N).lt.local_small) N_face_to_N=0.0D0
        if (abs(S_face_to_S).lt.local_small) S_face_to_S=0.0D0
        IF ((N_face_to_N.LE.ZERO).and.
     .      (j.eq.indexHigh(2)).and.(yplus)) then
          write(6,*) "grid stretching is too high near yplus boundary"
          call exit(0)
        ELSE IF (N_face_to_N.LE.ZERO) THEN
          FAC1NS(J) = (YY(J+2)-N_face)/DYNP(J+1)
          FAC2NS(J) = (N_face-YY(J+1))/DYNP(J+1)
          N_shift(J) = 2
        ELSE
          FAC1NS(J) = (N_face-YY(J))/DYNP(J)
          FAC2NS(J) = (YY(J+1)-N_face)/DYNP(J)
          N_shift(J) = 0
        END IF
        IF ((S_face_to_S.LE.ZERO).and.
     .      (j.eq.indexLow(2)).and.(yminus)) then
          write(6,*) "grid stretching is too high near yminus boundary"
          call exit(0)
        ELSE IF (S_face_to_S.LE.ZERO) THEN
          FAC3NS(J) = (S_face-YY(J-2))/DYPS(J-1)
          FAC4NS(J) = (YY(J-1)-S_face)/DYPS(J-1)
          S_shift(J) = -2
        ELSE
          FAC3NS(J) = (YY(J)-S_face)/DYPS(J)
          FAC4NS(J) = (S_face-YY(J-1))/DYPS(J)
          S_shift(J) = 0
        END IF
        if (abs(fac1ns(j)).lt.local_small) fac1ns(j)=0.0D0
        if (abs(fac2ns(j)).lt.local_small) fac2ns(j)=0.0D0
        if (abs(fac3ns(j)).lt.local_small) fac3ns(j)=0.0D0
        if (abs(fac4ns(j)).lt.local_small) fac4ns(j)=0.0D0
      END DO

*     factors for differencing schemes: v-cell

      DO 850 J = JSTV,JENDV
        RNSD = YV(J) + PT5*SNSV(J)
        SSD = YV(J) - PT5*SNSV(J)
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

C-----------------------------------------------------------------------
C     Z-DIRECTION
C-----------------------------------------------------------------------

*     p-cell distances and co-ordinates

      DO 250 K = kBegGhost,kEndGhost-1
        DZTP(K) = ZZ(K+1)-ZZ(K)
        DZPB(K+1) = DZTP(K)
  250 CONTINUE
      if (zminus) then
        DZPB(kBegGhost) = DZPB(kBegGhost+1) 
      else
        DZPB(kBegGhost) = ZZ(kBegGhost)-ZZ(kBegGhost-1)
      end if
      if (zplus) then
        DZTP(kEndGhost) = DZTP(kEndGhost-1) 
      else
        DZTP(kEndGhost) = ZZ(kEndGhost+1)-ZZ(kEndGhost)
      end if

      DO 400 K = kBegGhost,kEndGhost
        STB(K) = PT5*(DZTP(K)+DZPB(K))
  400 CONTINUE

*     w-cell distances and co-ordinates

      DO 725 K = kBegGhost+1,kEndGhost
         ZW(K) = PT5*(ZZ(K)+ZZ(K-1))
  725 CONTINUE
      if (zminus) then
        ZW(kBegGhost) = (THREE*ZZ(kBegGhost)-ZZ(kBegGhost+1))/TWO
      else
        ZW(kBegGhost) = PT5*(ZZ(kBegGhost)+ZZ(kBegGhost-1))
      end if
      if (zplus) then
        ZWzplus = (THREE*ZZ(kEndGhost)-ZZ(kEndGhost-1))/TWO
      end if

      DO 750 K = kBegGhost,kEndGhost-1
        DZTPW(K) = ZW(K+1)-ZW(K)
        DZPBW(K+1) = DZTPW(K)
  750 CONTINUE
      if (zminus) then
        DZPBW(kBegGhost) = DZTPW(kBegGhost)
      else
        DZPBW(kBegGhost) = ZW(kBegGhost)-
     -                     PT5*(ZZ(kBegGhost-1)+ZZ(kBegGhost-2))
      end if
      if (zplus) then
c because indexing used here is for CC variables
c this fix is needed to compute stbw correctly on zplus boundary face
        DZTPW(kEndGhost) = ZWzplus-ZW(kEndGhost)
      else
        DZTPW(kEndGhost) = PT5*(ZZ(kEndGhost+1)+ZZ(kEndGhost))-
     -                     ZW(kEndGhost)
      end if

      DO 775 K = kBegGhost,kEndGhost
        STBW(K) = PT5*(DZTPW(K)+DZPBW(K))
  775 CONTINUE

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
      DO 792 K = kBegGhost,kEndGhost-1
         BFAC(K) = (ZW(K+1)-ZZ(K))/STB(K)
 792  CONTINUE
      if (zplus) then
        BFAC(kEndGhost) = BFAC(kEndGhost-1)
      else
        BFAC(kEndGhost) = (PT5*(ZZ(kEndGhost+1)+ZZ(kEndGhost))-
     -                     ZZ(kEndGhost))/STB(kEndGhost)
      end if

*     factors for CC variable interpolation: p-cell

      DO K = indexLow(3),indexHigh(3)
        T_face = ZZ(K) + PT5*STB(K)
        B_face = ZZ(K) - PT5*STB(K)
        T_face_to_T = ZZ(K+1)-T_face
        B_face_to_B = B_face-ZZ(K-1)
        if (abs(T_face_to_T).lt.local_small) T_face_to_T=0.0D0
        if (abs(B_face_to_B).lt.local_small) B_face_to_B=0.0D0
        IF ((T_face_to_T.LE.ZERO).and.
     .      (k.eq.indexHigh(3)).and.(zplus)) then
          write(6,*) "grid stretching is too high near zplus boundary"
          call exit(0)
        ELSE IF (T_face_to_T.LE.ZERO) THEN
          FAC1TB(K) = (ZZ(K+2)-T_face)/DZTP(K+1)
          FAC2TB(K) = (T_face-ZZ(K+1))/DZTP(K+1)
          T_shift(K) = 2
        ELSE
          FAC1TB(K) = (T_face-ZZ(K))/DZTP(K)
          FAC2TB(K) = (ZZ(K+1)-T_face)/DZTP(K)
          T_shift(K) = 0
        END IF
        IF ((B_face_to_B.LE.ZERO).and.
     .      (k.eq.indexLow(3)).and.(zminus)) then
          write(6,*) "grid stretching is too high near zminus boundary"
          call exit(0)
        ELSE IF (B_face_to_B.LE.ZERO) THEN
          FAC3TB(K) = (B_face-ZZ(K-2))/DZPB(K-1)
          FAC4TB(K) = (ZZ(K-1)-B_face)/DZPB(K-1)
          B_shift(K) = -2
        ELSE
          FAC3TB(K) = (ZZ(K)-B_face)/DZPB(K)
          FAC4TB(K) = (B_face-ZZ(K-1))/DZPB(K)
          B_shift(K) = 0
        END IF
        if (abs(fac1tb(k)).lt.local_small) fac1tb(k)=0.0D0
        if (abs(fac2tb(k)).lt.local_small) fac2tb(k)=0.0D0
        if (abs(fac3tb(k)).lt.local_small) fac3tb(k)=0.0D0
        if (abs(fac4tb(k)).lt.local_small) fac4tb(k)=0.0D0
      END DO

*     factors for differencing schemes: w-cell

      DO 900 K = KSTW,KENDW
        TSD = ZW(K) + PT5*STBW(K)
        BSD = ZW(K) - PT5*STBW(K)
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
        IF (BBSD.LE.ZERO) THEN
          FAC3W(K) = (ZZ(K)-BSD)/DZPB(K)
          FAC4W(K) = (BSD-ZZ(K-1))/DZPB(K)
          KBSDW(K) = 0
        ELSE
          FAC3W(K) = (BSD-ZZ(K-2))/DZPB(K-1)
          FAC4W(K) = (ZZ(K-1)-BSD)/DZPB(K-1)
          KBSDW(K) = -2
        END IF
        if (abs(fac1w(k)).lt.local_small) fac1w(k)=0.0D0
        if (abs(fac2w(k)).lt.local_small) fac2w(k)=0.0D0
        if (abs(fac3w(k)).lt.local_small) fac3w(k)=0.0D0
        if (abs(fac4w(k)).lt.local_small) fac4w(k)=0.0D0
  900 CONTINUE

      RETURN
      END

