c*********************************************************************
c
c $Id$
c
c*********************************************************************

      SUBROUTINE uvelcoef(domLoU, domHiU, idxLoU, idxHiU, UU,
     &     cesav, cwsav, cnsav, cssav, ctsav, cbsav,
     &     ap, ae, aw, an, as, at, ab,
     &     domLoV, domHiV, VV,
     &     domLoW, domHiW, WW,
     &     domLo, domHi, DEN, VIS, deltaT,
     &     ceeu, cweu, cwwu, cnn, csn, css,
     &     ctt, cbt, cbb, sewu, sns, stb,
     &     dxepu, dynp, dztp, dxpw, fac1u,
     &     fac2u, fac3u, fac4u, iesdu, iwsdu,
     &     enfac, sfac, tfac, bfac,
     &     ierr)
C---------------------------------------------------------------------
C     THREE-DIMENSIONAL VERSION
C     THIS SUBROUTINE WILL compute matrix coeefecient for uvelocity
C---------------------------------------------------------------------

      implicit none

#include "param4.h"

c*********************************************************************
c     Arguments :
c*********************************************************************
      integer domLoU(3), domHiU(3), idxLoU(3), idxHiU(3)
      double precision UU(domLoU(1):domHiU(1), domLoU(2):domHiU(2), 
     &     domLoU(3):domHiU(3))
      double precision AP(domLoU(1):domHiU(1), domLoU(2):domHiU(2), 
     &     domLoU(3):domHiU(3))
      double precision AE(domLoU(1):domHiU(1), domLoU(2):domHiU(2), 
     &     domLoU(3):domHiU(3))
      double precision AW(domLoU(1):domHiU(1), domLoU(2):domHiU(2), 
     &     domLoU(3):domHiU(3))
      double precision AN(domLoU(1):domHiU(1), domLoU(2):domHiU(2), 
     &     domLoU(3):domHiU(3))
      double precision AS(domLoU(1):domHiU(1), domLoU(2):domHiU(2), 
     &     domLoU(3):domHiU(3))
      double precision AT(domLoU(1):domHiU(1), domLoU(2):domHiU(2), 
     &     domLoU(3):domHiU(3))
      double precision AB(domLoU(1):domHiU(1), domLoU(2):domHiU(2), 
     &     domLoU(3):domHiU(3))
      double precision CESAV(domLoU(1):domHiU(1), domLoU(2):domHiU(2), 
     &     domLoU(3):domHiU(3))
      double precision CWSAV(domLoU(1):domHiU(1), domLoU(2):domHiU(2), 
     &     domLoU(3):domHiU(3))
      double precision CNSAV(domLoU(1):domHiU(1), domLoU(2):domHiU(2), 
     &     domLoU(3):domHiU(3))
      double precision CSSAV(domLoU(1):domHiU(1), domLoU(2):domHiU(2), 
     &     domLoU(3):domHiU(3))
      double precision CTSAV(domLoU(1):domHiU(1), domLoU(2):domHiU(2), 
     &     domLoU(3):domHiU(3))
      double precision CBSAV(domLoU(1):domHiU(1), domLoU(2):domHiU(2), 
     &     domLoU(3):domHiU(3))
      integer domLoV(3), domHiV(3)
      double precision VV(domLoV(1):domHiV(1), domLoV(2):domHiV(2), 
     &     domLoV(3):domHiV(3))
      integer domLoW(3), domHiW(3)
      double precision WW(domLoW(1):domHiW(1), domLoW(2):domHiW(2), 
     &     domLoW(3):domHiW(3))
      integer domLo(3), domHi(3)
      integer DEN(domLo(1):domHi(1), domLo(2):domHi(2), 
     &     domLo(3):domHi(3))
      integer VIS(domLo(1):domHi(1), domLo(2):domHi(2), 
     &     domLo(3):domHi(3))
      double precision ceeu(domLo(1):domHi(1))
      double precision cweu(domLo(1):domHi(1))
      double precision cwwu(domLo(1):domHi(1))
      double precision cnn(domLo(2):domHi(2))
      double precision csn(domLo(2):domHi(2))
      double precision css(domLo(2):domHi(2))
      double precision ctt(domLo(3):domHi(3))
      double precision cbt(domLo(3):domHi(3))
      double precision cbb(domLo(3):domHi(3))
      double precision sewu(domLo(1):domHi(1))
      double precision sns(domLo(2):domHi(2))
      double precision stb(domLo(3):domHi(3))
      double precision dxepu(domLo(1):domHi(1))
      double precision dynp(domLo(2):domHi(2))
      double precision dztp(domLo(3):domHi(3))
      double precision dxpw(domLo(1):domHi(1))
      double rpecision fac1u(domLo(1):domHi(1))
      double precision fac2u(domLo(1):domHi(1))
      double precision fac3u(domLo(1):domHi(1))
      double precision fac4u(domLo(1):domHi(1))
      double precision iesdu(domLo(1):domHi(1))
      double precision iwsdu(domLo(1):domHi(1))
      double precision enfac(domLo(2):domHi(2))
      double precision sfac(domLo(2):domHi(2))
      double precision tfac(domLo(3):domHi(3))
      double precision bfac(domLo(3):domHi(3))
      integer ierr

c*********************************************************************
c     Local Variables :
c*********************************************************************
      integer iBegGhost, jBegGhost, kBegGhost
      integer iEndGhost, jEndGhost, kEndGhost
      integer iBegGhostU, jBegGhostU, kBegGhostU
      integer iEndGhostU, jEndGhostU, kEndGhostU
      integer ISTU, JSTU, KSTU, IENDU, JENDU, KENDU
      integer iBegGhostV, jBegGhostV, kBegGhostV
      integer iEndGhostV, jEndGhostV, kEndGhostV
      integer ISTV, JSTV, KSTV, IENDV, JENDV, KENDV
      integer iBegGhostW, jBegGhostW, kBegGhostW
      integer iEndGhostW, jEndGhostW, kEndGhostW
      integer ISTW, JSTW, KSTW, IENDW, JENDW, KENDW
      integer i, j, k
      double precision FLOW, RHO, AVGVELP, AVGVELS
      double precision YR, ZR, YVR, ZWR

c*********************************************************************
c     Start :
c*********************************************************************
      iBegGhost = domLo(1)
      jBegGhost = domLo(2)
      kBegGhost = domLo(3)
      iEndGhost = domHi(1)
      jEndGhost = domHi(2)
      kEndGhost = domHi(3)
      iBegGhostU = domLoU(1)
      jBegGhostU = domLoU(2)
      kBegGhostU = domLoU(3)
      iEndGhostU = domHiU(1)
      jEndGhostU = domHiU(2)
      kEndGhostU = domHiU(3)
      ISTU = idxLoU(1)
      JSTU = idxLoU(2)
      KSTU = idxLoU(3)
      IENDU = idxHiU(1)
      JENDU = idxHiU(2)
      KENDU = idxHiU(3)
      iBegGhostV = domLoV(1)
      jBegGhostV = domLoV(2)
      kBegGhostV = domLoV(3)
      iEndGhostV = domHiV(1)
      jEndGhostV = domHiV(2)
      kEndGhostV = domHiV(3)
      ISTV = idxLoV(1)
      JSTV = idxLoV(2)
      KSTV = idxLoV(3)
      IENDV = idxHiV(1)
      JENDV = idxHiV(2)
      KENDV = idxHiV(3)
      iBegGhostW = domLoW(1)
      jBegGhostW = domLoW(2)
      kBegGhostW = domLoW(3)
      iEndGhostW = domHiW(1)
      jEndGhostW = domHiW(2)
      kEndGhostW = domHiW(3)
      ISTW = idxLoW(1)
      JSTW = idxLoW(2)
      KSTW = idxLoW(3)
      IENDW = idxHiW(1)
      JENDW = idxHiW(2)
      KENDW = idxHiW(3)

      L2UP = .FALSE.
      TINY = 1.0E-20
      TCONS = 0.0
C      IF (LCONS) TCONS = 1.0
      TSUM = 0.0
      ISTART = 3
      DO 220 K = 2,NKM1
         DO 210 J = 2,NJM1
            DO 200 I = ISTART,NIM1   
C-----------------------------------------------------------------------
C     COMPUTE AREAS AND VOLUME
C-----------------------------------------------------------------------
               AREAN = RV(J+1)*SEWU(I)*STB(K)
               AREAS = RV(J)*SEWU(I)*STB(K)
               AREAEW = RA(J)*SNS(J)*STB(K)
               AREATB = SNS(J)*SEWU(I)
               VOL = RA(J)*SEWU(I)*SNS(J)*STB(K)
C-----------------------------------------------------------------------
C     CALCULATE CONVECTION COEFFICIENTS
C-----------------------------------------------------------------------
               GE = 0.5*(DEN(I+1,J,K)+DEN(I,J,K))*U(I+1,J,K)
               GP = 0.5*(DEN(I,J,K)+DEN(I-1,J,K))*U(I,J,K)
            GW = 0.5*(DEN(I-1,J,K)+DEN(I-2,J,K))*U(I-1,J,K)
            GN = 0.5*(DEN(I,J+1,K)+DEN(I,J,K))*V(I,J+1,K)
            GNW = 0.5*(DEN(I-1,J,K)+DEN(I-1,J+1,K))*V(I-1,J+1,K)
            GS = 0.5*(DEN(I,J-1,K)+DEN(I,J,K))*V(I,J,K)
            GSW = 0.5*(DEN(I-1,J,K)+DEN(I-1,J-1,K))*V(I-1,J,K)
            GT = 0.5*(DEN(I,J,K)+DEN(I,J,K+1))*W(I,J,K+1)
            GTW = 0.5*(DEN(I-1,J,K)+DEN(I-1,J,K+1))*W(I-1,J,K+1)
            GTP = 0.5*(DEN(I,J,K)+DEN(I,J,K-1))*W(I,J,K)
            GBW = 0.5*(DEN(I-1,J,K)+DEN(I-1,J,K-1))*W(I-1,J,K)
            CEO = 0.5*(GE+GP)*AREAEW
            CWO = 0.5*(GP+GW)*AREAEW
            CNO = (0.5*GN+0.5*GNW)*AREAN
            CSO = (0.5*GS+0.5*GSW)*AREAS
            CTO = (0.5*GT+0.5*GTW)*AREATB
            CBO = (0.5*GTP+0.5*GBW)*AREATB
C-----------------------------------------------------------------------
C      NEW DIFFERENCING STUFF
C-----------------------------------------------------------------------
            CE = CEEU(I)*CEO + CWEU(I)*CWO
            CW = CWWU(I)*CWO - CWEU(I)*CEO   
            CN = CNN(J)*CNO + CSN(J)*CSO
            CS = CSS(J)*CSO - CSN(J)*CNO
            CT = CTT(K)*CTO + CBT(K)*CBO
            CB = CBB(K)*CBO - CBT(K)*CTO
            CESAV(I,J,K) = CE
            CWSAV(I,J,K) = CW
            CNSAV(I,J,K) = CN
            CSSAV(I,J,K) = CS
            CTSAV(I,J,K) = CT
            CBSAV(I,J,K) = CB
C-----------------------------------------------------------------------
C      UPWIND DIFFERENCING STUFF
C-----------------------------------------------------------------------
            CPE = GP*VOL/SEW(I)
            CPW = GP*VOL/SEW(I-1)
            GVP = ENFAC(J)*(0.5*GN+0.5*GNW)+SFAC(J)*(0.5*GS+0.5*GSW)
            CPN = GVP*VOL/DYNP(J)
            CPS = GVP*VOL/DYPS(J)
            GWP = TFAC(K)*(0.5*GT+0.5*GTW)+BFAC(K)*(0.5*GTP+0.5*GBW)
            CPT = GWP*VOL/(RA(J)*DZTP(K))
            CPB = GWP*VOL/(RA(J)*DZPB(K))
C-----------------------------------------------------------------------
C     CALCULATE DIFFUSION COEFFICIENTS
C-----------------------------------------------------------------------
            VISE = FAC1U(I)*VIS(I,J,K)+FAC2U(I)*VIS(I+IESDU(I),J,K)
            VISW = FAC3U(I)*VIS(I-1,J,K)+FAC4U(I)*VIS(I+IWSDU(I),J,K)
            VISN = 0.5*(0.5*(VIS(I,J,K)+VIS(I,J+1,K))+
     &             0.5*(VIS(I-1,J,K)+VIS(I-1,J+1,K)))
            VISS = 0.5*(0.5*(VIS(I,J,K)+VIS(I,J-1,K))+
     &             0.5*(VIS(I-1,J,K)+VIS(I-1,J-1,K)))
            VIST = 0.5*(0.5*(VIS(I,J,K)+VIS(I,J,K+1))+
     &             0.5*(VIS(I-1,J,K)+VIS(I-1,J,K+1)))
            VISB = 0.5*(0.5*(VIS(I,J,K)+VIS(I,J,K-1))+
     &             0.5*(VIS(I-1,J,K)+VIS(I-1,J,K-1)))  
            DEO = VISE*AREAEW/DXEPU(I)
            DWO = VISW*AREAEW/DXPWU(I)
            DNO = VISN*AREAN/DYNP(J)
            DSO = VISS*AREAS/DYPS(J)
            DTO = VIST*AREATB/(RA(J)*DZTP(K))
            DBO = VISB*AREATB/(RA(J)*DZPB(K))
C-----------------------------------------------------------------------
C     NEW DIFFUSION TERMS
C-----------------------------------------------------------------------
            DE = CEEU(I)*DEO + CWEU(I)*DWO*DXPWU(I)/DXEPU(I)
            DW = CWWU(I)*DWO - CWEU(I)*DEO*DXEPU(I)/DXPWU(I)   
            DN = CNN(J)*DNO + CSN(J)*DSO*DYPS(J)/DYNP(J)
            DS = CSS(J)*DSO - CSN(J)*DNO*DYNP(J)/DYPS(J)
            DT = CTT(K)*DTO + CBT(K)*DBO*DZPB(K)/DZTP(K)
            DB = CBB(K)*DBO - CBT(K)*DTO*DZTP(K)/DZPB(K)
C-----------------------------------------------------------------------
C     CONVECTION DIFFERENCING SCHEME SELECTION
C-----------------------------------------------------------------------
            COEFE = DE - 0.5*ABS(CE)
            COEFW = DW - 0.5*ABS(CW)
            COEFN = DN - 0.5*ABS(CN)
            COEFS = DS - 0.5*ABS(CS)
            COEFT = DT - 0.5*ABS(CT)
            COEFB = DB - 0.5*ABS(CB)
            TEW = 1.0
            TNS = 1.0
            TTB = 1.0
            IF (COEFE.LT.0.0 .OR. COEFW.LT.0.0) TEW = 0.0
            IF (COEFN.LT.0.0 .OR. COEFS.LT.0.0) TNS = 0.0
            IF (COEFT.LT.0.0 .OR. COEFB.LT.0.0) TTB = 0.0
            AEC = -0.5*CE*TEW + MAX(0.0,-CPE)*(1.0-TEW)
            AWC = 0.5*CW*TEW + MAX(0.0,CPW)*(1.0-TEW)
            ANC = -0.5*CN*TNS + MAX(0.0,-CPN)*(1.0-TNS)
            ASC = 0.5*CS*TNS + MAX(0.0,CPS)*(1.0-TNS)
            ATC = -0.5*CT*TTB + MAX(0.0,-CPT)*(1.0-TTB)
            ABC = 0.5*CB*TTB + MAX(0.0,CPB)*(1.0-TTB)
            DU(I,J,K) = VOL/DXPW(I)
            TSUM = TSUM + TEW+TNS+TTB
C----------------------------------------------------------------------
C      PROVISIONS FOR VARIOUS UPWIND DIFFERENCING SCHEMES
C----------------------------------------------------------------------
            IF (LSMPT) THEN
              AE(I,J,K) = DE
              AW(I,J,K) = DW
              AN(I,J,K) = DN
              AS(I,J,K) = DS
              AT(I,J,K) = DT
              AB(I,J,K) = DB
              AC(I,J,K) = AEC+AWC+ANC+ASC+ATC+ABC
              SU(I,J,K) = AEC*U(I+1,J,K)+AWC*U(I-1,J,K) +
     &                    ANC*U(I,J+1,K)+ASC*U(I,J-1,K) +
     &                    ATC*U(I,J,K+1)+ABC*U(I,J,K-1)
            ELSE IF (LCEND) THEN
              AE(I,J,K) = DE-CE*0.5
              AW(I,J,K) = DW+CW*0.5
              AN(I,J,K) = DN-CN*0.5
              AS(I,J,K) = DS+CS*0.5
              AT(I,J,K) = DT-CT*0.5
              AB(I,J,K) = DB+CB*0.5
              AC(I,J,K) = 0.0
              SU(I,J,K) = 0.0
            ELSE IF (L2UP) THEN
              AE(I,J,K) = DE*(1.0-MAX(0.0,SIGN(1.0,COEFE)))
     &                    + MAX(0.0,COEFE) + MAX(0.0,-CE)
              AW(I,J,K) = DW*(1.0-MAX(0.0,SIGN(1.0,COEFW)))
     &                    + MAX(0.0,COEFW) + MAX(0.0,CW)
              AN(I,J,K) = DN*(1.0-MAX(0.0,SIGN(1.0,COEFN)))
     &                    + MAX(0.0,COEFN) + MAX(0.0,-CN)
              AS(I,J,K) = DS*(1.0-MAX(0.0,SIGN(1.0,COEFS)))
     &                    + MAX(0.0,COEFS) + MAX(0.0,CS)
              AT(I,J,K) = DT*(1.0-MAX(0.0,SIGN(1.0,COEFT)))
     &                    + MAX(0.0,COEFT) + MAX(0.0,-CT)
              AB(I,J,K) = DB*(1.0-MAX(0.0,SIGN(1.0,COEFB)))
     &                    + MAX(0.0,COEFB) + MAX(0.0,CB)
              AC(I,J,K) = 0.0
              SU(I,J,K) = 0.0
            ELSE
              AE(I,J,K) = AEC + DE
              AW(I,J,K) = AWC + DW
              AN(I,J,K) = ANC + DN
              AS(I,J,K) = ASC + DS
              AT(I,J,K) = ATC + DT
              AB(I,J,K) = ABC + DB
              AC(I,J,K) = 0.0
              SU(I,J,K) = 0.0
           END IF
 200    CONTINUE
 210  CONTINUE
 220  CONTINUE

      RETURN
      END

c*********************************************************************
c
c $Log$
c Revision 1.1  2000/07/07 23:07:48  rawat
c added inlet bc's
c
c*********************************************************************
