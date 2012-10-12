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
C 
c*********************************************************************
c
c
c*********************************************************************

#include <CCA/Components/Arches/fortran/uvelcoef_fort.h>
C---------------------------------------------------------------------
C     THREE-DIMENSIONAL VERSION
C     THIS SUBROUTINE WILL compute matrix coefficients for uvelocity
C---------------------------------------------------------------------

c*********************************************************************
c     Local Variables :
c*********************************************************************
      integer ISTU, JSTU, KSTU, IENDU, JENDU, KENDU
      integer i, j, k

      logical*1 L2UP

      double precision AREAN, AREAS, AREAEW, AREATB, VOL
      double precision GE, GP, GW, GN, GNW, GS, GSW, GT, GTW, GTP, GBW
      double precision CEO, CWO, CNO, CSO, CTO, CBO
      double precision CE, CW, CN, CS, CT, CB
      double precision CPE, CPW, GVP, CPN, CPS, CPT, CPB
      double precision CP, GEP, GWP
      double precision VISE, VISW, VISN, VISS, VIST, VISB
      double precision DEO, DWO, DNO, DSO, DTO, DBO
      double precision DE, DW, DN, DS, DT, DB
      double precision COEFE, COEFW, COEFN, COEFS, COEFT, COEFB
      double precision TEW, TNS, TTB
      double precision AEC, AWC, ANC, ASC, ATC, ABC

c*********************************************************************
c     Start :
c*********************************************************************
      ISTU = idxLoU(1)
      JSTU = idxLoU(2)
      KSTU = idxLoU(3)
      IENDU = idxHiU(1)
      JENDU = idxHiU(2)
      KENDU = idxHiU(3)

c*********************************************************************
c     Set all flags
c*********************************************************************

      L2UP = .false.
c*********************************************************************
c     Go thru all the cells
c*********************************************************************
      DO 220 K = KSTU,KENDU
         DO 210 J = JSTU,JENDU
            DO 200 I = ISTU,IENDU   
C-----------------------------------------------------------------------
C     COMPUTE AREAS AND VOLUME 
C-----------------------------------------------------------------------
               AREAN = SEWU(I)*STB(K)
               AREAS = SEWU(I)*STB(K)
               AREAEW = SNS(J)*STB(K)
               AREATB = SNS(J)*SEWU(I)
               VOL = SEWU(I)*SNS(J)*STB(K)
C-----------------------------------------------------------------------
C     CALCULATE CONVECTION COEFFICIENTS
C-----------------------------------------------------------------------
               GE = 0.5D0*(DEN(I+1,J,K)+DEN(I,J,K))*UU(I+1,J,K)
               GP = 0.5D0*(DEN(I,J,K)+DEN(I-1,J,K))*UU(I,J,K)
               GW = 0.5D0*(DEN(I-1,J,K)+DEN(I-2,J,K))*UU(I-1,J,K)
c               GE = DEN(I,J,K)*(UU(I+1,J,K)+UU(I,J,K))
c               GP = 0.0D0
c               GW = DEN(I-1,J,K)*(UU(I,J,K)+UU(I-1,J,K))
               GN = 0.5D0*(DEN(I,J+1,K)+DEN(I,J,K))*VV(I,J+1,K)
               GNW = 0.5D0*(DEN(I-1,J,K)+DEN(I-1,J+1,K))*VV(I-1,J+1,K)
               GS = 0.5D0*(DEN(I,J-1,K)+DEN(I,J,K))*VV(I,J,K)
               GSW = 0.5D0*(DEN(I-1,J,K)+DEN(I-1,J-1,K))*VV(I-1,J,K)
               GT = 0.5D0*(DEN(I,J,K)+DEN(I,J,K+1))*WW(I,J,K+1)
               GTW = 0.5D0*(DEN(I-1,J,K)+DEN(I-1,J,K+1))*WW(I-1,J,K+1)
               GTP = 0.5D0*(DEN(I,J,K)+DEN(I,J,K-1))*WW(I,J,K)
               GBW = 0.5D0*(DEN(I-1,J,K)+DEN(I-1,J,K-1))*WW(I-1,J,K)
               CEO = 0.5D0*(GE+GP)*AREAEW
               CWO = 0.5D0*(GP+GW)*AREAEW
               CNO = (0.5D0*GN+0.5D0*GNW)*AREAN
               CSO = (0.5D0*GS+0.5D0*GSW)*AREAS
               CTO = (0.5D0*GT+0.5D0*GTW)*AREATB
               CBO = (0.5D0*GTP+0.5D0*GBW)*AREATB
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
        GVP = NFAC(J)*(0.5D0*GN+0.5D0*GNW)+SFAC(J)*(0.5D0*GS+0.5D0*GSW)
               CPN = GVP*VOL/DYNP(J)
               CPS = GVP*VOL/DYPS(J)
        GWP = TFAC(K)*(0.5D0*GT+0.5D0*GTW)+BFAC(K)*(0.5D0*GTP+0.5D0*GBW)
               CPT = GWP*VOL/DZTP(K)
               CPB = GWP*VOL/DZPB(K)
C-----------------------------------------------------------------------
C     CALCULATE DIFFUSION COEFFICIENTS
C-----------------------------------------------------------------------
               VISE = FAC1U(I)*VIS(I,J,K)+FAC2U(I)*VIS(I+IESDU(I),J,K)
               VISW = FAC3U(I)*VIS(I-1,J,K)+FAC4U(I)*VIS(I+IWSDU(I),J,K)
c               VISN = 0.5D0*(0.5D0*(VIS(I,J,K)+VIS(I,J+1,K))+
c     &              0.5D0*(VIS(I-1,J,K)+VIS(I-1,J+1,K)))
c               VISS = 0.5D0*(0.5D0*(VIS(I,J,K)+VIS(I,J-1,K))+
c     &              0.5D0*(VIS(I-1,J,K)+VIS(I-1,J-1,K)))
c               VIST = 0.5D0*(0.5D0*(VIS(I,J,K)+VIS(I,J,K+1))+
c     &              0.5D0*(VIS(I-1,J,K)+VIS(I-1,J,K+1)))
c               VISB = 0.5D0*(0.5D0*(VIS(I,J,K)+VIS(I,J,K-1))+
c     &              0.5D0*(VIS(I-1,J,K)+VIS(I-1,J,K-1)))  
               VISN = FAC1NS(J)*0.5D0*(VIS(I,J+1,K)+VIS(I-1,J+1,K))+
     +                FAC2NS(J)*0.5D0*(VIS(I,J+N_shift(J),K)+
     +                                 VIS(I-1,J+N_shift(J),K))
               VISS = FAC3NS(J)*0.5D0*(VIS(I,J-1,K)+VIS(I-1,J-1,K))+
     +                FAC4NS(J)*0.5D0*(VIS(I,J+S_shift(J),K)+
     +                                 VIS(I-1,J+S_shift(J),K))
               VIST = FAC1TB(K)*0.5D0*(VIS(I,J,K+1)+VIS(I-1,J,K+1))+
     +                FAC2TB(K)*0.5D0*(VIS(I,J,K+T_shift(K))+
     +                                 VIS(I-1,J,K+T_shift(K)))
               VISB = FAC3TB(K)*0.5D0*(VIS(I,J,K-1)+VIS(I-1,J,K-1))+
     +                FAC4TB(K)*0.5D0*(VIS(I,J,K+B_shift(K))+
     +                                 VIS(I-1,J,K+B_shift(K)))
               DEO = VISE*AREAEW/DXEPU(I)
               DWO = VISW*AREAEW/DXPWU(I)
               DNO = VISN*AREAN/DYNP(J)
               DSO = VISS*AREAS/DYPS(J)
               DTO = VIST*AREATB/DZTP(K)
               DBO = VISB*AREATB/DZPB(K)
C-----------------------------------------------------------------------
C     NEW DIFFUSION TERMS
C-----------------------------------------------------------------------
c viscosity interpolation for weighted central differencing is now done
c directly to faces centered about P cell center, without intermediate
c interpolation to standard P cell faces (where velocities are stored)
c               DE = CEEU(I)*DEO + CWEU(I)*DWO*DXPWU(I)/DXEPU(I)
c               DW = CWWU(I)*DWO - CWEU(I)*DEO*DXEPU(I)/DXPWU(I)   
c               DN = CNN(J)*DNO + CSN(J)*DSO*DYPS(J)/DYNP(J)
c               DS = CSS(J)*DSO - CSN(J)*DNO*DYNP(J)/DYPS(J)
c               DT = CTT(K)*DTO + CBT(K)*DBO*DZPB(K)/DZTP(K)
c               DB = CBB(K)*DBO - CBT(K)*DTO*DZTP(K)/DZPB(K)
               DE = DEO
               DW = DWO
               DN = DNO
               DS = DSO
               DT = DTO
               DB = DBO
               if (de.lt.0.0D0) write(6,*) "DE SFCX is negative"
               if (dw.lt.0.0D0) write(6,*) "DW SFCX is negative"
               if (dn.lt.0.0D0) write(6,*) "DN SFCX is negative"
               if (ds.lt.0.0D0) write(6,*) "DS SFCX is negative"
               if (dt.lt.0.0D0) write(6,*) "DT SFCX is negative"
               if (db.lt.0.0D0) write(6,*) "DB SFCX is negative"
               if ((de.lt.0.0D0).or.(dw.lt.0.0D0).or.(dn.lt.0.0D0).or.
     .             (ds.lt.0.0D0).or.(dt.lt.0.0D0).or.(db.lt.0.0D0)) then
                 write(6,*) "Location indexes: ",i,j,k
                 write(6,*) "Possibly stretching is two high."
                 call exit(0)
               end if
C-----------------------------------------------------------------------
C     CONVECTION DIFFERENCING SCHEME SELECTION
C-----------------------------------------------------------------------
               COEFE = DE - 0.5D0*ABS(CE)
               COEFW = DW - 0.5D0*ABS(CW)
               COEFN = DN - 0.5D0*ABS(CN)
               COEFS = DS - 0.5D0*ABS(CS)
               COEFT = DT - 0.5D0*ABS(CT)
               COEFB = DB - 0.5D0*ABS(CB)
               TEW = 1.0D0
               TNS = 1.0D0
               TTB = 1.0D0
               IF (COEFE.LT.0.0D0 .OR. COEFW.LT.0.0D0) TEW = 0.0D0
               IF (COEFN.LT.0.0D0 .OR. COEFS.LT.0.0D0) TNS = 0.0D0
               IF (COEFT.LT.0.0D0 .OR. COEFB.LT.0.0D0) TTB = 0.0D0

               AEC = -0.5D0*CE*TEW + dMAX1(0.0D0,-CPE)*(1.0D0-TEW)
               AWC = 0.5D0*CW*TEW + dMAX1(0.0D0,CPW)*(1.0D0-TEW)
               ANC = -0.5D0*CN*TNS + dMAX1(0.0D0,-CPN)*(1.0D0-TNS)
               ASC = 0.5D0*CS*TNS + dMAX1(0.0D0,CPS)*(1.0D0-TNS)
               ATC = -0.5D0*CT*TTB + dMAX1(0.0D0,-CPT)*(1.0D0-TTB)
               ABC = 0.5D0*CB*TTB + dMAX1(0.0D0,CPB)*(1.0D0-TTB)
C----------------------------------------------------------------------
C      PROVISIONS FOR VARIOUS UPWIND DIFFERENCING SCHEMES
C----------------------------------------------------------------------
               IF (LCEND) THEN
                  AE(I,J,K) = DE-CE*0.5D0
                  AW(I,J,K) = DW+CW*0.5D0
                  AN(I,J,K) = DN-CN*0.5D0
                  AS(I,J,K) = DS+CS*0.5D0
                  AT(I,J,K) = DT-CT*0.5D0
                  AB(I,J,K) = DB+CB*0.5D0
               ELSE IF (L2UP) THEN
                  AE(I,J,K) = DE*(1.0D0-dMAX1(0.0D0,dSIGN(1.0D0,COEFE)))
     &                 + dMAX1(0.0D0,COEFE) + dMAX1(0.0D0,-CE)
                  AW(I,J,K) = DW*(1.0D0-dMAX1(0.0D0,dSIGN(1.0D0,COEFW)))
     &                 + dMAX1(0.0D0,COEFW) + dMAX1(0.0D0,CW)
                  AN(I,J,K) = DN*(1.0D0-dMAX1(0.0D0,dSIGN(1.0D0,COEFN)))
     &                 + dMAX1(0.0D0,COEFN) + dMAX1(0.0D0,-CN)
                  AS(I,J,K) = DS*(1.0D0-dMAX1(0.0D0,dSIGN(1.0D0,COEFS)))
     &                 + dMAX1(0.0D0,COEFS) + dMAX1(0.0D0,CS)
                  AT(I,J,K) = DT*(1.0D0-dMAX1(0.0D0,dSIGN(1.0D0,COEFT)))
     &                 + dMAX1(0.0D0,COEFT) + dMAX1(0.0D0,-CT)
                  AB(I,J,K) = DB*(1.0D0-dMAX1(0.0D0,dSIGN(1.0D0,COEFB)))
     &                 + dMAX1(0.0D0,COEFB) + dMAX1(0.0D0,CB)
               ELSE
                  AE(I,J,K) = AEC + DE
                  AW(I,J,K) = AWC + DW
                  AN(I,J,K) = ANC + DN
                  AS(I,J,K) = ASC + DS
                  AT(I,J,K) = ATC + DT
                  AB(I,J,K) = ABC + DB
               END IF
 200        CONTINUE
 210     CONTINUE
 220  CONTINUE

      RETURN
      END

