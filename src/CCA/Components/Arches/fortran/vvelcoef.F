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

#include <CCA/Components/Arches/fortran/vvelcoef_fort.h>
C---------------------------------------------------------------------
C     THREE-DIMENSIONAL VERSION
C     THIS SUBROUTINE WILL compute matrix coefficients for vvelocity
C---------------------------------------------------------------------

c*********************************************************************
c     Local Variables :
c*********************************************************************
      integer ISTV, JSTV, KSTV, IENDV, JENDV, KENDV
      integer i, j, k

      logical*1 L2UP

      double precision AREAN, AREAS, AREAEW, AREATB, VOL
      double precision GE, GP, GW, GN, GSE, GS, GSW, GT, GST, GSB, GB
      double precision GSP, GNP
      double precision CEO, CWO, CNO, CSO, CTO, CBO
      double precision CE, CW, CN, CS, CT, CB
      double precision GUP, CPE, CPW, CPN, CPS, GWP, CPT, CPB
      double precision VISE, VISW, VISN, VISS, VIST, VISB
      double precision DEO, DWO, DNO, DSO, DTO, DBO
      double precision DE, DW, DN, DS, DT, DB
      double precision COEFE, COEFW, COEFN, COEFS, COEFT, COEFB
      double precision TEW, TNS, TTB
      double precision AEC, AWC, ANC, ASC, ATC, ABC

c*********************************************************************
c     Start :
c*********************************************************************
      ISTV = idxLoV(1)
      JSTV = idxLoV(2)
      KSTV = idxLoV(3)
      IENDV = idxHiV(1)
      JENDV = idxHiV(2)
      KENDV = idxHiV(3)

c*********************************************************************
c     Set all flags
c*********************************************************************
      L2UP = .FALSE.
      

c*********************************************************************
c     Go thru all the cells
c*********************************************************************
      DO 220 K = KSTV,KENDV
         DO 210 J = JSTV,JENDV
            DO 200 I = ISTV,IENDV
C-----------------------------------------------------------------------
C     COMPUTE AREAS AND VOLUME
C-----------------------------------------------------------------------
               AREAN = STB(K)*SEW(I)
               AREAS = STB(K)*SEW(I)
               AREAEW = STB(K)*SNSV(J)
               AREATB = SNSV(J)*SEW(I)
               VOL = STB(K)*SEW(I)*SNSV(J)
C-----------------------------------------------------------------------
C     CALCULATE CONVECTION COEFFICIENTS
C-----------------------------------------------------------------------
               GE = 0.5D0*(DEN(I+1,J,K)+DEN(I,J,K))*UU(I+1,J,K)
               GSE = 0.5D0*(DEN(I,J-1,K)+DEN(I+1,J-1,K))*UU(I+1,J-1,K)
               GW = 0.5D0*(DEN(I,J,K)+DEN(I-1,J,K))*UU(I,J,K)
               GSW = 0.5D0*(DEN(I,J-1,K)+DEN(I-1,J-1,K))*UU(I,J-1,K)
               GN = 0.5D0*(DEN(I,J+1,K)+DEN(I,J,K))*VV(I,J+1,K)
               GP = 0.5D0*(DEN(I,J,K)+DEN(I,J-1,K))*VV(I,J,K)
               GS = 0.5D0*(DEN(I,J-1,K)+DEN(I,J-2,K))*VV(I,J-1,K)
c               GN = DEN(I,J,K)*(VV(I,J+1,K)+VV(I,J,K))
c               GP = 0.0D0
c               GS = DEN(I,J-1,K)*(VV(I,J,K)+VV(I,J-1,K))
               GT = 0.5D0*(DEN(I,J,K+1)+DEN(I,J,K))*WW(I,J,K+1)
               GST = 0.5D0*(DEN(I,J-1,K+1)+DEN(I,J-1,K))*WW(I,J-1,K+1)
               GB = 0.5D0*(DEN(I,J,K)+DEN(I,J,K-1))*WW(I,J,K)
               GSB = 0.5D0*(DEN(I,J-1,K)+DEN(I,J-1,K-1))*WW(I,J-1,K)
               CEO = (0.5D0*GE+0.5D0*GSE)*AREAEW
               CWO = (0.5D0*GW+0.5D0*GSW)*AREAEW
               CTO = (0.5D0*GT+0.5D0*GST)*AREATB
               CBO = (0.5D0*GB+0.5D0*GSB)*AREATB 
               CNO = 0.5D0*(GN+GP)*AREAN
               CSO = 0.5D0*(GP+GS)*AREAS
C-----------------------------------------------------------------------
C      NEW DIFFERENCING STUFF
C-----------------------------------------------------------------------
               CE = CEE(I)*CEO + CWE(I)*CWO
               CW = CWW(I)*CWO - CWE(I)*CEO   
               CN = CNNV(J)*CNO + CSNV(J)*CSO
               CS = CSSV(J)*CSO - CSNV(J)*CNO
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
         GUP = EFAC(I)*(0.5D0*GE+0.5D0*GSE)+WFAC(I)*(0.5D0*GW+0.5D0*GSW)
               CPE = GUP*VOL/DXEP(I)
               CPW = GUP*VOL/DXPW(I)
               CPN = GP*VOL/SNS(J)
               CPS = GP*VOL/SNS(J-1)
         GWP = TFAC(K)*(0.5D0*GT+0.5D0*GST)+BFAC(K)*(0.5D0*GB+0.5D0*GSB)
               CPT = GWP*VOL/DZTP(K)
               CPB = GWP*VOL/DZPB(K)
C-----------------------------------------------------------------------
C     CALCULATE DIFFUSION COEFFICIENTS
C-----------------------------------------------------------------------
c               VISE = 0.5D0*(0.5D0*(VIS(I,J,K)+VIS(I+1,J,K))+
c     &              0.5D0*(VIS(I,J-1,K)+VIS(I+1,J-1,K)))
c               VISW = 0.5D0*(0.5D0*(VIS(I,J,K)+VIS(I-1,J,K))+
c     &              0.5D0*(VIS(I,J-1,K)+VIS(I-1,J-1,K)))
               VISE = FAC1EW(I)*0.5D0*(VIS(I+1,J,K)+VIS(I+1,J-1,K))+
     +                FAC2EW(I)*0.5D0*(VIS(I+E_shift(I),J,K)+
     +                                 VIS(I+E_shift(I),J-1,K))
               VISW = FAC3EW(I)*0.5D0*(VIS(I-1,J,K)+VIS(I-1,J-1,K))+
     +                FAC4EW(I)*0.5D0*(VIS(I+W_shift(I),J,K)+
     +                                 VIS(I+W_shift(I),J-1,K))
               VISN = FAC1V(J)*VIS(I,J,K)+FAC2V(J)*VIS(I,J+JNSDV(J),K)
               VISS = FAC3V(J)*VIS(I,J-1,K)+FAC4V(J)*VIS(I,J+JSSDV(J),K)
c               VIST = 0.5D0*(0.5D0*(VIS(I,J,K)+VIS(I,J,K+1))+
c     &              0.5D0*(VIS(I,J-1,K)+VIS(I,J-1,K+1)))
c               VISB = 0.5D0*(0.5D0*(VIS(I,J,K)+VIS(I,J,K-1))+
c     &              0.5D0*(VIS(I,J-1,K)+VIS(I,J-1,K-1)))
               VIST = FAC1TB(K)*0.5D0*(VIS(I,J,K+1)+VIS(I,J-1,K+1))+
     +                FAC2TB(K)*0.5D0*(VIS(I,J,K+T_shift(K))+
     +                                 VIS(I,J-1,K+T_shift(K)))
               VISB = FAC3TB(K)*0.5D0*(VIS(I,J,K-1)+VIS(I,J-1,K-1))+
     +                FAC4TB(K)*0.5D0*(VIS(I,J,K+B_shift(K))+
     +                                 VIS(I,J-1,K+B_shift(K)))
               DEO = VISE*AREAEW/DXEP(I)
               DWO = VISW*AREAEW/DXPW(I)
               DNO = VISN*AREAN/DYNPV(J)
               DSO = VISS*AREAS/DYPSV(J)
               DTO = VIST*AREATB/DZTP(K)
               DBO = VISB*AREATB/DZPB(K)
C-----------------------------------------------------------------------
C     NEW DIFFUSION TERMS
C-----------------------------------------------------------------------
c viscosity interpolation for weighted central differencing is now done
c directly to faces centered about P cell center, without intermediate
c interpolation to standard P cell faces (where velocities are stored)
c               DE = CEE(I)*DEO + CWE(I)*DWO*DXPW(I)/DXEP(I)
c               DW = CWW(I)*DWO - CWE(I)*DEO*DXEP(I)/DXPW(I)   
c               DN = CNNV(J)*DNO + CSNV(J)*DSO*DYPSV(J)/DYNPV(J)
c               DS = CSSV(J)*DSO - CSNV(J)*DNO*DYNPV(J)/DYPSV(J)
c               DT = CTT(K)*DTO + CBT(K)*DBO*DZPB(K)/DZTP(K)
c               DB = CBB(K)*DBO - CBT(K)*DTO*DZTP(K)/DZPB(K)
               DE = DEO
               DW = DWO   
               DN = DNO
               DS = DSO
               DT = DTO
               DB = DBO
               if (de.lt.0.0D0) write(6,*) "DE SFCY is negative"
               if (dw.lt.0.0D0) write(6,*) "DW SFCY is negative"
               if (dn.lt.0.0D0) write(6,*) "DN SFCY is negative"
               if (ds.lt.0.0D0) write(6,*) "DS SFCY is negative"
               if (dt.lt.0.0D0) write(6,*) "DT SFCY is negative"
               if (db.lt.0.0D0) write(6,*) "DB SFCY is negative"
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
c ** This explicit double precision stuff is for the aix
               AEC = -0.5D0*CE*TEW + dMAX1(0.0D0,-CPE)*(1.0D0-TEW)
               AWC = 0.5D0*CW*TEW + dMAX1(0.0D0,CPW)*(1.0D0-TEW)
               ANC = -0.5D0*CN*TNS + dMAX1(0.0D0,-CPN)*(1.0D0-TNS)
               ASC = 0.5D0*CS*TNS + dMAX1(0.0D0,CPS)*(1.0D0-TNS)
               ATC = -0.5D0*CT*TTB + dMAX1(0.0D0,-CPT)*(1.0D0-TTB)
               ABC = 0.5D0*CB*TTB + dMAX1(0.0D0,CPB)*(1.0D0-TTB)
c               DV(I,J,K) = VOL/DYPS(J)

C----------------------------------------------------------------------
C      PROVISIONS FOR SIMPLEST
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

