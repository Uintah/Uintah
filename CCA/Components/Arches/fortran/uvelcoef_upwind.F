C UPWIND
C The MIT License
C
C Copyright (c) 1997-2012 The University of Utah
C
C Permission is hereby granted, free of charge, to any person obtaining a copy
C of this software and associated documentation files (the "Software"), to
C deal in the Software without restriction, including without limitation the
C rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
C sell copies of the Software, and to permit persons to whom the Software is
C furnished to do so, subject to the following conditions:
C
C The above copyright notice and this permission notice shall be included in
C all copies or substantial portions of the Software.
C
C THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
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

#include <CCA/Components/Arches/fortran/uvelcoef_upwind_fort.h>
C---------------------------------------------------------------------
C     THREE-DIMENSIONAL VERSION
C     THIS SUBROUTINE WILL compute matrix coefficients for uvelocity
C---------------------------------------------------------------------

c*********************************************************************
c     Local Variables :
c*********************************************************************
      integer ISTU, JSTU, KSTU, IENDU, JENDU, KENDU
      integer i, j, k

      double precision AREANS, AREAEW, AREATB, VOL
      double precision CE, CW, CN, CS, CT, CB
      double precision GE, GP, GW, GN, GNW, GS, GSW, GT
      double precision GTW, GTP, GBW
      double precision GWP
      double precision CPE, CPW, GVP, CPN, CPS, CPT, CPB
      double precision VISE, VISW, VISN, VISS, VIST, VISB
      double precision DE, DW, DN, DS, DT, DB
      double precision TEW, TNS, TTB
      double precision COEFE, COEFW, COEFN, COEFS, COEFT, COEFB
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

      AREANS = dx*dz
      AREAEW = dy*dz
      AREATB = dx*dy
      VOL = dx*dy*dz

      DO 220 K = KSTU,KENDU
         DO 210 J = JSTU,JENDU
            DO 200 I = ISTU,IENDU   

C-----------------------------------------------------------------------
C     CALCULATE CONVECTION COEFFICIENTS
C-----------------------------------------------------------------------
               GE = 0.5D0*(DEN(I+1,J,K)+DEN(I,J,K))*UU(I+1,J,K)
               GP = 0.5D0*(DEN(I,J,K)+DEN(I-1,J,K))*UU(I,J,K)
               GW = 0.5D0*(DEN(I-1,J,K)+DEN(I-2,J,K))*UU(I-1,J,K)
               GN = 0.5D0*(DEN(I,J+1,K)+DEN(I,J,K))*VV(I,J+1,K)
               GNW = 0.5D0*(DEN(I-1,J,K)+DEN(I-1,J+1,K))*VV(I-1,J+1,K)
               GS = 0.5D0*(DEN(I,J-1,K)+DEN(I,J,K))*VV(I,J,K)
               GSW = 0.5D0*(DEN(I-1,J,K)+DEN(I-1,J-1,K))*VV(I-1,J,K)
               GT = 0.5D0*(DEN(I,J,K)+DEN(I,J,K+1))*WW(I,J,K+1)
               GTW = 0.5D0*(DEN(I-1,J,K)+DEN(I-1,J,K+1))*WW(I-1,J,K+1)
               GTP = 0.5D0*(DEN(I,J,K)+DEN(I,J,K-1))*WW(I,J,K)
               GBW = 0.5D0*(DEN(I-1,J,K)+DEN(I-1,J,K-1))*WW(I-1,J,K)

               CE = 0.5D0*(GE+GP)*AREAEW
               CW = 0.5D0*(GP+GW)*AREAEW
               CN = (0.5D0*GN+0.5D0*GNW)*AREANS
               CS = (0.5D0*GS+0.5D0*GSW)*AREANS
               CT = (0.5D0*GT+0.5D0*GTW)*AREATB
               CB = (0.5D0*GTP+0.5D0*GBW)*AREATB

               CESAV(I,J,K) = CE
               CWSAV(I,J,K) = CW
               CNSAV(I,J,K) = CN
               CSSAV(I,J,K) = CS
               CTSAV(I,J,K) = CT
               CBSAV(I,J,K) = CB

C-----------------------------------------------------------------------
C      UPWIND DIFFERENCING 
C-----------------------------------------------------------------------
               CPE = GP*VOL/dx
               CPW = GP*VOL/dx
               GVP = 0.25*(GN+GNW+GS+GSW)
               CPN = GVP*VOL/dy
               CPS = GVP*VOL/dy
               GWP = .25*(GT+GTW+GTP+GBW)
               CPT = GWP*VOL/dz
               CPB = GWP*VOL/dz

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

C-----------------------------------------------------------------------
C     CALCULATE DIFFUSION COEFFICIENTS
C-----------------------------------------------------------------------
               VISE = VIS(I,J,K)
               VISW = VIS(I-1,J,K)
               VISN = 0.25D0*( VIS(I,J+1,K) + VIS(I-1,J+1,K))+
     +                0.25D0*( VIS(I,J,K)   + VIS(I-1,J,K))
               VISS = 0.25D0*( VIS(I,J-1,K) + VIS(I-1,J-1,K))+
     +                0.25D0*( VIS(I,J,K)   + VIS(I-1,J,K))
               VIST = 0.25D0*( VIS(I,J,K+1) + VIS(I-1,J,K+1))+
     +                0.25D0*( VIS(I,J,K)   + VIS(I-1,J,K) )
               VISB = 0.25D0*( VIS(I,J,K-1) + VIS(I-1,J,K-1))+
     +                0.25D0*( VIS(I,J,K)   + VIS(I-1,J,K) )

               DE = VISE*AREAEW/dx
               DW = VISW*AREAEW/dx
               DN = VISN*AREANS/dy
               DS = VISS*AREANS/dy
               DT = VIST*AREATB/dz
               DB = VISB*AREATB/dz

C-----------------------------------------------------------------------
C     CONVECTION DIFFERENCING SCHEME SELECTION
C-----------------------------------------------------------------------
               AE(I,J,K) = AEC + DE
               AW(I,J,K) = AWC + DW
               AN(I,J,K) = ANC + DN
               AS(I,J,K) = ASC + DS
               AT(I,J,K) = ATC + DT
               AB(I,J,K) = ABC + DB

 200        CONTINUE
 210     CONTINUE
 220  CONTINUE

      RETURN
      END

