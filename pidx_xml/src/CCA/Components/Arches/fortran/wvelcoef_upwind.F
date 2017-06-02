C
C The MIT License
C
C Copyright (c) 1997-2017 The University of Utah
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
c*********************************************************************
c
c
c*********************************************************************

#include <CCA/Components/Arches/fortran/wvelcoef_upwind_fort.h>
C---------------------------------------------------------------------
C     THREE-DIMENSIONAL VERSION
C     THIS SUBROUTINE WILL compute matrix coefficients for uvelocity
C---------------------------------------------------------------------

c*********************************************************************
c     Local Variables :
c*********************************************************************
      integer ISTW, JSTW, KSTW, IENDW, JENDW, KENDW
      integer i, j, k

      double precision AREANS, AREAEW, AREATB, VOL
      double precision GE, GEB, GW, GWB, GN, GNB, GS, GSB, GT, GPB, GB
      double precision CE, CW, CN, CS, CT, CB
      double precision GUP, CPE, CPW, GVP, CPN, CPS, CPT, CPB
      double precision VISE, VISW, VISN, VISS, VIST, VISB
      double precision DE, DW, DN, DS, DT, DB
      double precision COEFE, COEFW, COEFN, COEFS, COEFT, COEFB
      double precision TEW, TNS, TTB
      double precision AEC, AWC, ANC, ASC, ATC, ABC
      double precision DUDZE, DUDZW
      double precision DVDZN, DVDZS
      double precision DWDZB, DWDZT
      double precision PT5, PT25, small 
      double precision EPSN, EPSS, EPSE, EPSW, EPST, EPSB, EPSP

c*********************************************************************
c     Start :
c*********************************************************************
      ISTW = idxLoW(1)
      JSTW = idxLoW(2)
      KSTW = idxLoW(3)
      IENDW = idxHiW(1)
      JENDW = idxHiW(2)
      KENDW = idxHiW(3)

      PT5 = 0.5D0
      PT25 = 0.25D0
      small = 1.0D-10

      AREANS = dx*dz
      AREAEW = dy*dz
      AREATB = dx*dy
      VOL = dx*dy*dz

      DO 220 K = KSTW,KENDW
         DO 210 J = JSTW,JENDW
            DO 200 I = ISTW,IENDW   

              EPSP = eps(I,J,K)*eps(I,J,K-1)

              IF ( EPSP .gt. SMALL ) THEN 

              EPSE = eps(I+1,J,K)*eps(I+1,J,K-1)
              EPSW = eps(I-1,J,K)*eps(I-1,J,K-1)
              EPSN = eps(I,J+1,K)*eps(I,J+1,K-1)
              EPSS = eps(I,J-1,K)*eps(I,J-1,K-1)
              EPST = eps(I,J,K+1)
              EPSB = eps(I,J,K-2)
C-----------------------------------------------------------------------
C     CALCULATE CONVECTION COEFFICIENTS
C-----------------------------------------------------------------------
               GE = PT5*(DEN(I+1,J,K)+DEN(I,J,K))*UU(I+1,J,K)
               GEB = PT5*(DEN(I+1,J,K-1)+DEN(I,J,K-1))*UU(I+1,J,K-1)
               GW = PT5*(DEN(I-1,J,K)+DEN(I,J,K))*UU(I,J,K)
               GWB = PT5*(DEN(I,J,K-1)+DEN(I-1,J,K-1))*UU(I,J,K-1)
               GN = PT5*(DEN(I,J+1,K)+DEN(I,J,K))*VV(I,J+1,K)
               GNB = PT5*(DEN(I,J,K-1)+DEN(I,J+1,K-1))*VV(I,J+1,K-1)
               GS = PT5*(DEN(I,J-1,K)+DEN(I,J,K))*VV(I,J,K)
               GSB = PT5*(DEN(I,J,K-1)+DEN(I,J-1,K-1))*VV(I,J,K-1)
               GT = PT5*(DEN(I,J,K)+DEN(I,J,K+1))*WW(I,J,K+1)
               GPB = PT5*(DEN(I,J,K-1)+DEN(I,J,K))*WW(I,J,K)
               GB = PT5*(DEN(I,J,K-1)+DEN(I,J,K-2))*WW(I,J,K-1)

               CE = PT5*(GE+GEB)*AREAEW*EPSE
               CW = PT5*(GW+GWB)*AREAEW*EPSW
               CN = PT5*(GN+GNB)*AREANS*EPSN
               CS = PT5*(GS+GSB)*AREANS*EPSS
               CT = PT5*(GT+GPB)*AREATB*EPST
               CB = PT5*(GPB+GB)*AREATB*EPSB

               CESAV(I,J,K) = CE
               CWSAV(I,J,K) = CW
               CNSAV(I,J,K) = CN
               CSSAV(I,J,K) = CS
               CTSAV(I,J,K) = CT
               CBSAV(I,J,K) = CB

               GUP = 0.25*(GE+GEB+GW+GWB)
               CPE = GUP*VOL/DX
               CPW = GUP*VOL/DX
               GVP = 0.25*(GN+GNB+GS+GSB)
               CPN = GVP*VOL/DY
               CPS = GVP*VOL/DY
               CPT = GPB*VOL/DZ
               CPB = GPB*VOL/DZ

C-----------------------------------------------------------------------
C     CALCULATE DIFFUSION COEFFICIENTS
C-----------------------------------------------------------------------
               VISE = PT25*(VIS(I+1,J,K)+VIS(I+1,J,K-1)+
     +                VIS(I,J,K)+VIS(I,J,K-1))
               VISW = PT25*(VIS(I-1,J,K)+VIS(I-1,J,K-1)+
     +                VIS(I,J,K)+VIS(I,J,K-1))
               VISN = PT25*(VIS(I,J+1,K)+VIS(I,J+1,K-1)+
     +                VIS(I,J,K)+VIS(I,J,K-1))
               VISS = PT25*(VIS(I,J-1,K)+VIS(I,J-1,K-1)+
     +                VIS(I,J,K)+VIS(I,J,K-1))
               VIST = VIS(I,J,K)
               VISB = VIS(I,J,K-1)

               DE = VISE*AREAEW/DX*EPSE
               DW = VISW*AREAEW/DX*EPSW
               DN = VISN*AREANS/DY*EPSN
               DS = VISS*AREANS/DY*EPSS
               DT = VIST*AREATB/DZ*EPST
               DB = VISB*AREATB/DZ*EPSB

               COEFE = DE - PT5*ABS(CE)
               COEFW = DW - PT5*ABS(CW)
               COEFN = DN - PT5*ABS(CN)
               COEFS = DS - PT5*ABS(CS)
               COEFT = DT - PT5*ABS(CT)
               COEFB = DB - PT5*ABS(CB)

               TEW = 1.0D0
               TNS = 1.0D0
               TTB = 1.0D0
               IF (COEFE.LT.0.0D0 .OR. COEFW.LT.0.0D0) TEW = 0.0D0
               IF (COEFN.LT.0.0D0 .OR. COEFS.LT.0.0D0) TNS = 0.0D0
               IF (COEFT.LT.0.0D0 .OR. COEFB.LT.0.0D0) TTB = 0.0D0

               AEC = -PT5*CE*TEW + dMAX1(0.0D0,-CPE)*(1.0D0-TEW)
               AWC = PT5*CW*TEW + dMAX1(0.0D0,CPW)*(1.0D0-TEW)
               ANC = -PT5*CN*TNS + dMAX1(0.0D0,-CPN)*(1.0D0-TNS)
               ASC = PT5*CS*TNS + dMAX1(0.0D0,CPS)*(1.0D0-TNS)
               ATC = -PT5*CT*TTB + dMAX1(0.0D0,-CPT)*(1.0D0-TTB)
               ABC = PT5*CB*TTB + dMAX1(0.0D0,CPB)*(1.0D0-TTB)


C----------------------------------------------------------------------
C      PROVISIONS FOR VARIOUS CONVECTION DIFFERENCING SCHEMES
C----------------------------------------------------------------------
               AE(I,J,K) = AEC + DE
               AW(I,J,K) = AWC + DW
               AN(I,J,K) = ANC + DN
               AS(I,J,K) = ASC + DS
               AT(I,J,K) = ATC + DT
               AB(I,J,K) = ABC + DB

C----------------------------------------------------------------------
C      Velocity source term calculation 
C----------------------------------------------------------------------
               DUDZE = (UU(I+1,J,K)-UU(I+1,J,K-1))/DZ
               DUDZW = (UU(I,J,K)-UU(I,J,K-1))/DZ
               DVDZN = (VV(I,J+1,K)-VV(I,J+1,K-1))/DZ
               DVDZS = (VV(I,J,K)-VV(I,J,K-1))/DZ
               DWDZT = (WW(I,J,K+1)-WW(I,J,K))/DZ
               DWDZB = (WW(I,J,K)-WW(I,J,K-1))/DZ

               SU(I,J,K) = SU(I,J,K)+
     &              ((VISE*DUDZE-VISW*DUDZW)/DX +
     &               (VISN*DVDZN-VISS*DVDZS)/DY  +
     &               (VIST*DWDZT-VISB*DWDZB)/DZ  +
     &               pt5*(DEN(I,J,K)+DEN(i,j,k-1))*
     $               (WW(I,J,K))/DELTAT 
     &               + (PT5*(DEN(I,J,K-1)+DEN(I,J,K))-
     $               pt5*(den_ref(i,j,k) + den_ref(i,j,k-1)))*
     $               GRAV ) * VOL*EPSP

                SU(I,J,K) = SU(I,J,K) - WW(I,J,K)*(
     $               CE-CW+CN-CS+CT-CB)

              else 
                
                CESAV(I,J,K) = 0.0D0
                CWSAV(I,J,K) = 0.0D0
                CNSAV(I,J,K) = 0.0D0
                CSSAV(I,J,K) = 0.0D0
                CTSAV(I,J,K) = 0.0D0
                CBSAV(I,J,K) = 0.0D0
                AE(I,J,K) = 0.0D0
                AW(I,J,K) = 0.0D0
                AN(I,J,K) = 0.0D0
                AS(I,J,K) = 0.0D0
                AT(I,J,K) = 0.0D0
                AB(I,J,K) = 0.0D0
                SU(I,J,K) = 0.0D0

              endif 

 200        CONTINUE
 210     CONTINUE
 220  CONTINUE

      RETURN
      END

