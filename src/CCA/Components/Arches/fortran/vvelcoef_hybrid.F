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
C 
c*********************************************************************
c
c
c*********************************************************************

#include <CCA/Components/Arches/fortran/vvelcoef_hybrid_fort.h>
C---------------------------------------------------------------------
C     THREE-DIMENSIONAL VERSION
C     THIS SUBROUTINE WILL compute matrix coefficients for vvelocity
C---------------------------------------------------------------------

c*********************************************************************
c     Local Variables :
c*********************************************************************
      integer ISTV, JSTV, KSTV, IENDV, JENDV, KENDV
      integer i, j, k

      double precision AREANS, AREAEW, AREATB, VOL
      double precision CE, CW, CN, CS, CT, CB
      double precision GE, GP, GW, GN, GSE, GS, GSW, GT, GST, GSB, GB
      double precision GSP, GNP, GUP
      double precision GWP
      double precision VISE, VISW, VISN, VISS, VIST, VISB
      double precision DE, DW, DN, DS, DT, DB
      double precision TE, TW, TN, TS, TT, TB
      double precision COEFE, COEFW, COEFN, COEFS, COEFT, COEFB
      double precision AEC, AWC, ANC, ASC, ATC, ABC
      double precision DUDYE, DUDYW
      double precision DVDYN, DVDYS
      double precision DWDYB, DWDYT
      double precision PT5, PT25, small
      double precision EPSN, EPSS, EPSE, EPSW, EPST, EPSB, EPSP

c*********************************************************************
c     Start :
c*********************************************************************
      ISTV = idxLoV(1)
      JSTV = idxLoV(2)
      KSTV = idxLoV(3)
      IENDV = idxHiV(1)
      JENDV = idxHiV(2)
      KENDV = idxHiV(3)

      PT5 = 0.5D0
      PT25 = 0.25D0
      small = 1.0D-10

      AREANS = dx*dz
      AREAEW = dy*dz
      AREATB = dx*dy
      VOL = dx*dy*dz


      DO 220 K = KSTV,KENDV
         DO 210 J = JSTV,JENDV
            DO 200 I = ISTV,IENDV

              EPSP = eps(I,J,K)*eps(I,J-1,K)

            IF ( EPSP .gt. small ) THEN 

              EPSE = eps(I+1,J,K)*eps(I+1,J-1,K)
              EPSW = eps(I-1,J,K)*eps(I-1,J-1,K)
              EPSN = eps(I,J+1,K)
              EPSS = eps(I,J-2,K)
              EPST = eps(I,J,K+1)*eps(I,J-1,K+1)
              EPSB = eps(I,J,K-1)*eps(I,J-1,K-1)

C-----------------------------------------------------------------------
C     CALCULATE CONVECTION COEFFICIENTS
C-----------------------------------------------------------------------
               GE = PT5*(DEN(I+1,J,K)+DEN(I,J,K))*UU(I+1,J,K)
               GSE = PT5*(DEN(I,J-1,K)+DEN(I+1,J-1,K))*UU(I+1,J-1,K)
               GW = PT5*(DEN(I,J,K)+DEN(I-1,J,K))*UU(I,J,K)
               GSW = PT5*(DEN(I,J-1,K)+DEN(I-1,J-1,K))*UU(I,J-1,K)
               GN = PT5*(DEN(I,J+1,K)+DEN(I,J,K))*VV(I,J+1,K)
               GP = PT5*(DEN(I,J,K)+DEN(I,J-1,K))*VV(I,J,K)
               GS = PT5*(DEN(I,J-1,K)+DEN(I,J-2,K))*VV(I,J-1,K)
               GT = PT5*(DEN(I,J,K+1)+DEN(I,J,K))*WW(I,J,K+1)
               GST = PT5*(DEN(I,J-1,K+1)+DEN(I,J-1,K))*WW(I,J-1,K+1)
               GB = PT5*(DEN(I,J,K)+DEN(I,J,K-1))*WW(I,J,K)
               GSB = PT5*(DEN(I,J-1,K)+DEN(I,J-1,K-1))*WW(I,J-1,K)

               CE = PT5*(GE+GSE)*AREAEW*EPSE
               CW = PT5*(GW+GSW)*AREAEW*EPSW
               CT = PT5*(GT+GST)*AREATB*EPST
               CB = PT5*(GB+GSB)*AREATB*EPSB
               CN = PT5*(GN+GP)*AREANS*EPSN
               CS = PT5*(GP+GS)*AREANS*EPSS

               CESAV(I,J,K) = CE
               CWSAV(I,J,K) = CW
               CNSAV(I,J,K) = CN
               CSSAV(I,J,K) = CS
               CTSAV(I,J,K) = CT
               CBSAV(I,J,K) = CB

C-----------------------------------------------------------------------
C     CALCULATE DIFFUSION COEFFICIENTS
C-----------------------------------------------------------------------
               VISE = PT25*(VIS(I+1,J,K)+VIS(I+1,J-1,K)+
     +                VIS(I,J,K)+ VIS(I,J-1,K))
               VISW = PT25*(VIS(I-1,J,K)+VIS(I-1,J-1,K)+
     +                VIS(I,J,K)+VIS(I,J-1,K))
               VISN = VIS(I,J,K)
               VISS = VIS(I,J-1,K)
               VIST = PT25*(VIS(I,J,K+1)+VIS(I,J-1,K+1)+
     +                VIS(I,J,K)+VIS(I,J-1,K))
               VISB = PT25*(VIS(I,J,K-1)+VIS(I,J-1,K-1)+
     +                VIS(I,J,K)+VIS(I,J-1,K))

               DE = VISE*AREAEW/dx*EPSE
               DW = VISW*AREAEW/dx*EPSW
               DN = VISN*AREANS/dy*EPSN
               DS = VISS*AREANS/dy*EPSS
               DT = VIST*AREATB/dz*EPST
               DB = VISB*AREATB/dz*EPSB


C-----------------------------------------------------------------------
C     CONVECTION DIFFERENCING SCHEME SELECTION
C-----------------------------------------------------------------------
               COEFE = DE - ABS(CE)/Relimit
               COEFW = DW - ABS(CW)/Relimit
               COEFN = DN - ABS(CN)/Relimit
               COEFS = DS - ABS(CS)/Relimit
               COEFT = DT - ABS(CT)/Relimit
               COEFB = DB - ABS(CB)/Relimit
               TE = 1.0D0
               TW = 1.0D0
               TN = 1.0D0
               TS = 1.0D0
               TT = 1.0D0
               TB = 1.0D0
               IF (COEFE.LT.0.0D0) TE = 0.0D0
               IF (COEFW.LT.0.0D0) TW = 0.0D0
               IF (COEFN.LT.0.0D0) TN = 0.0D0
               IF (COEFS.LT.0.0D0) TS = 0.0D0
               IF (COEFT.LT.0.0D0) TT = 0.0D0
               IF (COEFB.LT.0.0D0) TB = 0.0D0
               IF (TE.EQ.1.0D0) CVSY(I,J,K) = CVSY(I,J,K)+1.0d0/6.0d0
               IF (TW.EQ.1.0D0) CVSY(I,J,K) = CVSY(I,J,K)+1.0d0/6.0d0
               IF (TN.EQ.1.0D0) CVSY(I,J,K) = CVSY(I,J,K)+1.0d0/6.0d0
               IF (TS.EQ.1.0D0) CVSY(I,J,K) = CVSY(I,J,K)+1.0d0/6.0d0
               IF (TT.EQ.1.0D0) CVSY(I,J,K) = CVSY(I,J,K)+1.0d0/6.0d0
               IF (TB.EQ.1.0D0) CVSY(I,J,K) = CVSY(I,J,K)+1.0d0/6.0d0
               AEC = -PT5*CE*TE + dMAX1(0.0D0,-CE)*(1.0D0-TE)
               AWC = PT5*CW*TW + dMAX1(0.0D0,CW)*(1.0D0-TW)
               ANC = -PT5*CN*TN + dMAX1(0.0D0,-CN)*(1.0D0-TN)
               ASC = PT5*CS*TS + dMAX1(0.0D0,CS)*(1.0D0-TS)
               ATC = -PT5*CT*TT + dMAX1(0.0D0,-CT)*(1.0D0-TT)
               ABC = PT5*CB*TB + dMAX1(0.0D0,CB)*(1.0D0-TB)
               AE(I,J,K) = AEC + DE
               AW(I,J,K) = AWC + DW
               AN(I,J,K) = ANC + DN
               AS(I,J,K) = ASC + DS
               AT(I,J,K) = ATC + DT
               AB(I,J,K) = ABC + DB

C-----------------------------------------------------------------------
C     Compute the velocity source
C-----------------------------------------------------------------------
               DUDYE = (UU(I+1,J,K)-UU(I+1,J-1,K))/DY
               DUDYW = (UU(I,J,K)-UU(I,J-1,K))/DY
               DVDYN  = (VV(I,J+1,K)-VV(I,J,K))/DY
               DVDYS  = (VV(I,J,K)-VV(I,J-1,K))/DY
               DWDYT = (WW(I,J,K+1)-WW(I,J-1,K+1))/DY
               DWDYB = (WW(I,J,K)-WW(I,J-1,K))/DY

               SU(I,J,K) = SU(I,J,K)+
     &              ( (VISE*DUDYE-VISW*DUDYW)/DX +
     &                (VISN*DVDYN-VISS*DVDYS)/DY +
     &                (VIST*DWDYT-VISB*DWDYB)/DZ +
     &                pt5*(DEN(I,J,K)+DEN(I,J-1,K))*
     $                (VV(I,J,K))/DELTAT
     &              + (PT5*(DEN(I,J-1,K)+DEN(I,J,K))-
     $              pt5*(DEN_REF(i,j,k)+DEN_REF(i,j-1,k)))*
     $              GRAV )*VOL*EPSP

                SU(I,J,K) = SU(I,J,K) - VV(I,J,K)*(
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
                CVSY(I,J,K) = 0.0d0

              endif 


 200        CONTINUE
 210     CONTINUE
 220  CONTINUE

      RETURN
      END

