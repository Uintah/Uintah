C
C The MIT License
C
C Copyright (c) 1997-2020 The University of Utah
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

#include <CCA/Components/Arches/fortran/uvelcoef_hybrid_fort.h>
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
      double precision VISE, VISW, VISN, VISS, VIST, VISB
      double precision DE, DW, DN, DS, DT, DB
      double precision TE, TW, TN, TS, TT, TB
      double precision COEFE, COEFW, COEFN, COEFS, COEFT, COEFB
      double precision AEC, AWC, ANC, ASC, ATC, ABC
      double precision DUDXE, DUDXW
      double precision DVDXN, DVDXS
      double precision DWDXB, DWDXT
      double precision PT5, PT25, small
      double precision EPSN, EPSS, EPSE, EPSW, EPST, EPSB
      double precision EPSP

c*********************************************************************
c     Start :
c*********************************************************************
      ISTU = idxLoU(1)
      JSTU = idxLoU(2)
      KSTU = idxLoU(3)
      IENDU = idxHiU(1)
      JENDU = idxHiU(2)
      KENDU = idxHiU(3)

      PT5 = 0.5D0
      PT25 = 0.25D0
      small = 1.0D-10

      AREANS = dx*dz
      AREAEW = dy*dz
      AREATB = dx*dy
      VOL = dx*dy*dz

      DO 220 K = KSTU,KENDU
         DO 210 J = JSTU,JENDU
            DO 200 I = ISTU,IENDU

               EPSP = eps(I,J,K)*eps(I-1,J,K)

            IF ( EPSP .gt. SMALL ) THEN

              EPSE = eps(I+1,J,K)
              EPSW = eps(I-2,J,K)
              EPSN = eps(I,J+1,K)*eps(I-1,J+1,K)
              EPSS = eps(I,J-1,K)*eps(I-1,J-1,K)
              EPST = eps(I,J,K+1)*eps(I-1,J,K+1)
              EPSB = eps(I,J,K-1)*eps(I-1,J,K-1)
C-----------------------------------------------------------------------
C     CALCULATE CONVECTION COEFFICIENTS
C-----------------------------------------------------------------------
               GE = PT5*(DEN(I+1,J,K)+DEN(I,J,K))*UU(I+1,J,K)
               GP = PT5*(DEN(I,J,K)+DEN(I-1,J,K))*UU(I,J,K)
               GW = PT5*(DEN(I-1,J,K)+DEN(I-2,J,K))*UU(I-1,J,K)
               GN = PT5*(DEN(I,J+1,K)+DEN(I,J,K))*VV(I,J+1,K)
               GNW = PT5*(DEN(I-1,J,K)+DEN(I-1,J+1,K))*VV(I-1,J+1,K)
               GS = PT5*(DEN(I,J-1,K)+DEN(I,J,K))*VV(I,J,K)
               GSW = PT5*(DEN(I-1,J,K)+DEN(I-1,J-1,K))*VV(I-1,J,K)
               GT = PT5*(DEN(I,J,K)+DEN(I,J,K+1))*WW(I,J,K+1)
               GTW = PT5*(DEN(I-1,J,K)+DEN(I-1,J,K+1))*WW(I-1,J,K+1)
               GTP = PT5*(DEN(I,J,K)+DEN(I,J,K-1))*WW(I,J,K)
               GBW = PT5*(DEN(I-1,J,K)+DEN(I-1,J,K-1))*WW(I-1,J,K)

               CE = PT5*(GE+GP)*AREAEW*EPSE
               CW = PT5*(GP+GW)*AREAEW*EPSW
               CN = PT5*(GN+GNW)*AREANS*EPSN
               CS = PT5*(GS+GSW)*AREANS*EPSS
               CT = PT5*(GT+GTW)*AREATB*EPST
               CB = PT5*(GTP+GBW)*AREATB*EPSB

               CESAV(I,J,K) = CE
               CWSAV(I,J,K) = CW
               CNSAV(I,J,K) = CN
               CSSAV(I,J,K) = CS
               CTSAV(I,J,K) = CT
               CBSAV(I,J,K) = CB

C-----------------------------------------------------------------------
C     CALCULATE DIFFUSION COEFFICIENTS
C-----------------------------------------------------------------------
               VISE = VIS(I,J,K)
               VISW = VIS(I-1,J,K)
               VISN = PT25*( VIS(I,J+1,K) + VIS(I-1,J+1,K)+
     +                VIS(I,J,K)   + VIS(I-1,J,K))
               VISS = PT25*( VIS(I,J-1,K) + VIS(I-1,J-1,K)+
     +                VIS(I,J,K)   + VIS(I-1,J,K))
               VIST = PT25*( VIS(I,J,K+1) + VIS(I-1,J,K+1)+
     +                VIS(I,J,K)   + VIS(I-1,J,K) )
               VISB = PT25*( VIS(I,J,K-1) + VIS(I-1,J,K-1)+
     +                VIS(I,J,K)   + VIS(I-1,J,K) )

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
               IF (TE.EQ.1.0D0) CVSX(I,J,K) = CVSX(I,J,K)+1.0d0/6.0d0
               IF (TW.EQ.1.0D0) CVSX(I,J,K) = CVSX(I,J,K)+1.0d0/6.0d0
               IF (TN.EQ.1.0D0) CVSX(I,J,K) = CVSX(I,J,K)+1.0d0/6.0d0
               IF (TS.EQ.1.0D0) CVSX(I,J,K) = CVSX(I,J,K)+1.0d0/6.0d0
               IF (TT.EQ.1.0D0) CVSX(I,J,K) = CVSX(I,J,K)+1.0d0/6.0d0
               IF (TB.EQ.1.0D0) CVSX(I,J,K) = CVSX(I,J,K)+1.0d0/6.0d0
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
C     velocity source
C-----------------------------------------------------------------------
               DUDXE  = (UU(I+1,J,K)-UU(I,J,K))/dx
               DUDXW  = (UU(I,J,K)-UU(I-1,J,K))/dx
               DVDXN  = (VV(I,J+1,K)-VV(I-1,J+1,K))/dx
               DVDXS  = (VV(I,J,K)-VV(I-1,J,K))/dx
               DWDXT  = (WW(I,J,K+1)-WW(I-1,J,K+1))/dx
               DWDXB  = (WW(I,J,K)-WW(I-1,J,K))/dx

               SU(I,J,K) = SU(I,J,K)+
     &              ((VISE*DUDXE - VISW*DUDXW)/dx +
     &              ( VISN*DVDXN - VISS*DVDXS)/dy +
     &              ( VIST*DWDXT - VISB*DWDXB)/dz +
     &              PT5*(DEN(I,J,K) + DEN(I-1,J,K))*
     $              (UU(I,J,K))/DELTAT
     &              + (PT5*(DEN(I,J,K)+DEN(I-1,J,K))-
     $              PT5*(den_ref(i,j,k)+den_ref(i-1,j,k)))*
     $              GRAV )*VOL*EPSP

                SU(I,J,K) = SU(I,J,K) - UU(I,J,K)*(
     $               CE - CW + CN - CS + CT - CB )

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
               CVSX(I,J,K) = 0.0d0

             endif


 200        CONTINUE
 210     CONTINUE
 220  CONTINUE

      RETURN
      END
