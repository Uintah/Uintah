C
C The MIT License
C
C Copyright (c) 1997-2018 The University of Utah
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

#include <CCA/Components/Arches/fortran/uvelcoef_central_fort.h>
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
      double precision GE, GP, GW, GN, GNW, GS, GSW, GT, GTW, GTP, GBW
      double precision VISE, VISW, VISN, VISS, VIST, VISB
      double precision DE, DW, DN, DS, DT, DB
      double precision COEFE, COEFW, COEFN, COEFS, COEFT, COEFB
      double precision CE, CW, CN, CS, CT, CB
      double precision DUDXE, DUDXW
      double precision DVDXN, DVDXS
      double precision DWDXB, DWDXT
      double precision PT5, PT25, small, tiny
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
      tiny = 1.0D-20

c*********************************************************************
c     Go thru all the cells
c*********************************************************************
      AREANS = dx*dz
      AREAEW = dy*dz
      AREATB = dx*dy
      VOL = dx*dy*dz

      DO 220 K = KSTU,KENDU
         DO 210 J = JSTU,JENDU
            DO 200 I = ISTU,IENDU

              EPSP = eps(I,J,K)*eps(I-1,J,K)

            IF ( EPSP .gt. small ) THEN

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

               CE = PT5*(GE+GP)*AREAEW * EPSE
               CW = PT5*(GP+GW)*AREAEW * EPSW
               CN = (PT5*GN+PT5*GNW)*AREANS * EPSN
               CS = (PT5*GS+PT5*GSW)*AREANS * EPSS
               CT = (PT5*GT+PT5*GTW)*AREATB * EPST
               CB = (PT5*GTP+PT5*GBW)*AREATB * EPSB

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
               VISN = PT25*( VIS(I,J+1,K) + VIS(I-1,J+1,K) +
     +                         VIS(I,J,K) + VIS(I-1,J,K) )
               VISS = PT25*( VIS(I,J-1,K) + VIS(I-1,J-1,K) +
     +                         VIS(I,J,K)   + VIS(I-1,J,K) )
               VIST = PT25*( VIS(I,J,K+1) + VIS(I-1,J,K+1) +
     +                         VIS(I,J,K)   + VIS(I-1,J,K) )
               VISB = PT25*( VIS(I,J,K-1) + VIS(I-1,J,K-1) +
     +                         VIS(I,J,K)   + VIS(I-1,J,K) )

               DE = VISE*AREAEW/dx * EPSE
               DW = VISW*AREAEW/dx * EPSW
               DN = VISN*AREANS/dy * EPSN
               DS = VISS*AREANS/dy * EPSS
               DT = VIST*AREATB/dz * EPST
               DB = VISB*AREATB/dz * EPSB

C-----------------------------------------------------------------------
C     CONVECTION DIFFERENCING SCHEME SELECTION
C-----------------------------------------------------------------------
               AE(I,J,K) = DE-CE*PT5
               AW(I,J,K) = DW+CW*PT5
               AN(I,J,K) = DN-CN*PT5
               AS(I,J,K) = DS+CS*PT5
               AT(I,J,K) = DT-CT*PT5
               AB(I,J,K) = DB+CB*PT5

C-----------------------------------------------------------------------
C     velocity source
C-----------------------------------------------------------------------
               DUDXE  = (UU(I+1,J,K)-UU(I,J,K))/dx
               DUDXW  = (UU(I,J,K)-UU(I-1,J,K))/dx
               DVDXN  = (VV(I,J+1,K)-VV(I-1,J+1,K))/dx
               DVDXS  = (VV(I,J,K)-VV(I-1,J,K))/dx
               DWDXT  = (WW(I,J,K+1)-WW(I-1,J,K+1))/dx
               DWDXB  = (WW(I,J,K)-WW(I-1,J,K))/dx

               SU(I,J,K) =
     &              ((VISE*DUDXE - VISW*DUDXW)/dx +
     &               (VISN*DVDXN - VISS*DVDXS)/dy +
     &               (VIST*DWDXT - VISB*DWDXB)/dz +
     &              PT5*(old_DEN(I,J,K) + old_DEN(I-1,J,K))*
     $              (old_UU(I,J,K))/DELTAT
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

             endif

 200        CONTINUE
 210     CONTINUE
 220  CONTINUE

      RETURN
      END

