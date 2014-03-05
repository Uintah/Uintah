C
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
c     Go thru all the cells
c*********************************************************************
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
               CESAV(I,J,K) = 0.5D0*(GE+GP)*AREAEW
               CWSAV(I,J,K) = 0.5D0*(GP+GW)*AREAEW
               CNSAV(I,J,K) = (0.5D0*GN+0.5D0*GNW)*AREANS
               CSSAV(I,J,K) = (0.5D0*GS+0.5D0*GSW)*AREANS
               CTSAV(I,J,K) = (0.5D0*GT+0.5D0*GTW)*AREATB
               CBSAV(I,J,K) = (0.5D0*GTP+0.5D0*GBW)*AREATB

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
               AE(I,J,K) = DE-CESAV(I,J,K)*0.5D0
               AW(I,J,K) = DW+CWSAV(I,J,K)*0.5D0
               AN(I,J,K) = DN-CNSAV(I,J,K)*0.5D0
               AS(I,J,K) = DS+CSSAV(I,J,K)*0.5D0
               AT(I,J,K) = DT-CTSAV(I,J,K)*0.5D0
               AB(I,J,K) = DB+CBSAV(I,J,K)*0.5D0

 200        CONTINUE
 210     CONTINUE
 220  CONTINUE

      RETURN
      END

