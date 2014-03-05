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

#include <CCA/Components/Arches/fortran/vvelcoef_central_fort.h>
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
      double precision GE, GP, GW, GN, GSE, GS, GSW, GT, GST, GSB, GB
      double precision VISE, VISW, VISN, VISS, VIST, VISB
      double precision DE, DW, DN, DS, DT, DB

c*********************************************************************
c     Start :
c*********************************************************************
      ISTV = idxLoV(1)
      JSTV = idxLoV(2)
      KSTV = idxLoV(3)
      IENDV = idxHiV(1)
      JENDV = idxHiV(2)
      KENDV = idxHiV(3)

      AREANS = dx*dz
      AREAEW = dy*dz
      AREATB = dx*dy
      VOL = dx*dy*dz

c*********************************************************************
c     Go thru all the cells
c*********************************************************************
      DO 220 K = KSTV,KENDV
         DO 210 J = JSTV,JENDV
            DO 200 I = ISTV,IENDV

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
               GT = 0.5D0*(DEN(I,J,K+1)+DEN(I,J,K))*WW(I,J,K+1)
               GST = 0.5D0*(DEN(I,J-1,K+1)+DEN(I,J-1,K))*WW(I,J-1,K+1)
               GB = 0.5D0*(DEN(I,J,K)+DEN(I,J,K-1))*WW(I,J,K)
               GSB = 0.5D0*(DEN(I,J-1,K)+DEN(I,J-1,K-1))*WW(I,J-1,K)
               CESAV(I,J,K) = (0.5D0*GE+0.5D0*GSE)*AREAEW
               CWSAV(I,J,K) = (0.5D0*GW+0.5D0*GSW)*AREAEW
               CTSAV(I,J,K) = (0.5D0*GT+0.5D0*GST)*AREATB
               CBSAV(I,J,K) = (0.5D0*GB+0.5D0*GSB)*AREATB 
               CNSAV(I,J,K) = 0.5D0*(GN+GP)*AREANS
               CSSAV(I,J,K) = 0.5D0*(GP+GS)*AREANS

C-----------------------------------------------------------------------
C     CALCULATE DIFFUSION COEFFICIENTS
C-----------------------------------------------------------------------
               VISE = 0.25D0*(VIS(I+1,J,K)+VIS(I+1,J-1,K))+
     +                0.25D0*(VIS(I,J,K)+ VIS(I,J-1,K))
               VISW = 0.25D0*(VIS(I-1,J,K)+VIS(I-1,J-1,K))+
     +                0.25D0*(VIS(I,J,K)+VIS(I,J-1,K))
               VISN = VIS(I,J,K)
               VISS = VIS(I,J-1,K)
               VIST = 0.25D0*(VIS(I,J,K+1)+VIS(I,J-1,K+1))+
     +                0.25D0*(VIS(I,J,K)+VIS(I,J-1,K))
               VISB = 0.25D0*(VIS(I,J,K-1)+VIS(I,J-1,K-1))+
     +                0.25D0*(VIS(I,J,K)+VIS(I,J-1,K))
               DE = VISE*AREAEW/DX
               DW = VISW*AREAEW/DX
               DN = VISN*AREANS/DY
               DS = VISS*AREANS/DY
               DT = VIST*AREATB/DZ
               DB = VISB*AREATB/DZ
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

