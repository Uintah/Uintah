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
c*********************************************************************
c
c
c*********************************************************************

#include <CCA/Components/Arches/fortran/wvelcoef_central_fort.h>
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
      double precision VISE, VISW, VISN, VISS, VIST, VISB
      double precision DE, DW, DN, DS, DT, DB

c*********************************************************************
c     Start :
c*********************************************************************
      ISTW = idxLoW(1)
      JSTW = idxLoW(2)
      KSTW = idxLoW(3)
      IENDW = idxHiW(1)
      JENDW = idxHiW(2)
      KENDW = idxHiW(3)

      AREANS = dx*dz
      AREAEW = dy*dz
      AREATB = dx*dy
      VOL = dx*dy*dz

      DO 220 K = KSTW,KENDW
         DO 210 J = JSTW,JENDW
            DO 200 I = ISTW,IENDW   

C-----------------------------------------------------------------------
C     CALCULATE CONVECTION COEFFICIENTS
C-----------------------------------------------------------------------
               GE = 0.5D0*(DEN(I+1,J,K)+DEN(I,J,K))*UU(I+1,J,K)
               GEB = 0.5D0*(DEN(I+1,J,K-1)+DEN(I,J,K-1))*UU(I+1,J,K-1)
               GW = 0.5D0*(DEN(I-1,J,K)+DEN(I,J,K))*UU(I,J,K)
               GWB = 0.5D0*(DEN(I,J,K-1)+DEN(I-1,J,K-1))*UU(I,J,K-1)
               GN = 0.5D0*(DEN(I,J+1,K)+DEN(I,J,K))*VV(I,J+1,K)
               GNB = 0.5D0*(DEN(I,J,K-1)+DEN(I,J+1,K-1))*VV(I,J+1,K-1)
               GS = 0.5D0*(DEN(I,J-1,K)+DEN(I,J,K))*VV(I,J,K)
               GSB = 0.5D0*(DEN(I,J,K-1)+DEN(I,J-1,K-1))*VV(I,J,K-1)
               GT = 0.5D0*(DEN(I,J,K)+DEN(I,J,K+1))*WW(I,J,K+1)
               GPB = 0.5D0*(DEN(I,J,K-1)+DEN(I,J,K))*WW(I,J,K)
               GB = 0.5D0*(DEN(I,J,K-1)+DEN(I,J,K-2))*WW(I,J,K-1)

               CESAV(I,J,K) = (0.5D0*GE+0.5D0*GEB)*AREAEW
               CWSAV(I,J,K) = (0.5D0*GW+0.5D0*GWB)*AREAEW
               CNSAV(I,J,K) = (0.5D0*GN+0.5D0*GNB)*AREANS
               CSSAV(I,J,K) = (0.5D0*GS+0.5D0*GSB)*AREANS
               CTSAV(I,J,K) = 0.5D0*(GT+GPB)*AREATB
               CBSAV(I,J,K) = 0.5D0*(GPB+GB)*AREATB

C-----------------------------------------------------------------------
C     CALCULATE DIFFUSION COEFFICIENTS
C-----------------------------------------------------------------------
               VISE = 0.25D0*(VIS(I+1,J,K)+VIS(I+1,J,K-1))+
     +                0.25D0*(VIS(I,J,K)+VIS(I,J,K-1))
               VISW = 0.25D0*(VIS(I-1,J,K)+VIS(I-1,J,K-1))+
     +                0.25D0*(VIS(I,J,K)+VIS(I,J,K-1))
               VISN = 0.25D0*(VIS(I,J+1,K)+VIS(I,J+1,K-1))+
     +                0.25D0*(VIS(I,J,K)+VIS(I,J,K-1))
               VISS = 0.25D0*(VIS(I,J-1,K)+VIS(I,J-1,K-1))+
     +                0.25D0*(VIS(I,J,K)+VIS(I,J,K-1))
               VIST = VIS(I,J,K)
               VISB = VIS(I,J,K-1)

               DE = VISE*AREAEW/DX
               DW = VISW*AREAEW/DX
               DN = VISN*AREANS/DY
               DS = VISS*AREANS/DY
               DT = VIST*AREATB/DZ
               DB = VISB*AREATB/DZ

C----------------------------------------------------------------------
C      PROVISIONS FOR VARIOUS CONVECTION DIFFERENCING SCHEMES
C----------------------------------------------------------------------
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

