C
C The MIT License
C
C Copyright (c) 1997-2012 The University of Utah
C
C Permission is hereby granted, free of charge, to any person obtaining a copy
C of this software and associated documentation files (the "Software"), to
C deal in the Software without restriction, including without limitation the
C rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
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

#include <CCA/Components/Arches/fortran/prescoef_var_fort.h>
#include "param4.h"
*----------------------------------------------------------------------- 
*     Written by Rajesh Rawat, Jan 29, 2002
*     Last modified, September 16, 1998
*
*     Computes variable pressure coefficients
*
*-----------------------------------------------------------------------

c*********************************************************************
c     Locals :
c*********************************************************************
      double precision AREAN,AREAS,AREATB,AREAEW,EPSG,VOL 
      double precision DENE,DENW,DENN,DENS,DENT,DENB
      integer i,j,k
      integer IST, JST, KST, IEND, JEND, KEND

c*********************************************************************
c     Start :
c*********************************************************************
      IST = idxLo(1)
      JST = idxLo(2)
      KST = idxLo(3)
      IEND = idxHi(1)
      JEND = idxHi(2)
      KEND = idxHi(3)
C-----------------------------------------------------------------------
C     ASSEMBLY OF COEFFICIENTS  
C-----------------------------------------------------------------------
      DO 120 k = KST,KEND
         DO 110 j = JST,JEND
            DO 100 i = IST,IEND
C-----------------------------------------------------------------------
C     COMPUTE AREAS AND VOLUME
C-----------------------------------------------------------------------
               AREAN = SEW(i)*STB(k)
               AREAS = SEW(i)*STB(k)
               AREAEW = SNS(j)*STB(k)
               AREATB = SNS(j)*SEW(i)
               VOL = SNS(j)*SEW(i)*STB(k)
               EPSG = ONE
c              IF (LCALP .AND. .NOT.LP) THEN
c                 CALL CALEPS(domLo, domHi, 
c    &                 DENP,PDEN,I,J,K,EPSG)
c              ENDIF
C-----------------------------------------------------------------------
C     CALCULATE COEFFICIENTS
C-----------------------------------------------------------------------
               DENE = PT5*(DEN(i,j,k)+DEN(i+1,j,k))
               DENW = PT5*(DEN(i,j,k)+DEN(i-1,j,k))
               DENN = PT5*(DEN(i,j,k)+DEN(i,j+1,k))
               DENS = PT5*(DEN(i,j,k)+DEN(i,j-1,k))
               DENT = PT5*(DEN(i,j,k)+DEN(i,j,k+1))
               DENB = PT5*(DEN(i,j,k)+DEN(i,j,k-1))

               AE(i,j,k) = AREAEW/(DENE*DXEP(i))*EPSG

               AW(i,j,k) = AREAEW/(DENW*DXPW(i))*EPSG

               AN(i,j,k) = AREAN/(DENN*DYNP(j))*EPSG

               AS(i,j,k) = AREAS/(DENS*DYPS(j))*EPSG

               AT(i,j,k) = AREATB/(DENT*DZTP(k))*EPSG

               AB(i,j,k) = AREATB/(DENB*DZPB(k)) *EPSG
 100        CONTINUE
 110     CONTINUE
 120  CONTINUE

      RETURN
      END


