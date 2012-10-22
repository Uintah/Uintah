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
C 
c*********************************************************************
c
c
c*********************************************************************

#include <CCA/Components/Arches/fortran/inlpresbcinout_fort.h>
#include "param4.h"
C-----------------------------------------------------------------------

c*********************************************************************
c     Local Variables :
c*********************************************************************
      integer IST, JST, KST, IEND, JEND, KEND
      integer i, j, k

c*********************************************************************
c     Start :
c*********************************************************************
      IST = idxLo(1)
      JST = idxLo(2)
      KST = idxLo(3)
      IEND = idxHi(1)
      JEND = idxHi(2)
      KEND = idxHi(3)
      if (xminus) then
         I = ist
         DO 140 K=KST,KEND
            DO 130 J=JST,JEND
               IF (PCELL(I-1,J,K) .EQ. PFIELD) THEN
                  flowin = flowin + pt5*(den(i-1,j,k)+den(i,j,k))*
     $                     sns(j)*stb(k)*max(0.0D0,uu(i,j,k))
                  flowout = flowout - pt5*(den(i-1,j,k)+den(i,j,k))*
     $                      sns(j)*stb(k)*min(0.0D0,uu(i,j,k))
                 if (cb) then
                  co2in = co2in + pt5*(den(i-1,j,k)+den(i,j,k))*
     $                     sns(j)*stb(k)*max(0.0D0,uu(i,j,k))*
     $                     pt5*(co2(i-1,j,k)+co2(i,j,k))
                  co2out = co2out - pt5*(den(i-1,j,k)+den(i,j,k))*
     $                      sns(j)*stb(k)*min(0.0D0,uu(i,j,k))*
     $                     pt5*(co2(i-1,j,k)+co2(i,j,k))
                 endif
               endif
 130        CONTINUE
 140     CONTINUE
      endif
      if (xplus) then
         I = iend+1
         DO 190 K=KST,KEND
            DO 180 J=JST,JEND
               IF (PCELL(I,J,K) .EQ. PFIELD) THEN
                  flowin = flowin - pt5*(den(i-1,j,k)+den(i,j,k))*
     $                     sns(j)*stb(k)*min(0.0D0,uu(i,j,k))
                  flowout = flowout + pt5*(den(i-1,j,k)+den(i,j,k))*
     $                      sns(j)*stb(k)*max(0.0D0,uu(i,j,k))
                 if (cb) then
                  co2in = co2in - pt5*(den(i-1,j,k)+den(i,j,k))*
     $                     sns(j)*stb(k)*min(0.0D0,uu(i,j,k))*
     $                     pt5*(co2(i-1,j,k)+co2(i,j,k))
                  co2out = co2out + pt5*(den(i-1,j,k)+den(i,j,k))*
     $                      sns(j)*stb(k)*max(0.0D0,uu(i,j,k))*
     $                     pt5*(co2(i-1,j,k)+co2(i,j,k))
                 endif
               END IF
 180        CONTINUE
 190     CONTINUE
      endif
C
C	Compute PBC for J=1 (South) side of domain
C
      if (yminus) then
         J = jst
         DO 240 K=KST,KEND
            DO 230 I=IST,IEND
               IF (PCELL(I,J-1,K) .EQ. PFIELD) THEN
                  flowin = flowin + pt5*(den(i,j-1,k)+den(i,j,k))*
     $                     sew(i)*stb(k)*max(0.0D0,vv(i,j,k))
                  flowout = flowout - pt5*(den(i,j-1,k)+den(i,j,k))*
     $                      sew(i)*stb(k)*min(0.0D0,vv(i,j,k))
                 if (cb) then
                  co2in = co2in + pt5*(den(i,j-1,k)+den(i,j,k))*
     $                     sew(i)*stb(k)*max(0.0D0,vv(i,j,k))*
     $                     pt5*(co2(i,j-1,k)+co2(i,j,k))
                  co2out = co2out - pt5*(den(i,j-1,k)+den(i,j,k))*
     $                      sew(i)*stb(k)*min(0.0D0,vv(i,j,k))*
     $                     pt5*(co2(i,j-1,k)+co2(i,j,k))
                 endif
               END IF
 230        CONTINUE
 240     CONTINUE
      endif
C     
C     Compute PBC for J=NJ (North) side of domain
C     
      if (yplus) then
         J = jend+1
         DO 290 K=KST,KEND
            DO 280 I=IST,IEND
               IF (PCELL(I,J,K) .EQ. PFIELD) THEN
                  flowin = flowin - pt5*(den(i,j-1,k)+den(i,j,k))*
     $                     sew(i)*stb(k)*min(0.0D0,vv(i,j,k))
                  flowout = flowout + pt5*(den(i,j-1,k)+den(i,j,k))*
     $                      sew(i)*stb(k)*max(0.0D0,vv(i,j,k))
                 if (cb) then
                  co2in = co2in - pt5*(den(i,j-1,k)+den(i,j,k))*
     $                     sew(i)*stb(k)*min(0.0D0,vv(i,j,k))*
     $                     pt5*(co2(i,j-1,k)+co2(i,j,k))
                  co2out = co2out + pt5*(den(i,j-1,k)+den(i,j,k))*
     $                      sew(i)*stb(k)*max(0.0D0,vv(i,j,k))*
     $                     pt5*(co2(i,j-1,k)+co2(i,j,k))
                 endif
               END IF
 280        CONTINUE
 290     CONTINUE
      endif

C
C	Calculate PBC for K=1 (Bottom) side of domain
C
      if (zminus) then
         K = kst
         DO 340 J=JST,JEND
            DO 330 I=IST,IEND
               IF (PCELL(I,J,K-1) .EQ. PFIELD) THEN
                  flowin = flowin + pt5*(den(i,j,k-1)+den(i,j,k))*
     $                     sew(i)*sns(j)*max(0.0D0,ww(i,j,k))
                  flowout = flowout - pt5*(den(i,j,k-1)+den(i,j,k))*
     $                      sew(i)*sns(j)*min(0.0D0,ww(i,j,k))
                 if (cb) then
                  co2in = co2in + pt5*(den(i,j,k-1)+den(i,j,k))*
     $                     sew(i)*sns(j)*max(0.0D0,ww(i,j,k))*
     $                     pt5*(co2(i,j,k-1)+co2(i,j,k))
                  co2out = co2out - pt5*(den(i,j,k-1)+den(i,j,k))*
     $                      sew(i)*sns(j)*min(0.0D0,ww(i,j,k))*
     $                     pt5*(co2(i,j,k-1)+co2(i,j,k))
                 endif
               END IF
 330        CONTINUE
 340     CONTINUE
      endif
C
C	Compute PBC for K=NK (Top) side of domain
C
      if (zplus) then
         K = kend+1
         DO 390 J=JST,JEND
            DO 380 I=IST,IEND
               IF (PCELL(I,J,K) .EQ. PFIELD) THEN
                  flowin = flowin - pt5*(den(i,j,k-1)+den(i,j,k))*
     $                     sew(i)*sns(j)*min(0.0D0,ww(i,j,k))
                  flowout = flowout + pt5*(den(i,j,k-1)+den(i,j,k))*
     $                      sew(i)*sns(j)*max(0.0D0,ww(i,j,k))
                 if (cb) then
                  co2in = co2in - pt5*(den(i,j,k-1)+den(i,j,k))*
     $                     sew(i)*sns(j)*min(0.0D0,ww(i,j,k))*
     $                     pt5*(co2(i,j,k-1)+co2(i,j,k))
                  co2out = co2out + pt5*(den(i,j,k-1)+den(i,j,k))*
     $                      sew(i)*sns(j)*max(0.0D0,ww(i,j,k))*
     $                     pt5*(co2(i,j,k-1)+co2(i,j,k))
                 endif
               END IF
 380        CONTINUE
 390     CONTINUE
      endif
C     
      RETURN
      END

