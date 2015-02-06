C 
C 
C The MIT License
C 
C Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
C Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
C University of Utah.
C 
C License for the specific language governing rights and limitations under
C Permission is hereby granted, free of charge, to any person obtaining a 
C copy of this software and associated documentation files (the "Software"),
C to deal in the Software without restriction, including without limitation 
C the rights to use, copy, modify, merge, publish, distribute, sublicense, 
C and/or sell copies of the Software, and to permit persons to whom the 
C Software is furnished to do so, subject to the following conditions:
C 
C The above copyright notice and this permission notice shall be included 
C in all copies or substantial portions of the Software.
C 
C THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
C OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
C FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
C THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
C LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
C FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
C DEALINGS IN THE SOFTWARE.
C 
C
C 
C 
#include <CCA/Components/Models/Radiation/fortran/m_rdombc_fort.h>
      integer I,J,K,IST,IEND,JST,JEND,KST,KEND

      double precision PI,PI1,SIG

      DATA SIG/5.67D-08/

      pi = dacos(-1.0D0)
      PI1 = 1.0d0 / PI

      IEND=idxhi(1)
      JEND=idxhi(2)
      KEND=idxhi(3)
      IST=idxlo(1)
      JST=idxlo(2)
      KST=idxlo(3)

c     Specification of boundary values of temperature and absorption
c     coefficient for real (fire) problems.  Temperature is set at 293 K
c     because of a validation case in which the temperature at the radiometer
c     location was 293 K. The emissivity at the boundary is assumed to be one,
c     because otherwise the different directions will depend on each other
c     (rho + epsilon = 1, rho = reflectivity).  Here the absorption 
c     coefficient is arbitrarily set to one; the value is not really used
c     anywhere, only makes the graphing pretty.

      if (xminus) then

         DO 30 K=KST,KEND
            DO 20 J=JST,JEND
               i = ist -1
               IF (PCELL(I,J,K).NE.FFIELD) THEN
                  ABSKG(I,J,K) = 1.0d0  
                  tg(i,j,k) = 293.0d0
               END IF
 20         CONTINUE
 30      CONTINUE

      endif

      if (xplus) then

         DO 130 K=KST,KEND
            DO 120 J=JST,JEND
               i = iend + 1
               IF (PCELL(I,J,K).NE.FFIELD) THEN
                  ABSKG(I,J,K) = 1.0d0
                  tg(i,j,k) = 293.0d0
               END IF
 120         CONTINUE
 130      CONTINUE

      endif

      if (yminus) then
            
         DO 50 K=KST,KEND
            DO 40 I=IST,IEND
               j = jst -1
               IF (PCELL(I,J,K).NE.FFIELD) THEN
                  ABSKG(I,J,K) = 1.0d0
                  tg(i,j,k) = 293.0d0
               END IF
 40         CONTINUE
 50      CONTINUE

      endif

      if (yplus) then

         DO 150 K=KST,KEND
            DO 140 I=IST,IEND
               j = jend + 1
               IF (PCELL(I,J,K).NE.FFIELD) THEN
                  ABSKG(I,J,K) = 1.0d0
                  tg(i,j,k) = 293.0d0
               END IF
 140         CONTINUE
 150      CONTINUE

      endif
         
      if (zminus) then

         DO 70 J=JST,JEND
            DO 60 I=IST,IEND
               k = kst -1
               IF (PCELL(I,J,K).NE.FFIELD) THEN
                  ABSKG(I,J,K) = 1.0d0 
                  tg(i,j,k) = 293.0d0
               END IF
 60         CONTINUE
 70      CONTINUE

      endif

      if (zplus) then

         DO 170 J=JST,JEND
            DO 160 I=IST,IEND
               k = kend + 1
               IF (PCELL(I,J,K).NE.FFIELD) THEN
                  ABSKG(I,J,K) = 1.0d0 
                  tg(i,j,k) = 293.0d0
               END IF
 160         CONTINUE
 170      CONTINUE

      endif

      if (l_testproblems) then

c     Specification for test problems.  Problems 1 and 3 have zero
c     temperature at the boundaries, and Problem 2 has a specified temperature
c     that may be different from zero.

         if (xminus) then
            
            DO 230 K=KST,KEND
               DO 220 J=JST,JEND
                  i = ist -1
                  IF (PCELL(I,J,K).NE.FFIELD) THEN
                     if (lprobone.or.lprobthree) tg(i,j,k) = 0.0d0
                     if (lprobtwo)then
                        tg(i,j,k) = 0.0d0
                     end if
                  END IF
 220            CONTINUE
 230         CONTINUE

         endif

         if (xplus) then
            
            DO 330 K=KST,KEND
               DO 320 J=JST,JEND
                  i = iend + 1
                  IF (PCELL(I,J,K).NE.FFIELD) THEN
                     if (lprobone.or.lprobthree) tg(i,j,k) = 0.0d0
                     if (lprobtwo)then
                        tg(i,j,k) = 0.0d0
                     end if
                  END IF
 320            CONTINUE
 330         CONTINUE

         endif

         if (yminus) then
            DO 250 K=KST,KEND
               DO 240 I=IST,IEND
                  j = jst -1
                  IF (PCELL(I,J,K).NE.FFIELD) THEN
                     if (lprobone.or.lprobthree) tg(i,j,k) = 0.0d0
                     if (lprobtwo)then
                        tg(i,j,k) = 0.0d0
                     end if
                  END IF
 240            CONTINUE
 250         CONTINUE
         endif

         if (yplus) then
            DO 350 K=KST,KEND
               DO 340 I=IST,IEND
                  j = jend + 1
                  IF (PCELL(I,J,K).NE.FFIELD) THEN
                     if (lprobone.or.lprobthree) tg(i,j,k) = 0.0d0
                     if (lprobtwo)then
                        tg(i,j,k) = 0.0d0
                     end if
                  END IF
 340           CONTINUE
 350        CONTINUE
         endif

         if (zminus) then
            DO 270 J=JST,JEND
               DO 260 I=IST,IEND
                  k = kst -1
                  IF (PCELL(I,J,K).NE.FFIELD) THEN
                     if (lprobone.or.lprobthree) tg(i,j,k) = 0.0d0
                     if (lprobtwo)then
                        tg(i,j,k) = 0.0d0
                     end if
                  END IF
 260           CONTINUE
 270        CONTINUE
         endif

         if (zplus) then
            DO 370 J=JST,JEND
               DO 360 I=IST,IEND
                  k = kend + 1
                  IF (PCELL(I,J,K).NE.FFIELD) THEN
                     if (lprobone.or.lprobthree) tg(i,j,k) = 0.0d0
                     if (lprobtwo)then
                        tg(i,j,k) = 0.0d0
                     end if
                  END IF
 360            CONTINUE
 370         CONTINUE
         endif

      endif

      RETURN
      end
