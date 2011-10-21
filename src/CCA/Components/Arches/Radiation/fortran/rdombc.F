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
#include <CCA/Components/Arches/Radiation/fortran/rdombc_fort.h>
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

      if (xminus.or.xplus) then
         DO 30 K=KST,KEND
            DO 20 J=JST,JEND
               if (xminus) then
                  i = ist -1
                  IF (PCELL(I,J,K).NE.FFIELD) THEN

                     if (lprobone.or.lprobthree) tg(i,j,k) = 0.0d0
                     if (lprobtwo)then
                     tg(i,j,k) = 0.0d0
                     end if
                     ABSKG(I,J,K) = wall_abskg
                     tg(i,j,k) = wall_temp

                 END IF
               endif
               if (xplus) then
                  i = iend + 1
                  IF (PCELL(I,J,K).NE.FFIELD) THEN

                     if (lprobone.or.lprobthree) tg(i,j,k) = 0.0d0
                     if (lprobtwo)then
                     tg(i,j,k) = 0.0d0
                     end if
                     ABSKG(I,J,K) = wall_abskg
                     tg(i,j,k) = wall_temp

                  END IF
               endif
 20         CONTINUE
 30      CONTINUE
      endif
      if (yminus.or.yplus) then
         DO 50 K=KST,KEND
            DO 40 I=IST,IEND
               if (yminus) then
                  j = jst -1
                  IF (PCELL(I,J,K).NE.FFIELD) THEN

                     if (lprobone.or.lprobthree) tg(i,j,k) = 0.0d0
                     if (lprobtwo)then
                     tg(i,j,k) = 0.0d0
                     end if
                     ABSKG(I,J,K) = wall_abskg
                     tg(i,j,k) = wall_temp

                  END IF
               endif
               if (yplus) then
                  j = jend + 1
                  IF (PCELL(I,J,K).NE.FFIELD) THEN

                     if (lprobone.or.lprobthree) tg(i,j,k) = 0.0d0
                     if (lprobtwo)then
                     tg(i,j,k) = 0.0d0
                     end if
                     ABSKG(I,J,K) = wall_abskg
                     tg(i,j,k) = wall_temp

                  END IF
               endif
 40         CONTINUE
 50      CONTINUE
      endif
      if (zminus.or.zplus) then
         DO 70 J=JST,JEND
            DO 60 I=IST,IEND
               if (zminus) then
                  k = kst -1
                  IF (PCELL(I,J,K).NE.FFIELD) THEN

                     if (lprobone.or.lprobthree) tg(i,j,k) = 0.0d0
                     if (lprobtwo)then
                     tg(i,j,k) = 0.0d0
                     end if
                     ABSKG(I,J,K) = wall_abskg
                     tg(i,j,k) = wall_temp
 
                 END IF
               endif
               if (zplus) then
                  k = kend + 1
                  IF (PCELL(I,J,K).NE.FFIELD) THEN

                     if (lprobone.or.lprobthree) tg(i,j,k) = 0.0d0
                     if (lprobtwo)then
                     tg(i,j,k) = 0.0d0
                     end if
                     ABSKG(I,J,K) = wall_abskg
                     tg(i,j,k) = wall_temp

                  END IF
               endif
 60         CONTINUE
 70      CONTINUE
      endif
      RETURN
      end








