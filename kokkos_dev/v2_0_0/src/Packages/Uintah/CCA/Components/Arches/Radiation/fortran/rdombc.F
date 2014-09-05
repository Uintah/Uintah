#include <Packages/Uintah/CCA/Components/Arches/Radiation/fortran/rdombc_fort.h>
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

                     if (lprobone.or.lprobthree) tg(i,j,k) = 0.0
                     if (lprobtwo)then
                     tg(i,j,k) = 0.0
                     end if
                     ABSKG(I,J,K) = 1.0D0

                  END IF
               endif
               if (xplus) then
                  i = iend + 1
                  IF (PCELL(I,J,K).NE.FFIELD) THEN

                     if (lprobone.or.lprobthree) tg(i,j,k) = 0.0
                     if (lprobtwo)then
                     tg(i,j,k) = 0.0
                     end if
                     ABSKG(I,J,K) = 1.0D0

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

                     if (lprobone.or.lprobthree) tg(i,j,k) = 0.0
                     if (lprobtwo)then
                     tg(i,j,k) = 0.0
                     end if
                     ABSKG(I,J,K) = 1.0D0

                  END IF
               endif
               if (yplus) then
                  j = jend + 1
                  IF (PCELL(I,J,K).NE.FFIELD) THEN

                     if (lprobone.or.lprobthree) tg(i,j,k) = 0.0
                     if (lprobtwo)then
                     tg(i,j,k) = 0.0
                     end if
                     ABSKG(I,J,K) = 1.0D0

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

                     if (lprobone.or.lprobthree) tg(i,j,k) = 0.0
                     if (lprobtwo)then
                     tg(i,j,k) = 0.0
                     end if
                     ABSKG(I,J,K) = 1.0D0

                  END IF
               endif
               if (zplus) then
                  k = kend + 1
                  IF (PCELL(I,J,K).NE.FFIELD) THEN

                     if (lprobone.or.lprobthree) tg(i,j,k) = 0.0
                     if (lprobtwo)then
                     tg(i,j,k) = 0.0
                     end if
                     ABSKG(I,J,K) = 1.0D0

                  END IF
               endif
 60         CONTINUE
 70      CONTINUE
      endif
      RETURN
      end








