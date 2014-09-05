#include <Packages/Uintah/CCA/Components/Arches/Radiation/fortran/radarray_fort.h>
c      SUBROUTINE radarray(RGAMMA,SD15,SD,SD7,SD3)


      Integer I,J

C     Open different files and read different arrays

      OPEN (UNIT=15,FILE='RGAMMA',STATUS='OLD')

         DO 10 J=1,7
            READ(15,*)(RGAMMA(4*(J-1)+I),I=1,4)
 10      CONTINUE

      CLOSE (UNIT=15)

      OPEN (UNIT=25,FILE='SDCOTWO',STATUS='OLD')

         DO 20 J=1,80
            READ(25,*)(SD15(6*(J-1)+I),I=1,6)
 20      CONTINUE

      CLOSE (UNIT=25)

      OPEN (UNIT=35,FILE='SDWATER',STATUS='OLD')

         DO 30 J=1,376
            READ(35,*)(SD(6*(J-1)+I),I=1,6)
 30      CONTINUE

      CLOSE (UNIT=35)

c      OPEN (UNIT=45,FILE='SDFUEL',STATUS='OLD')

c         DO 40 J=1,16
c            READ(45,*)(SD7(I),I=1,3)
c 40      CONTINUE

c         DO 50 J=1,32
c            READ(45,*)(SD3(I),I=1,3)
c 50      CONTINUE

c      CLOSE (UNIT=45)

c     Tested: Values of rgamma, sd15 and sd are being read correctly 

      RETURN
      END

