#include <Packages/Uintah/CCA/Components/Arches/Radiation/fortran/radarray_fort.h>
c      SUBROUTINE radarray(RGAMMA,SD15,SD,SD7,SD3)


      Integer I,J

C     Open different files and read different arrays

C BROKEN!!!!!!!!!!!!!!!!!!! NEEDS TO BE FIXED.
      OPEN (UNIT=15,FILE='RGAMMA',STATUS='OLD')

         DO 10 J=1,7
            READ(15,*)(RGAMMA(I),I=1,4)
 10      CONTINUE

      CLOSE (UNIT=15)

      OPEN (UNIT=25,FILE='SDCOTWO',STATUS='OLD')

         DO 20 J=1,80
            READ(25,*)(SD15(I),I=1,6)
 20      CONTINUE

      CLOSE (UNIT=25)

      OPEN (UNIT=35,FILE='SDWATER',STATUS='OLD')

         DO 30 J=1,376
            READ(35,*)(SD(I),I=1,6)
 30      CONTINUE

      CLOSE (UNIT=35)

      OPEN (UNIT=45,FILE='SDFUEL',STATUS='OLD')

         DO 40 J=1,16
            READ(45,*)(SD7(I),I=1,3)
 40      CONTINUE

         DO 50 J=1,32
            READ(45,*)(SD3(I),I=1,3)
 50      CONTINUE

      CLOSE (UNIT=45)

c      write(6,*)RGAMMA(2,2),SD15(3,8),SD(3,3),SD3(3,3)

      RETURN
      END

