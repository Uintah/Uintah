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
#include <CCA/Components/Arches/Radiation/fortran/radarray_fort.h>
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

