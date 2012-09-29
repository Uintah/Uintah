C
C Copyright (c) 1997-2012 The University of Utah
C
C Permission is hereby granted, free of charge, to any person obtaining a copy
C of this software and associated documentation files (the \"Software\"), to
C deal in the Software without restriction, including without limitation the
C rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
C sell copies of the Software, and to permit persons to whom the Software is
C furnished to do so, subject to the following conditions:
C
C The above copyright notice and this permission notice shall be included in
C all copies or substantial portions of the Software.
C
C THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
C IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
C FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
C AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
C LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
C FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
C IN THE SOFTWARE.
C
C 
#include <CCA/Components/Arches/Radiation/fortran/rordrss_fort.h>
C***********************************************************************
C     SUBROUTINE RORDR (RORDR stands for "Radiation-ORDinates-Rectangular")
C     Generates the table of direction cosines and quadrature weights for 
C     discrete ordinates calculations in rectangular co-ordinates 
C     This is called before the intensity iterations.
C     The following is the spherical surface symmetric equally dividing 
C     quadrature scheme,Journal of Heat Transfer,Vol.124,p.482
C     Written by Gautham 01/03.
C***********************************************************************
C      dimension ORD(SN),OXI(SN*(SN+2)),OMU(SN*(SN+2)),
C     &     OETA(SN*(SN+2)),WT(SN*(SN+2))
c*********************************************************************
c     Local Variables :
c*********************************************************************

      integer TOTORD, i

      TOTORD = SN*(SN+2)

C     Open different files and read different arrays

      IF (SN.EQ.4) THEN
         OPEN (UNIT=15,FILE='SSSETS',STATUS='OLD')
         READ(15,*)(OMU(I),OETA(I),OXI(I),I=1,24)
         CLOSE (UNIT=15)
      ELSE IF (SN.EQ.6) THEN
         OPEN (UNIT=15,FILE='SSSETS',STATUS='OLD')
         READ(15,10)
 10      FORMAT(24(/))
         READ(15,*)(OMU(I),OETA(I),OXI(I),I=1,48)
         CLOSE (UNIT=15)
      ELSE 
         OPEN (UNIT=15,FILE='SSSETS',STATUS='OLD')
         READ(15,20)
 20      FORMAT(73(/))
         READ(15,*)(OMU(I),OETA(I),OXI(I),I=1,80)
         CLOSE (UNIT=15)
      ENDIF

c       Generate weights "WT"

        do i=1,TOTORD

           if (SN.EQ.8) then
              WT(i)=0.1570796d0

           else if (SN.EQ.6) then
              WT(i)=0.2617994d0

           else 
              WT(i)=0.5235988d0

           endif
c              write(6,*)OMU(i),OETA(i),OXI(i)
        enddo

        RETURN
        end





