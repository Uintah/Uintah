#include <Packages/Uintah/CCA/Components/Arches/Radiation/fortran/rordrtn_fort.h>
C***********************************************************************
C     SUBROUTINE RORDR (RORDR stands for "Radiation-ORDinates-Rectangular")
C     Generates the table of direction cosines and quadrature weights for 
C     discrete ordinates calculations in rectangular co-ordinates 
C     This is called before the intensity iterations.
C     The following is the "TN" quadrature set, Journal of Heat Transfer,
C     Vol.120, p.514
C     Written by Gautham 01/03.
C***********************************************************************
C      dimension ORD(SN),OXI(SN*(SN+2)),OMU(SN*(SN+2)),
C     &     OETA(SN*(SN+2)),WT(SN*(SN+2))
c*********************************************************************
c     Local Variables :
c*********************************************************************

      integer i

C     Open different files and read different arrays

      IF (SN.EQ.2) THEN
         OPEN (UNIT=15,FILE='TNSETS',STATUS='OLD')
         READ(15,*)(OMU(I),OETA(I),OXI(I),WT(I),I=1,32)
         CLOSE (UNIT=15)
      ELSE IF (SN.EQ.3) THEN
         OPEN (UNIT=15,FILE='TNSETS',STATUS='OLD')
         READ(15,10)
 10      FORMAT(32(/))
         READ(15,*)(OMU(I),OETA(I),OXI(I),WT(I),I=1,72)
         CLOSE (UNIT=15)
      ELSE 
         OPEN (UNIT=15,FILE='TNSETS',STATUS='OLD')
         READ(15,20)
 20      FORMAT(105(/))
         READ(15,*)(OMU(I),OETA(I),OXI(I),WT(I),I=1,128)
         CLOSE (UNIT=15)
      ENDIF

c      do i=1,128
c              write(6,*)OMU(i),OETA(i),OXI(i)
c      enddo

        RETURN
        end





