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
C 
#include <CCA/Components/Arches/Radiation/fortran/rordr_fort.h>
C***********************************************************************
C     SUBROUTINE RORDR (RORDR stands for "Radiation-ORDinates-Rectangular")
C     Generates the table of direction cosines and quadrature weights for 
C     discrete ordinates calculations in rectangular co-ordinates 
C     This is called before the intensity iterations.
C     The following is the Carlson and Lathrop quadrature set.
C     Written by Gautham 12/01.
C***********************************************************************
C      dimension ORD(SN),OXI(SN*(SN+2)),OMU(SN*(SN+2)),
C     &     OETA(SN*(SN+2)),WT(SN*(SN+2))
c*********************************************************************
c     Local Variables :
c*********************************************************************

      integer TOTORD, ORDOCT, i, j, total,count,m
      double precision ORD(SN/2)
      
        TOTORD = SN*(SN+2)
        ORDOCT = SN*(SN+2)/8     

c       Initialize the different values that the direction cosines can take
	
        IF (SN .EQ. 8) THEN
	  ORD(1) = 0.979554d0
	  ORD(2) = 0.804009d0
	  ORD(3) = 0.577350d0
	  ORD(4) = 0.142256d0
        
	ELSE IF (SN .EQ. 6) THEN
	  ORD(1) = 0.965601d0
	  ORD(2) = 0.695051d0
	  ORD(3) = 0.183867d0
	  
	ELSE IF (SN .EQ. 4) THEN
	  ORD(1) = 0.908248d0
	  ORD(2) = 0.295876d0
	
	ELSE
	  ORD(1) = 0.577350d0

	END IF
       
c       Generate values of "Xi" for one octant

        count=1
        do j=SN/2,1,-1
           do i=1,j
           OXI(count)=ORD(j)
           count=count+1
           enddo
        enddo

c       Generate values of "Xi" for all 8 octants

        do i=1,TOTORD
         if (i.le.(TOTORD/2))OXI(i+ORDOCT)= OXI(i)
         if ((i.gt.(TOTORD/2))) OXI(i)= -OXI(i-(TOTORD/2))
        enddo

c       Generate values of "Eta" for one octant

        count=1
        m=1
        do j=1,SN/2
           do i=SN/2,m,-1
              OETA(count)=ORD(i)
              count=count+1
           enddo
           m=m+1
        enddo
        
c     Generate values of "Eta" for all 8 octants

        do i=1,TOTORD
         if (i.le.(7*ORDOCT))OETA(i+ORDOCT)=OETA(i)
         if ((i.gt.(TOTORD/4)).AND.(i.le.(TOTORD/2))) OETA(i)= 
     1   -OETA(i)
         if (i.gt.(3*TOTORD/4))OETA(i)= -OETA(i)
        enddo

c       Generate values of "Mu" for one octant

        count=1
        m=1
        do j=1,SN/2
           do i=m,SN/2
              OMU(count)=ORD(i)
              count=count+1
           enddo
           m=m+1
        enddo

c       Generate values of "Mu" for all 8 octants

        do i=1,TOTORD

           if (i.le.(7*ORDOCT))OMU(i+ORDOCT)=OMU(i)
           if ((i.gt.(ORDOCT)).AND.(i.le.(2*ORDOCT))) 
     1	   OMU(i)= -OMU(i)
           if ((i.gt.(3*ORDOCT)).AND.(i.le.(4*ORDOCT))) 
     1	   OMU(i)= -OMU(i)
           if ((i.gt.(5*ORDOCT)).AND.(i.le.(6*ORDOCT))) 
     1	   OMU(i)= -OMU(i)
           if ((i.gt.(7*ORDOCT)).AND.(i.le.(8*ORDOCT))) 
     1	   OMU(i)= -OMU(i)

        enddo

c       Generate weights "WT"

        do i=1,TOTORD
           if (SN.EQ.2) then
              WT(i)=1.5707963d0

           else if (SN.EQ.4) then
              WT(i)=0.5235987d0

           else if (SN.EQ.6) then
              WT(i)=0.1609517d0

              if ((ABS(OXI(i)).EQ.ORD(2)).OR.
     1  (ABS(OETA(i)).EQ.ORD(2)).OR.
     1  (ABS(OMU(i)).EQ.ORD(2))) WT(i)=0.3626469d0
          
           else if (SN.EQ.8) then
              WT(i)=0.0992284d0

              if ((ABS(OXI(i)).EQ.ORD(1)).OR.
     1  (ABS(OETA(i)).EQ.ORD(1)).OR.
     1  (ABS(OMU(i)).EQ.ORD(1)))
     1  WT(i)=0.1712359d0

              if ((ABS(OXI(i)).EQ.ORD(3)).AND.
     1            (ABS(OETA(i)).EQ.ORD(3)).AND.
     1            (ABS(OMU(i)).EQ.ORD(3))) WT(i)=0.4617179d0

              endif
c              write(6,*)OMU(i),OETA(i),OXI(i),WT(i)
        enddo

        RETURN
        end





