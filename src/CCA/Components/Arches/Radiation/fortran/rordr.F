C
C The MIT License
C
C Copyright (c) 1997-2020 The University of Utah
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
#include <CCA/Components/Arches/Radiation/fortran/rordr_fort.h>
C***********************************************************************
C     SUBROUTINE RORDR (RORDR stands for "Radiation-ORDinates-Rectangular")
C     Generates the table of direction cosines and quadrature weights for 
C     discrete ordinates calculations in rectangular co-ordinates 
C     This is called before the intensity iterations.
C     The following is the Carlson and Lathrop quadrature set.
C     Written by Gautham 12/01.
C//----------------------------------------------------------------------//   
C     UPDATE: 
C     Added quadrature sets to build weight and ordinance table for SN=16.
C     Used Carlson and Lathrop 1965 while satisfying even moment conditions.
C     Update by Derek 5/14
C***********************************************************************
C      dimension ORD(SN),OXI(SN*(SN+2)),OMU(SN*(SN+2)),
C     &     OETA(SN*(SN+2)),WT(SN*(SN+2))
c*********************************************************************
c     Local Variables :
c*********************************************************************

      integer TOTORD, ORDOCT, i, j, total,count,m
      double precision ORD(SN/2)
      double precision p_i(SN/2)
      
        TOTORD = SN*(SN+2)
        ORDOCT = SN*(SN+2)/8     
c       Initialize the different values that the direction cosines can take

      IF (SN .EQ. 16) THEN
        ORD(1) = 0.9805009d0  ! Direction ordinates in reverse order, in accordance with Gauthams 
        ORD(2) = 0.9092855d0  ! algorithm (reordered to match Carlson and Lathrop down below)
        ORD(3) = 0.8319966d0  ! Due to ordinates being symetrical these values are used several 
        ORD(4) = 0.7467506d0  ! times to generate the desired directions. 
        ORD(5) = 0.6504264d0  
        ORD(6) = 0.5370966d0  ! i.e.  x=ORD(1) y=ORD(8) z=ORD(8)  or  x=ORD(5) y=ORD(4) z=ORD(8)  
        ORD(7) = 0.3922893d0  ! Note that all positive values correspond to the x+ y+ and z+ octant  
        ORD(8) = 0.1389568d0  ! the signs of ORD are changed to generated directions for other octants.
                              ! Only combinations of the ORD that meet the following constraint are considered:
                              !//-----------------       x^2 + y^2 + z^2 =1    -------------------//

        p_i(1) =0.0769489138199670d0  ! Point weights - defined in Carlson and Lathrop 1965 
        p_i(2) =0.0649203838679024d0  ! values are scaled by a factor of pi / 2  
        p_i(3) =0.0333520900883053d0  ! (the solid angle of a single octant)
        p_i(4) =0.0402449014499140d0 
        p_i(5) =0.0566250084660985d0  
        p_i(6) =0.0227119870094947d0 
        p_i(7) =0.0541858759298514d0   
        p_i(8) =0.0133798860320062d0  


      ELSE IF (SN .EQ. 8) THEN
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

      ELSE IF (SN .EQ. 2) THEN
        ORD(1) = 0.577350d0
      ELSE
        PRINT *, "//-----------------ERROR!!!!!!!----------------//"
        PRINT *, "Please set DOM to have 2,4,6,8 or 16 ordinates!"
        PRINT *, "//---------------------------------------------//"
        STOP
      END IF
       
c       Generate values of "Xi" for one octant 

        count=1
        do j=SN/2,1,-1
           do i=1,j
           OXI(count)=ORD(j)
           count=count+1
           enddo
        enddo
c       Generate values of "Xi" for all octants 

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
        
c     Generate values of "Eta" for all octant

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

c       Generate values of "Mu" for all octants 

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
        IF (SN .EQ. 16) THEN  ! Reorder to match ordering in paper
          ORD(8) = 0.9805009d0
          ORD(7) = 0.9092855d0
          ORD(6) = 0.8319966d0
          ORD(5) = 0.7467506d0
          ORD(4) = 0.6504264d0
          ORD(3) = 0.5370966d0
          ORD(2) = 0.3922893d0
          ORD(1) = 0.1389568d0
        END IF

        do i=1,TOTORD
           if (SN.EQ.2) then                ! SN = 2
              WT(i)=1.5707963d0

           else if (SN.EQ.4) then           ! SN = 4
              WT(i)=0.5235987d0

           else if (SN.EQ.6) then           ! SN = 6
              WT(i)=0.1609517d0

              if ((ABS(OXI(i)).EQ.ORD(2)).OR.
     1            (ABS(OETA(i)).EQ.ORD(2)).OR.
     1            (ABS(OMU(i)).EQ.ORD(2))) WT(i)=0.3626469d0
          
           else if (SN.EQ.8) then           ! SN = 8
              WT(i)=0.0992284d0

              if ((ABS(OXI(i)).EQ.ORD(1)).OR.
     1            (ABS(OETA(i)).EQ.ORD(1)).OR.
     1            (ABS(OMU(i)).EQ.ORD(1)))
     1            WT(i)=0.1712359d0

              if ((ABS(OXI(i)).EQ.ORD(3)).AND.
     1            (ABS(OETA(i)).EQ.ORD(3)).AND.
     1            (ABS(OMU(i)).EQ.ORD(3))) WT(i)=0.4617179d0

           else if (SN.EQ.16) then         !  SN = 16

              if (((ABS(OXI(i)).EQ.ORD(3)).OR.       ! 3 per octant
     1            (ABS(OETA(i)).EQ.ORD(3)).OR.
     1            (ABS(OMU(i)).EQ.ORD(3))).AND. 
     1            ((ABS(OXI(i)).EQ.ORD(4)).OR.
     1            (ABS(OETA(i)).EQ.ORD(4)).OR.
     1            (ABS(OMU(i)).EQ.ORD(4)))) then
                  WT(i)=p_i(8)
              else if (((ABS(OXI(i)).EQ.ORD(4)).OR.  ! 3 per octant 
     1            (ABS(OETA(i)).EQ.ORD(4)).OR.
     1            (ABS(OMU(i)).EQ.ORD(4))).AND. 
     1            ((ABS(OXI(i)).EQ.ORD(2)).OR.
     1            (ABS(OETA(i)).EQ.ORD(2)).OR.
     1            (ABS(OMU(i)).EQ.ORD(2)))) then
                  WT(i)=p_i(7)
              else if (((ABS(OXI(i)).EQ.ORD(5)).OR.  ! 6 per octant note: paper reports 4, there is a mistake such that it is 6
     1            (ABS(OETA(i)).EQ.ORD(5)).OR.
     1            (ABS(OMU(i)).EQ.ORD(5))).AND. 
     1            ((ABS(OXI(i)).EQ.ORD(3)).OR.
     1            (ABS(OETA(i)).EQ.ORD(3)).OR.
     1            (ABS(OMU(i)).EQ.ORD(3))).AND. 
     1            ((ABS(OXI(i)).EQ.ORD(2)).OR.
     1            (ABS(OETA(i)).EQ.ORD(2)).OR.
     1            (ABS(OMU(i)).EQ.ORD(2)))) then 
                  WT(i)=p_i(6)
              else if (((ABS(OXI(i)).EQ.ORD(6)).OR.  ! 3 per octant
     1            (ABS(OETA(i)).EQ.ORD(6)).OR.
     1            (ABS(OMU(i)).EQ.ORD(6))).AND. 
     1            ((ABS(OXI(i)).EQ.ORD(2)).OR.
     1            (ABS(OETA(i)).EQ.ORD(2)).OR.
     1            (ABS(OMU(i)).EQ.ORD(2)))) then
                   WT(i)=p_i(5)
              else if (((ABS(OXI(i)).EQ.ORD(5)).OR.  ! 6 per octant
     1            (ABS(OETA(i)).EQ.ORD(5)).OR.
     1            (ABS(OMU(i)).EQ.ORD(5))).AND. 
     1            ((ABS(OXI(i)).EQ.ORD(4)).OR.
     1            (ABS(OETA(i)).EQ.ORD(4)).OR.
     1            (ABS(OMU(i)).EQ.ORD(4))).AND. 
     1            ((ABS(OXI(i)).EQ.ORD(1)).OR.
     1            (ABS(OETA(i)).EQ.ORD(1)).OR.
     1            (ABS(OMU(i)).EQ.ORD(1)))) then
                   WT(i)=p_i(4)
             else if (((ABS(OXI(i)).EQ.ORD(6)).OR.  ! 6 per octant
     1            (ABS(OETA(i)).EQ.ORD(6)).OR.
     1            (ABS(OMU(i)).EQ.ORD(6))).AND. 
     1            ((ABS(OXI(i)).EQ.ORD(3)).OR.
     1            (ABS(OETA(i)).EQ.ORD(3)).OR.
     1            (ABS(OMU(i)).EQ.ORD(3))).AND. 
     1            ((ABS(OXI(i)).EQ.ORD(1)).OR.
     1            (ABS(OETA(i)).EQ.ORD(1)).OR.
     1            (ABS(OMU(i)).EQ.ORD(1)))) then
                  WT(i)=p_i(3)
              else if (((ABS(OXI(i)).EQ.ORD(7)).OR.  ! 6 per octant
     1            (ABS(OETA(i)).EQ.ORD(7)).OR.
     1            (ABS(OMU(i)).EQ.ORD(7)))) then 
                  WT(i)=p_i(2)
              else if ((ABS(OXI(i)).EQ.ORD(8)).OR.   ! 3 per octant
     1            (ABS(OETA(i)).EQ.ORD(8)).OR.
     1            (ABS(OMU(i)).EQ.ORD(8))) then 
                  WT(i)=p_i(1)
              else
        PRINT *, "//--ERROR! unsassigned direction (WT set to zero)--//"
              endif   ! End of if statement  SN = 16

              endif   ! End of if statement  SN = 2

        enddo         ! End of do statment i=1,TOTORD

        RETURN
        end





