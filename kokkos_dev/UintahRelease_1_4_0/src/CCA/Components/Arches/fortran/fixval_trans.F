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
c*********************************************************************
c
c
c*********************************************************************

      SUBROUTINE fixval_trans(VAL,
     &     AE,AW,AN,AS,AT,AB,SU,SP,
     &     CONST)
C-----------------------------------------------------------------------
C     WRITTEN BY SESHADRI KUMAR, OCTOBER 14, 1996
C
C     fixval FIXES THE VALUE OF A VARIABLE AT THE VALUE VAL AT THE
C     POSITION (I,J,K) BY SETTING THE OFF-DIAGONALS, AE,AW,AN,AS,AT,AB,
C     OF THAT EQUATION TO ZERO AND THE DIAGONAL, AP, TO THAT VALUE.
C     
C-----------------------------------------------------------------------
      implicit none

#include "param4.h"

c*********************************************************************
c     Arguments :
c*********************************************************************
      double precision VAL
      double precision AE
      double precision AW
      double precision AN
      double precision AS
      double precision AT
      double precision AB
      double precision SU
      double precision SP
      double precision CONST

c*********************************************************************
c     Local Variables :
c*********************************************************************

c*********************************************************************
c     Start
c*********************************************************************
C-----------------------------------------------------------------------
      SU = CONST*VAL
      SP = ZERO
      AE = ZERO
      AW = ZERO
      AN = ZERO
      AS = ZERO
      AT = ZERO
      AB = ZERO
C-----------------------------------------------------------------------
      RETURN
      END

c*********************************************************************
c
c Revision 1.2  2000/07/14 03:45:48  rawat
c completed velocity bc and fixed some bugs
c
c Revision 1.1  2000/04/12 20:31:45  rawat
c modified PressureSolver and added fortran subroutines
c
c
c*********************************************************************
