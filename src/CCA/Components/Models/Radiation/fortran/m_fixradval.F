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
      subroutine m_fixradval(val,
     $     aw, as, ab, 
     $     ae, an, at,
     $     ap, su, const)


c*********************************************************************
c
c
c*********************************************************************
C-----------------------------------------------------------------------
C     WRITTEN BY Gautham, OCTOBER 14, 2002
c
c     Modified, April 29, 2005, by Seshadri Kumar: Added ae, an, and
c     at to work for the case where there is a non-ffield cellType within
c     the domain
C
C     fixval FIXES THE VALUE OF A VARIABLE AT THE VALUE VAL AT THE
C     POSITION (I,J,K) BY SETTING THE OFF-DIAGONALS, AE,AW,AN,AS,AT,AB,
C     OF THAT EQUATION TO ZERO AND THE DIAGONAL, AP, TO THAT VALUE.
C     
C-----------------------------------------------------------------------
      implicit none


c*********************************************************************
c     Arguments :
c*********************************************************************
      double precision VAL
      double precision AW
      double precision AS
      double precision AB
      double precision ae
      double precision an
      double precision at
      double precision AP
      double precision SU
      double precision CONST
c     Local Variables :
c*********************************************************************

c*********************************************************************
c     Start
c*********************************************************************
C-----------------------------------------------------------------------
      SU = CONST*VAL
      AP = CONST
      AW = 0.0d0
      AS = 0.0d0
      AB = 0.0d0
      ae = 0.0d0
      an = 0.0d0
      as = 0.0d0
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
