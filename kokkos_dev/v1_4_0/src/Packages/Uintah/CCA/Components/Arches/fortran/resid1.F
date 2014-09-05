c*********************************************************************
c
c
c*********************************************************************

#if 1
#include <Packages/Uintah/CCA/Components/Arches/fortran/resid1_fort.h>
#else
      SUBROUTINE resid1(R0,X,Y,SCF,NVAR,IJKST,
     $     IST,JST,KST,IEND,JEND,KEND,
     $     ISTR,JSTR,KSTR,IENDR,JENDR,KENDR,IJ,IK)
C-----------------------------------------------------------------------
C     Written by: Seshadri Kumar, April 1995
C     Last modified, December 1, 1997
C      
C     Calculates the residual vector in 1D as the difference between
C     the 1D vector X and its corresponding 3D counterpart Y:
C     used for u-M(u) computations, where u is stored in a 1D vector
C     function X and M(u) is in grid functions Y.  The vector Y is
C     typically of dimension NIxNJxNK, being a grid function.  The vector
C     X is a vector function that stores a number of grid functions Y.
C     There is thus the need for an offset location IJKST which indicates 
C     how much offset should be used in the 1D vector (to account for 
C     previous grid functions stored in X and correspondingly R0) 
C     before accessing values of X for comparison with Y.
C      
C     R0(IJK) = (X(IJK) - Y(I,J,K))*(FAC/SCF(IJK))
C      
C     EXPLANATION OF SYMBOLS:
C       R0      1D residual array (o/p)
C       X       1D vector function array (i/p)
C       Y       3D grid function array (i/p)
C	SCF	Scaling Factor Matrix for the entire equation system
C		(i/p)
C	NVAR	Total number of variables (total number of points
C		in the domain times number of equations) used in
C		nonlinear GMRES calculations (i/p)
C       IJ,IK   Temporary arrays to calculate J- and K-offsets
C               while doing 3D-1D or 1D-3D operations
C       IJKST   Starting location from which to access data in 
C               the 1D array when performing 1D-3D or 3D-1D operations
C               (i/p) (o/p)
C	IST,JST,KST,IEND,JEND,KEND (i/p)
C		Locations of the starting and ending indices for the
C		values of Y that need to be written into.
C	ISTR,JSTR,KSTR,IENDR,JENDR,KENDR (i/p)
C		Locations of the starting and ending indices for
C		variable X: these define the range of values in each
C               direction that need to be considered while calculating
C               offsets for storing 3D vectors in 1D.
C               Note: these indices must always be greater than or
C               equal to the write indices, otherwise the wrong 
C               variables will be accessed and there can be bounds
C               violations.  When they are greater than
C               the write indices, then the extra space is padded by
C               whatever was in array X before this operation.
*-----------------------------------------------------------------------
      implicit none

#include "param4.h"

*-----------------------------------------------------------------------
*     passed variables

      integer nvar,ijkst,ist,jst,kst,iend,jend,kend,istr,jstr,kstr,
     $     iendr,jendr,kendr
      double precision 
     $     Y(ISTR:IENDR,JSTR:JENDR,KSTR:KENDR),X(*),R0(*),SCF(*)
      integer IJ(JSTR:JENDR), IK(KSTR:KENDR)
*-----------------------------------------------------------------------
*     local variables

      double precision fac
      integer ni,nj,nk,nij,nijk,i,j,k,ijk
C-----------------------------------------------------------------------
C     EXECUTABLE STATEMENTS
C-----------------------------------------------------------------------
      NI = IENDR - ISTR + 1
      NJ = JENDR - JSTR + 1
      NK = KENDR - KSTR + 1
      IF ((NI.LT.(IEND-IST+1)).OR.(NJ.LT.(JEND-JST+1)).OR.
     $     (NK.LT.(KEND-KST+1))) STOP 'Error in resid1; bounds exceeded'
      NIJ = NI*NJ
      NIJK = NIJ*NK
C-----------------------------------------------------------------------
      DO 100 K=KST,KEND
         IK(K) = (K-KSTR)*NIJ
 100  CONTINUE
      DO 200 J=JST,JEND
         IJ(J) = (J-JSTR)*NI
 200  CONTINUE
C-----------------------------------------------------------------------
      FAC = ONE
C-----------------------------------------------------------------------
      DO 1200 K=KST,KEND
         DO 1100 J=JST,JEND
            DO 1000 I=IST,IEND
               IJK = I-ISTR+1 + IJ(J) + IK(K) + IJKST
               R0(IJK) = FAC*(X(IJK) - Y(I,J,K))
               R0(IJK) = R0(IJK)/SCF(IJK)
 1000       CONTINUE
 1100    CONTINUE
 1200 CONTINUE
      IJKST = IJKST + NIJK
C-----------------------------------------------------------------------
      RETURN
      END      

c*********************************************************************
c
c Revision 1.1  2000/04/12 20:31:50  rawat
c modified PressureSolver and added fortran subroutines
c
c
c*********************************************************************

