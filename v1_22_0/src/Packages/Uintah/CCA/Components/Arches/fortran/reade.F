c*********************************************************************
c
c
c*********************************************************************

#if 1
#include <Packages/Uintah/CCA/Components/Arches/fortran/reade_fort.h>
#else
      SUBROUTINE reade(X,Y,NVAR,IJKST,
     $     IST,JST,KST,IEND,JEND,KEND,
     $     ISTR,JSTR,KSTR,IENDR,JENDR,KENDR,
     $     IJ, IK)
C-----------------------------------------------------------------------
C     Reads values from a 3D array Y into one consolidated 1D vector X
C     Written by Seshadri Kumar, April 1995
C     Last modified: Nov. 26, 1997
C      
C     EXPLANATION OF SYMBOLS:
C       X       1D array (o/p)
C       Y       3D array (i/p)
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
C		values of Y that need to be read into X
C	ISTR,JSTR,KSTR,IENDR,JENDR,KENDR (i/p)
C		Locations of the starting and ending indices for
C		variable X: these define the range of values in each
C               direction that need to be considered while calculating
C               offsets for storing 3D vectors in 1D.
C               Note: these indices must always be greater than or
C               equal to the read indices, otherwise the wrong 
C               variables will be accessed and there can be bounds
C               violations.  When they are greater than
C               the read indices, then the extra space is padded by
C               whatever was in array X before this operation.
C-----------------------------------------------------------------------
      implicit none

*-----------------------------------------------------------------------
*     passed variables

      double precision
     $     Y(ISTR:IENDR,JSTR:JENDR,KSTR:KENDR),X(*)
      integer nvar,ijkst,ist,jst,kst,iend,jend,kend,
     $     istr,jstr,kstr,iendr,jendr,kendr
      integer IK(KSTR:KENDR),IJ(JSTR:JENDR)
*-----------------------------------------------------------------------
*     local variables

      integer ni,nj,nk,nij,nijk,i,j,k,ijk
*-----------------------------------------------------------------------
*     EXECUTABLE STATEMENTS
*-----------------------------------------------------------------------
      NI = IENDR - ISTR + 1
      NJ = JENDR - JSTR + 1
      NK = KENDR - KSTR + 1
      IF ((NI.LT.(IEND-IST+1)).OR.(NJ.LT.(JEND-JST+1)).OR.
     $     (NK.LT.(KEND-KST+1))) STOP 'Error in reade; bounds exceeded'
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
      DO 3000 K=KST,KEND
         DO 2000 J=JST,JEND
            DO 1000 I=IST,IEND
               IJK = I-ISTR+1 + IJ(J) + IK(K) + IJKST
               X(IJK) = Y(I,J,K)
 1000       CONTINUE
 2000    CONTINUE
 3000 CONTINUE
      IJKST = IJKST + NIJK
C-----------------------------------------------------------------------
      RETURN
      END

c*********************************************************************
c
c $Log $
c
c*********************************************************************

