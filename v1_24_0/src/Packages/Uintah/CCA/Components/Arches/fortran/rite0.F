c*********************************************************************
c
c
c*********************************************************************

#if 1
#include <Packages/Uintah/CCA/Components/Arches/fortran/rite0_fort.h>
#else
      SUBROUTINE rite0(X,Y,NVAR,IJKST,
     $     IST,JST,KST,IEND,JEND,KEND,
     $     ISTR,JSTR,KSTR,IENDR,JENDR,KENDR, IJ, IK)
C-----------------------------------------------------------------------
C     Written by: Seshadri Kumar, April 1995
C     Last modified, November 26, 1997
C      
C     Writes values into the 3D array Y from the 1D vector X
C      
C     EXPLANATION OF SYMBOLS:
C       X       1D array (i/p)
C       Y       3D array (o/p)
C	NVAR	Total number of variables (total number of points
C		in the domain times number of equations) used in
C		nonlinear GMRES calculations (i/p)
C       IJ,IK   Temporary arrays to calculate J- and K-offsets
C               while doing 3D-1D or 1D-3D operations
C       IJKST   Starting location from which to access data in 
C               the 1D array when performing 1D-3D or 3D-1D operations
C               (i/p) (o/p)
C	IST,JST,KST,IEND,JEND,KEND 
C		Locations of the starting and ending indices for the
C		values of Y that need to be written into (i/p)
C	ISTR,JSTR,KSTR,IENDR,JENDR,KENDR 
C		Locations of the starting and ending indices for
C		variable X: these define the range of values in each
C               direction that need to be considered while calculating
C               offsets for storing 3D vectors in 1D (i/p)
C               Note: these indices must always be greater than or
C               equal to the write indices, otherwise the wrong 
C               variables will be accessed and there can be bounds
C               violations.  When they are greater than
C               the write indices, then the extra space is padded by
C               whatever was in array X before this operation.
*-----------------------------------------------------------------------
      implicit none

*-----------------------------------------------------------------------
*     passed variables

      double precision
     $     Y(ISTR:IENDR,JSTR:JENDR,KSTR:KENDR),X(*)
      integer nvar,ijkst,ist,jst,kst,iend,jend,kend,
     $     istr,jstr,kstr,iendr,jendr,kendr
      integer IJ(JSTR:JENDR), IK(KSTR:KENDR)
*-----------------------------------------------------------------------
*     local variables

      integer i,j,k,ni,nj,nk,nij,nijk,ijk
C-----------------------------------------------------------------------
C     EXECUTABLE STATEMENTS
C-----------------------------------------------------------------------
      NI = IENDR - ISTR + 1
      NJ = JENDR - JSTR + 1
      NK = KENDR - KSTR + 1
      IF ((NI.LT.(IEND-IST+1)).OR.(NJ.LT.(JEND-JST+1)).OR.
     $     (NK.LT.(KEND-KST+1))) STOP 'Error in rite0; bounds exceeded'
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
      DO 600 K=KST,KEND
         DO 500 J=JST,JEND
            DO 400 I=IST,IEND
               IJK = I-ISTR+1 + IJ(J) + IK(K) + IJKST
               Y(I,J,K) = X(IJK)
 400       CONTINUE
 500    CONTINUE
 600  CONTINUE
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

