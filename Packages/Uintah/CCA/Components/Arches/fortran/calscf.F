c*********************************************************************
c
c
c*********************************************************************

#include <Packages/Uintah/CCA/Components/Arches/fortran/calscf_fort.h>
C-----------------------------------------------------------------------
C     Written by: Seshadri Kumar, ca. August 1996
C     Last modified, November 26, 1997
C      
C     Calculates components of the scaling factor (1D) array SCF
C     corresponding to the (3D) variable PHI.  Currently coded in 
C     to do block scaling, but can be easily modified to do point
C     scaling.
C      
C     EXPLANATION OF SYMBOLS:
C       SCF     1D scaling factor array (o/p)
C       PHI     3D variable array (i/p)
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
C		values of PHI for which values of SCF need to be
C               assigned (i/p)
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
c*********************************************************************
c     Local Variables :
c*********************************************************************
      integer ISTR,JSTR,KSTR,IENDR,JENDR,KENDR
      integer IST,JST,KST,IEND,JEND,KEND
      double precision PHIL1,PHIAVE,PHIMAX
      integer NI,NJ,NK,NIJ,NIJK,IJK,i,j,k

c*********************************************************************
c     Start :
c*********************************************************************
      ISTR = indexLow(1)
      JSTR = indexLow(2)
      KSTR = indexLow(3)
      IENDR = indexHigh(1)
      JENDR = indexHigh(2)
      KENDR = indexHigh(3)
      IST = ISTR + 1
      JST = JSTR + 1
      KST = KSTR + 1
      IEND = IENDR - 1
      JEND = JENDR - 1
      KEND = KENDR - 1
*-----------------------------------------------------------------------
      
C-----Determine Scaling factor norms------------------------------------

      CALL ARRL1(PHIL1, PHIAVE,
     & PASS3A(PHI)
     & PASS3B(PHI),
     & indexLow, indexHigh)
     &     PHI,PHIL1,PHIAVE)
      CALL ARRMAX(indexLow, indexHigh, 
     & PASS3A(PHI)
     & PASS3B(PHI),
     & PHIMAX)
      IF (PHIAVE.LT.SMALL) PHIAVE = ONE
      if (phimax.lt.small) phimax = one
C      PHICUT = 1.0D-5
      
C-----Determine 3D variable range for 1D offsets------------------------
      
      NI = IENDR - ISTR + 1
      NJ = JENDR - JSTR + 1
      NK = KENDR - KSTR + 1
      IF ((NI.LT.(IEND-IST+1)).OR.(NJ.LT.(JEND-JST+1)).OR.
     $     (NK.LT.(KEND-KST+1))) STOP 'Error in calscf; bounds exceeded'
      NIJ = NI*NJ
      NIJK = NIJ*NK
C-----------------------------------------------------------------------
      DO 100 K=KST,KEND
         IK(K) = (K-KSTR)*NIJ
 100  CONTINUE
      DO 200 J=JST,JEND
         IJ(J) = (J-JSTR)*NI
 200  CONTINUE
      
C-----Calculate and assign scaling factor-------------------------------

      DO 500 K=KST,KEND
         DO 400 J=JST,JEND
            DO 300 I=IST,IEND
               IJK = IK(K) + IJ(J) + I-ISTR+1 + IJKST
               SCF(IJK) = PHIMAX
c               SCF(IJK) = PHIAVE
C               IF (ABS(PHI(I,J,K)).LT.PHICUT*PHIMAX) THEN
C                  SCF(IJK) = GREAT
C               ENDIF
 300        CONTINUE
 400     CONTINUE
 500  CONTINUE
      IJKST = IJKST + NIJK
C-----------------------------------------------------------------------
      RETURN
      END

c*********************************************************************
c
c Revision 1.1  2000/04/12 20:31:44  rawat
c modified PressureSolver and added fortran subroutines
c
c
c*********************************************************************

