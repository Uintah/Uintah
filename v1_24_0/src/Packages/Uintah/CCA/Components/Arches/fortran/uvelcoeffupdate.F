c
c $Id $
c 

#include <Packages/Uintah/CCA/Components/Arches/fortran/uvelcoeffupdate_fort.h>
#include "param4.h"
C--------------------------------------------------------------------
C     WRITTEN BY SESHADRI KUMAR, OCTOBER 14, 1996
*     last modified, September 15, 1998
C
C     THIS SUBROUTINE CALCULATES THE MASS SOURCE TERM DUE TO 
C     CONTINUITY AND THE USE OF THE CONSERVATIVE FORM IN THE 
C     PARTIAL DIFFERENTIAL EQUATION
C
C     PASSED VARIABLES:
C
C	PHI	Variable for which equation is being solved (i/p)
C	AE,AW,AN,AS,AT,AB,AP
C		Coefficients of the seven-diagonal finite difference
C		equation (o/p)
C	SU	Nonlinear source term, array (i/p and o/p)
C	SP 	Linear portion of the source term, array (i/p and o/p)
*       spm     external mass source for phase under consideration (i/p)
*       sign    sign of the mass source: one means it is a source, -one
*               means it is a sink.  This is useful in a two-phase
*               Eulerian calculation, wherein we may have the same spm
*               term in both phases, but with opposite sign (i/p)
C       TCONS   Determines whether (TCONS=1) or not (TCONS=0)
C               the conservative form of the PHI-equation is being
C               solved for (i/p)
C	valid_lo
C		Starting indices for the calculation (i/p)
C	valid_hi
C		Ending indices for calculation (i/p)
C	CESAV,CWSAV,CNSAV,CSSAV,CTSAV,CBSAV
C		Convection fluxes on the east,west,north,south,top,
C		and bottom faces of the P-cell (i/p)
C
C--------------------------------------------------------------------

*
*     declare local variables
*


      integer i
      integer j
      integer k


      do 520 k = valid_lo(3), valid_hi(3)
         do 510 j = valid_lo(2), valid_hi(2)
            do 500 i = valid_lo(1), valid_hi(1)
               AE(I,J,K) = AE(i,j,k)-cesav(i,j,k)*0.5D0
               AW(I,J,K) = AW(I,J,K)+CWSAV(I,J,K)*0.5D0
               AN(I,J,K) = AN(I,J,K)-CNSAV(I,J,K)*0.5D0
               AS(I,J,K) = AS(I,J,K)+CSSAV(I,J,K)*0.5D0
               AT(I,J,K) = AT(I,J,K)-CTSAV(I,J,K)*0.5D0
               AB(I,J,K) = AB(I,J,K)+CBSAV(I,J,K)*0.5D0
               AP(I,J,K) = AP(I,J,K)+0.5D0*(CESAV(I,J,K)-CWSAV(I,J,K)+
     $                  CNSAV(I,J,K)-CSSAV(I,J,K)+
     $                  CTSAV(I,J,K)-CBSAV(I,J,K))

 500        continue
 510     continue
 520  continue

c--------------------------------------------------------------------
      return
      end

c
c $Log $
c 
