c*********************************************************************
c
c
c*********************************************************************

#if 1
#include <Packages/Uintah/CCA/Components/Arches/fortran/fncd_fort.h>
#else
      SUBROUTINE fncd(X,FX,DFX,RLAMB1,RLAMB2)

      implicit none

#include "param4.h"
c
c     based on Jenkins and Savage (1983) model (D)
c
c      FX = X**3*(RLAMB1-ONE)+X**2*(TWO-THREE*RLAMB1)+
c      THREE*RLAMB1*X-RLAMB1
c      DFX = THREE*X**2*(RLAMB1-ONE)+TWO*X*(TWO-THREE*RLAMB1)+
c      THREE*RLAMB1
c
c     based on Jenkins and Savage expression for FP but using a different
c     form of g0(nu) [ g0(nu) = (ONE - (nu/numax)**(1/3))**(-1)] (B)
c
c      FX = RLAMB1*X**2 + (1.5537601*X)**(ONE/THREE) - ONE
c      DFX = TWO*RLAMB1*X + (1.5537601/THREE)*(1.5537601*X)**
c      (-TWO/THREE)
c
c     based on Ding and Gidaspow (1990) model (C)
c

c*********************************************************************
c     Arguments :
c*********************************************************************
      double precision X, FX, DFX, RLAMB1, RLAMB2

c*********************************************************************
c     Locals :
c*********************************************************************
      double precision RL3

c*********************************************************************
c     Start :
c*********************************************************************
      RL3 = 1.5537601D0**(ONE/THREE)
      FX = RLAMB2*X**2 - RL3*X**(FOUR/THREE) + X + 
     &     RLAMB1*RL3*X**(ONE/THREE) - RLAMB1
      DFX = TWO*RLAMB2*X - RL3*(FOUR/THREE)*X**(ONE/THREE) + 
     &      ONE + RLAMB1*RL3*X**(-TWO/THREE)/THREE
c
c     based on Ding and Gidaspow (1990) expression for FP but using 
c     Jenkins and Savage's (1983) expression for g0(nu) (A)
c
c      FX = TWO*X**4-(SIX-RLAMB2+TWO*RLAMB1)*X**3+(SIX*RLAMB1-
c     &     TWO*RLAMB2+SIX)*X**2-(SIX*RLAMB1+TWO)*X+TWO*RLAMB1
c      DFX = 8.*X**3-THREE*(SIX-RLAMB2+TWO*RLAMB1)*X**2+TWO*
c     &      (SIX*RLAMB1+SIX-TWO*RLAMB2)*X-(SIX*RLAMB1+TWO)
c
      RETURN
      END

c*********************************************************************
c
c Revision 1.1  2000/04/12 20:31:45  rawat
c modified PressureSolver and added fortran subroutines
c
c
c*********************************************************************
