c*********************************************************************
c
c
c*********************************************************************
c GENERAL INFORMATION
c 
c    mmcelltypeInit.F
c 
c    Author: Rajesh Rawat (rawat@crsim.utah.edu)
c    University of Utah
c 
c    Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
c   
c    Copyright (C) 2000 University of Utah
c 
c KEYWORDS
c    Boundary Initialization
c 
c DESCRIPTION
c    Initializes cell type
c   
c WARNING
c    None
c   
c*********************************************************************

#include <Packages/Uintah/CCA/Components/Arches/fortran/mmcelltypeinit_fort.h>
C-----------------------------------------------------------------------
C     THREE-DIMENSIONAL VERSION
C     INITIALIZATION ROUTINE     
C-----------------------------------------------------------------------

c*********************************************************************
c     Locals :
c*********************************************************************
      integer i, j, k
      integer iBeg, iEnd
      integer jBeg, jEnd
      integer kBeg, kEnd

      double precision fcutoff

C-----------------------------------------------------------------------
C     Set up the locals
C-----------------------------------------------------------------------
      iBeg = indexLow(1)
      jBeg = indexLow(2)
      kBeg = indexLow(3)
      iEnd = indexHigh(1)
      jEnd = indexHigh(2)
      kEnd = indexHigh(3)

C-----------------------------------------------------------------------
C     RESET all mmcell type to flowtype
C-----------------------------------------------------------------------
      DO 300 K = kBeg,kEnd
         DO 200 J = jBeg,jEnd
            DO 100 I = iBeg,iEnd
               if (PCELL(I,J,K).eq.MMWALLID)
     $              PCELL(I,J,K) = FLOWCELLID

 100        CONTINUE
 200     CONTINUE
 300  CONTINUE

      DO 310 K = kBeg,kEnd
         DO 210 J = jBeg,jEnd
            DO 110 I = iBeg,iEnd

               if (VOIDFRAC(i,j,k).le.cutoff) then

                  PCELL(I,J,K) = MMWALLID
                  VOIDFRAC(I,J,K) = 0.0D0

               else

                  voidfrac(i,j,k) = 1.0d0

               endif

 110        CONTINUE
 210     CONTINUE
 310  CONTINUE

      RETURN
      END

