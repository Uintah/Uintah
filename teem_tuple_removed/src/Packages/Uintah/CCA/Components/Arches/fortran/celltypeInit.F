c*********************************************************************
c
c
c*********************************************************************
c GENERAL INFORMATION
c 
c    celltypeInit.F
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

#include <Packages/Uintah/CCA/Components/Arches/fortran/celltypeInit_fort.h>
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
C     INITIALIZE ALL CARDINAL VARIABLES  
C-----------------------------------------------------------------------
      DO 300 K = kBeg,kEnd
         DO 200 J = jBeg,jEnd
            DO 100 I = iBeg,iEnd
               PCELL(I,J,K) = CELLVAL
 100        CONTINUE
 200     CONTINUE
 300  CONTINUE

      RETURN
      END

c*********************************************************************
c
c Revision 1.3  2000/08/04 02:14:37  bbanerje
c Added debug statements.
c
c Revision 1.2  2000/06/28 08:14:55  bbanerje
c Changed the init routines a bit.
c
c Revision 1.1  2000/06/14 23:07:59  bbanerje
c Added celltypeInit.F and sub.mk
c
c
c*********************************************************************
