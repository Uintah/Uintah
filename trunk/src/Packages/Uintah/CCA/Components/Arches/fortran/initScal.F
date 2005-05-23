c*********************************************************************
c
c
c*********************************************************************
c GENERAL INFORMATION
c 
c    InitScal.F
c 
c    Author: Biswajit Banerjee (bbanerje@crsim.utah.edu)
c    University of Utah
c 
c    Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
c   
c    Copyright (C) 2000 University of Utah
c 
c KEYWORDS
c    Arches Initialization
c 
c DESCRIPTION
c    Initializes scalars
c   
c WARNING
c    Other variables initialized in init.F
c   
c*********************************************************************

#include <Packages/Uintah/CCA/Components/Arches/fortran/initScal_fort.h>
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

c*********************************************************************
c     Get the indices of interest
c*********************************************************************
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
               SCALAR(I,J,K) = SVAL
 100        CONTINUE
 200     CONTINUE
 300  CONTINUE

      RETURN
      END

c*********************************************************************
c
c Revision 1.3  2000/09/25 16:29:24  rawat
c modified requires in PressureSolver for multiple patches
c
c Revision 1.2  2000/06/28 08:14:55  bbanerje
c Changed the init routines a bit.
c
c Revision 1.1  2000/06/13 19:52:58  bbanerje
c Initialize scalars
c
c
c*********************************************************************
