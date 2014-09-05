#ifndef UINTAH_HOMEBREW_BNRREGRIDDER_H
#define UINTAH_HOMEBREW_BNRREGRIDDER_H

#include <Packages/Uintah/CCA/Components/Regridder/RegridderCommon.h>

namespace Uintah {

/**************************************

CLASS
   BNRRegridder
   
   Short description...

GENERAL INFORMATION

   BNRRegridder.h

   Bryan Worthen
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   BNRRegridder

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

  //! Takes care of AMR Regridding, using the Berger-Rigoutsos algorithm
  class BNRRegridder : public RegridderCommon {
  public:
    BNRRegridder(const ProcessorGroup* pg);
    virtual ~BNRRegridder();

    //! Create a new Grid
    virtual Grid* regrid(Grid* oldGrid, SchedulerP& sched);
  };

} // End namespace Uintah

#endif
