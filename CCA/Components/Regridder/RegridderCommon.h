#ifndef UINTAH_HOMEBREW_REGRIDDERCOMMON_H
#define UINTAH_HOMEBREW_REGRIDDERCOMMON_H

#include <Packages/Uintah/CCA/Ports/Regridder.h>
#include <Packages/Uintah/Core/Parallel/UintahParallelComponent.h>
#include <Core/Geometry/IntVector.h>
#include <vector>

using std::vector;
using namespace SCIRun;

namespace Uintah {

/**************************************

CLASS
   RegridderCommon
   
   Short description...

GENERAL INFORMATION

   RegridderCommon.h

   Bryan Worthen
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   RegridderCommon

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

  class ProcessorGroup;

  //! Takes care of AMR Regridding.  Parent class which takes care
  //! of common regridding functionality.
  class RegridderCommon : public Regridder, public UintahParallelComponent {
  public:
    RegridderCommon(ProcessorGroup* pg);
    virtual ~RegridderCommon();

    //! Initialize with regridding parameters from ups file
    virtual void problemSetup(const ProblemSpecP& params);

    //! Asks if we need to recompile the task graph.
    //! Will return true if we did a regrid
    virtual bool needRecompile(double time, double delt,
			       const GridP& grid);

  private:
    IntVector d_minPatchSize;
    IntVector d_maxPatchSize;
    int d_safetyLayers; // ?
    vector<int> d_dividingLattice;

    bool newGrid;
  };

} // End namespace Uintah

#endif
