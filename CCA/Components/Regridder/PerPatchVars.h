
#ifndef Uintah_Component_Regridder_PerPatchVars_h
#define Uintah_Component_Regridder_PerPatchVars_h

/**************************************

   Bryan Worthen
   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
   University of Utah
  
   Copyright (C) 2000 University of Utah

DESCRIPTION
   Some structs to help the regridder on a per-patch basis
  
WARNING
  
****************************************/

#include <Packages/Uintah/Core/ProblemSpec/RefCounted.h>
#include <Packages/Uintah/Core/ProblemSpec/Handle.h>

namespace Uintah {

struct PatchFlag : public RefCounted {
  inline PatchFlag() { flag = false; }
  inline void set() { flag = true; }
  bool flag;
};

typedef Handle<PatchFlag> PatchFlagP;

}

#endif
