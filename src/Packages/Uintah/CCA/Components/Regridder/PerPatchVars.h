
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
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Core/Geometry/IntVector.h>

namespace Uintah {

struct PatchFlag : public RefCounted {
  inline PatchFlag() { flag = false; }
  inline void set() { flag = true; }
  bool flag;
};

// used to indicate which sub patches within a patch will be used to create the next level
struct SubPatchFlag : public RefCounted {
  inline SubPatchFlag(const IntVector& low, const IntVector& high) { rewindow(low, high); }
  inline SubPatchFlag() {};
  inline void rewindow(const IntVector& low, const IntVector& high) {     
    subpatches.rewindow(low, high);
    subpatches.initialize(0);
  }
  inline void set(IntVector& index) { subpatches[index] = 1;}
  inline void clear(IntVector& index) { subpatches[index] = 0;}
  inline IntVector getLowIndex() { return subpatches.getLowIndex(); }
  inline IntVector getHighIndex() { return subpatches.getHighIndex(); }
  inline const int operator[](const IntVector& idx) const {
    int x = subpatches[idx];
    return x;
  }

  CCVariable<int> subpatches;
};

typedef Handle<PatchFlag> PatchFlagP;
typedef Handle<SubPatchFlag> SubPatchFlagP;
}

#endif
