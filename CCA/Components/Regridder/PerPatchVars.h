
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
#include <Core/Exceptions/InternalError.h>

namespace Uintah {

struct PatchFlag : public RefCounted {
  inline PatchFlag() { flag = false; }
  inline void set() { flag = true; }
  bool flag;
};

// used to indicate which sub patches within a patch will be used to create the next level
struct SubPatchFlag : public RefCounted {
  inline SubPatchFlag(const IntVector& low, const IntVector& high) { 
    initialize(low, high);
  }
  inline SubPatchFlag() {subpatches = 0; delSubpatches = false;}
  inline ~SubPatchFlag() { if (delSubpatches) delete subpatches; }
  
  // initialize with our own memory
  inline void initialize(const IntVector& low, const IntVector& high) {     
    low_ = low; 
    high_ = high;
    range_ = high - low; 
    size_ = range_.x() * range_.y() * range_.z();
    subpatches = scinew int[size_];
    delSubpatches = true;
    for (int i = 0; i < size_; i++)
      subpatches[i] = 0;
  }

  // use external memory
  inline void initialize(const IntVector& low, const IntVector& high, int* data) {
    if (delSubpatches)
      throw InternalError("Memory already allocated for this subpatch flag");
    low_ = low; 
    high_ = high;
    range_ = high - low; 
    size_ = range_.x() * range_.y() * range_.z();
    subpatches = data;

  }
  inline int getIndex(const IntVector& index) const { 
    IntVector tmp(index-low_);
    return tmp.x() * range_.y() * range_.z() + tmp.y() * range_.z() + tmp.z(); 
  }
  inline void set(IntVector& index) { subpatches[getIndex(index)] = 1; }
  inline void clear(IntVector& index) { subpatches[getIndex(index)] = 0;}
  inline const int operator[](const IntVector& idx) const {
    int x = subpatches[getIndex(idx)];
    return x;
  }
  // array
  int *subpatches;
  IntVector low_, high_, range_;
  int size_;

  // whether we allocated the memory ourselves or not.
  bool delSubpatches;
};

typedef Handle<PatchFlag> PatchFlagP;
typedef Handle<SubPatchFlag> SubPatchFlagP;
}

#endif
