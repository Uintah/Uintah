
#ifndef UINTAH_HOMEBREW_VarLabelMatlPatch_H
#define UINTAH_HOMEBREW_VarLabelMatlPatch_H

#include <Packages/Uintah/Core/Grid/VarLabel.h>
#include <Packages/Uintah/Core/Grid/Patch.h>

namespace Uintah {


    /**************************************
      
      struct
        VarLabelMatlPatch
      
        VarLabel, Material, and Patch
	
      
      GENERAL INFORMATION
      
        VarLabelMatlPatch.h
      
        Wayne Witzel
        Department of Computer Science
        University of Utah
      
        Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
      
        Copyright (C) 2002 SCI Group
      
      KEYWORDS
        VarLabel, Material, Patch
      
      DESCRIPTION
        Specifies a VarLabel on a specific patch and for a specific material
	with an operator< defined so this can be used as a key in a map.
      
      WARNING
      
      ****************************************/

struct VarLabelMatlPatch {
  VarLabelMatlPatch(const VarLabel* label, int matlIndex, const Patch* patch)
    : label_(label), matlIndex_(matlIndex), patch_(patch) {}
  VarLabelMatlPatch(const VarLabelMatlPatch& copy)
    : label_(copy.label_), matlIndex_(copy.matlIndex_), patch_(copy.patch_)
  {}
  VarLabelMatlPatch& operator=(const VarLabelMatlPatch& copy)
  {
    label_=copy.label_; matlIndex_=copy.matlIndex_; patch_=copy.patch_;
    return *this;
  }
  
  bool operator<(const VarLabelMatlPatch& other) const;
  const VarLabel* label_;
  int matlIndex_;
  const Patch* patch_;    
};  

} // End namespace Uintah

#endif
