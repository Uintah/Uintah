
#ifndef UINTAH_HOMEBREW_VarLabelMatlPatchDW_H
#define UINTAH_HOMEBREW_VarLabelMatlPatchDW_H

#include <Packages/Uintah/Core/Grid/VarLabel.h>
#include <Packages/Uintah/Core/Grid/Patch.h>

namespace Uintah {


    /**************************************
      
      struct
        VarLabelMatlPatchDW
      
        VarLabel, Material, and Patch
	
      
      GENERAL INFORMATION
      
        VarLabelMatlPatchDW.h
      
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

  struct VarLabelMatlPatchDW {
    VarLabelMatlPatchDW(const VarLabel* label, int matlIndex,
			const Patch* patch, int dw)
      : label_(label), patch_(patch), matlIndex_(matlIndex), dw_(dw)
    {}
    VarLabelMatlPatchDW(const VarLabelMatlPatchDW& copy)
      : label_(copy.label_), patch_(copy.patch_),
	 matlIndex_(copy.matlIndex_), dw_(copy.dw_)
    {}
    VarLabelMatlPatchDW& operator=(const VarLabelMatlPatchDW& copy)
    {
      label_=copy.label_;
      matlIndex_=copy.matlIndex_;
      patch_=copy.patch_;
      dw_=copy.dw_;
      return *this;
    }
  
    bool operator<(const VarLabelMatlPatchDW& other) const;
    const VarLabel* label_;
    const Patch* patch_;    
    int matlIndex_;
    int dw_;
  };  
} // End namespace Uintah

#endif
