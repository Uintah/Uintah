
#ifndef UINTAH_HOMEBREW_VarLabelLevelDW_H
#define UINTAH_HOMEBREW_VarLabelLevelDW_H

#include <Packages/Uintah/Core/Grid/VarLabel.h>
#include <Packages/Uintah/Core/Grid/Patch.h>

namespace Uintah {


    /**************************************
      
      struct
        VarLabelLevelDW
      
        VarLabel, Material, and Patch
	
      
      GENERAL INFORMATION
      
        VarLabelLevelDW.h
      
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

struct VarLabelLevelDW {
  VarLabelLevelDW(const VarLabel* label, const Level* level, int dw)
    : label_(label), level_(level), dw_(dw)
  {}
  VarLabelLevelDW(const VarLabelLevelDW& copy)
    : label_(copy.label_), level_(copy.level_), dw_(copy.dw_)
  {}
  VarLabelLevelDW& operator=(const VarLabelLevelDW& copy)
  {
    label_=copy.label_; level_=copy.level_; dw_=copy.dw_;
    return *this;
  }
  
  bool operator<(const VarLabelLevelDW& other) const;
  const VarLabel* label_;
  const Level* level_;
  int dw_;
};  

} // End namespace Uintah

#endif
