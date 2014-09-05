
#ifndef UINTAH_HOMEBREW_VarLabelMatlLevel_H
#define UINTAH_HOMEBREW_VarLabelMatlLevel_H

#include <Packages/Uintah/Core/Grid/Variables/VarLabel.h>
#include <Packages/Uintah/Core/Grid/Level.h>

namespace Uintah {


    /**************************************
      
      struct
        VarLabelMatlLevel
      
        VarLabel, Material, and Level
	
      
      GENERAL INFORMATION
      
        VarLabelMatlLevel.h
      
        Wayne Witzel
        Department of Computer Science
        University of Utah
      
        Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
      
        Copyright (C) 2002 SCI Group
      
      KEYWORDS
        VarLabel, Material, Level
      
      DESCRIPTION
        Specifies a VarLabel on a specific level and for a specific material
	with an operator< defined so this can be used as a key in a map.
      
      WARNING
      
      ****************************************/

struct VarLabelMatlLevel {
  VarLabelMatlLevel(const VarLabel* label, int matlIndex, const Level* level)
    : label_(label), matlIndex_(matlIndex), level_(level) {}
  VarLabelMatlLevel(const VarLabelMatlLevel& copy)
    : label_(copy.label_), matlIndex_(copy.matlIndex_), level_(copy.level_)
  {}
  VarLabelMatlLevel& operator=(const VarLabelMatlLevel& copy)
  {
    label_=copy.label_; matlIndex_=copy.matlIndex_; level_=copy.level_;
    return *this;
  }
  
  bool operator<(const VarLabelMatlLevel& other) const;
  const VarLabel* label_;
  int matlIndex_;
  const Level* level_;    
};  

} // End namespace Uintah

#endif
