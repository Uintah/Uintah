
#ifndef UINTAH_HOMEBREW_VarLabelMatl_H
#define UINTAH_HOMEBREW_VarLabelMatl_H

#include <Packages/Uintah/Core/Grid/Variables/VarLabel.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/Level.h>

namespace Uintah {


    /**************************************
      
      struct
        VarLabelMatl
      
        VarLabel, Material, and Domain
	
      
      GENERAL INFORMATION
      
        VarLabelMatl.h
      
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

template<class DomainType> struct VarLabelMatl {
  VarLabelMatl(const VarLabel* label, int matlIndex, const DomainType* domain)
    : label_(label), matlIndex_(matlIndex), domain_(domain) {}
  VarLabelMatl(const VarLabelMatl<DomainType>& copy)
    : label_(copy.label_), matlIndex_(copy.matlIndex_), domain_(copy.domain_)
  {}
  VarLabelMatl<DomainType>& operator=(const VarLabelMatl<DomainType>& copy)
  {
    label_=copy.label_; matlIndex_=copy.matlIndex_; domain_=copy.domain_;
    return *this;
  }
  
  bool operator<(const VarLabelMatl<DomainType>& other) const
  {
    if (label_->equals(other.label_)) {
      if (matlIndex_ == other.matlIndex_)
	return domain_ < other.domain_;
      else
	return matlIndex_ < other.matlIndex_;
    }
    else {
      VarLabel::Compare comp;
      return comp(label_, other.label_);
    }
  };
 
  const VarLabel* label_;
  int matlIndex_;
  const DomainType* domain_;    
};  

} // End namespace Uintah

#endif
