
#ifndef UINTAH_HOMEBREW_VarLabelMatlDW_H
#define UINTAH_HOMEBREW_VarLabelMatlDW_H

#include <Packages/Uintah/Core/Grid/Variables/VarLabel.h>

namespace Uintah {


    /**************************************
      
      struct
        VarLabelMatlDW
      
        VarLabel, Material, and Domain
	
      
      GENERAL INFORMATION
      
        VarLabelMatlDW.h
      
        Wayne Witzel
        Department of Computer Science
        University of Utah
      
        Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
      
        Copyright (C) 2002 SCI Group
      
      KEYWORDS
        VarLabel, Material, Patch
      
      DESCRIPTION
        Specifies a VarLabel on a specific domain and for a specific material
	with an operator< defined so this can be used as a key in a map.
      
      WARNING
      
      ****************************************/

  template <class DomainType> struct VarLabelMatlDW {
    VarLabelMatlDW(const VarLabel* label, int matlIndex,
			const DomainType* domain, int dw)
      : label_(label), domain_(domain), matlIndex_(matlIndex), dw_(dw)
    {}
    VarLabelMatlDW(const VarLabelMatlDW<DomainType>& copy)
      : label_(copy.label_), domain_(copy.domain_),
	 matlIndex_(copy.matlIndex_), dw_(copy.dw_)
    {}
    VarLabelMatlDW<DomainType>& operator=(const VarLabelMatlDW<DomainType>& copy)
    {
      label_=copy.label_;
      matlIndex_=copy.matlIndex_;
      domain_=copy.domain_;
      dw_=copy.dw_;
      return *this;
    }
  
    bool operator<(const VarLabelMatlDW<DomainType>& other) const {
      if(dw_ < other.dw_)
	return true;
      else if(dw_ > other.dw_)
	return false;
      
      if(matlIndex_ < other.matlIndex_)
	return true;
      else if(matlIndex_ > other.matlIndex_)
	return false;
      
      if(domain_ < other.domain_)
	return true;
      else if(domain_ > other.domain_)
	return false;
      
      VarLabel::Compare comp;
      return comp(label_, other.label_);
           
    };

    const VarLabel* label_;
    const DomainType* domain_;    
    int matlIndex_;
    int dw_;
  };  
} // End namespace Uintah

#endif
