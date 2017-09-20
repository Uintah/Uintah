/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */


#ifndef UINTAH_HOMEBREW_VarLabelMatl_H
#define UINTAH_HOMEBREW_VarLabelMatl_H

#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Level.h>


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
      
      
      KEYWORDS
        VarLabel, Material, Patch
      
      DESCRIPTION

        Specifies a VarLabel on a specific material and a specific
        patch or level and optionally with another domain (data
        warehouse) with an operator< defined so this can be used as a
        key in a map.
      
      WARNING
      
      ****************************************/
template<class DomainType0, class DomainType1 = void> struct VarLabelMatl
{
  VarLabelMatl(const VarLabel* label, int matlIndex, const DomainType0* domain0, const DomainType1* domain1)
    : label_(label), matlIndex_(matlIndex), domain0_(domain0), domain1_(domain1) {}
  VarLabelMatl(const VarLabelMatl<DomainType0, DomainType1>& copy)
    : label_(copy.label_), matlIndex_(copy.matlIndex_), domain0_(copy.domain0_), domain1_(copy.domain1_)
  {}
  VarLabelMatl<DomainType0, DomainType1>& operator=(const VarLabelMatl<DomainType0, DomainType1>& copy)
  {
    label_=copy.label_; matlIndex_=copy.matlIndex_; domain0_=copy.domain0_; domain1_=copy.domain1_;
    return *this;
  }
  
  bool operator<(const VarLabelMatl<DomainType0, DomainType1>& other) const
  {
    if (label_->equals(other.label_)) {
      if (matlIndex_ == other.matlIndex_)
	if ( domain0_ == other.domain0_)
	  return domain1_ < other.domain1_;
	else
	  return domain0_ < other.domain0_;
      else
        return matlIndex_ < other.matlIndex_;
    }
    else {
      VarLabel::Compare comp;
      return comp(label_, other.label_);
    }
  }
  
  bool operator==(const VarLabelMatl<DomainType0, DomainType1>& other) const
  {
    return ((label_->equals(other.label_)) && (matlIndex_ == other.matlIndex_) && (domain0_ == other.domain0_) && (domain1_ == other.domain1_));
  }

  const VarLabel* label_;
  int matlIndex_;
  const DomainType0* domain0_;
  const DomainType1* domain1_;
};  


template<class DomainType> struct VarLabelMatl<DomainType, void>
{
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
  }
  
  bool operator==(const VarLabelMatl<DomainType>& other) const
  {
    return ((label_->equals(other.label_)) && (matlIndex_ == other.matlIndex_) && (domain_ == other.domain_));
  }

  const VarLabel* label_;
  int matlIndex_;
  const DomainType* domain_;
};  


} // End namespace Uintah

#endif
