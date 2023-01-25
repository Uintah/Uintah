/*
 * The MIT License
 *
 * Copyright (c) 1997-2019 The University of Utah
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

#ifndef UINTAH_HOMEBREW_VarLabelMatlMemspace_H
#define UINTAH_HOMEBREW_VarLabelMatlMemspace_H

#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Level.h>

namespace Uintah {


    /**************************************

      struct
        VarLabelMatlMemspace

        VarLabel, Material, and Domain

     DESCRIPTION

        The On Demand Data Warehouse needs knowledge of a 4-tuple, the
        VarLabel, Material, Domain, and the Memory Space.  This struct
        supplies that 4 tuple.

      WARNING

      ****************************************/

template<class DomainType, class MemSpace>
struct VarLabelMatlMemspace
{
  VarLabelMatlMemspace(const VarLabel* label, int matlIndex, const DomainType* domain, const MemSpace& memorySpace)
    : label_(label), matlIndex_(matlIndex), domain_(domain), memorySpace_(memorySpace) {}
  VarLabelMatlMemspace(const VarLabelMatlMemspace<DomainType, MemSpace>& copy)
    : label_(copy.label_), matlIndex_(copy.matlIndex_), domain_(copy.domain_), memorySpace_(copy.memorySpace_)
  {}
  VarLabelMatlMemspace<DomainType, MemSpace>& operator=(const VarLabelMatlMemspace<DomainType, MemSpace>& copy)
  {
    label_=copy.label_; matlIndex_=copy.matlIndex_; domain_=copy.domain_; memorySpace_=copy.memorySpace_;
    return *this;
  }

  bool operator<(const VarLabelMatlMemspace<DomainType, MemSpace>& other) const
  {
    if (label_->equals(other.label_)) {
      if (matlIndex_ == other.matlIndex_)
        if ( domain_ == other.domain_)
          return memorySpace_ < other.memorySpace_;
        else
          return domain_ < other.domain_;
      else
        return matlIndex_ < other.matlIndex_;
    }
    else {
      VarLabel::Compare comp;
      return comp(label_, other.label_);
    }
  }

  bool operator==(const VarLabelMatlMemspace<DomainType, MemSpace>& other) const
  {
    return ((label_->equals(other.label_)) && (matlIndex_ == other.matlIndex_) && (domain_ == other.domain_) && (memorySpace_ == other.memorySpace_));
  }

  const VarLabel* label_;
  int matlIndex_;
  const DomainType* domain_;
  const MemSpace memorySpace_;
};

template<class DomainType, class MemSpace>
struct VarLabelMatlMemspaceHasher {
  size_t operator()(const VarLabelMatlMemspace<DomainType, MemSpace>& v) const
  {
    size_t h=0;
    char *str =const_cast<char*> (v.label_->getName().data());
    while (int c = *str++) h = h*7+c;
    return ( ( ((size_t)v.label_) << (sizeof(size_t)/2) ^ ((size_t)v.label_) >> (sizeof(size_t)/2) )
             ^ (size_t)v.domain_ ^ (size_t)v.matlIndex_ );
  }
};

} // End namespace Uintah

#endif
