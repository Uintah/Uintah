/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
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


#ifndef Uintah_Component_Regridder_PerPatchVars_h
#define Uintah_Component_Regridder_PerPatchVars_h

/**************************************

   Bryan Worthen
   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
   University of Utah
  
   
DESCRIPTION
   Some structs to help the regridder on a per-patch basis
  
WARNING
  
****************************************/

#include <Core/Util/RefCounted.h>
#include <Core/Util/Handle.h>
#include <Core/Grid/Variables/CCVariable.h>
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
  inline SubPatchFlag() {subpatches_ = 0; delSubpatches_ = false;}
  SubPatchFlag( const SubPatchFlag& ) { throw InternalError( "SubPatchFlag( const SubPatchFlag& ) not implemented!", __FILE__, __LINE__ ); }
  SubPatchFlag& operator=( const SubPatchFlag& ) { throw InternalError( "SubPatchFlag& operator=( const SubPatchFlag& ) not implemented!", __FILE__, __LINE__ );}
  inline ~SubPatchFlag() { if (delSubpatches_) delete subpatches_; }
  
  // initialize with our own memory
  inline void initialize(const IntVector& low, const IntVector& high) {     
    low_ = low; 
    high_ = high;
    range_ = high - low; 
    size_ = range_.x() * range_.y() * range_.z();
    subpatches_ = scinew int[size_];
    delSubpatches_ = true;
    for (int i = 0; i < size_; i++)
      subpatches_[i] = 0;
  }

  // use external memory
  inline void initialize(const IntVector& low, const IntVector& high, int* data) {
    if (delSubpatches_)
      throw InternalError("Memory already allocated for this subpatch flag", __FILE__, __LINE__);
    low_ = low; 
    high_ = high;
    range_ = high - low; 
    size_ = range_.x() * range_.y() * range_.z();
    subpatches_ = data;

  }
  inline int getIndex(const IntVector& index) const { 
    IntVector tmp(index-low_);
    return tmp.x() * range_.y() * range_.z() + tmp.y() * range_.z() + tmp.z(); 
  }
  inline void set(IntVector& index) { subpatches_[getIndex(index)] = 1; }
  inline void clear(IntVector& index) { subpatches_[getIndex(index)] = 0;}
  inline int operator[](const IntVector& idx) const {
    int x = subpatches_[getIndex(idx)];
    return x;
  }
  // array
  int *subpatches_;
  IntVector low_, high_, range_;
  int size_;

  // whether we allocated the memory ourselves or not.
  bool delSubpatches_;
};

typedef Handle<PatchFlag> PatchFlagP;
typedef Handle<SubPatchFlag> SubPatchFlagP;
}

#endif
