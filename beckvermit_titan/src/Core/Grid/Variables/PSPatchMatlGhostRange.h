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


#ifndef UINTAH_HOMEBREW_PSPatchMatlGhostRange_H
#define UINTAH_HOMEBREW_PSPatchMatlGhostRange_H

#include <Core/Grid/Patch.h>
#include <Core/Geometry/IntVector.h>

using SCIRun::IntVector;

namespace Uintah {


    /**************************************
      
      struct
        PSPatchMatlGhostRange
      
        Patch, Material, Ghost, Range info
        
      
      GENERAL INFORMATION
      
        PSPatchMatlGhostRange.h
      
        Bryan Worthen
        Department of Computer Science
        University of Utah
      
        Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
      
      
      KEYWORDS
        Patch, Material, Ghost, Range
      
      DESCRIPTION
        Has all the important information for identifying a particle subset
          patch, material, and ghost properties
      
      WARNING
      
      ****************************************/

struct  PSPatchMatlGhostRange {
  PSPatchMatlGhostRange(const Patch* patch, int matl, 
                   IntVector low, IntVector high, int dwid, int count=1)
    : patch_(patch), matl_(matl), low_(low), high_(high), dwid_(dwid), count_(count)
  {}
  PSPatchMatlGhostRange(const PSPatchMatlGhostRange& copy)
    : patch_(copy.patch_), matl_(copy.matl_), low_(copy.low_), 
       high_(copy.high_), dwid_(copy.dwid_), count_(copy.count_)
  {}
  
  bool operator<(const PSPatchMatlGhostRange& other) const;
  bool operator==(const PSPatchMatlGhostRange& other) const
  {
    return patch_==other.patch_ && matl_ == other.matl_ && low_ == other.low_ && high_ == other.high_ && dwid_ == other.dwid_;
  }
  bool operator!=(const PSPatchMatlGhostRange& other) const
  {
    return !operator==(other);
  }
  const Patch* patch_;
  int matl_;
  IntVector low_;
  IntVector high_;
  int dwid_;
  mutable int count_; //a count of how many times this has been created
};  

  ostream& operator<<(ostream &out, const PSPatchMatlGhostRange &pmg);
} // End namespace Uintah

#endif
