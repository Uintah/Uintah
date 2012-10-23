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


#ifndef UINTAH_HOMEBREW_PSPatchMatlGhost_H
#define UINTAH_HOMEBREW_PSPatchMatlGhost_H

#include <Core/Grid/Patch.h>
#include <Core/Geometry/IntVector.h>

using SCIRun::IntVector;

namespace Uintah {


    /**************************************
      
      struct
        PSPatchMatlGhost
      
        Patch, Material, Ghost info
        
      
      GENERAL INFORMATION
      
        PSPatchMatlGhost.h
      
        Bryan Worthen
        Department of Computer Science
        University of Utah
      
        Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
      
      
      KEYWORDS
        Patch, Material, Ghost
      
      DESCRIPTION
        Has all the important information for identifying a particle subset
          patch, material, and ghost properties
      
      WARNING
      
      ****************************************/

struct  PSPatchMatlGhost {
  PSPatchMatlGhost(const Patch* patch, int matl, int dwid, int count=1)
    : patch_(patch), matl_(matl), dwid_(dwid), count_(count)
  {}
  PSPatchMatlGhost(const PSPatchMatlGhost& copy)
    : patch_(copy.patch_), matl_(copy.matl_), dwid_(copy.dwid_), count_(copy.count_)
  {}
  
  bool operator<(const PSPatchMatlGhost& other) const;
  bool operator==(const PSPatchMatlGhost& other) const
  {
    return patch_==other.patch_ && matl_ == other.matl_ && dwid_ == other.dwid_;
  }
  bool operator!=(const PSPatchMatlGhost& other) const
  {
    return !operator==(other);
  }
  const Patch* patch_;
  int matl_;
  int dwid_;
  mutable int count_; //a count of how many times this has been created
};  

  ostream& operator<<(ostream &out, const PSPatchMatlGhost &pmg);
} // End namespace Uintah

#endif
