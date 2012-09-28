/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


#ifndef PATCH_BVH_BASE_H
#define PATCH_BVH_BASE_H


#include <Core/Geometry/IntVector.h>
#include <Core/Malloc/Allocator.h>

#include <Core/Grid/Patch.h>
#include <Core/Grid/Level.h>

namespace Uintah {

  /**************************************

    CLASS
      PatchBVHBase

      A Bounding Volume Hiearchy for querying patches that are 
      within a given range.  This class is a the base class for leafs and nodes.

    GENERAL INFORMATION

      PatchBVHBase.h

    Justin Luitjens
    Department of Computer Science
    University of Utah

    Center for the Simulation of Accidental Fires and Explosions (C-SAFE)

    Copyright (C) 2008 - University of Utah

    KEYWORDS
      PatchBVH

    DESCRIPTION
      The PatchBVH is used for querying patches within a given range.

    WARNING

   ****************************************/

  struct PatchKeyVal
  {
    const Patch* patch;
    IntVector center2; //twice the center of the patch be be used by sorting
  };
  
  inline bool PatchKeyCompare0(const PatchKeyVal& p1, const PatchKeyVal &p2)
  {
      return p1.center2[0]<p2.center2[0];
  }
  inline bool PatchKeyCompare1(const PatchKeyVal& p1, const PatchKeyVal &p2)
  {
      return p1.center2[1]<p2.center2[1];
  }
  inline bool PatchKeyCompare2(const PatchKeyVal& p1, const PatchKeyVal &p2)
  {
      return p1.center2[2]<p2.center2[2];
  }
  
  class PatchBVHBase 
  {
  public:
    PatchBVHBase() {};

    virtual ~PatchBVHBase() {} ;

    virtual void query(const IntVector& low, const IntVector& high, Level::selectType& patches, bool includeExtraCells)=0;

    static unsigned int getLeafSize() { return leafSize_; }
    static void setLeafSize(int leafSize) { leafSize_=leafSize; }

  protected:

    friend class PatchBVH;

    IntVector low_, high_;  //the bounding box for this node/leaf

    static unsigned int leafSize_;      //the number of patches in a leaf
  };
} // end namespace Uintah

#endif
