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


#include <Core/Containers/RangeTree.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Malloc/Allocator.h>

#include <Core/Grid/Patch.h>
#include <Core/Grid/Level.h>


namespace Uintah {
  using namespace SCIRun;

/**************************************

CLASS
   PatchRangeTree

   Uses the RangeTree template to make a RangeTree to be used
   to query for patches within a given range.  Used by
   Level::selectPatches.

GENERAL INFORMATION

   PatchRangeTree.h

   Wayne Witzel
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   PatchRangeTree

DESCRIPTION
   The RangeTree is used to query the patches based on the
   patch centers, then each of those patches are checked using
   their dimensions to see if they are actually within the
   query range.
   
   Assuming that the average patch dimensions of the set of
   patches is not much less than the maximum patch dimensions
   the query is performed in about O(log(n)^2 + k) time where
   n is the number of patches to be searched and k is the number
   of those patches within the query.  The initialization time
   and storage space requirements are both O(n*log(n)^2).  
  
WARNING
  
****************************************/

class PatchRangeTree
{
public:
  PatchRangeTree(const std::vector<Patch*>& patches);
  PatchRangeTree(const std::vector<const Patch*>& patches);

  ~PatchRangeTree();
  
  void query(const IntVector& low, const IntVector& high,
             Level::selectType& patches);
private:
  class PatchPoint
  {
  public:
    PatchPoint()
      : d_patch(NULL) { }

    PatchPoint(IntVector centerTimes2)
      : d_patch(NULL), d_centerTimes2(centerTimes2) { }
    
    void setPatch(const Patch* patch)
    {
      d_patch = patch;
      d_centerTimes2 = patch->getNodeLowIndex() + patch->getNodeHighIndex();
    }
    
    int operator[](int i) const
    { return d_centerTimes2[i]; }

    const Patch* getPatch() const
    { return d_patch; }
  private:
    const Patch* d_patch;
    
    // center of the patch multiplied by 2
    IntVector d_centerTimes2;
  };

  RangeTree<PatchPoint, int>* d_rangeTree;
  IntVector d_maxPatchDimensions;

  // PatchPoint's vector is kept here mostly for memory management
  PatchPoint* d_patchPoints;
  int d_numPatches;
};

} // end namespace Uintah
