#include <Core/Containers/RangeTree.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Malloc/Allocator.h>

#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/Level.h>

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
      d_centerTimes2 = patch->getInteriorNodeLowIndex() + patch->getInteriorNodeHighIndex();
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
