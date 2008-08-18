#ifndef PATCH_BVH_BASE_H
#define PATCH_BVH_BASE_H


#include <Core/Geometry/IntVector.h>
#include <Core/Malloc/Allocator.h>

#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/Level.h>

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

  class PatchBVHBase 
  {
  public:
    PatchBVHBase() {};

    virtual ~PatchBVHBase() {} ;

    virtual void query(const IntVector& low, const IntVector& high, Level::selectType& patches)=0;

    static unsigned int getLeafSize() { return leafSize_; }
    static void setLeafSize(int leafSize) { leafSize_=leafSize; }

  protected:

    friend class PatchBVH;
    struct PatchKeyVal
    {
      bool operator<(const PatchKeyVal& p) const
      {
        return center2[PatchBVHBase::sortDim_]<p.center2[PatchBVHBase::sortDim_];
      }
      const Patch* patch;
      IntVector center2; //twice the center of the patch be be used by sorting
    };
    /**
     * Returns true if the given range intersects my volume
     */
    inline bool intersects(const IntVector& low, const IntVector &high)
    {
      return intersects(low,high,low_,high_);
    }

    /**
     * Returns true if the given ranges intersect
     */
    static inline bool intersects(const IntVector& low1, const IntVector &high1, const IntVector& low2, const IntVector high2)
    {
      return low1.x()<high2.x() && low1.y()<high2.y() && low1.z()<high2.z()    // intersect if low1 is less than high2 
        && high1.x()>low2.x() && high1.y()>low2.y() && high1.z()>low2.z();  // and high1 is greater than their low2
    }
    IntVector low_, high_;  //the bounding box for this node/leaf

    static unsigned int leafSize_;      //the number of patches in a leaf
    static int sortDim_;               //the current sorting dimension
  };
} // end namespace Uintah

#endif
