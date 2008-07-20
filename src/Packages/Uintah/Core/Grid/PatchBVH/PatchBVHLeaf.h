#ifndef PATCH_BVH_LEAF_H
#define PATCH_BVH_LEAF_H


#include <Packages/Uintah/Core/Grid/PatchBVH/PatchBVHBase.h>
#include <vector>

namespace Uintah {

  /**************************************

    CLASS
    PatchBVHLeaf

    A Bounding Volume Hiearchy for querying patches that are 
    within a given range.  This class is a leaf of the tree.

    GENERAL INFORMATION

    PatchBVHLeaf.h

    Justin Luitjens
    Department of Computer Science
    University of Utah

    Center for the Simulation of Accidental Fires and Explosions (C-SAFE)

    Copyright (C) 2008 SCI Group

    KEYWORDS
    PatchBVH

    DESCRIPTION
    The PatchBVH is used for querying patches within a given range.
    WARNING

   ****************************************/

  class PatchBVHLeaf : public PatchBVHBase
  {
    public:
      PatchBVHLeaf(std::vector<PatchKeyVal>::iterator begin, std::vector<PatchKeyVal>::iterator end);

      ~PatchBVHLeaf();

      void query(const IntVector& low, const IntVector& high, Level::selectType& patches);
    private:
      std::vector<PatchKeyVal>::iterator begin_, end_;

  };

} // end namespace Uintah

#endif
