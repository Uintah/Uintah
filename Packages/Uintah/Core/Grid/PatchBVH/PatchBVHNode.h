#ifndef PATCH_BVH_NODE_H
#define PATCH_BVH_NODE_H


#include <Packages/Uintah/Core/Grid/PatchBVH/PatchBVHBase.h>
#include <vector>
namespace Uintah {

  /**************************************

    CLASS
    PatchBVHNode

    A Bounding Volume Hiearchy for querying patches that are 
    within a given range.  This class is a general node of the tree.

    GENERAL INFORMATION

    PatchBVHNode.h

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

  class PatchBVHNode : public PatchBVHBase
  {
    public:
      PatchBVHNode(std::vector<PatchKeyVal>::iterator begin, std::vector<PatchKeyVal>::iterator end);

      ~PatchBVHNode();

      void query(const IntVector& low, const IntVector& high, Level::selectType& patches);
    private:

      PatchBVHBase *left_, *right_;
  };

} // end namespace Uintah

#endif
