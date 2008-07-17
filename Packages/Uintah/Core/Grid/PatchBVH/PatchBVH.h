#ifndef PATCH_BVH_H
#define PATCH_BVH_H

#include <Packages/Uintah/Core/Grid/PatchBVH/PatchBVHBase.h>
#include <vector>

#include <iostream>
using namespace std;
namespace Uintah {

  /**************************************

    CLASS
    PatchBVH

    A Bounding Volume Hiearchy for querying patches that are 
    within a given range.

    GENERAL INFORMATION

    PatchBVH.h

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

  class PatchBVH
  {
    public:
      PatchBVH(const std::vector<const Patch*>& patches);

      ~PatchBVH();

      void query(const IntVector& low, const IntVector& high, Level::selectType& patches);
      
    private:

      PatchBVHBase *root_; //the root of the BVH tree
      std::vector<PatchBVHBase::PatchKeyVal>  patches_; //a copy of the patches passed in that can be reordered
  };

} // end namespace Uintah

#endif
