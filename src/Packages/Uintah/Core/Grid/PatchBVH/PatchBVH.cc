
#include <Packages/Uintah/Core/Grid/PatchBVH/PatchBVH.h>
#include <Packages/Uintah/Core/Grid/PatchBVH/PatchBVHNode.h>
#include <Packages/Uintah/Core/Grid/PatchBVH/PatchBVHLeaf.h>
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
  PatchBVH::PatchBVH(const std::vector<const Patch*>& patches)
  {
    for(std::vector<const Patch*>::const_iterator iter=patches.begin();iter<patches.end();iter++)
    {
      PatchBVHBase::PatchKeyVal key;

      key.patch=*iter;
      key.center= ((*iter)->getCellLowIndex__New()+(*iter)->getCellHighIndex__New())*IntVector(.5,.5,.5);

      patches_.push_back(key);
    }


    if(patches_.size()>PatchBVHBase::getLeafSize())
    {
      //create a node
      root_=new PatchBVHNode(patches_.begin(), patches_.end());
    }
    else
    {
      //create a leaf
      root_=new PatchBVHLeaf(patches_.begin(), patches_.end());
    }
  }
  
  PatchBVH::PatchBVH(const std::vector<Patch*>& patches)
  {
    for(std::vector<Patch*>::const_iterator iter=patches.begin();iter<patches.end();iter++)
    {
      PatchBVHBase::PatchKeyVal key;

      key.patch=*iter;
      key.center= ((*iter)->getCellLowIndex__New()+(*iter)->getCellHighIndex__New())*IntVector(.5,.5,.5);

      patches_.push_back(key);
    }


    if(patches_.size()>PatchBVHBase::getLeafSize())
    {
      //create a node
      root_=new PatchBVHNode(patches_.begin(), patches_.end());
    }
    else
    {
      //create a leaf
      root_=new PatchBVHLeaf(patches_.begin(), patches_.end());
    }
  }

  PatchBVH::~PatchBVH()
  {
    if(root_!=NULL)
      delete root_;

    patches_.clear();
  }

  void PatchBVH::query(const IntVector& low, const IntVector& high, Level::selectType& patches)
  {
    //verify query range is valid
    if(high.x()<=low.x() || high.y()<=low.y() || high.z()<=low.z())
      return;

    patches.resize(0);
    root_->query(low,high,patches);
  }

} // end namespace Uintah
