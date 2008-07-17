
#include <Packages/Uintah/Core/Grid/PatchBVH/PatchBVHLeaf.h>

namespace Uintah {

  /**************************************

    CLASS
    PatchBVHLeaf

    A Bounding Volume Hiearchy for querying patches that are 
    within a given range.  This class is a leaf of the tree.

    GENERAL INFORMATION

    PatchBVHLeaf.cc

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

  PatchBVHLeaf::PatchBVHLeaf(std::vector<PatchKeyVal>::iterator begin, std::vector<PatchKeyVal>::iterator end) : begin_(begin), end_(end)
  {
    //set bounding box
    low_=begin->patch->getExtraNodeLowIndex__New();
    high_=begin->patch->getExtraNodeHighIndex__New();

    for(std::vector<PatchKeyVal>::iterator iter=begin+1; iter<end; iter++)
    {
      low_=Min(low_,iter->patch->getExtraNodeLowIndex__New());
      high_=Max(high_,iter->patch->getExtraNodeHighIndex__New());
    }

  }

  PatchBVHLeaf::~PatchBVHLeaf()
  {
    //no need to delete anything
  }

  void PatchBVHLeaf::query(const IntVector& low, const IntVector& high, Level::selectType& patches)
  {
    //check that the query intersects my bounding box
    if(!intersects(low,high,low_,high_))
      return;

    //loop through lists individually
    for(std::vector<PatchKeyVal>::iterator iter=begin_;iter<end_;iter++)
    {
      //if patch intersects range
      if(intersects(low,high, iter->patch->getExtraNodeLowIndex__New(), iter->patch->getExtraNodeHighIndex__New()))
        patches.push_back(iter->patch); //add it to the list
    }
  }

} // end namespace Uintah
