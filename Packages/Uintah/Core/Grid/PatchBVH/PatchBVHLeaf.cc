
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

  PatchBVHLeaf::PatchBVHLeaf(std::vector<PatchKeyVal> &patches, unsigned int begin, unsigned int end) : patches_(patches), begin_(begin), end_(end)
  {
    //set bounding box
    low_=patches[begin].patch->getCellLowIndex__New();
    high_=patches[begin].patch->getCellHighIndex__New();

    for(unsigned int i=begin+1; i<end; i++)
    {
      low_=Min(low_,patches[i].patch->getCellLowIndex__New());
      high_=Max(high_,patches[i].patch->getCellHighIndex__New());
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
    for(int i=begin_;i<end_;i++)
    {
      //if patch intersects range
      if(intersects(low,high, patches_[i].patch->getCellLowIndex__New(), patches_[i].patch->getCellHighIndex__New()))
        patches.push_back(patches_[i].patch); //add it to the list
    }
  }

} // end namespace Uintah
