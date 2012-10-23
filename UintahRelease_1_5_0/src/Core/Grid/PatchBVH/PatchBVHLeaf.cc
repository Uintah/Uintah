/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include <Core/Grid/PatchBVH/PatchBVHLeaf.h>

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


    KEYWORDS
    PatchBVH

    DESCRIPTION
    The PatchBVH is used for querying patches within a given range.
    WARNING

   ****************************************/

  PatchBVHLeaf::PatchBVHLeaf(std::vector<PatchKeyVal>::iterator begin, std::vector<PatchKeyVal>::iterator end) : begin_(begin), end_(end)
  {
    //set bounding box
    low_=begin->patch->getExtraCellLowIndex();
    high_=begin->patch->getExtraCellHighIndex();

    for(std::vector<PatchKeyVal>::iterator iter=begin+1; iter<end; iter++)
    {
      low_=Min(low_,iter->patch->getExtraCellLowIndex());
      high_=Max(high_,iter->patch->getExtraCellHighIndex());
    }

  }

  PatchBVHLeaf::~PatchBVHLeaf()
  {
    //no need to delete anything
  }

  void PatchBVHLeaf::query(const IntVector& low, const IntVector& high, Level::selectType& patches, bool includeExtraCells)
  {
    //check that the query intersects my bounding box
    if(!doesIntersect(low,high,low_,high_))
      return;

    //loop through lists individually
    for(std::vector<PatchKeyVal>::iterator iter=begin_;iter<end_;iter++)
    {
      if(includeExtraCells)
      {
        //if patch intersects range
        if(doesIntersect(low,high, iter->patch->getExtraCellLowIndex(), iter->patch->getExtraCellHighIndex()))
          patches.push_back(iter->patch); //add it to the list
      }
      else
      {
        //if patch intersects range
        if(doesIntersect(low,high, iter->patch->getCellLowIndex(), iter->patch->getCellHighIndex()))
          patches.push_back(iter->patch); //add it to the list
      }
    }
  }

} // end namespace Uintah
