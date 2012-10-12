/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include <Core/Grid/PatchBVH/PatchBVHNode.h>
#include <Core/Grid/PatchBVH/PatchBVHLeaf.h>

namespace Uintah {

  /**************************************

    CLASS
    PatchBVHNode

    A Bounding Volume Hiearchy for querying patches that are 
    within a given range.  This class is a general node of the tree.

    GENERAL INFORMATION

    PatchBVHNode.cc

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

  PatchBVHNode::PatchBVHNode(std::vector<PatchKeyVal>::iterator begin, std::vector<PatchKeyVal>::iterator end) : left_(NULL), right_(NULL)
  {
    //set bounding box
    low_=begin->patch->getExtraCellLowIndex();
    high_=begin->patch->getExtraCellHighIndex();

    for(std::vector<PatchKeyVal>::iterator iter=begin+1; iter<end; iter++)
    {
       low_=Min(low_,iter->patch->getExtraCellLowIndex());
       high_=Max(high_,iter->patch->getExtraCellHighIndex());
    }

    //find maximum dimension
    IntVector range=high_-low_;
    int maxd=0;

    if(range[1]>range[maxd])
      maxd=1;
    if(range[2]>range[maxd])
      maxd=2;
     
    //sort on maiximum dimension
    switch(maxd)
    {
      case 0:
        sort(begin,end,PatchKeyCompare0);
        break;
      case 1:
        sort(begin,end,PatchKeyCompare1);
        break;
      case 2: 
        sort(begin,end,PatchKeyCompare2);
        break;
      default:
        //should not be possible
        break;
    }
    //split the list in half

    //create left and right nodes/leafs
    unsigned int size=(end-begin);
    unsigned int left_size=size/2;
    unsigned int right_size=size-left_size;

    if(left_size>PatchBVHBase::getLeafSize())
    {
      //create new node
      left_=new PatchBVHNode(begin,begin+left_size);
    }
    else
    {
      //create new leaf
      left_=new PatchBVHLeaf(begin,begin+left_size);
    }

    if(right_size>PatchBVHBase::getLeafSize())
    {
      //create new node
      right_=new PatchBVHNode(begin+left_size,end);
    }
    else
    {
      //create new leaf
      right_=new PatchBVHLeaf(begin+left_size,end);
    }
  }

  PatchBVHNode::~PatchBVHNode()
  {
    //this class should only be made if there are more than 2 objects in the list thus both sides should exist
    ASSERT(left_!=NULL);
    delete left_;
    
    ASSERT(right_!=NULL);
    delete right_;
  }

  void PatchBVHNode::query(const IntVector& low, const IntVector& high, Level::selectType& patches,bool includeExtraCells)
  {
    //check that the query intersects my bounding box
    if(!doesIntersect(low,high,low_,high_)) {
      return;
    }
    //intersect with left and right trees
    ASSERT(left_!=NULL);
    left_->query(low,high,patches,includeExtraCells);
    ASSERT(right_!=NULL);
    right_->query(low,high,patches,includeExtraCells);
  }

} // end namespace Uintah
