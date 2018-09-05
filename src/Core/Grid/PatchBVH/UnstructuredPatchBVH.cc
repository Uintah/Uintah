/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
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


#include <Core/Grid/PatchBVH/UnstructuredPatchBVH.h>
#include <Core/Grid/PatchBVH/UnstructuredPatchBVHNode.h>
#include <Core/Grid/PatchBVH/UnstructuredPatchBVHLeaf.h>
namespace Uintah {
 

  /**************************************

    CLASS
    UnstructuredPatchBVH

    A Bounding Volume Hiearchy for querying patches that are 
    within a given range.

    GENERAL INFORMATION

    UnstructuredPatchBVH.h

    Justin Luitjens
    Department of Computer Science
    University of Utah

    Center for the Simulation of Accidental Fires and Explosions (C-SAFE)


    KEYWORDS
    UnstructuredPatchBVH

    DESCRIPTION
    The UnstructuredPatchBVH is used for querying patches within a given range.
    WARNING

   ****************************************/
  UnstructuredPatchBVH::UnstructuredPatchBVH(const std::vector<const UnstructuredPatch*>& patches) : root_(nullptr)
  {
    if(patches.size()==0)
      return;
    
    for(std::vector<const UnstructuredPatch*>::const_iterator iter=patches.begin();iter<patches.end();iter++)
    {
      UnstructuredPatchKeyVal key;

      key.patch=*iter;
      key.center2=(*iter)->getExtraCellLowIndex()+(*iter)->getExtraCellHighIndex();

      patches_.push_back(key);
    }


    if(patches_.size()>UnstructuredPatchBVHBase::getLeafSize())
    {
      //create a node
      root_=new UnstructuredPatchBVHNode(patches_.begin(), patches_.end());
    }
    else
    {
      //create a leaf
      root_=new UnstructuredPatchBVHLeaf(patches_.begin(), patches_.end());
    }
  }
  
  UnstructuredPatchBVH::UnstructuredPatchBVH(const std::vector<UnstructuredPatch*>& patches) : root_(nullptr)
  {
    if(patches.size()==0)
      return;
    
    for(std::vector<UnstructuredPatch*>::const_iterator iter=patches.begin();iter<patches.end();iter++)
    {
      UnstructuredPatchKeyVal key;

      key.patch=*iter;
      key.center2=(*iter)->getExtraCellLowIndex()+(*iter)->getExtraCellHighIndex();

      patches_.push_back(key);
    }


    if(patches_.size()>UnstructuredPatchBVHBase::getLeafSize())
    {
      //create a node
      root_=new UnstructuredPatchBVHNode(patches_.begin(), patches_.end());
    }
    else
    {
      //create a leaf
      root_=new UnstructuredPatchBVHLeaf(patches_.begin(), patches_.end());
    }
  }

  UnstructuredPatchBVH::~UnstructuredPatchBVH()
  {
    if(root_!=nullptr)
      delete root_;

    patches_.clear();
  }

  void UnstructuredPatchBVH::query(const IntVector& low, const IntVector& high, UnstructuredLevel::selectType& patches, bool includeExtraCells)
  {
    //verify query range is valid
    if(high.x()<=low.x() || high.y()<=low.y() || high.z()<=low.z())
      return;

    if(root_==nullptr)
      return;
    
    root_->query(low,high,patches,includeExtraCells);
  }

} // end namespace Uintah
