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

#ifndef PATCH_UNSTRUCTURED_BVH_LEAF_H
#define PATCH_UNSTRUCTURED_BVH_LEAF_H


#include <Core/Grid/PatchBVH/UnstructuredPatchBVHBase.h>
#include <vector>

namespace Uintah {

  /**************************************

    CLASS
    UnstructuredPatchBVHLeaf

    A Bounding Volume Hiearchy for querying patches that are 
    within a given range.  This class is a leaf of the tree.

    GENERAL INFORMATION

    UnstructuredPatchBVHLeaf.h

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

  class UnstructuredPatchBVHLeaf : public UnstructuredPatchBVHBase
  {
    public:
      UnstructuredPatchBVHLeaf(std::vector<UnstructuredPatchKeyVal>::iterator begin, std::vector<UnstructuredPatchKeyVal>::iterator end);

      ~UnstructuredPatchBVHLeaf();

      void query(const IntVector& low, const IntVector& high, UnstructuredLevel::selectType& patches,bool includeExtraCells);
    private:
      std::vector<UnstructuredPatchKeyVal>::iterator begin_, end_;

  };

} // end namespace Uintah

#endif
