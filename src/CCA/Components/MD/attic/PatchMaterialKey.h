/*
 * The MIT License
 *
 * Copyright (c) 1997-2013 The University of Utah
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

#ifndef UINTAH_HMD_PATCHMATERIALKEY_H
#define UINTAH_HMD_PATCHMATERIALKEY_H

#include <Core/Grid/Patch.h>

namespace Uintah {

  struct PatchMaterialKey {

      PatchMaterialKey(const Patch* patchID,
                       int matlIndex) :
          d_patchID(patchID), d_matlIndex(matlIndex)
      {
      }

      PatchMaterialKey(const PatchMaterialKey& copy) :
          d_patchID(copy.d_patchID), d_matlIndex(copy.d_matlIndex)
      {
      }

      PatchMaterialKey& operator=(const PatchMaterialKey& copy)
      {
        d_patchID = copy.d_patchID;
        d_matlIndex = copy.d_matlIndex;
        return *this;
      }

      bool operator<(const PatchMaterialKey& other) const
      {
        if (d_patchID == other.d_patchID) {
          return d_matlIndex < other.d_matlIndex;
        } else {
          return (d_patchID < other.d_patchID);
        }
      };

      bool operator==(const PatchMaterialKey& other) const
      {
        return (d_patchID == other.d_patchID) && (d_matlIndex == other.d_matlIndex);
      };

      const Patch* d_patchID;
      int d_matlIndex;
  };

}  // End namespace Uintah

#endif
