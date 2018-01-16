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


#ifndef UINTAH_HOMEBREW_CutCellInfoP_H
#define UINTAH_HOMEBREW_CutCellInfoP_H

#include <CCA/Components/MPMArches/CutCellInfo.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Util/Handle.h>

namespace Uintah {

template<class T> class Handle;
class CutCellInfo;
typedef Handle<CutCellInfo> CutCellInfoP;


  template<>
  inline void PerPatch<CutCellInfoP>::readNormal(std::istream& in, bool swapBytes)
  {
    // Note if swap bytes for CutCellInfoP is implemente then this
    // template specialization can be deleted as the general template
    // will work.
    SCI_THROW(InternalError("Swap bytes for CutCellInfoP is not implemented", __FILE__, __LINE__));

    ssize_t linesize = (ssize_t)(sizeof(CutCellInfoP));
    
    CutCellInfoP val;
    
    in.read((char*) &val, linesize);
    
    // if (swapBytes)
    //   Uintah::swapbytes(val);
    
    value = std::make_shared<CutCellInfoP>(val);
  }
} // End namespace Uintah

#endif
