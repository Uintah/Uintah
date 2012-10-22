/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
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

#ifndef Uintah_Core_Grid_ComputeSet_special_cc
#define Uintah_Core_Grid_ComputeSet_special_cc

//
// ComputeSet_special.cc
//
//    Note, ComputeSet_special.cc is #include'd direclty in
// ComputeSet.h when using the PGI compilers...
//

#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Grid/Patch.h>
#include <algorithm>

namespace Uintah {
  
template<>
void ComputeSubset<const Patch*>::sort()
{
  std::sort(items.begin(), items.end(), Patch::Compare());
}

template<>  
bool ComputeSubset<const Patch*>::compareElems( const Patch* e1,
                                                const Patch* e2 )
{
  return Patch::Compare()(e1, e2); }
}

#endif // #ifdef Uintah_Core_Grid_ComputeSet_special_cc
       // this file needs to be included to instantiate the templates
       // for some compilers

