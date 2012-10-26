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


#include <Core/Grid/Variables/Stencil4.h>
#include <Core/Disclosure/TypeDescription.h>
#include <Core/Util/Endian.h>
#include <Core/Util/FancyAssert.h>
#include <Core/Malloc/Allocator.h>
#include <sci_defs/mpi_defs.h> // For MPIPP_H on SGI

using namespace Uintah;

namespace Uintah {
  static MPI_Datatype makeMPI_Stencil4()
  {
    ASSERTEQ(sizeof(Stencil4), sizeof(double)*4);
    MPI_Datatype mpitype;
    MPI_Type_vector(1, 4, 4, MPI_DOUBLE, &mpitype);
    MPI_Type_commit(&mpitype);
    return mpitype;
  }
  
  const TypeDescription* fun_getTypeDescription(Stencil4*)
  {
    static TypeDescription* td = 0;
    if(!td){
      td = scinew TypeDescription(TypeDescription::Other,
                                  "Stencil4", true, 
                                  &makeMPI_Stencil4);
    }
    return td;
  }
  
  std::ostream & operator << (std::ostream &out, const Uintah::Stencil4 &a) {
    out << "A.p: " << a.p << " A.w: " << a.w << " A.s: " << a.s 
    << " A.b: " << a.b ;
    return out;
  }
  
  
}

namespace SCIRun {
  
  void swapbytes( Stencil4& a) {
    SWAP_8(a.p);
    SWAP_8(a.w);
    SWAP_8(a.s);
    SWAP_8(a.b);
  }
  
} // namespace SCIRun
